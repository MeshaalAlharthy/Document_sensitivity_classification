from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedGroupKFold

logger = logging.getLogger(__name__)

LABELS = ["سري للغاية", "سري", "مقيد", "عام"]

_DIACRITICS = re.compile(r"[ؗ-ًؚ-ْٰ]")
_ALEF = re.compile(r"[إأآٱ]")
_SPACES = re.compile(r"\s+")

MISSING = {"", "-", "—", "nan", "none", "null", "n/a", "na"}
UNKNOWN_DEPARTMENT = "غير محدد"
CORE_FIELDS = ("وصف العملية", "الكلمات المفتاحية")

DEPARTMENT_PREFIX = "الإدارة"
TARGET_PREFIX = "رأي اللجنة"
FIELD_PREFIXES = [
    ("القسم", "القسم"),
    ("وصف العملية", "وصف العملية"),
    ("النماذج المرتبطة", "النماذج"),
    ("هل تحتوي البيانات", "تحتوي بيانات شخصية"),
    ("نوع البيانات الشخصية", "نوع البيانات الشخصية"),
    ("المعالجة و الاستخدام", "استخدام البيانات الشخصية"),
    ("نوع التأثير", "نوع التأثير"),
    ("توضيح الكلمات", "الكلمات المفتاحية"),
    ("هل يتم مشاركة", "مشاركة خارجية"),
    ("من المخول", "المخول بالمشاركة"),
    ("هل يتم رفع", "أنظمة خارجية"),
    ("اسم الجهة الخارجية", "الجهة الخارجية"),
]


def normalize(text: str) -> str:
    text = _DIACRITICS.sub("", text).replace("ـ", "")
    text = _ALEF.sub("ا", text)
    return _SPACES.sub(" ", text).strip()


def department_key(name: str) -> str:
    key = normalize(name).replace("ة", "ه").replace("ى", "ي")
    return _SPACES.sub(" ", re.sub(r"\d+", "", key)).strip()


def cell(value: object) -> str:
    if pd.isna(value):
        return ""
    text = normalize(str(value))
    return "" if text.lower() in MISSING else text


def find_column(columns: list[str], prefix: str) -> str:
    target = normalize(prefix)
    matches = [c for c in columns if normalize(c).startswith(target)]
    if len(matches) != 1:
        raise KeyError(f"expected one column starting with '{prefix}', found {len(matches)}")
    return matches[0]


def build_text(row: pd.Series, fields: list[tuple[str, str]]) -> str:
    parts = [f"{name}: {row[column]}" for column, name in fields if row[column]]
    return " [SEP] ".join(parts)


def duplicate_groups(texts: pd.Series, threshold: float) -> np.ndarray:
    vectors = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit_transform(texts)
    adjacency = cosine_similarity(vectors) >= threshold
    _, groups = connected_components(adjacency, directed=False)
    return groups


def clean(
    raw: pd.DataFrame,
    department: str,
    target: str,
    fields: list[tuple[str, str]],
) -> tuple[pd.DataFrame, dict[str, int]]:
    table = raw.astype(object).map(cell)
    report = {"rows_in": len(table)}

    deduplicated = table.drop_duplicates()
    report["exact_duplicates"] = len(table) - len(deduplicated)
    table = deduplicated

    label_ids = {normalize(name): i for i, name in enumerate(LABELS)}
    labels = table[target].map(label_ids)
    report["missing_label"] = int(labels.isna().sum())
    table = table[labels.notna()].assign(**{target: labels[labels.notna()].astype(int)})

    core = [column for column, name in fields if name in CORE_FIELDS]
    empty_core = (table[core] == "").all(axis=1)
    report["empty_core_fields"] = int(empty_core.sum())
    table = table[~empty_core]

    report["missing_department"] = int((table[department] == "").sum())
    table[department] = table[department].replace("", UNKNOWN_DEPARTMENT)

    report["rows_out"] = len(table)
    return table.reset_index(drop=True), report


def load_data(path: str, duplicate_threshold: float = 0.9) -> pd.DataFrame:
    raw = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=True)
    columns = list(raw.columns)
    department = find_column(columns, DEPARTMENT_PREFIX)
    target = find_column(columns, TARGET_PREFIX)
    fields = [(find_column(columns, prefix), name) for prefix, name in FIELD_PREFIXES]

    table, report = clean(raw, department, target, fields)
    logger.info("cleaning report: %s", report)

    df = pd.DataFrame({
        "row_id": np.arange(len(table)),
        "text": table.apply(build_text, axis=1, fields=fields),
        "department": table[department].map(department_key),
        "label": table[target].astype(int).to_numpy(),
    })
    df["group"] = duplicate_groups(df["text"], duplicate_threshold)
    logger.info("%d rows, %d departments, %d duplicate groups", len(df), df["department"].nunique(), df["group"].nunique())
    return df


@dataclass(frozen=True)
class Fold:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def department_units(df: pd.DataFrame) -> np.ndarray:
    unknown = df["department"] == department_key(UNKNOWN_DEPARTMENT)
    units = pd.factorize(df["department"].where(~unknown, "row " + df["row_id"].astype(str)))[0]
    groups = units.max() + 1 + df["group"].to_numpy()
    size = groups.max() + 1
    links = coo_matrix((np.ones(len(df)), (units, groups)), shape=(size, size))
    _, components = connected_components(links, directed=False)
    return components[units]


def make_folds(
    df: pd.DataFrame,
    n_splits: int = 5,
    val_fraction: float = 0.15,
    seed: int = 42,
    by_department: bool = False,
) -> list[Fold]:
    labels = df["label"].to_numpy()
    groups = df["group"].to_numpy()
    outer_groups = department_units(df) if by_department else groups
    outer = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    inner = StratifiedGroupKFold(n_splits=round(1 / val_fraction), shuffle=True, random_state=seed)
    folds = []
    for dev, test in outer.split(np.zeros(len(df)), labels, outer_groups):
        train, val = next(inner.split(np.zeros(len(dev)), labels[dev], groups[dev]))
        folds.append(Fold(train=dev[train], val=dev[val], test=test))
    return folds


def oversample(train: pd.DataFrame, ratio: float, seed: int) -> pd.DataFrame:
    counts = train["label"].value_counts()
    target = int(counts.max() * ratio)
    extra = [
        train[train["label"] == label].sample(target - count, replace=True, random_state=seed)
        for label, count in counts.items()
        if count < target
    ]
    return pd.concat([train, *extra], ignore_index=True)
