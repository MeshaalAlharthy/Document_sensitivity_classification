from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import DEPARTMENT_PREFIX, FIELD_PREFIXES, LABELS, UNKNOWN_DEPARTMENT, build_text, cell, department_key, find_column
from src.models import BertClassifier, apply_temperature


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    columns = list(raw.columns)
    department = find_column(columns, DEPARTMENT_PREFIX)
    fields = [(find_column(columns, prefix), name) for prefix, name in FIELD_PREFIXES]
    table = raw.astype(object).map(cell)
    return pd.DataFrame({
        "text": table.apply(build_text, axis=1, fields=fields),
        "department": table[department].replace("", UNKNOWN_DEPARTMENT).map(department_key),
    })


def predict(df: pd.DataFrame, model_dirs: list[Path]) -> np.ndarray:
    folds = sorted(fold for model_dir in model_dirs for fold in model_dir.glob("fold_*"))
    if not folds:
        raise FileNotFoundError(f"no fold_* models found in {[str(d) for d in model_dirs]}")
    total = np.zeros((len(df), len(LABELS)))
    for fold in folds:
        model = BertClassifier.load(fold)
        total += apply_temperature(model.predict_proba(df), model.temperature)
    return total / len(folds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument(
        "--models", nargs="+", default=["results/arabert_focal", "results/arabert_focal_seed1", "results/arabert_focal_seed2"]
    )
    parser.add_argument("--output", default="predictions_new.csv")
    parser.add_argument("--review-below", type=float, default=0.6)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    raw = pd.read_csv(args.input, encoding="utf-8-sig", dtype=str)
    probs = predict(prepare(raw), [Path(m) for m in args.models])
    confidence = probs.max(axis=1)
    result = pd.DataFrame({
        "department": raw[find_column(list(raw.columns), DEPARTMENT_PREFIX)],
        "predicted_level": [LABELS[i] for i in probs.argmax(axis=1)],
        "confidence": confidence.round(2),
        "needs_review": np.where(confidence < args.review_below, "yes", "no"),
    })
    pd.concat([raw, result.drop(columns="department")], axis=1).to_csv(args.output, index=False, encoding="utf-8-sig")
    print(result.to_string(index=False))
    print(f"\nsaved to {args.output}")


if __name__ == "__main__":
    main()
