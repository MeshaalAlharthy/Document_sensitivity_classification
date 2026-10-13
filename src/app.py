from __future__ import annotations

import argparse
import time
from html import escape
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

from src.data import LABELS, find_column
from src.models import BertClassifier, apply_temperature
from src.predict import prepare

EXAMPLES = Path("examples/new_transactions.csv")
BATCH_EXAMPLE = Path("examples/batch_transactions.csv")
FONTS = Path(__file__).resolve().parent.parent / "assets" / "fonts"
NAVY = "#1b2a41"
FONT_FACES = "".join(
    f"@font-face {{ font-family: 'Plex Arabic'; font-weight: {weight}; src: url('/gradio_api/file={FONTS / name}') format('truetype'); }}\n"
    for name, weight in [("IBMPlexSansArabic-Regular.ttf", 400), ("IBMPlexSansArabic-Medium.ttf", 500), ("IBMPlexSansArabic-Bold.ttf", 700)]
)
CSS = FONT_FACES + """
* { font-family: 'Plex Arabic', sans-serif !important; letter-spacing: 0 !important; }
.gradio-container { direction: rtl; max-width: 1320px !important; margin: auto; }
footer { display: none !important; }
#header { direction: rtl; text-align: right; background: #1b2a41; color: #ffffff; padding: 22px 28px; border-radius: 10px; border-bottom: 4px solid #b08d57; }
#header h1 { color: #ffffff; margin: 0 0 6px 0; font-size: 26px; }
#header p { color: #c9d3e0; margin: 0; font-size: 14px; }
.result { direction: rtl; text-align: right; background: #ffffff; border: 1px solid #d5dbe3; border-radius: 10px; padding: 26px 30px; color: #1b2a41; }
.result .caption { font-size: 15px; color: #5b6b80; margin-bottom: 4px; }
.result .level { font-size: 40px; font-weight: 700; margin-bottom: 14px; }
.result .decision { font-size: 17px; font-weight: 600; padding: 12px 16px; border-radius: 8px; margin-bottom: 24px; }
.result .ok { background: #e7f3ec; color: #1d5c3a; }
.result .review { background: #fdf1e3; color: #8a4b0f; }
.result .row { margin-bottom: 16px; }
.result .head { display: flex; justify-content: space-between; font-size: 17px; margin-bottom: 6px; }
.result .track { background: #e9edf2; border-radius: 6px; height: 14px; overflow: hidden; }
.result .fill { height: 100%; border-radius: 6px; }
.result .empty { font-size: 17px; color: #5b6b80; text-align: center; padding: 60px 0; }
.result .stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 10px 0 20px; }
.result .stat { background: #f4f6f9; border-radius: 8px; padding: 14px 16px; }
.result .stat .value { font-size: 28px; font-weight: 700; }
.result .stat .label { font-size: 13px; color: #5b6b80; }
.result .chips { display: flex; flex-wrap: wrap; gap: 10px; }
.result .chip { color: #ffffff; padding: 6px 14px; border-radius: 16px; font-size: 15px; font-weight: 600; }
@media (max-width: 700px) { .result .stats { grid-template-columns: repeat(2, 1fr); } }
.batch { direction: rtl; background: #ffffff; border: 1px solid #d5dbe3; border-radius: 10px; overflow-x: auto; }
.batch table { direction: rtl; width: 100%; border-collapse: collapse; color: #1b2a41; font-size: 15px; }
.batch th { background: #f4f6f9; text-align: right; font-weight: 700; padding: 13px 16px; border-bottom: 1px solid #d5dbe3; white-space: nowrap; }
.batch td { text-align: right; padding: 12px 16px; border-bottom: 1px solid #e9edf2; }
.batch tr:last-child td { border-bottom: none; }
.batch tbody tr:nth-child(even) td { background: #fafbfc; }
.batch .num { color: #5b6b80; width: 40px; }
.batch .level { font-weight: 700; white-space: nowrap; }
.batch .ok { color: #1d5c3a; white-space: nowrap; }
.batch .review { color: #8a4b0f; font-weight: 600; white-space: nowrap; }
.note { direction: rtl; text-align: right; background: #f4f6f9; border-right: 3px solid #b08d57; border-radius: 6px; padding: 10px 14px; font-size: 14px; color: #5b6b80; }
"""
FORCE_LIGHT = """() => {
  const url = new URL(window.location);
  if (url.searchParams.get('__theme') !== 'light') { url.searchParams.set('__theme', 'light'); window.location.href = url.href; }
}"""
COLORS = {"سري للغاية": "#8b1e2d", "سري": "#b5651d", "مقيد": "#1f4e79", "عام": "#2e7d4f"}
PLACEHOLDER = "<div class='result'><div class='empty'>أدخل بيانات المعاملة أو اختر مثالًا، ثم اضغط «تصنيف المعاملة»</div></div>"
BATCH_PLACEHOLDER = "<div class='result'><div class='empty'>أرفق سجل معاملات بصيغة CSV أو اختر الملف التوضيحي، ثم اضغط «تصنيف السجل»</div></div>"
NOTE = "<div class='note'>ملاحظة: جميع البيانات في الأمثلة اصطناعية لغرض الاستعراض، ولا تتبع لأي جهة.</div>"


def render(probs: np.ndarray, review_below: float) -> str:
    best = int(probs.argmax())
    confidence = probs[best]
    if confidence < review_below:
        decision = (
            f"<div class='decision review'>⚠️ تُحال للجنة للمراجعة: الثقة <bdi>{confidence:.0%}</bdi> "
            f"أقل من حد المراجعة <bdi>{review_below:.0%}</bdi></div>"
        )
    else:
        decision = f"<div class='decision ok'>✓ تصنيف تلقائي بثقة <bdi>{confidence:.0%}</bdi></div>"
    rows = "".join(
        f"<div class='row'><div class='head'><span>{LABELS[i]}</span><bdi>{probs[i]:.0%}</bdi></div>"
        f"<div class='track'><div class='fill' style='width:{probs[i] * 100:.1f}%;background:{COLORS[LABELS[i]]}'></div></div></div>"
        for i in np.argsort(-probs)
    )
    return (
        f"<div class='result'><div class='caption'>التصنيف المتوقع</div>"
        f"<div class='level' style='color:{COLORS[LABELS[best]]}'>{LABELS[best]}</div>{decision}{rows}</div>"
    )


def render_batch(levels: list[str], confidence: np.ndarray, elapsed: float, review_below: float) -> str:
    stats = [
        (len(levels), "عدد المعاملات"),
        (f"{elapsed:.1f} ث", "الزمن الكلي"),
        (f"{elapsed / len(levels):.2f} ث", "متوسط المعاملة"),
        (int((confidence < review_below).sum()), "تُحال للمراجعة"),
    ]
    cards = "".join(f"<div class='stat'><div class='value'><bdi>{v}</bdi></div><div class='label'>{k}</div></div>" for v, k in stats)
    chips = "".join(
        f"<span class='chip' style='background:{COLORS[name]}'>{name}: <bdi>{levels.count(name)}</bdi></span>"
        for name in LABELS if levels.count(name)
    )
    return f"<div class='result'><div class='caption'>ملخص السجل</div><div class='stats'>{cards}</div><div class='chips'>{chips}</div></div>"


def render_table(departments: list[str], descriptions: list[str], levels: list[str], confidence: np.ndarray, review_below: float) -> str:
    head = "".join(f"<th>{name}</th>" for name in ["#", "الإدارة", "وصف العملية", "التصنيف المتوقع", "الثقة", "القرار"])
    rows = "".join(
        f"<tr><td class='num'><bdi>{i}</bdi></td><td>{escape(dept)}</td><td>{escape(desc)}</td>"
        f"<td class='level' style='color:{COLORS[level]}'>{level}</td><td><bdi>{conf:.0%}</bdi></td>"
        + ("<td class='review'>⚠️ مراجعة اللجنة</td>" if conf < review_below else "<td class='ok'>✓ تلقائي</td>")
        + "</tr>"
        for i, (dept, desc, level, conf) in enumerate(zip(departments, descriptions, levels, confidence), start=1)
    )
    return f"<div class='batch'><table dir='rtl'><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>"
THEME = gr.themes.Base(primary_hue="slate", secondary_hue="slate", neutral_hue="slate", radius_size="sm").set(
    button_primary_background_fill=NAVY,
    button_primary_background_fill_hover="#2c4166",
    button_primary_text_color="#ffffff",
    color_accent_soft="#e3e8ef",
)
DEFAULT_MODELS = ["results/arabert_focal", "results/arabert_focal_seed1", "results/arabert_focal_seed2"]
INPUTS = [
    ("الإدارة", "الإدارة"),
    ("القسم", "القسم"),
    ("وصف العملية", "وصف العملية"),
    ("الكلمات المفتاحية", "توضيح الكلمات"),
    ("نوع البيانات الشخصية", "نوع البيانات الشخصية"),
    ("نوع التأثير المتوقع", "نوع التأثير"),
    ("مشاركة البيانات مع جهة خارجية", "هل يتم مشاركة"),
    ("الجهة الخارجية", "اسم الجهة الخارجية"),
]


class Ensemble:
    def __init__(self, model_dirs: list[str]) -> None:
        paths = sorted(fold for d in model_dirs for fold in Path(d).glob("fold_*"))
        if not paths:
            raise FileNotFoundError(f"no fold_* models found in {model_dirs}")
        self.model = BertClassifier.load(paths[0])
        self.members = [torch.load(p / "model.pt", map_location="cpu") for p in paths]

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        total = np.zeros((len(df), len(LABELS)))
        for member in self.members:
            self.model.network.load_state_dict(member["state"], strict=False)
            total += apply_temperature(self.model.predict_proba(df), member["temperature"])
        return total / len(self.members)


def ensemble_row(model_dirs: list[str]) -> dict | None:
    files = [Path(d) / "predictions.csv" for d in model_dirs]
    if not all(f.exists() for f in files):
        return None
    columns = [f"p_{k}" for k in range(len(LABELS))]
    runs = [pd.read_csv(f).set_index("row_id").sort_index() for f in files]
    merged = runs[0][["fold", "label"]].copy()
    merged["pred"] = np.mean([r[columns].to_numpy() for r in runs], axis=0).argmax(axis=1)
    scores = {"macro_f1": [], "accuracy": [], **{f"f1 {name}": [] for name in LABELS}}
    for _, fold in merged.groupby("fold"):
        scores["macro_f1"].append(f1_score(fold["label"], fold["pred"], labels=range(len(LABELS)), average="macro"))
        scores["accuracy"].append(accuracy_score(fold["label"], fold["pred"]))
        per_class = f1_score(fold["label"], fold["pred"], labels=range(len(LABELS)), average=None, zero_division=0)
        for name, score in zip(LABELS, per_class):
            scores[f"f1 {name}"].append(score)
    return {"experiment": "arabert_ensemble (final)", **{k: f"{np.mean(v):.3f} ± {np.std(v):.3f}" for k, v in scores.items()}}


def results_table(model_dirs: list[str]) -> pd.DataFrame:
    path = Path("results/summary.csv")
    table = pd.read_csv(path) if path.exists() else pd.DataFrame()
    columns = ["experiment", "macro_f1", "accuracy"] + [f"f1 {name}" for name in LABELS]
    table = table[[c for c in columns if c in table.columns]]
    row = ensemble_row(model_dirs)
    if row:
        table = pd.concat([pd.DataFrame([row]), table], ignore_index=True)
    names = {"experiment": "التجربة", "macro_f1": "Macro-F1", "accuracy": "الدقة"}
    return table.rename(columns=lambda c: names.get(c, c.replace("f1 ", "F1 ")))


def build(model_dirs: list[str], review_below: float) -> gr.Blocks:
    ensemble = Ensemble(model_dirs)
    header = list(pd.read_csv(EXAMPLES, encoding="utf-8-sig", nrows=0).columns)
    targets = [find_column(header, prefix) for _, prefix in INPUTS]
    has_pd = find_column(header, "هل تحتوي البيانات")
    examples = pd.read_csv(EXAMPLES, encoding="utf-8-sig", dtype=str).fillna("")[targets].values.tolist()

    def classify(*values: str) -> str:
        row = dict.fromkeys(header, "")
        row.update(zip(targets, values))
        pd_text = row[targets[4]].strip()
        row[has_pd] = "نعم" if pd_text and pd_text != "لا يوجد" else "لا"
        return render(ensemble.predict(prepare(pd.DataFrame([row])))[0], review_below)

    def classify_file(path: str | None):
        if not path:
            raise gr.Error("أرفق ملف CSV أولًا")
        try:
            raw = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
            columns = list(raw.columns)
            department, description = find_column(columns, "الإدارة"), find_column(columns, "وصف العملية")
        except (KeyError, ValueError, UnicodeDecodeError, pd.errors.ParserError):
            raise gr.Error("تعذّر قراءة الملف: تأكد أنه CSV بنفس أعمدة نموذج المعاملات")
        if raw.empty:
            raise gr.Error("الملف لا يحتوي على معاملات")
        start = time.perf_counter()
        probs = ensemble.predict(prepare(raw))
        elapsed = time.perf_counter() - start
        confidence = probs.max(axis=1)
        levels = [LABELS[i] for i in probs.argmax(axis=1)]
        table = render_table(
            raw[department].fillna("").tolist(), raw[description].fillna("").tolist(), levels, confidence, review_below
        )
        return render_batch(levels, confidence, elapsed, review_below), table

    with gr.Blocks(title="Arabic Sensitivity Classification") as demo:
        gr.HTML(
            "<div id='header'><h1>نظام تصنيف حساسية المعاملات</h1>"
            "<p>المنصة الذكية لاكتشاف وتصنيف سرية الوثائق حسب حساسية المحتويات: سري للغاية · سري · مقيد · عام</p></div>"
        )
        with gr.Tab("تصنيف معاملة"):
            with gr.Row():
                with gr.Column(scale=5):
                    fields = []
                    for label, _ in INPUTS:
                        if label.startswith("مشاركة"):
                            fields.append(gr.Radio(["نعم", "لا"], value="لا", label=label))
                        else:
                            fields.append(gr.Textbox(label=label, rtl=True, lines=2 if label == "وصف العملية" else 1))
                    button = gr.Button("تصنيف المعاملة", variant="primary")
                with gr.Column(scale=6):
                    result = gr.HTML(PLACEHOLDER)
                    gr.HTML(NOTE)
            gr.Examples(examples, inputs=fields, label="أمثلة توضيحية")
            button.click(classify, inputs=fields, outputs=result)
        with gr.Tab("تصنيف سجل معاملات"):
            with gr.Row():
                with gr.Column(scale=5):
                    upload = gr.File(label="سجل المعاملات (CSV)", file_types=[".csv"], type="filepath")
                    batch_button = gr.Button("تصنيف السجل", variant="primary")
                    gr.Examples([[str(BATCH_EXAMPLE)]], inputs=upload, label="ملف توضيحي")
                with gr.Column(scale=6):
                    summary = gr.HTML(BATCH_PLACEHOLDER)
                    gr.HTML(NOTE)
            batch_table = gr.HTML("")
            batch_button.click(classify_file, inputs=upload, outputs=[summary, batch_table])
        with gr.Tab("نتائج التقييم"):
            gr.Markdown("تقييم بالتحقق المتقاطع (5 أجزاء) على معاملات لم يرها النموذج، مع منع تسرب المعاملات المتشابهة بين التدريب والاختبار.", rtl=True)
            gr.Dataframe(results_table(model_dirs), interactive=False)
    return demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--review-below", type=float, default=0.6)
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    gr.set_static_paths(paths=[FONTS])
    build(args.models, args.review_below).launch(server_port=args.port, css=CSS, theme=THEME, js=FORCE_LIGHT)


if __name__ == "__main__":
    main()
