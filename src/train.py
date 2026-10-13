from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from src.data import LABELS, load_data, make_folds, oversample
from src.models import apply_temperature, build_model, fit_temperature

logger = logging.getLogger("train")

SCALARS = ["macro_f1", "accuracy", "under_classification_rate", "top_secret_public_confusions", "ece", "ece_raw"]


def load_config(path: str, overrides: list[str]) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for item in overrides:
        key, value = item.split("=", 1)
        *parents, leaf = key.split(".")
        node = cfg
        for part in parents:
            node = node[part]
        node[leaf] = yaml.safe_load(value)
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence > low) & (confidence <= high)
        if mask.any():
            error += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(error)


def evaluate(labels: np.ndarray, probs: np.ndarray) -> dict:
    preds = probs.argmax(axis=1)
    classes = list(range(len(LABELS)))
    precision, recall, f1, support = precision_recall_fscore_support(labels, preds, labels=classes, zero_division=0)
    matrix = confusion_matrix(labels, preds, labels=classes)
    return {
        "macro_f1": float(f1_score(labels, preds, labels=classes, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(labels, preds)),
        "under_classification_rate": float(np.mean(preds > labels)),
        "top_secret_public_confusions": int(matrix[0, 3] + matrix[3, 0]),
        "ece": expected_calibration_error(probs, labels),
        "per_class": {
            name: {"precision": float(p), "recall": float(r), "f1": float(s), "support": int(n)}
            for name, p, r, s, n in zip(LABELS, precision, recall, f1, support)
        },
        "confusion_matrix": matrix.tolist(),
    }


def summarize(fold_metrics: list[dict]) -> dict:
    values = {key: [m[key] for m in fold_metrics] for key in SCALARS}
    for name in LABELS:
        values[f"f1 {name}"] = [m["per_class"][name]["f1"] for m in fold_metrics]
    return {key: {"mean": float(np.mean(v)), "std": float(np.std(v))} for key, v in values.items()}


def update_leaderboard(output_dir: Path, name: str, summary: dict) -> None:
    path = output_dir / "summary.csv"
    row = {"experiment": name, **{key: f"{s['mean']:.3f} ± {s['std']:.3f}" for key, s in summary.items()}}
    table = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if not table.empty:
        table = table[table["experiment"] != name]
    pd.concat([table, pd.DataFrame([row])], ignore_index=True).to_csv(path, index=False, encoding="utf-8-sig")


def run(cfg: dict, name: str) -> dict:
    set_seed(cfg["seed"])
    df = load_data(cfg["data"]["path"], cfg["data"]["duplicate_threshold"])
    split = cfg["split"]
    folds = make_folds(df, split["n_splits"], split["val_fraction"], split["seed"], split["by_department"])

    output_dir = Path(cfg["output_dir"])
    run_dir = output_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)

    fold_metrics, predictions = [], []
    for i, fold in enumerate(folds):
        train, val, test = df.iloc[fold.train], df.iloc[fold.val], df.iloc[fold.test]
        if cfg["model"]["imbalance"] == "oversample":
            train = oversample(train, cfg["model"]["oversample_ratio"], cfg["seed"])
        model = build_model(cfg["model"], cfg["seed"]).fit(train, val)
        model.temperature = fit_temperature(model.predict_proba(val), val["label"].to_numpy())
        if cfg["model"]["type"] == "arabert":
            model.save(run_dir / f"fold_{i + 1}")
        raw = model.predict_proba(test)
        probs = apply_temperature(raw, model.temperature)
        metrics = evaluate(test["label"].to_numpy(), probs)
        metrics["ece_raw"] = expected_calibration_error(raw, test["label"].to_numpy())
        metrics["temperature"] = model.temperature
        fold_metrics.append(metrics)
        logger.info("fold %d: macro_f1=%.3f", i + 1, metrics["macro_f1"])
        predictions.append(pd.DataFrame({
            "row_id": test["row_id"].to_numpy(),
            "fold": i + 1,
            "label": test["label"].to_numpy(),
            "pred": probs.argmax(axis=1),
            **{f"p_{k}": probs[:, k] for k in range(len(LABELS))},
        }))

    oof = pd.concat(predictions, ignore_index=True)
    prob_columns = [f"p_{k}" for k in range(len(LABELS))]
    result = {
        "summary": summarize(fold_metrics),
        "pooled": evaluate(oof["label"].to_numpy(), oof[prob_columns].to_numpy()),
        "folds": fold_metrics,
    }

    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    oof.to_csv(run_dir / "predictions.csv", index=False)
    update_leaderboard(output_dir, name, result["summary"])

    for key, s in result["summary"].items():
        logger.info("%-30s %.3f ± %.3f", key, s["mean"], s["std"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--name", required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    for name in ("train", "src"):
        logging.getLogger(name).setLevel(logging.INFO)
    run(load_config(args.config, args.overrides), args.name)


if __name__ == "__main__":
    main()
