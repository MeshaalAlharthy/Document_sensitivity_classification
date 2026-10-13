from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from transformers.utils import logging as transformers_logging

from src.data import LABELS

logger = logging.getLogger(__name__)
transformers_logging.set_verbosity_error()
transformers_logging.disable_progress_bar()


def fit_temperature(probs: np.ndarray, labels: np.ndarray) -> float:
    logits = np.log(np.clip(probs, 1e-12, 1.0))
    rows = np.arange(len(labels))
    result = minimize_scalar(
        lambda t: -log_softmax(logits / t, axis=1)[rows, labels].mean(), bounds=(0.05, 20.0), method="bounded"
    )
    return float(result.x)


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    return softmax(np.log(np.clip(probs, 1e-12, 1.0)) / temperature, axis=1)


class TfidfClassifier:
    def __init__(self, use_department: bool, weighted: bool, c_grid: list[float], seed: int) -> None:
        self.use_department = use_department
        self.class_weight = "balanced" if weighted else None
        self.c_grid = c_grid
        self.seed = seed
        self.temperature = 1.0
        self.pipeline: Pipeline | None = None
        self.best_c: float | None = None

    def _build(self, c: float) -> Pipeline:
        text = FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2)),
        ])
        transformers = [("text", text, "text")]
        if self.use_department:
            transformers.append(("department", OneHotEncoder(handle_unknown="ignore"), ["department"]))
        classifier = LogisticRegression(C=c, class_weight=self.class_weight, max_iter=3000, random_state=self.seed)
        return Pipeline([("features", ColumnTransformer(transformers)), ("classifier", classifier)])

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> TfidfClassifier:
        best_score = -1.0
        for c in self.c_grid:
            pipeline = self._build(c).fit(train, train["label"])
            score = f1_score(val["label"], pipeline.predict(val), average="macro")
            if score > best_score:
                best_score, self.best_c, self.pipeline = score, c, pipeline
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        probs = np.zeros((len(df), len(LABELS)))
        probs[:, self.pipeline.classes_] = self.pipeline.predict_proba(df)
        return probs


def focal_loss(logits: torch.Tensor, labels: torch.Tensor, weight: torch.Tensor | None, gamma: float) -> torch.Tensor:
    log_pt = F.log_softmax(logits, dim=-1).gather(1, labels.unsqueeze(1)).squeeze(1)
    scale = (1 - log_pt.exp()) ** gamma
    if weight is not None:
        scale = scale * weight[labels]
    return -(scale * log_pt).mean()


class AdminAwareBert(nn.Module):
    def __init__(self, base: str, lora_r: int, departments: int) -> None:
        super().__init__()
        encoder = AutoModel.from_pretrained(base)
        lora = LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.1, target_modules=["query", "value"])
        self.encoder = get_peft_model(encoder, lora)
        self.department = nn.Embedding(departments, 32) if departments else None
        size = 2 * encoder.config.hidden_size + (32 if departments else 0)
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(size, len(LABELS)))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, department: torch.Tensor) -> torch.Tensor:
        states = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(states.dtype)
        mean = (states * mask).sum(dim=1) / mask.sum(dim=1)
        features = [states[:, 0], mean]
        if self.department is not None:
            features.append(self.department(department))
        return self.head(torch.cat(features, dim=1))


class BertClassifier:
    def __init__(self, cfg: dict, seed: int) -> None:
        self.cfg = cfg
        self.bert = cfg["bert"]
        self.seed = seed
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert["base"])
        self.temperature = 1.0
        self.departments: list[str] = []
        self.index: dict[str, int] = {}
        self.network: AdminAwareBert | None = None

    def _build(self, departments: list[str]) -> None:
        self.departments = departments if self.cfg["use_department"] else []
        self.index = {name: i + 1 for i, name in enumerate(self.departments)}
        size = len(self.departments) + 1 if self.departments else 0
        self.network = AdminAwareBert(self.bert["base"], self.bert["lora_r"], size).to(self.device)

    def _batch(self, df: pd.DataFrame) -> dict[str, torch.Tensor]:
        tokens = self.tokenizer(
            df["text"].tolist(), padding=True, truncation=True, max_length=self.bert["max_length"], return_tensors="pt"
        )
        department = torch.tensor([self.index.get(name, 0) for name in df["department"]])
        return {
            "input_ids": tokens["input_ids"].to(self.device),
            "attention_mask": tokens["attention_mask"].to(self.device),
            "department": department.to(self.device),
        }

    def _autocast(self) -> torch.autocast:
        return torch.autocast(self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda")

    def _trainable_state(self) -> dict[str, torch.Tensor]:
        return {name: p.detach().cpu().clone() for name, p in self.network.named_parameters() if p.requires_grad}

    def _loss(self, logits: torch.Tensor, labels: torch.Tensor, weight: torch.Tensor | None) -> torch.Tensor:
        if self.cfg["imbalance"] == "focal":
            return focal_loss(logits, labels, weight, self.bert["focal_gamma"])
        return F.cross_entropy(logits, labels, weight=weight)

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> BertClassifier:
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self._build(sorted(train["department"].unique()))

        labels = train["label"].to_numpy()
        weight = None
        if self.cfg["imbalance"] in ("weighted", "focal"):
            values = compute_class_weight("balanced", classes=np.arange(len(LABELS)), y=labels)
            weight = torch.tensor(values, dtype=torch.float, device=self.device)

        parameters = [p for p in self.network.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=self.bert["lr"], weight_decay=0.01)
        batch_size = self.bert["batch_size"]
        total_steps = self.bert["epochs"] * math.ceil(len(train) / batch_size)
        scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)
        scaler = torch.amp.GradScaler(self.device.type, enabled=self.device.type == "cuda")

        best_score, best_state, waited = -1.0, None, 0
        for epoch in range(1, self.bert["epochs"] + 1):
            self.network.train()
            order = rng.permutation(len(train))
            losses = []
            for start in range(0, len(order), batch_size):
                rows = train.iloc[order[start:start + batch_size]]
                batch = self._batch(rows)
                if self.departments:
                    dropped = torch.rand(len(rows), device=self.device) < 0.1
                    batch["department"] = batch["department"].masked_fill(dropped, 0)
                target = torch.tensor(rows["label"].to_numpy(), device=self.device)
                with self._autocast():
                    logits = self.network(**batch)
                loss = self._loss(logits.float(), target, weight)
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(parameters, 1.0)
                scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= scale:
                    scheduler.step()
                losses.append(loss.item())

            score = f1_score(val["label"], self.predict_proba(val).argmax(axis=1), average="macro")
            logger.info("  epoch %2d: loss=%.4f val_macro_f1=%.3f", epoch, np.mean(losses), score)
            if score > best_score:
                best_score, best_state, waited = score, self._trainable_state(), 0
            else:
                waited += 1
                if waited >= self.bert["patience"]:
                    break

        self.network.load_state_dict(best_state, strict=False)
        return self

    @torch.no_grad()
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        self.network.eval()
        probs = []
        for start in range(0, len(df), 64):
            with self._autocast():
                logits = self.network(**self._batch(df.iloc[start:start + 64]))
            probs.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
        return np.concatenate(probs)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "config": self.cfg,
            "departments": self.departments,
            "temperature": self.temperature,
            "state": self._trainable_state(),
        }
        torch.save(checkpoint, path / "model.pt")

    @classmethod
    def load(cls, path: Path) -> BertClassifier:
        checkpoint = torch.load(path / "model.pt", map_location="cpu")
        model = cls(checkpoint["config"], seed=0)
        model._build(checkpoint["departments"])
        model.network.load_state_dict(checkpoint["state"], strict=False)
        model.temperature = checkpoint["temperature"]
        return model


def build_model(cfg: dict, seed: int) -> TfidfClassifier | BertClassifier:
    if cfg["type"] == "tfidf":
        return TfidfClassifier(cfg["use_department"], cfg["imbalance"] == "weighted", cfg["c_grid"], seed)
    if cfg["type"] == "arabert":
        return BertClassifier(cfg, seed)
    raise ValueError(f"unknown model type: {cfg['type']}")
