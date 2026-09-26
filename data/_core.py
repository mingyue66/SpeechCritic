from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SEED = 20260724
CLASS_ORDER = ("B", "tie", "A")
CLASS_TO_INT = {label: index for index, label in enumerate(CLASS_ORDER)}


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path) as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    if not rows:
        raise ValueError(f"{path}: no records")
    return rows


def write_json(path: str | Path, value: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_model() -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("multinomial", LogisticRegression(
            penalty="l2", C=1.0, solver="lbfgs", max_iter=5000,
            random_state=SEED,
        )),
    ])


def align_probability(model: Pipeline, probability: np.ndarray) -> np.ndarray:
    aligned = np.zeros((len(probability), len(CLASS_ORDER)))
    classes = model.named_steps["multinomial"].classes_
    for source_index, class_value in enumerate(classes):
        aligned[:, int(class_value)] = probability[:, source_index]
    return aligned


def multiclass_brier(target: np.ndarray, probability: np.ndarray) -> float:
    one_hot = np.eye(len(CLASS_ORDER))[target]
    return float(np.mean(np.sum((probability - one_hot) ** 2, axis=1)))


def metric_summary(target: np.ndarray, probability: np.ndarray) -> dict:
    prediction = probability.argmax(axis=1)
    return {
        "n": int(len(target)),
        "accuracy": float(accuracy_score(target, prediction)),
        "macro_f1": float(f1_score(
            target, prediction, labels=list(range(len(CLASS_ORDER))),
            average="macro", zero_division=0,
        )),
        "log_loss": float(log_loss(
            target, probability, labels=list(range(len(CLASS_ORDER))),
        )),
        "brier": multiclass_brier(target, probability),
    }


def human_label(row: dict, dimension: str) -> str | None:
    labels = row.get("human_labels") or {}
    if dimension in labels:
        label = labels[dimension]
        if label is None:
            return None
    else:
        votes = (row.get("human_votes") or {}).get(dimension)
        if votes is None:
            return None
        try:
            counts = {label: int(votes.get(label, 0)) for label in CLASS_ORDER}
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{row.get('pair_id', '<unknown>')}: human_votes.{dimension} "
                "must contain integer A/tie/B counts"
            ) from exc
        total = sum(counts.values())
        if total <= 0 or min(counts.values()) < 0:
            raise ValueError(
                f"{row.get('pair_id', '<unknown>')}: human_votes.{dimension} "
                "must contain non-negative counts with a positive total"
            )
        label = max(CLASS_ORDER, key=counts.get)
        if counts[label] <= total / 2:
            return None
    if label not in CLASS_TO_INT:
        raise ValueError(
            f"{row.get('pair_id', '<unknown>')}: human_labels.{dimension} "
            f"must be one of {list(CLASS_ORDER)} or null"
        )
    return label


def metric_gap(row: dict, dimension: str, metric: str) -> float | None:
    gaps = (row.get("metric_gaps") or {}).get(dimension) or {}
    if metric in gaps:
        value = gaps[metric]
    else:
        values = ((row.get("metric_values") or {}).get(dimension) or {}).get(metric)
        if values is None:
            return None
        try:
            value = float(values["A"]) - float(values["B"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{row.get('pair_id', '<unknown>')}: metric_values.{dimension}.{metric} "
                "must contain numeric A and B values"
            ) from exc
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{row.get('pair_id', '<unknown>')}: metric_gaps.{dimension}.{metric} "
            "must be numeric or null"
        ) from exc
    if not math.isfinite(value):
        return None
    return value


def validate_dev_rows(rows: list[dict]) -> None:
    group_to_fold = {}
    has_fold = [row.get("fold") is not None for row in rows]
    if any(has_fold) and not all(has_fold):
        raise ValueError("fold must be present on every DEV row or omitted from every DEV row")
    for row in rows:
        pair_id = row.get("pair_id")
        if not pair_id:
            raise ValueError("every calibration record requires pair_id")
        if row.get("split") != "dev":
            raise ValueError(f"{pair_id}: calibration input must contain DEV rows only")
        if row.get("group_id") is None:
            raise ValueError(f"{pair_id}: group_id is required for grouped out-of-fold evaluation")
        if row.get("fold") is None:
            continue
        group = str(row["group_id"])
        fold = str(row["fold"])
        previous = group_to_fold.setdefault(group, fold)
        if previous != fold:
            raise ValueError(f"group_id {group!r} crosses folds {previous!r} and {fold!r}")


def available_metrics(rows: list[dict], dimension: str) -> list[str]:
    names = set()
    for row in rows:
        names.update(((row.get("metric_gaps") or {}).get(dimension) or {}).keys())
        names.update(((row.get("metric_values") or {}).get(dimension) or {}).keys())
    return sorted(names)


def available_dimensions(rows: list[dict]) -> list[str]:
    names = set()
    for row in rows:
        for field in ("human_labels", "human_votes", "metric_gaps", "metric_values"):
            names.update((row.get(field) or {}).keys())
    return sorted(names)


def serialize_model(model: Pipeline) -> dict:
    scaler = model.named_steps["scale"]
    logistic = model.named_steps["multinomial"]
    return {
        "classes": [int(value) for value in logistic.classes_],
        "scaler_mean": float(scaler.mean_[0]),
        "scaler_scale": float(scaler.scale_[0]),
        "coef": [float(value) for value in logistic.coef_[:, 0]],
        "intercept": [float(value) for value in logistic.intercept_],
    }


def predict_serialized(model: dict, value: float) -> dict[str, float]:
    classes = model["classes"]
    scale = model["scaler_scale"] or 1.0
    standardized = (float(value) - model["scaler_mean"]) / scale
    logits = np.asarray(model["intercept"]) + np.asarray(model["coef"]) * standardized
    if len(classes) == 2:
        positive = 1.0 / (1.0 + np.exp(-logits[0]))
        probability = np.asarray([1.0 - positive, positive])
    elif len(classes) == len(CLASS_ORDER):
        exponent = np.exp(logits - logits.max())
        probability = exponent / exponent.sum()
    else:
        raise ValueError("saved calibrator must contain two or three preference classes")
    aligned = np.zeros(len(CLASS_ORDER))
    for source_index, class_value in enumerate(classes):
        aligned[int(class_value)] = probability[source_index]
    return {label: float(aligned[index]) for index, label in enumerate(CLASS_ORDER)}
