from __future__ import annotations

import argparse

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from _core import (CLASS_ORDER, CLASS_TO_INT, align_probability,
                   available_dimensions, available_metrics, human_label,
                   make_model, metric_gap, metric_summary, read_jsonl,
                   validate_dev_rows, write_json)


def fold_assignments(rows: list[dict], dimension: str, requested_folds: int) -> tuple[list[str], str]:
    if all(row.get("fold") is not None for row in rows):
        return [str(row["fold"]) for row in rows], "provided"
    target = np.asarray([CLASS_TO_INT[human_label(row, dimension)] for row in rows], dtype=int)
    groups = np.asarray([str(row["group_id"]) for row in rows])
    class_counts = np.bincount(target, minlength=len(CLASS_ORDER))
    positive_counts = class_counts[class_counts > 0]
    n_splits = min(requested_folds, len(set(groups)), int(positive_counts.min()))
    if n_splits < 2:
        raise ValueError("not enough groups or class examples to create two grouped folds")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=20260724)
    assigned = np.full(len(rows), -1, dtype=int)
    for fold, (_, validation) in enumerate(splitter.split(np.zeros((len(rows), 1)), target, groups)):
        assigned[validation] = fold
    if (assigned < 0).any():
        raise ValueError("failed to assign every row to a grouped fold")
    return [str(value) for value in assigned], "generated_stratified_group"


def evaluate(rows: list[dict], dimension: str, metric: str, requested_folds: int) -> dict:
    usable = [
        row for row in rows
        if human_label(row, dimension) is not None
        and metric_gap(row, dimension, metric) is not None
    ]
    assigned, fold_source = fold_assignments(usable, dimension, requested_folds)
    indexed = list(zip(usable, assigned))
    folds = sorted(set(assigned))
    if len(folds) < 2:
        raise ValueError("fewer than two populated folds")

    target, model_probability, baseline_probability = [], [], []
    for fold in folds:
        training = [row for row, row_fold in indexed if row_fold != fold]
        validation = [row for row, row_fold in indexed if row_fold == fold]
        if not training or not validation:
            continue
        train_y = np.asarray([
            CLASS_TO_INT[human_label(row, dimension)] for row in training
        ], dtype=int)
        if len(set(train_y.tolist())) < 2:
            raise ValueError(f"fold {fold}: training partition contains fewer than two classes")
        train_x = np.asarray([
            metric_gap(row, dimension, metric) for row in training
        ], dtype=float).reshape(-1, 1)
        valid_x = np.asarray([
            metric_gap(row, dimension, metric) for row in validation
        ], dtype=float).reshape(-1, 1)

        model = make_model().fit(train_x, train_y)
        model_probability.append(align_probability(model, model.predict_proba(valid_x)))
        counts = np.bincount(train_y, minlength=len(CLASS_ORDER)).astype(float)
        baseline_probability.append(np.tile(counts / counts.sum(), (len(validation), 1)))
        target.extend(CLASS_TO_INT[human_label(row, dimension)] for row in validation)

    target_array = np.asarray(target, dtype=int)
    model_array = np.concatenate(model_probability)
    baseline_array = np.concatenate(baseline_probability)
    model_metrics = metric_summary(target_array, model_array)
    baseline_metrics = metric_summary(target_array, baseline_array)
    return {
        "status": "ok",
        "folds": len(folds),
        "fold_source": fold_source,
        "model": model_metrics,
        "class_frequency_baseline": baseline_metrics,
        "delta_macro_f1": model_metrics["macro_f1"] - baseline_metrics["macro_f1"],
        "delta_log_loss": model_metrics["log_loss"] - baseline_metrics["log_loss"],
        "delta_brier": model_metrics["brier"] - baseline_metrics["brier"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Screen candidate domain metrics with grouped out-of-fold human labels."
    )
    parser.add_argument(
        "--input", required=True,
        help="DEV JSONL with human preferences, metric values/gaps, group_id, and optional fold",
    )
    parser.add_argument("--output", required=True, help="selection report JSON")
    parser.add_argument("--folds", type=int, default=5,
                        help="grouped folds to create when the input has no fold field (default: 5)")
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    validate_dev_rows(rows)
    report = {
        "method": {
            "scope": "development set only",
            "model": "StandardScaler + L2 multinomial logistic regression",
            "selection_measure": "grouped out-of-fold macro-F1",
            "baseline": "training-fold class frequency",
            "class_order": list(CLASS_ORDER),
            "note": (
                "The suggested metric is the highest-OOF-macro-F1 candidate that improves over "
                "the baseline. It covers only automatic screening; any listening evidence must "
                "be considered separately."
            ),
        },
        "dimensions": {},
    }
    for dimension in available_dimensions(rows):
        metric_names = available_metrics(rows, dimension)
        comparable_rows = [
            row for row in rows
            if human_label(row, dimension) is not None
            and all(metric_gap(row, dimension, metric) is not None for metric in metric_names)
        ]
        candidates = {}
        for metric in metric_names:
            try:
                candidates[metric] = evaluate(comparable_rows, dimension, metric, args.folds)
            except ValueError as exc:
                candidates[metric] = {"status": "not_evaluable", "reason": str(exc)}
        supported = [
            (name, result) for name, result in candidates.items()
            if result.get("status") == "ok" and result["delta_macro_f1"] > 0
        ]
        best_score = max(
            (result["model"]["macro_f1"] for _, result in supported),
            default=None,
        )
        best = [
            name for name, result in supported
            if np.isclose(result["model"]["macro_f1"], best_score)
        ] if best_score is not None else []
        suggested = best[0] if len(best) == 1 else None
        report["dimensions"][dimension] = {
            "suggested_metric": suggested,
            "retained_metric": suggested,
            "comparison_n": len(comparable_rows),
            "suggestion_reason": (
                "unique highest OOF macro-F1 above baseline" if suggested
                else "no unique candidate improved over baseline"
            ),
            "candidates": candidates,
        }
        print(f"[{dimension}] suggested={suggested or 'none'} candidates={len(candidates)}")
    write_json(args.output, report)


if __name__ == "__main__":
    main()
