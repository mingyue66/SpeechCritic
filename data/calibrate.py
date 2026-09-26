from __future__ import annotations

import argparse
import json

import numpy as np

from _core import (CLASS_ORDER, CLASS_TO_INT, human_label, make_model,
                   metric_gap, read_jsonl, serialize_model, validate_dev_rows,
                   write_json)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refit retained one-metric multinomial mappings on the complete DEV set."
    )
    parser.add_argument("--input", required=True, help="the same DEV JSONL used by select_metrics.py")
    parser.add_argument("--selection", required=True, help="selection report produced by select_metrics.py")
    parser.add_argument("--output", required=True, help="frozen calibration JSON")
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    validate_dev_rows(rows)
    selection = json.load(open(args.selection))
    output = {
        "method": "one_metric_multinomial",
        "fit_scope": "complete development set",
        "class_order": list(CLASS_ORDER),
        "dimensions": {},
    }
    for dimension, selection_entry in (selection.get("dimensions") or {}).items():
        selected = selection_entry.get("retained_metric", selection_entry.get("suggested_metric"))
        if not selected:
            output["dimensions"][dimension] = {
                "status": "unknown", "metric": None,
                "reason": "no metric was retained after selection and review",
            }
            continue
        training = [
            row for row in rows
            if human_label(row, dimension) is not None
            and metric_gap(row, dimension, selected) is not None
        ]
        labels = [human_label(row, dimension) for row in training]
        observed = set(labels)
        if len(observed) < 2:
            raise ValueError(
                f"{dimension}/{selected}: full DEV fit requires at least two preference classes; "
                f"observed {sorted(observed)}"
            )
        features = np.asarray([
            metric_gap(row, dimension, selected) for row in training
        ], dtype=float).reshape(-1, 1)
        target = np.asarray([CLASS_TO_INT[label] for label in labels], dtype=int)
        model = make_model().fit(features, target)
        candidate_report = selection["dimensions"][dimension]["candidates"][selected]
        output["dimensions"][dimension] = {
            "status": "calibrated",
            "metric": selected,
            "orientation": "signed A-minus-B gap; positive favors A",
            "n_train": len(training),
            "observed_classes": sorted(observed, key=CLASS_TO_INT.get),
            "selection_oof": candidate_report,
            **serialize_model(model),
        }
        print(f"[{dimension}] metric={selected} n={len(training)}")
    write_json(args.output, output)


if __name__ == "__main__":
    main()
