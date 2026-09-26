from __future__ import annotations

import argparse
import json

from _core import metric_gap, predict_serialized, read_jsonl, write_jsonl


def unknown(dimension: str, reason: str) -> dict:
    return {
        "dimension": dimension,
        "state": "unknown",
        "metric": None,
        "p_A": None,
        "p_Tie": None,
        "p_B": None,
        "confidence": None,
        "reason": reason,
    }


def render_dimension(hint: dict) -> str:
    if hint["state"] == "unknown":
        return (
            f"{hint['dimension']}:\n- State: unknown\n- Suggested direction: none\n"
            "- Confidence: none\n- Probabilities: none"
        )
    return (
        f"{hint['dimension']}:\n- State: soft distribution\n- Suggested direction: none\n"
        f"- Confidence: {hint['confidence']:.4f}\n"
        f"- Probabilities: A={hint['p_A']:.4f}, Tie={hint['p_Tie']:.4f}, "
        f"B={hint['p_B']:.4f}"
    )


def render_block(hints: dict[str, dict], dimensions: list[str]) -> str:
    sections = ["[Optional Prior Evidence]"]
    sections.extend(render_dimension(hints[dimension]) for dimension in dimensions)
    sections.append("No aggregate prior is provided.")
    return "\n\n".join(sections)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply frozen DEV mappings to an unlabeled comparison pool."
    )
    parser.add_argument(
        "--input", required=True,
        help="JSONL containing pair_id and metric_values or metric_gaps",
    )
    parser.add_argument("--calibration", required=True, help="calibration JSON from calibrate.py")
    parser.add_argument("--output", required=True, help="JSONL with probability hints")
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    calibration = json.load(open(args.calibration))
    dimensions = list((calibration.get("dimensions") or {}).keys())
    if not dimensions:
        raise ValueError("calibration contains no dimensions")
    outputs = []
    for row in rows:
        if not row.get("pair_id"):
            raise ValueError("every scaling record requires pair_id")
        hints = {}
        for dimension in dimensions:
            fitted = (calibration.get("dimensions") or {}).get(dimension) or {}
            if fitted.get("status") != "calibrated":
                hints[dimension] = unknown(dimension, "no retained calibrator")
                continue
            metric = fitted["metric"]
            gap = metric_gap(row, dimension, metric)
            if gap is None:
                hints[dimension] = unknown(dimension, f"missing metric gap: {metric}")
                continue
            probability = predict_serialized(fitted, gap)
            hints[dimension] = {
                "dimension": dimension,
                "state": "soft_distribution",
                "metric": metric,
                "p_A": probability["A"],
                "p_Tie": probability["tie"],
                "p_B": probability["B"],
                "confidence": max(probability.values()),
            }
        output = {
            "pair_id": row["pair_id"],
            "domain_hints": hints,
            "optional_prior_evidence": render_block(hints, dimensions),
        }
        if row.get("uid") is not None:
            output["uid"] = row["uid"]
        outputs.append(output)
    write_jsonl(args.output, outputs)
    print(f"[scale] wrote {len(outputs)} records to {args.output}")


if __name__ == "__main__":
    main()
