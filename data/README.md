# Prepare human-calibrated domain hints

Prepare the following for a small human-labeled development set:

1. Pairwise examples with Candidate A and Candidate B.
2. A `group_id` shared by examples that must remain in the same validation
   fold, such as the same source utterance, subject, scene, or original image.
3. Human `A` / `tie` / `B` preferences for each task-specific dimension,
   provided either as majority labels or as vote counts.
4. One or more scalar domain measurements for each dimension. Orient every
   measurement so that a larger value is better, then provide its value for A
   and B or provide the signed `A - B` gap directly.

Because the pipeline assumes only pairwise human preferences and scalar domain
measurements, it can be reused across perception tasks rather than being tied
to speech. For vision, for example, users can substitute measurements such as
CLIP similarity or layout scores and apply the same select--calibrate--scale
workflow.

## Input format

Each line of the development JSONL represents one comparison:

```json
{
  "pair_id": "dev_example_0001",
  "split": "dev",
  "group_id": "source_0042",
  "human_votes": {
    "appearance": {"A": 2, "tie": 1, "B": 0},
    "composition": {"A": 0, "tie": 3, "B": 0}
  },
  "metric_values": {
    "appearance": {
      "embedding_similarity": {"A": 0.82, "B": 0.61}
    },
    "composition": {
      "layout_score": {"A": 0.70, "B": 0.72}
    }
  }
}
```

`human_labels` may be used instead of vote counts:

```json
"human_labels": {"appearance": "A", "composition": "tie"}
```

The code derives a strict majority from `human_votes`; a vote pattern without
a strict majority is excluded for that dimension. `metric_gaps` may be used
instead of candidate values:

```json
"metric_gaps": {"appearance": {"embedding_similarity": 0.21}}
```

If a lower raw measurement is better, negate or otherwise orient it before
providing the values. An optional `fold` field can supply an existing grouped
split. When it is absent, `select_metrics.py` creates deterministic stratified
grouped folds from `group_id`.

## Select

`select_metrics.py` fits each candidate metric independently with
`StandardScaler + L2 multinomial logistic regression`. It evaluates held-out
predictions with grouped OOF macro-F1 and compares them with the corresponding
training-fold class-frequency baseline. Candidate metrics within a dimension
are evaluated on the same complete-case examples.

The report writes both `suggested_metric` and `retained_metric`. They initially
match when there is a unique best candidate above the baseline. Review
`retained_metric` before calibration: replace it when separate domain evidence
supports another candidate, or set it to `null` to omit that dimension.

```bash
python data/select_metrics.py \
  --input PATH/TO/dev_preferences_and_metrics.jsonl \
  --output outputs/metric_selection.json
```

## Calibrate

`calibrate.py` refits each retained one-metric mapping on all usable DEV
examples and saves the scaler and logistic-regression parameters. It supports
both A/Tie/B datasets and datasets containing only two observed preference
classes.

```bash
python data/calibrate.py \
  --input PATH/TO/dev_preferences_and_metrics.jsonl \
  --selection outputs/metric_selection.json \
  --output outputs/calibration.json
```

## Scale

Prepare the larger unlabeled pool with `pair_id` and the same `metric_values`
or `metric_gaps` fields. Human labels, groups, and folds are not required.
`scale.py` applies the frozen mappings and emits `P(A)`, `P(Tie)`, and `P(B)`
plus a prompt-ready optional-evidence block. A missing or uncalibrated
dimension is marked `unknown`.

```bash
python data/scale.py \
  --input PATH/TO/unlabeled_metrics.jsonl \
  --calibration outputs/calibration.json \
  --output outputs/domain_hints.jsonl
```

## Runnable example

[`example_preferences.jsonl`](example_preferences.jsonl) is a small fictional
two-dimension dataset that runs through all three commands directly:

```bash
python data/select_metrics.py \
  --input data/example_preferences.jsonl \
  --output outputs/example_selection.json
python data/calibrate.py \
  --input data/example_preferences.jsonl \
  --selection outputs/example_selection.json \
  --output outputs/example_calibration.json
python data/scale.py \
  --input data/example_preferences.jsonl \
  --calibration outputs/example_calibration.json \
  --output outputs/example_hints.jsonl
```

[`example_train_manifest.jsonl`](example_train_manifest.jsonl) separately shows
the final SpeechCritic verdict-and-rationale format consumed by `train/`.
