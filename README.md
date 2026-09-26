# SpeechCritic

SpeechCritic is an end-to-end recipe for building reference-conditioned speech
critics from limited pairwise human judgments. It first turns task-specific
measurements into human-calibrated probabilistic hints, then studies three
complementary ways to train a critic that returns an Overall preference,
dimension-level judgments, and an acoustically grounded rationale.

```text
human preferences + domain measurements
    -> select reliable measurements
    -> calibrate P(A / Tie / B)
    -> scale hints to an unlabeled pool
    -> create verdict-and-rationale supervision
    -> train with SFT, OPD, or RL
```

## What is included

- `data/`: a runnable, modality-agnostic **select--calibrate--scale** pipeline.
  Selection uses grouped out-of-fold validation; calibration learns an
  uncertainty-preserving multinomial mapping; scaling freezes and applies that
  mapping to new pairs. Vision or other perception tasks can substitute their
  own measurements, such as CLIP similarity or layout scores.
- `train/`: LoRA recipes for **supervised fine-tuning (SFT)**, **on-policy
  distillation (OPD)**, and verdict-reward **reinforcement learning (RL)**.
  SFT and RL use MS-Swift; OPD uses a compact distributed
  PyTorch/Transformers/PEFT loop. The OPD design follows
  [OPSD](https://github.com/siyan-zhao/OPSD), adapted here to multimodal speech,
  separate frozen teachers, and privileged acoustic rationales.
- `train/lang/`: language-pair metadata and the language-agnostic SpeechCritic
  prompt adapted from Appendix C.2 of the paper.
- `docs/index.html`: a self-contained interactive demo with embedded audio,
  suitable for local viewing or publication from the repository's `docs/`
  directory with GitHub Pages.

The training recipes support frozen or trainable audio encoders, optional
transcripts, privileged or vanilla OPD teachers, four composable verdict reward
designs, and GRPO/DAPO optimization.

## Demo

Open `docs/index.html` in a browser. The page contains its audio inline and does
not require a server or external media files.

## Setup

```bash
conda env create -f environment.yaml
conda activate speechcritic
```

The default student is `Qwen/Qwen2.5-Omni-7B`; the OPD configurations also
support a Qwen3-Omni teacher. The RL configuration enables colocated vLLM, so
install a vLLM build compatible with the selected CUDA and MS-Swift versions
when using that option.

## Prepare data

Training stages consume JSON Lines records containing a reference, two
candidates, optional transcripts, and the targets required by that stage:

```json
{
  "pair_id": "example_en_ja_0001",
  "audios": {
    "reference": "audio/example/reference_en.wav",
    "candidate_a": "audio/example/candidate_a_ja.wav",
    "candidate_b": "audio/example/candidate_b_ja.wav"
  },
  "transcripts": {
    "source": "Please wait here until the rain stops.",
    "target": "雨が止むまで、ここで待っていてください。"
  },
  "labels": {
    "overall": "b",
    "dims": {"D1": "tie", "D2": "b", "D3": "b", "D4": "tie", "D5": "b"}
  },
  "rationale": "[Reference Anchor]\n...\n[Overall]\n...\nFinal answer: [[B]]"
}
```

The complete fictional record is in
[`data/example_train_manifest.jsonl`](data/example_train_manifest.jsonl).
SFT requires `rationale`; RL requires `labels`; privileged OPD additionally
uses `rationale` as teacher-only information.

To run the calibration example end to end:

```bash
python data/select_metrics.py --input data/example_preferences.jsonl --output outputs/selection.json
python data/calibrate.py --input data/example_preferences.jsonl --selection outputs/selection.json --output outputs/calibration.json
python data/scale.py --input data/example_preferences.jsonl --calibration outputs/calibration.json --output outputs/hints.jsonl
```

See [`data/README.md`](data/README.md) for the reusable input contract and the
meaning of each step.

## Train

Set `data.train_jsonl`, `data.audio_root`, checkpoint paths, and `output.dir` in
the selected YAML file, then use the common launcher:

```bash
bash train/run.sh sft train/sft/config.yaml
bash train/run.sh opd train/opd/config_privileged_sft.yaml
bash train/run.sh rl train/rl/config.yaml
```

Use `NPROC_PER_NODE=8` for an eight-GPU OPD run. SFT and RL accept `--dry-run`
to validate the manifest and print the generated MS-Swift command without
starting optimization. Any YAML value can be overridden from the command line,
for example `--override train.lr=1e-5`.

The YAML files preserve the paper's optimization, LoRA, sequence-length,
rollout, and reward settings. The reported runs used eight NVIDIA A100 GPUs for
SFT/RL and eight NVIDIA H200 GPUs for OPD. More details are in
[`train/README.md`](train/README.md).

## Citation and upstream projects

The SpeechCritic citation will be added after anonymous review. This release
builds on [MS-Swift](https://github.com/modelscope/ms-swift),
[Qwen2.5-Omni](https://github.com/QwenLM/Qwen2.5-Omni), and
[Qwen3-Omni](https://github.com/QwenLM/Qwen3-Omni). The OPD implementation is
inspired by [OPSD](https://github.com/siyan-zhao/OPSD). Please cite the
applicable projects and follow their model and software licenses; BibTeX entries
are collected in [`CITATIONS.bib`](CITATIONS.bib).
