# Training

SpeechCritic uses [MS-Swift](https://github.com/modelscope/ms-swift) for SFT and
RL. OPD uses the same model stack through a distributed
PyTorch/Transformers/PEFT training loop, following the on-policy trajectory
distillation design of [OPSD](https://github.com/siyan-zhao/OPSD).

First edit the selected YAML file and set:

- `data.train_jsonl` and, when needed, `data.audio_root`;
- model or adapter checkpoint paths;
- `output.dir`.

For example, an SFT configuration should point to the prepared JSONL manifest:

```yaml
data:
  train_jsonl: /path/to/train.jsonl
  audio_root: /path/to/audio
output:
  dir: outputs/sft
```

Then run a stage through the common launcher:

```bash
bash train/run.sh sft train/sft/config.yaml
bash train/run.sh opd train/opd/config_privileged_sft.yaml
bash train/run.sh rl train/rl/config.yaml
```

Set the number of OPD workers with `NPROC_PER_NODE`, for example:

```bash
NPROC_PER_NODE=8 bash train/run.sh opd train/opd/config_privileged_sft.yaml
```

SFT can also be launched directly. Use `--dry-run` first to validate the data
and print the generated MS-Swift command:

```bash
python train/sft/train.py --config train/sft/config.yaml --dry-run
python train/sft/train.py --config train/sft/config.yaml
```

Create and activate the environment from the repository root before running:

```bash
conda env create -f environment.yaml
conda activate speechcritic
```
