from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

def parser(description: str = "") -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description=description, allow_abbrev=False)

def add_common_args(ap: argparse.ArgumentParser, config_dir: Path, dry_run: bool = True) -> None:
    ap.add_argument("--config", default=str(config_dir / "config.yaml"))
    ap.add_argument("--override", action="append", default=[], metavar="dotted.key=value")
    if dry_run:
        ap.add_argument("--dry-run", action="store_true",
                        help="build the dataset and print the command without launching")

def load_config(path: str, overrides: list[str]) -> dict:
    import yaml
    cfg = yaml.safe_load(open(path))
    for ov in overrides:
        key, _, val = ov.partition("=")
        node, parts = cfg, key.split(".")
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = yaml.safe_load(val)
    return cfg

def require(cfg: dict, dotted: str):
    node = cfg
    for p in dotted.split("."):
        node = node[p]
    if node is None:
        raise SystemExit(f"config: '{dotted}' is required (currently null)")
    return node

def write_training_metadata(out_dir, cfg: dict, stage: str, stage_dir: Path,
                     extra: dict | None = None) -> None:
    sp = cfg["model"].get("system_prompt")
    rec = {"stage": stage,
           "base": cfg["model"]["base"],

           "system_prompt": str((stage_dir / sp).resolve().name) if sp else None,
           "encoder_trainable": bool(cfg.get("lora", {}).get("encoder_trainable", False)),
           "seed": cfg["train"]["seed"]}
    rec.update(extra or {})
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(out_dir) / "training_metadata.json", "w") as f:
        json.dump(rec, f, indent=2)

def swift_exe() -> str:
    cand = Path(sys.executable).parent / "swift"
    if cand.is_file():
        return str(cand)
    raise SystemExit(
        "could not find the MS-Swift CLI next to the active Python executable. "
        "Install the training environment (conda env create -f environment.yaml) "
        "and activate it. This check intentionally ignores a system `swift` binary, "
        "which may be the Apple Swift compiler."
    )

def world_size() -> int:
    n = os.environ.get("NPROC_PER_NODE") or os.environ.get("WORLD_SIZE")
    if n:
        return int(n)
    try:
        import torch
        return max(torch.cuda.device_count(), 1)
    except Exception:
        return 1
