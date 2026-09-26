from __future__ import annotations

import json
from pathlib import Path

THINKER_PREFIX = "base_model.model.thinker."
PLAIN_PREFIX = "base_model.model."

def thinker_relative(adapter_dir: str, workdir: Path, label: str = "adapter") -> str:
    from safetensors.torch import load_file, save_file

    src = Path(adapter_dir)
    weights = src / "adapter_model.safetensors"
    if not weights.exists():
        raise SystemExit(f"no adapter_model.safetensors in {src}")
    st = load_file(str(weights))
    if not any(k.startswith(THINKER_PREFIX) for k in st):
        return str(src)

    dst = Path(workdir) / f"thinker_relative_{src.name}"
    dst.mkdir(parents=True, exist_ok=True)

    save_file({(PLAIN_PREFIX + k[len(THINKER_PREFIX):] if k.startswith(THINKER_PREFIX) else k):
               v.contiguous().clone() for k, v in st.items()},
              str(dst / "adapter_model.safetensors"))
    cfg = json.load(open(src / "adapter_config.json"))
    tm = cfg.get("target_modules")
    if isinstance(tm, str):
        cfg["target_modules"] = tm.replace(r"thinker\.", "").replace("thinker.", "")
    elif isinstance(tm, list):
        cfg["target_modules"] = [m.replace("thinker.", "") for m in tm]
    json.dump(cfg, open(dst / "adapter_config.json", "w"), indent=2)
    print(f"[adapter] {label}: stripped `thinker.` prefix from {src.name} ({len(st)} tensors)")
    return str(dst)
