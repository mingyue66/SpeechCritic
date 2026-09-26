from __future__ import annotations
import json
import os
from pathlib import Path

DIMS = ("D1", "D2", "D3", "D4", "D5")

from common.prompt_format import (AUDIO_KEYS, audio_paths, build_segments,
                                  to_structured_content, to_swift_content)

def load(path: str | Path, audio_root: str | None = None, require: tuple[str, ...] = ()) -> list[dict]:
    recs = []
    with open(path) as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "pair_id" not in r:
                raise ValueError(f"{path}:{ln} missing 'pair_id'")
            au = r.get("audios") or {}
            missing = [k for k in AUDIO_KEYS if not au.get(k)]
            if missing:
                raise ValueError(f"{path}:{ln} ({r['pair_id']}) missing audios: {missing}")
            if audio_root:
                r["audios"] = {k: (v if os.path.isabs(v) else os.path.join(audio_root, v))
                               for k, v in au.items()}
            for k in AUDIO_KEYS:
                p = r["audios"][k]
                if not os.path.exists(p):
                    raise FileNotFoundError(f"{path}:{ln} ({r['pair_id']}) audio not found: {p}")
            if "rationale" in require and not r.get("rationale"):
                raise ValueError(f"{path}:{ln} ({r['pair_id']}) missing 'rationale' (required by this stage)")
            if "labels" in require:
                lab = r.get("labels") or {}
                if lab.get("overall") not in ("a", "b"):
                    raise ValueError(f"{path}:{ln} ({r['pair_id']}) labels.overall must be 'a' or 'b'")
                dims = lab.get("dims") or {}
                bad = [d for d in DIMS if dims.get(d) not in ("a", "b", "tie")]
                if bad:
                    raise ValueError(f"{path}:{ln} ({r['pair_id']}) labels.dims invalid/missing: {bad}")
            recs.append(r)
    if not recs:
        raise ValueError(f"{path}: no records")
    return recs

def read_prompt(path: str | Path) -> str:
    return Path(path).read_text().strip()
