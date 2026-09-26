"""MS-Swift reward plugins for SpeechCritic.

Rewards score the parsed Overall verdict, the mean correctness over decisive
(non-Tie) dimensions, or additive/gated combinations of the two. Tie-labeled
dimensions are omitted because they provide no directional A/B signal.
"""
from __future__ import annotations
import os
import sys
from typing import List

from swift.rewards.orm import ORM, orms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.output_format import DIMS, parse_dims, parse_overall

DIM_WEIGHT = float(os.environ.get("SPEECHCRITIC_DIM_WEIGHT", "0.3"))

def _r_overall(completion: str, label: str) -> float:
    pred = parse_overall(completion)
    gold = str(label).strip().lower()
    return 1.0 if (pred in ("a", "b") and pred == gold) else -1.0

def _r_dim(completion: str, per_dim) -> float:
    gold = {k.upper(): str(v).lower() for k, v in dict(per_dim or {}).items()}
    pred = parse_dims(completion)
    scores = [1.0 if pred[d] == gold[d] else -1.0
              for d in DIMS if gold.get(d) in ("a", "b")]
    return sum(scores) / len(scores) if scores else 0.0

class Overall(ORM):
    def __call__(self, completions: List[str], label: List[str], **kw) -> List[float]:
        return [_r_overall(c, y) for c, y in zip(completions, label)]

class Dim(ORM):
    def __call__(self, completions: List[str], per_dim: List, **kw) -> List[float]:
        return [_r_dim(c, p) for c, p in zip(completions, per_dim)]

class OverallPlusDim(ORM):
    def __call__(self, completions: List[str], label: List[str], per_dim: List, **kw) -> List[float]:
        return [_r_overall(c, y) + DIM_WEIGHT * _r_dim(c, p)
                for c, y, p in zip(completions, label, per_dim)]

class OverallGatedDim(ORM):
    def __call__(self, completions: List[str], label: List[str], per_dim: List, **kw) -> List[float]:
        out = []
        for c, y, p in zip(completions, label, per_dim):
            ro = _r_overall(c, y)
            out.append(ro + (DIM_WEIGHT * _r_dim(c, p) if ro > 0 else 0.0))
        return out

orms["speechcritic_overall"] = Overall
orms["speechcritic_dim_only"] = Dim
orms["speechcritic_overall_plus_dim"] = OverallPlusDim
orms["speechcritic_overall_gated_dim"] = OverallGatedDim
