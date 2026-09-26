from __future__ import annotations
import re

DIMS = ("D1", "D2", "D3", "D4", "D5")
LABELS = ("a", "b", "tie")

_RE_FINAL = re.compile(r"(?:Final answer|Overall Judgment)[\s\S]*?\[\[(A|B|tie)\]\]", re.I)
_RE_ANY   = re.compile(r"\[\[(A|B|tie)\]\]", re.I)
_RE_DHEAD = re.compile(r"\[D([1-5])\]")
_RE_WINNER = re.compile(r"(?:Dimension winner|Winner)\s*:?\s*\[\[(A|B|tie)\]\]", re.I)
_RE_TIENOTE = re.compile(r"Tie note\s*:?\s*\[\[(A|B|tie)\]\]", re.I)

def parse_overall(text: str) -> str | None:
    text = text or ""
    m = _RE_FINAL.findall(text)
    if m:
        return m[-1].lower()
    any_ = _RE_ANY.findall(text)
    return any_[-1].lower() if any_ else None

def parse_dims(text: str) -> dict[str, str | None]:
    text = text or ""
    heads = list(_RE_DHEAD.finditer(text))
    out: dict[str, str | None] = {d: None for d in DIMS}
    for i, h in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        section = text[h.end():end]
        m = _RE_WINNER.search(section) or _RE_TIENOTE.search(section)
        key = f"D{h.group(1)}"
        if m and out[key] is None:
            out[key] = m.group(1).lower()
    return out

def parse(text: str) -> tuple[str | None, dict[str, str | None]]:
    return parse_overall(text), parse_dims(text)

def is_complete(text: str) -> bool:
    ov, dims = parse(text)
    return ov is not None and all(dims[d] is not None for d in DIMS)
