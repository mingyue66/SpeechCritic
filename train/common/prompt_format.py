from __future__ import annotations

CLIP_LABELS = (
    ("reference",   "[Reference]:"),
    ("candidate_a", "[Candidate A]:"),
    ("candidate_b", "[Candidate B]:"),
)
AUDIO_KEYS = tuple(k for k, _ in CLIP_LABELS)

DEFAULT_SOURCE_LABEL = "[English reference transcript]:"
DEFAULT_TARGET_LABEL = "[Target Japanese transcript]:"

class Language:

    def __init__(self, name: str = "en_ja",
                 source_label: str = DEFAULT_SOURCE_LABEL,
                 target_label: str = DEFAULT_TARGET_LABEL,
                 system_prompt: str | None = None):
        self.name = name
        self.source_label = source_label
        self.target_label = target_label
        self.system_prompt = system_prompt

    def labels(self):
        return (("source", self.source_label), ("target", self.target_label))

    def __repr__(self):
        return f"Language({self.name!r})"

EN_JA = Language()

def load_language(spec: str | None, search_dir=None) -> Language:
    if not spec:
        return EN_JA
    from pathlib import Path as _Path
    import yaml
    base = _Path(search_dir) if search_dir else _Path(__file__).resolve().parents[1] / "lang"
    cand = _Path(spec) if str(spec).endswith((".yaml", ".yml")) else base / str(spec) / "config.yaml"
    if not cand.exists():
        available = sorted(p.name for p in base.iterdir() if (p / "config.yaml").exists()) \
            if base.exists() else []
        raise SystemExit(f"language pair {spec!r} not found (looked for {cand}). "
                         f"Available in {base}: {available or 'none'}")
    cfg = yaml.safe_load(open(cand)) or {}
    missing = [k for k in ("source_label", "target_label") if not cfg.get(k)]
    if missing:
        raise SystemExit(f"{cand}: language pack is missing {missing}")
    sp = cfg.get("system_prompt")
    if sp and not _Path(sp).is_absolute():
        sp = str((cand.parent / sp).resolve())
    return Language(name=cfg.get("name") or cand.stem, source_label=cfg["source_label"],
                    target_label=cfg["target_label"], system_prompt=sp)

def transcript_prefix(rec: dict, lang: Language = EN_JA) -> str:
    t = rec.get("transcripts") or {}
    return "".join(f"{label} {t[key]}\n" for key, label in lang.labels() if t.get(key))

def build_segments(rec: dict, include_transcripts: bool = True,
                   lang: Language = EN_JA) -> list[tuple[str, str]]:
    head = transcript_prefix(rec, lang) if include_transcripts else ""
    segs: list[tuple[str, str]] = []
    for i, (key, label) in enumerate(CLIP_LABELS):

        segs.append(("text", f"{head}{label}\n" if i == 0 else f"\n{label}\n"))
        segs.append(("audio", rec["audios"][key]))
    return segs

def to_structured_content(segments) -> list[dict]:
    return [{"type": "text", "text": v} if kind == "text" else {"type": "audio", "audio": v}
            for kind, v in segments]

def to_swift_content(segments) -> str:
    return "".join(v if kind == "text" else "<audio>" for kind, v in segments)

def audio_paths(segments) -> list[str]:
    return [v for kind, v in segments if kind == "audio"]
