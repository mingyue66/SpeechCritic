from __future__ import annotations
import json, random, subprocess, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import data as judge_data
from common import prompt_format as prompt_layout
from common import runtime as rt

def system_prompt_path(cfg: dict, lang) -> Path:
    explicit = cfg["model"].get("system_prompt")
    if explicit:
        return Path(__file__).parent / explicit
    if not lang.system_prompt:
        raise SystemExit(f"language pack {lang.name!r} defines no system_prompt, and "
                         f"model.system_prompt is unset -- set one of them")
    return Path(lang.system_prompt)

def build_dataset(cfg: dict, out_path: Path, lang) -> int:
    recs = judge_data.load(rt.require(cfg, "data.train_jsonl"),
                       audio_root=cfg["data"].get("audio_root"),
                       require=("rationale",))
    frac = float(cfg["data"].get("fraction", 1.0))
    if not 0 < frac <= 1:
        raise SystemExit("data.fraction must be in (0, 1]")
    if frac < 1.0:
        rng = random.Random(cfg["train"]["seed"])
        recs = rng.sample(recs, int(round(len(recs) * frac)))
    system = judge_data.read_prompt(system_prompt_path(cfg, lang))
    n = 0
    with open(out_path, "w") as f:
        for r in recs:
            segs = judge_data.build_segments(r, cfg["data"].get("include_transcripts", True), lang)
            f.write(json.dumps({
                "messages": [
                    {"role": "system", "content": system},

                    {"role": "user", "content": judge_data.to_swift_content(segs)},
                    {"role": "assistant", "content": r["rationale"]},
                ],
                "audios": judge_data.audio_paths(segs),
            }, ensure_ascii=False) + "\n")
            n += 1
    return n

TARGETS_WITH_ENCODER = (
    r"^(thinker\.model(?=\.).*\.(o_proj|q_proj|down_proj|k_proj|gate_proj|v_proj|up_proj)"
    r"|thinker\.audio_tower(?=\.).*\.(fc1|q_proj|k_proj|out_proj|proj|v_proj|fc2))$"
)

def main() -> None:
    ap = rt.parser("SpeechCritic SFT")
    rt.add_common_args(ap, Path(__file__).parent)
    args = ap.parse_args()

    cfg = rt.load_config(args.config, args.override)
    out_dir = Path(rt.require(cfg, "output.dir")); out_dir.mkdir(parents=True, exist_ok=True)
    staged = out_dir / "sft_dataset.jsonl"
    lang = prompt_layout.load_language(cfg["data"].get("language"))
    n = build_dataset(cfg, staged, lang)
    print(f"[data] {n} examples -> {staged}")
    rt.write_training_metadata(out_dir, cfg, "sft", Path(__file__).parent,
                     {"n_train": n, "data_fraction": cfg["data"].get("fraction", 1.0),
                      "language": cfg["data"].get("language") or "en_ja",
                      "system_prompt": system_prompt_path(cfg, lang).name})

    enc = bool(cfg["lora"]["encoder_trainable"])
    cmd = [
        rt.swift_exe(), "sft",
        "--model", rt.require(cfg, "model.base"),
        "--dataset", str(staged),
        "--tuner_type", "lora",
        "--lora_rank", str(cfg["lora"]["rank"]),
        "--lora_alpha", str(cfg["lora"]["alpha"]),
        "--lora_dropout", str(cfg["lora"]["dropout"]),

        *(["--target_regex", cfg["lora"].get("target_regex") or TARGETS_WITH_ENCODER]
          if enc else ["--target_modules", "all-linear"]),
        "--freeze_vit", "false" if enc else "true",
        "--freeze_llm", "false",
        "--torch_dtype", "bfloat16" if cfg["train"]["bf16"] else "float32",
        "--save_steps", str(cfg["train"]["save_every"]),
        "--per_device_train_batch_size", str(cfg["train"]["batch_size"]),
        "--gradient_accumulation_steps", str(cfg["train"]["grad_accum"]),
        "--learning_rate", str(cfg["train"]["lr"]),
        "--lr_scheduler_type", str(cfg["train"]["lr_scheduler"]),
        "--warmup_ratio", str(cfg["train"]["warmup_ratio"]),
        "--weight_decay", str(cfg["train"]["weight_decay"]),
        "--gradient_checkpointing", "true" if cfg["train"]["gradient_checkpointing"] else "false",
        "--max_length", str(cfg["train"]["max_length"]),
        "--split_dataset_ratio", "0",
        "--dataset_shuffle", "true",
        "--lazy_tokenize", "true",
        "--dataloader_num_workers", "4",
        "--seed", str(cfg["train"]["seed"]),
        "--output_dir", str(out_dir),
        "--save_total_limit", "-1",
        "--logging_steps", "5",
    ]
    if cfg["train"].get("deepspeed"):
        cmd += ["--deepspeed", str(cfg["train"]["deepspeed"])]

    epochs, max_steps = cfg["train"].get("epochs"), cfg["train"].get("max_steps")
    if epochs and max_steps:
        raise SystemExit("set train.epochs or train.max_steps, not both (max_steps would win and the "
                         "epoch count would be silently ignored)")
    if not epochs and not max_steps:
        raise SystemExit("set train.epochs or train.max_steps")
    if epochs:
        cmd += ["--num_train_epochs", str(epochs)]
    else:
        cmd += ["--max_steps", str(max_steps)]

    print(f"[lora] encoder_trainable={enc}  targets="
          + ("regex(thinker.model + audio_tower incl. proj)" if enc else "all-linear"))
    print(f"[lang] {lang.name}  system_prompt={system_prompt_path(cfg, lang)}")
    print("[train] " + (f"{epochs} epochs" if epochs else f"{max_steps} steps"))
    print("[cmd] " + " ".join(cmd))
    if args.dry_run:
        return
    raise SystemExit(subprocess.call(cmd))

if __name__ == "__main__":
    main()
