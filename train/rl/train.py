from __future__ import annotations
import json, os, subprocess, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import data as judge_data
from common import prompt_format as prompt_layout
from common import runtime as rt
from common.output_format import DIMS

TARGETS_LM_ONLY = (
    r"^(thinker\.model(?=\.).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj))$"
)
TARGETS_WITH_ENCODER = (
    r"^(thinker\.model(?=\.).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
    r"|thinker\.audio_tower(?=\.).*\.(q_proj|k_proj|v_proj|out_proj|fc1|fc2|proj))$"
)

def system_prompt_path(cfg: dict, lang) -> Path:
    explicit = cfg["model"].get("system_prompt")
    if explicit:
        return Path(__file__).parent / explicit
    if not lang.system_prompt:
        raise SystemExit(f"language pack {lang.name!r} defines no system_prompt, and "
                         f"model.system_prompt is unset -- set one of them")
    return Path(lang.system_prompt)

def build_dataset(cfg, out_path, lang) -> int:
    recs = judge_data.load(rt.require(cfg, "data.train_jsonl"),
                       audio_root=cfg["data"].get("audio_root"),
                       require=("labels",))
    system = judge_data.read_prompt(system_prompt_path(cfg, lang))
    with open(out_path, "w") as f:
        for r in recs:
            segs = judge_data.build_segments(r, cfg["data"].get("include_transcripts", True), lang)
            f.write(json.dumps({
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": judge_data.to_swift_content(segs)},
                ],
                "audios": judge_data.audio_paths(segs),

                "label": r["labels"]["overall"],
                "per_dim": {d: r["labels"]["dims"][d] for d in DIMS},
            }, ensure_ascii=False) + "\n")
    return len(recs)

def merge_init_adapter(base: str, adapter: str, work: Path) -> str:
    merged = work / "base_with_init"
    if merged.exists():
        print(f"[init] reusing {merged}")
        return str(merged)
    print(f"[init] merging {adapter} into {base} -> {merged}")
    subprocess.run([rt.swift_exe(), "export", "--model", base, "--adapters", adapter,
                    "--merge_lora", "true", "--output_dir", str(merged),
                    "--torch_dtype", "bfloat16"], check=True)
    return str(merged)

def main() -> None:
    ap = rt.parser("SpeechCritic RL")
    rt.add_common_args(ap, Path(__file__).parent)
    args = ap.parse_args()

    cfg = rt.load_config(args.config, args.override)
    out_dir = Path(rt.require(cfg, "output.dir")); out_dir.mkdir(parents=True, exist_ok=True)
    staged = out_dir / "rl_dataset.jsonl"
    lang = prompt_layout.load_language(cfg["data"].get("language"))
    n = build_dataset(cfg, staged, lang)
    print(f"[data] {n} prompts -> {staged}")
    rt.write_training_metadata(out_dir, cfg, "rl", Path(__file__).parent, {
        "n_train": n,

        "merge_first": cfg["model"].get("init_adapter"),
        "algo": cfg["train"]["algo"],
        "language": cfg["data"].get("language") or "en_ja",
                      "system_prompt": system_prompt_path(cfg, lang).name,
        "reward_funcs": rt.require(cfg, "reward.funcs"),
        "dim_weight": cfg["reward"].get("dim_weight")})

    model = rt.require(cfg, "model.base")
    if cfg["model"].get("init_adapter"):
        if args.dry_run:

            print(f"[init] --dry-run: would merge {cfg['model']['init_adapter']} into {model}")
            model = "<base with init_adapter merged>"
        else:
            model = merge_init_adapter(model, cfg["model"]["init_adapter"], out_dir)

    ngen = int(cfg["train"]["num_generations"])
    gen_batch = int(cfg["train"]["batch_size"]) * int(cfg["train"]["grad_accum"]) * rt.world_size()
    if gen_batch % ngen:
        raise SystemExit(
            f"train.batch_size x train.grad_accum x (GPUs) = {gen_batch} must be divisible by "
            f"train.num_generations = {ngen}. Adjust grad_accum or num_generations "
            f"(the defaults, 32 and 8, satisfy this)."
        )

    funcs = rt.require(cfg, "reward.funcs")
    if isinstance(funcs, str):
        funcs = [funcs]
    enc = bool(cfg["lora"]["encoder_trainable"])
    env = dict(
        os.environ,
        SPEECHCRITIC_DIM_WEIGHT=str(cfg["reward"].get("dim_weight", 0.3)),
    )

    algo = str(cfg["train"]["algo"]).lower()
    if algo not in ("dapo", "grpo"):
        raise SystemExit(f"train.algo must be 'dapo' or 'grpo', got {algo!r}")
    cmd = [
        rt.swift_exe(), "rlhf",
        "--rlhf_type", "grpo",
        "--model", model,
        "--dataset", str(staged),
        "--external_plugins", str(Path(__file__).parent / "rewards.py"),
        "--reward_funcs", *funcs,
        "--tuner_type", "lora",
        "--lora_rank", str(cfg["lora"]["rank"]),
        "--lora_alpha", str(cfg["lora"]["alpha"]),
        "--lora_dropout", str(cfg["lora"]["dropout"]),

        *(["--target_regex", cfg["lora"].get("target_regex") or TARGETS_WITH_ENCODER]
          if enc else ["--target_modules", "all-linear"]),
        "--freeze_vit", "false" if enc else "true",
        "--freeze_aligner", "false" if enc else "true",
        "--torch_dtype", "bfloat16",
        "--num_generations", str(cfg["train"]["num_generations"]),
        "--per_device_train_batch_size", str(cfg["train"]["batch_size"]),
        "--gradient_accumulation_steps", str(cfg["train"]["grad_accum"]),
        "--learning_rate", str(cfg["train"]["lr"]),
        "--temperature", str(cfg["train"]["temperature"]),
        "--top_p", str(cfg["train"]["top_p"]),
        "--max_completion_length", str(cfg["train"]["max_completion_length"]),
        "--max_length", str(cfg["train"]["max_length"]),
        "--max_steps", str(cfg["train"]["max_steps"]),
        "--save_steps", str(cfg["train"]["save_every"]),
        "--save_total_limit", "-1",
        "--logging_steps", "1",
        "--seed", str(cfg["train"]["seed"]),
        "--output_dir", str(out_dir),
    ]

    cmd += ["--beta", str(cfg["train"]["kl_beta"]),
            "--epsilon", str(cfg["train"]["clip_epsilon"])]
    if algo == "dapo":
        cmd += ["--epsilon_high", str(cfg["train"]["clip_epsilon_high"]),
                "--dynamic_sample", str(cfg["train"]["dynamic_sample"]).lower(),
                "--overlong_filter", str(cfg["train"]["overlong_filter"]).lower(),
                "--loss_type", str(cfg["train"]["loss_type"]),
                "--importance_sampling_level", str(cfg["train"]["importance_sampling_level"])]

    v = cfg["train"].get("vllm") or {}
    if v.get("enabled", True):

        cmd += ["--use_vllm", "true",
                "--vllm_mode", str(v.get("mode", "colocate")),
                "--vllm_gpu_memory_utilization", str(v.get("gpu_memory_utilization", 0.35))]

    weights = cfg["reward"].get("weights") or []
    if weights:
        if len(weights) != len(funcs):
            raise SystemExit("reward.weights must be empty or the same length as reward.funcs")
        cmd += ["--reward_weights", *[str(w) for w in weights]]

    print(f"[reward] {funcs}  dim_weight={cfg['reward'].get('dim_weight')}")
    print(f"[lora] encoder_trainable={enc}")
    print("[cmd] " + " ".join(cmd))
    if args.dry_run:
        return
    raise SystemExit(subprocess.call(cmd, env=env))

if __name__ == "__main__":
    main()
