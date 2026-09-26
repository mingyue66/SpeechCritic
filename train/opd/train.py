"""On-policy distillation for SpeechCritic.

The student generates a trajectory from the ordinary speech-comparison input.
A frozen teacher scores those same tokens, optionally with the reference
rationale as private evidence, and the student minimizes token-level
Jensen--Shannon divergence from the teacher distribution.

The design follows OPSD (Zhao et al., 2026), adapted to multimodal speech and
separate student/teacher models: https://github.com/siyan-zhao/OPSD
"""
from __future__ import annotations
import os, random, shutil, sys, tempfile, time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from peft import LoraConfig, PeftModel, get_peft_model

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import data as judge_data
from common import prompt_format as prompt_layout
from common import runtime as rt
from common.adapters import thinker_relative

TARGETS_LM_ONLY = r"^(model(?=\.).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj))$"
TARGETS_WITH_ENCODER = (
    r"^(model(?=\.).*\.(up_proj|o_proj|gate_proj|v_proj|q_proj|down_proj|k_proj)"
    r"|audio_tower(?=\.).*\.(v_proj|q_proj|fc2|out_proj|fc1|proj|k_proj))$"
)

def init_dist() -> tuple[int, int, torch.device]:
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", rank))
    else:
        rank, world, local = 0, 1, 0
    torch.cuda.set_device(local)
    return rank, world, torch.device(f"cuda:{local}")

def rank0(rank: int) -> bool:
    return rank == 0

def average_grads(model, world: int) -> None:
    if world == 1:
        return
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad /= world

def broadcast_trainable_parameters(model, world: int) -> None:
    if world == 1:
        return
    for parameter in model.parameters():
        if parameter.requires_grad:
            dist.broadcast(parameter.data, src=0)

def system_prompt_path(cfg: dict, lang) -> Path:
    explicit = cfg["model"].get("system_prompt")
    if explicit:
        return Path(__file__).parent / explicit
    if not lang.system_prompt:
        raise SystemExit(f"language pack {lang.name!r} defines no system_prompt, and "
                         f"model.system_prompt is unset -- set one of them")
    return Path(lang.system_prompt)

def student_messages(rec: dict, system: str, include_transcripts: bool, lang) -> list[dict]:
    return [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user", "content": judge_data.to_structured_content(
            judge_data.build_segments(rec, include_transcripts, lang))},
    ]

def teacher_messages(rec: dict, system: str, include_transcripts: bool, lang,
                     privileged: bool) -> list[dict]:
    msgs = student_messages(rec, system, include_transcripts, lang)
    if privileged:
        msgs[-1]["content"].append({"type": "text", "text":
            "Private guidance for this training example. Use it to inform your judgment, but express the "
            "rationale in natural evaluator language rather than quoting this text:\n" + rec["rationale"]})
    return msgs

def encode(processor, conversations: list[list[dict]], device, bf16_floats: bool = False) -> dict:
    from qwen_omni_utils import process_mm_info
    text = processor.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True)
    audios, images, videos = process_mm_info(conversations, use_audio_in_video=False)
    inputs = processor(text=text, audio=audios, images=images, videos=videos,
                       return_tensors="pt", padding=True, use_audio_in_video=False)
    out = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    if bf16_floats:
        out = {k: (v.to(torch.bfloat16) if torch.is_tensor(v) and v.is_floating_point() else v)
               for k, v in out.items()}
    return out

def append_tokens(inputs: dict, tokens: torch.Tensor) -> dict:
    out = dict(inputs)
    out["input_ids"] = torch.cat([inputs["input_ids"], tokens], dim=1)
    if "attention_mask" in inputs:
        pad = torch.ones_like(tokens)
        out["attention_mask"] = torch.cat([inputs["attention_mask"], pad], dim=1)
    return out

def jsd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    log_p_s = torch.log_softmax(student_logits.float(), dim=-1)
    log_p_t = torch.log_softmax(teacher_logits.float(), dim=-1)
    p_s, p_t = log_p_s.exp(), log_p_t.exp()
    log_m = torch.log(0.5 * (p_s + p_t) + 1e-12)
    jsd = 0.5 * (p_t * (log_p_t - log_m)).sum(-1) + 0.5 * (p_s * (log_p_s - log_m)).sum(-1)
    return jsd.mean()

def build_student(cfg, device, workdir):
    from transformers import AutoProcessor, Qwen2_5OmniThinkerForConditionalGeneration
    base = rt.require(cfg, "model.base")
    processor = AutoProcessor.from_pretrained(base, trust_remote_code=True)
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        base, torch_dtype=torch.bfloat16, attn_implementation=cfg["train"]["attn"],
        device_map={"": device})
    init = cfg["model"].get("student_init_adapter")
    if init:

        model = PeftModel.from_pretrained(model, thinker_relative(init, workdir, "student_init"),
                                          is_trainable=True)
    else:
        enc = bool(cfg["lora"]["encoder_trainable"])
        model = get_peft_model(model, LoraConfig(
            r=cfg["lora"]["rank"], lora_alpha=cfg["lora"]["alpha"],
            lora_dropout=cfg["lora"]["dropout"], bias="none", task_type="CAUSAL_LM",
            target_modules=TARGETS_WITH_ENCODER if enc else TARGETS_LM_ONLY))
    model.train()
    return model, processor

def build_teacher(cfg, device, workdir):
    from transformers import AutoProcessor, Qwen2_5OmniThinkerForConditionalGeneration
    ttype = str(cfg["model"].get("teacher_type") or "qwen2_5")
    base = cfg["model"].get("teacher_base") or rt.require(cfg, "model.base")

    if ttype not in ("qwen2_5", "qwen3_omni_moe"):
        raise SystemExit(f"model.teacher_type must be 'qwen2_5' or 'qwen3_omni_moe', got {ttype!r}")
    if ttype == "qwen3_omni_moe":
        if not cfg["model"].get("teacher_base"):
            raise SystemExit("teacher_type=qwen3_omni_moe needs model.teacher_base -- the cross-model "
                             "teacher has its own weights, it is not the student's base")
        if cfg["model"].get("teacher_adapter"):
            raise SystemExit("teacher_adapter applies only to a qwen2_5 teacher; a cross-model teacher "
                             "is used as-is")

    if ttype == "qwen3_omni_moe":
        from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            base, torch_dtype=torch.bfloat16, attn_implementation=cfg["train"]["attn"])
        for m in ("disable_talker", "disable_token2wav"):
            if hasattr(model, m):
                try:
                    getattr(model, m)()
                except Exception:
                    pass
        model = model.to(device)
        processor = Qwen3OmniMoeProcessor.from_pretrained(base)
        processor.tokenizer.padding_side = "left"
    elif ttype == "qwen2_5":
        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            base, torch_dtype=torch.bfloat16, attn_implementation=cfg["train"]["attn"],
            device_map={"": device})
        adapter = cfg["model"].get("teacher_adapter")
        if adapter:
            model = PeftModel.from_pretrained(
                model, thinker_relative(adapter, workdir, "teacher")).merge_and_unload()
        processor = AutoProcessor.from_pretrained(base, trust_remote_code=True)
    else:
        raise SystemExit(f"unhandled teacher_type {ttype!r}")

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    callable_module = getattr(model, "thinker", model)
    return model, processor, ttype, callable_module

def main() -> None:
    ap = rt.parser("SpeechCritic OPD")
    rt.add_common_args(ap, Path(__file__).parent, dry_run=False)
    args = ap.parse_args()
    cfg = rt.load_config(args.config, args.override)

    rank, world, device = init_dist()
    torch.manual_seed(cfg["train"]["seed"] + rank)
    random.seed(cfg["train"]["seed"] + rank)

    out_dir = Path(rt.require(cfg, "output.dir"))
    if rank0(rank):
        out_dir.mkdir(parents=True, exist_ok=True)

    lang = prompt_layout.load_language(cfg["data"].get("language"))
    privileged = bool(cfg["model"].get("teacher_privileged", True))
    required = ("rationale",) if privileged else ()
    recs = judge_data.load(rt.require(cfg, "data.train_jsonl"),
                       audio_root=cfg["data"].get("audio_root"), require=required)
    if rank0(rank):
        rt.write_training_metadata(out_dir, cfg, "opd", Path(__file__).parent, {
            "n_train": len(recs),
            "teacher_type": cfg["model"].get("teacher_type") or "qwen2_5",
            "teacher_base": cfg["model"].get("teacher_base") or cfg["model"]["base"],
            "teacher_adapter": cfg["model"].get("teacher_adapter"),
            "teacher_privileged": privileged,
            "student_init_adapter": cfg["model"].get("student_init_adapter"),
            "language": cfg["data"].get("language") or "en_ja",
                      "system_prompt": system_prompt_path(cfg, lang).name})
    order = list(range(len(recs)))
    random.Random(cfg["train"]["seed"]).shuffle(order)
    if rank0(rank):
        print(f"[data] {len(recs)} examples")

    work = Path(tempfile.mkdtemp(prefix=f"opd_rank{rank}_"))

    student, s_proc = build_student(cfg, device, work)
    broadcast_trainable_parameters(student, world)
    torch.manual_seed(cfg["train"]["seed"] + rank)
    teacher, t_proc, t_type, teacher_fwd = build_teacher(cfg, device, work)

    def _vocab(m):
        for attr in ("lm_head", "thinker"):
            sub = getattr(m, attr, None)
            if sub is not None:
                head = getattr(sub, "lm_head", sub)
                w = getattr(head, "weight", None)
                if w is not None:
                    return w.shape[0]
        cfgm = getattr(m, "config", None)
        return getattr(cfgm, "vocab_size", None)
    v_s, v_t = _vocab(student), _vocab(teacher)
    if v_s and v_t and v_s != v_t:
        raise SystemExit(f"student vocabulary is {v_s} tokens, teacher is {v_t}. A JSD between them is "
                         f"undefined -- a cross-model teacher must share the student's vocabulary.")
    if rank0(rank):
        print(f"[vocab] student and teacher agree on {v_s} tokens")
    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    if rank0(rank):
        print(f"[student] trainable params: {trainable:,}  encoder_trainable={cfg['lora']['encoder_trainable']}")
        print(f"[teacher] frozen; type={t_type} "
              f"base={cfg['model'].get('teacher_base') or cfg['model']['base']} "
              f"adapter={cfg['model'].get('teacher_adapter') or '(none)'} "
              f"privileged={privileged}")

    opt = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=float(cfg["train"]["lr"]),
        betas=(float(cfg["train"]["adam_beta1"]), float(cfg["train"]["adam_beta2"])),
        eps=float(cfg["train"]["adam_eps"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    system = judge_data.read_prompt(system_prompt_path(cfg, lang))
    incl = cfg["data"].get("include_transcripts", True)
    if int(cfg["train"].get("batch_size", 1)) != 1:
        raise SystemExit("the released OPD loop currently requires train.batch_size=1")
    accum = int(cfg["train"]["grad_accum"])
    lmax = int(cfg["train"]["loss_max_tokens"])
    cursor = rank
    t0 = time.time()

    for step in range(1, int(cfg["train"]["max_steps"]) + 1):
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(accum):
            rec = recs[order[cursor % len(order)]]
            cursor += world

            s_inputs = encode(s_proc, [student_messages(rec, system, incl, lang)], device)
            with torch.no_grad():
                gen = student.generate(**s_inputs, max_new_tokens=int(cfg["train"]["max_new_tokens"]),
                                       do_sample=False, eos_token_id=None)
            prompt_len = s_inputs["input_ids"].shape[1]
            completion = gen[:, prompt_len:]
            if completion.shape[1] == 0:
                continue
            if lmax > 0:
                completion = completion[:, :lmax]

            s_scored = append_tokens(s_inputs, completion)
            s_logits = student(**s_scored).logits[:, prompt_len - 1:-1, :]

            t_inputs = encode(t_proc, [teacher_messages(rec, system, incl, lang, privileged)], device,
                              bf16_floats=(t_type == "qwen3_omni_moe"))
            t_prompt_len = t_inputs["input_ids"].shape[1]
            t_scored = append_tokens(t_inputs, completion)
            with torch.no_grad():

                t_logits = teacher_fwd(**t_scored).logits[:, t_prompt_len - 1:-1, :]

            n = min(s_logits.shape[1], t_logits.shape[1])
            loss = jsd_loss(s_logits[:, :n, :], t_logits[:, :n, :]) / accum
            loss.backward()
            total += loss.item()

        average_grads(student, world)
        torch.nn.utils.clip_grad_norm_(
            [p for p in student.parameters() if p.requires_grad],
            float(cfg["train"]["max_grad_norm"]),
        )
        opt.step()

        if rank0(rank):
            el = time.time() - t0
            per = el / step
            left = per * (int(cfg["train"]["max_steps"]) - step)
            mem = torch.cuda.max_memory_allocated(device) / 2**30
            print(f"[step {step}/{cfg['train']['max_steps']}] jsd={total:.4f} "
                  f"{per:.1f}s/step elapsed={el / 3600:.1f}h eta={left / 3600:.1f}h "
                  f"peak={mem:.1f}GiB", flush=True)
        if rank0(rank) and step % int(cfg["train"]["save_every"]) == 0:
            ck = out_dir / f"checkpoint-{step}"
            student.save_pretrained(str(ck))
            print(f"[save] {ck}", flush=True)

    if rank0(rank):
        student.save_pretrained(str(out_dir / f"checkpoint-{cfg['train']['max_steps']}"))
    shutil.rmtree(work, ignore_errors=True)
    if world > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
