"""Collect expert-routing traces with HF transformers (GPU; one forward per sequence for the recording).

  generate  prompts.jsonl ({"id", "prompt", "source"?, "thinking"?} per line) -> chat template -> model.generate
            (sampling settings below) -> replay the full token sequence once with the Recorder
  replay    existing token sequences (a native traces dir, or a split-layout dir with ``seqs/``) -> Recorder

Recording runs as a single teacher-forced forward (prefill) over prompt + generation, so router logits and
predictions come from the same pass. Output: one ``<id>.pt`` Trace per sequence (moe_trace.traces).

    CUDA_VISIBLE_DEVICES=<idle gpu> python -m moe_trace collect generate --model Qwen/Qwen3.6-35B-A3B \
        --prompts prompts.jsonl --out traces/q36 [--max-new-tokens 1024] [--cpu-offload-gib 60]
"""
from __future__ import annotations

import getpass
import json
import os
import subprocess
import time
from pathlib import Path

import torch

from moe_trace.recorder import Recorder
from moe_trace.traces import Trace, load_dir, save

# Qwen-recommended sampling (thinking / non-thinking); use --greedy for deterministic decoding
SAMPLING = {True: dict(temperature=0.6, top_p=0.95, top_k=20), False: dict(temperature=0.7, top_p=0.8, top_k=20)}


def check_gpus_idle() -> None:
    """Refuse to run on GPUs where other users have processes (shared server)."""
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not vis:
        raise SystemExit("set CUDA_VISIBLE_DEVICES explicitly after checking nvidia-smi")
    q = lambda s: subprocess.check_output(["nvidia-smi", f"--query-{s}", "--format=csv,noheader"], text=True).split("\n")
    bus = dict(x.split(", ") for x in q("gpu=index,pci.bus_id") if x)
    busy = []
    for line in q("compute-apps=gpu_bus_id,pid"):
        if not line:
            continue
        b, pid = line.split(", ")
        user = subprocess.run(["ps", "-o", "user=", "-p", pid], capture_output=True, text=True).stdout.strip()
        idx = next((i for i, x in bus.items() if x == b), "?")
        if idx in vis.split(",") and user != getpass.getuser():
            busy.append(f"GPU {idx}: {user} (pid {pid})")
    if busy:
        raise SystemExit("GPUs in use by others:\n  " + "\n  ".join(busy))


def load_model(name: str, cpu_offload_gib: float = 0.0):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    mm = {i: f"{int(torch.cuda.mem_get_info(i)[0] / 2**30) - 4}GiB" for i in range(torch.cuda.device_count())}
    mm["cpu"] = f"{int(cpu_offload_gib)}GiB"
    kw = dict(dtype=torch.bfloat16, device_map="auto", max_memory=mm)
    try:
        model = AutoModelForCausalLM.from_pretrained(name, **kw)
    except ValueError:  # multimodal checkpoints (e.g. Qwen3.6) only map to the image-text class
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(name, **kw)
    model.eval()
    return model, AutoTokenizer.from_pretrained(name)


@torch.inference_mode()
def record(model, rec: Recorder, tokens: torch.Tensor) -> dict:
    body = getattr(model, "model", model)  # skip the lm_head: only the routing is needed
    body(input_ids=tokens[None].to(rec.dev))
    return rec.pack()


def to_trace(name, tokens, P, lg, k, **meta) -> Trace:
    act = lg["z"].float().topk(k, -1).indices
    pred = {}
    for kind in ("hop1", "pre"):
        pred[kind] = lg[kind]
    return Trace(name=name, prompt_len=P, act=act, num_experts=lg["z"].shape[-1], z_logits=lg["z"],
                 pred_logits=pred, tokens=tokens, **meta)


def generate(args):
    check_gpus_idle()
    model, tok = load_model(args.model, args.cpu_offload_gib)
    rec = Recorder(model)
    k = int(getattr(model.config, "num_experts_per_tok", 0) or getattr(model.config.get_text_config(), "num_experts_per_tok", 8))
    out = Path(args.out)
    rows = [json.loads(x) for x in open(args.prompts)]
    rows = rows[:args.limit] if args.limit else rows
    t0 = time.time()
    for i, r in enumerate(rows):
        if (out / f"{r['id']}.pt").exists():
            continue
        think = bool(r.get("thinking", False))
        ids = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], add_generation_prompt=True,
                                      return_tensors="pt", enable_thinking=think)
        ids = ids["input_ids"] if isinstance(ids, dict) or hasattr(ids, "keys") else ids
        torch.manual_seed(args.seed + i)
        samp = dict(SAMPLING[think]) if not args.greedy else dict(do_sample=False)
        if not args.greedy:
            samp["do_sample"] = True
        gen = model.generate(ids.to(rec.dev), max_new_tokens=args.max_new_tokens, **samp)[0].cpu()
        P = ids.shape[1]
        lg = record(model, rec, gen)
        save(to_trace(r["id"], gen, P, lg, k, source=r.get("source", ""), thinking=think, model=args.model), out)
        print(f"[{i + 1}/{len(rows)}] {r['id']}: {P} prompt + {len(gen) - P} new tokens ({time.time() - t0:.0f}s)",
              flush=True)


def replay(args):
    check_gpus_idle()
    model, _ = load_model(args.model, args.cpu_offload_gib)
    rec = Recorder(model)
    out = Path(args.out)
    src = load_dir(args.traces, limit=args.limit)
    t0 = time.time()
    for i, tr in enumerate(src):
        if (out / f"{tr.name}.pt").exists() or tr.tokens is None:
            continue
        lg = record(model, rec, tr.tokens)
        save(to_trace(tr.name, tr.tokens, tr.prompt_len, lg, tr.k, source=tr.source, thinking=tr.thinking,
                      model=args.model), out)
        print(f"[{i + 1}/{len(src)}] {tr.name} ({time.time() - t0:.0f}s)", flush=True)
