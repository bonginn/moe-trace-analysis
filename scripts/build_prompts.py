"""Build the multi-domain prompt set (1714 prompts; math / code / English chat / Chinese) from public HF datasets.

Deterministic: for each source, the first N records (streaming order) whose prompt is 20..6000 characters long.
Math and code prompts are marked ``thinking: true`` (reasoning-mode generation), chat / zh ``false``.

    pip install datasets
    python scripts/build_prompts.py --out prompts.jsonl                                   # all 1714
    python scripts/build_prompts.py --out prompts_96.jsonl --subset scripts/subset_96.txt  # the 96-sequence set

Needs network access (datasets are streamed from the HF Hub). Each source dataset keeps its own license
(see data/README.md); the 96-prompt subset is also shipped as data/prompts_96.jsonl.
"""
import argparse
import json

TH = "\nPlease reason step by step."
SOURCES = [  # name, domain, load_dataset args, kwargs, prompt fn, n, thinking, max_tokens
    ("math500", "math", ("HuggingFaceH4/MATH-500",), {"split": "test"}, lambda x: x["problem"] + TH, 300, True, 4096),
    ("gsm8k", "math", ("openai/gsm8k", "main"), {"split": "train"}, lambda x: x["question"], 300, True, 3072),
    ("mbpp", "code", ("google-research-datasets/mbpp", "full"), {"split": "train"},
     lambda x: x["text"] + "\nWrite a Python function and explain it. Example test: " + x["test_list"][0], 300, True, 4096),
    ("humaneval", "code", ("openai/openai_humaneval",), {"split": "test"},
     lambda x: "Complete the following Python function and explain your solution.\n\n" + x["prompt"], 164, True, 4096),
    ("ultrachat", "chat", ("HuggingFaceH4/ultrachat_200k",), {"split": "train_sft"}, lambda x: x["prompt"], 300, False, 2048),
    ("openorca", "chat", ("Open-Orca/OpenOrca",), {"split": "train"},
     lambda x: (x["system_prompt"] + "\n\n" if x["system_prompt"] else "") + x["question"], 150, False, 2048),
    ("zh_alpaca", "zh", ("shibing624/alpaca-zh",), {"split": "train"},
     lambda x: x["instruction"] + ("\n" + x["input"] if x.get("input") else ""), 200, False, 2048),
]


def build(only=None):
    from datasets import load_dataset

    rows = []
    for name, dom, ds_args, kw, fn, n, think, mt in SOURCES:
        got = 0
        for x in load_dataset(*ds_args, streaming=True, **kw):
            p = fn(x).strip()
            if 20 < len(p) < 6000:
                rid = f"{name}_{got:04d}"
                if only is None or rid in only:
                    rows.append({"id": rid, "source": name, "domain": dom, "thinking": think, "max_tokens": mt, "prompt": p})
                got += 1
            if got >= n:
                break
        print(f"{name}: {got}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--subset", help="file with one prompt id per line (e.g. scripts/subset_96.txt)")
    a = ap.parse_args()
    only = {x.strip() for x in open(a.subset) if x.strip()} if a.subset else None
    rows = build(only)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} prompts to {a.out}")


if __name__ == "__main__":
    main()
