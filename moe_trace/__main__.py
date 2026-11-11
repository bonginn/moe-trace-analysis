"""moe_trace CLI.

  python -m moe_trace collect generate --model M --prompts P.jsonl --out DIR      (GPU)
  python -m moe_trace collect replay   --model M --traces SRC --out DIR          (GPU)
  python -m moe_trace report DIR [--cache 0.25 0.40] [--B 0 2] [--out report.txt]   (CPU)

``report`` = locality / skew, cross-layer prediction recall (hop1 / hop-1.5),
and per-layer cache hit rates (pure LRU, LRU + prefetch of B, LRFU+protect + prefetch) at the given cache sizes.
DIR is a native trace dir or the split layout (a dir with seqs/ and optionally logits/).
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict


def _pct(f):  # 0.125 -> "12.5%", 0.25 -> "25%"
    return f"{100 * f:g}%"


def report(a):
    from moe_trace import cachesim, metrics
    from moe_trace.traces import load_dir

    t0 = time.time()
    trs = load_dir(a.traces, limit=a.limit)
    L, E, k = trs[0].L, trs[0].num_experts, trs[0].k
    ndec = sum(t.T - t.prompt_len for t in trs)
    src = defaultdict(int)
    for t in trs:
        src[t.source] += 1
    w = a.workers
    out = [f"traces {a.traces}: {len(trs)} seqs, {ndec} decode tokens, {L} MoE layers x {E} experts, top-{k}",
           "sources: " + ", ".join(f"{s or '?'} {n}" for s, n in sorted(src.items())), ""]

    loc = metrics.locality(trs)
    out += ["## expert usage (decode)",
            f"  overlap@1 (|S_t ∩ S_t+1| / {k})  mean {loc['overlap@1'].mean():.1%}  (random {loc['random_overlap@1']:.1%})"
            f"   L0 {loc['overlap@1'][0]:.1%}  last {loc['overlap@1'][-1]:.1%}",
            "  " + "   ".join(f"{key} {v.mean():.1%}" for key, v in loc.items() if key.startswith("reuse@")),
            f"  skew: share of accesses to the top-10% experts {loc['skew_top10']:.1%} (uniform 10%)", ""]

    per, by_src = metrics.recall(trs)
    out.append(f"## cross-layer prediction recall (decode, layers 1..L-1; truth = router top-{k})")
    for key in sorted(per):
        kind, kk = key
        r = per[key][1:]
        q = [r[i * len(r) // 4:(i + 1) * len(r) // 4].mean().item() for i in range(4)] if len(r) >= 4 else []
        out.append(f"  {kind:5s} @{kk:<2d} mean {r.mean():6.1%}" +
                   ("   by layer quarter " + " ".join(f"{x:6.1%}" for x in q) if q else "") + f"   layer 1 {r[0]:6.1%}")
    if len(by_src) > 1:
        keys = sorted(k_ for k_ in per if k_[1] == 8)
        out.append("  by source (@8): " + " | ".join(kk[0] for kk in keys))
        for s in sorted(by_src):
            out.append(f"    {s or '?':12s} " + "  ".join(f"{by_src[s].get(kk, float('nan')):6.1%}" for kk in keys))
    out.append("")

    curve = cachesim.lru_curve(trs, workers=w)
    caps = [int(round(f * E)) for f in a.cache]
    out.append("## per-layer cache hit rate (uniform capacity per layer; prompt warms; decode hits, mean over layers)")
    out.append("  pure LRU:        " + "   ".join(f"{_pct(f)} ({c}/layer) {curve[:, min(c, E)].mean():.1%}"
                                                 for f, c in zip(a.cache, caps)))
    kind = "hop1"
    pf = cachesim.prefetch_hit(trs, caps, Bs=a.B, kind=kind, workers=w)
    for bi, B in enumerate(a.B):
        out.append(f"  LRU + {kind} B={B}: " + "   ".join(f"{_pct(f)} {pf[bi, ci].mean():.1%}" for ci, f in enumerate(a.cache)))
    if not a.no_integrated:
        ig = cachesim.integrated(trs, caps, Bs=a.B, kind=kind, workers=w)
        for pi, pol in enumerate(cachesim.POLICIES):
            for bi, B in enumerate(a.B):
                out.append(f"  {pol:12s} B={B}: " + "   ".join(
                    f"{_pct(f)} {ig['hit'][pi, bi, ci].mean():.1%} (fetch/tok {ig['fetch_per_tok'][pi, bi, ci]:.1f})"
                    for ci, f in enumerate(a.cache)))
    out.append(f"\n({time.time() - t0:.0f}s)")
    text = "\n".join(out)
    print(text)
    if a.out:
        open(a.out, "w").write(text + "\n")


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m moe_trace")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    cs = c.add_subparsers(dest="mode", required=True)
    for mode in ("generate", "replay"):
        m = cs.add_parser(mode)
        m.add_argument("--model", required=True)
        m.add_argument("--out", required=True)
        m.add_argument("--limit", type=int, default=0)
        m.add_argument("--cpu-offload-gib", type=float, default=0.0)
        if mode == "generate":
            m.add_argument("--prompts", required=True)
            m.add_argument("--max-new-tokens", type=int, default=1024)
            m.add_argument("--seed", type=int, default=0)
            m.add_argument("--greedy", action="store_true")
        else:
            m.add_argument("--traces", required=True)
    r = sub.add_parser("report")
    r.add_argument("traces")
    r.add_argument("--cache", type=float, nargs="+", default=[0.25, 0.40])
    r.add_argument("--B", type=int, nargs="+", default=[0, 2])
    r.add_argument("--no-integrated", action="store_true")
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--workers", type=int, default=None)
    r.add_argument("--out", default="")
    a = p.parse_args(argv)
    if a.cmd == "report":
        report(a)
    else:
        from moe_trace import collect

        getattr(collect, a.mode)(a)


if __name__ == "__main__":
    sys.exit(main())
