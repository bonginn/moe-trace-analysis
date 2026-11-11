"""Prediction accuracy and expert-usage statistics on traces (CPU).

recall@k   = |predicted top-k ∩ routed set| / topk over decode positions, per target layer (layer 0 not scored);
  topk = the model's experts per token (8 for Qwen3, 4 for gpt-oss, ...). Truth = top-topk of the stored router logits
  when the trace has them (the same forward as the predictor logits), else ``act``.
overlap@1  = |S_t ∩ S_{t+1}| / topk, adjacent decode tokens, per layer.
reuse@d    = P(a routed expert was routed in the previous d decode tokens), per layer.
skew       = share of routed accesses that go to the top 10% experts of the layer (per sequence, then averaged).
"""
from __future__ import annotations

from collections import defaultdict

import torch


def _truth(tr):
    if tr.z_logits is not None:
        return tr.z_logits.float().topk(tr.k, -1).indices
    return tr.act


def recall_one(tr, kinds=("hop1", "pre"), ks=(8, 16)):
    """-> {(kind, k): per-layer hits [L]}, decode tokens x topk."""
    t = _truth(tr)[tr.prompt_len:, 1:]
    acc = {}
    for kind in kinds:
        if kind not in tr.pred_logits and kind not in tr.pred_rank:
            continue
        for k in ks:  # top-k taken per k (not a slice of the top-16: fp16 ties may order differently)
            r = tr.rank(kind, k)[tr.prompt_len:, 1:]
            hit = torch.zeros(tr.L, dtype=torch.float64)
            hit[1:] = (r[..., :k, None] == t[..., None, :]).any(-1).sum((0, 2)).double()
            acc[(kind, k)] = hit
    return acc, (tr.T - tr.prompt_len) * tr.k


def recall(traces, kinds=("hop1", "pre"), ks=(8, 16)):
    """Pooled per-layer recall -> ({(kind, k): tensor [L] in [0, 1], layer 0 = nan}, {source: {(kind, k): mean}})."""
    tot, n = defaultdict(float), 0
    by_src = defaultdict(lambda: [defaultdict(float), 0])
    for tr in traces:
        acc, m = recall_one(tr, kinds, ks)
        for key, v in acc.items():
            tot[key] = tot[key] + v
            by_src[tr.source][0][key] = by_src[tr.source][0][key] + v
        n += m
        by_src[tr.source][1] += m
    per_layer = {key: v / n for key, v in tot.items()}
    for v in per_layer.values():
        v[0] = float("nan")
    src = {s: {key: (v[1:].sum() / (m * (len(v) - 1))).item() for key, v in a.items()} for s, (a, m) in by_src.items()}
    return per_layer, src


def locality(traces, ds=(1, 4, 16)):
    """Per-layer overlap@1 and reuse@d over decode tokens, and per-sequence skew (mean). Token-weighted."""
    L, E, K = traces[0].L, traces[0].num_experts, traces[0].k
    ov = torch.zeros(L, dtype=torch.float64); n_ov = 0
    reuse = {d: torch.zeros(L, dtype=torch.float64) for d in ds}; n_re = {d: 0 for d in ds}
    skews = []
    for tr in traces:
        a = tr.act[tr.prompt_len:].long()
        T = a.shape[0]
        if T < 2:
            continue
        oh = torch.zeros(T, L, E, dtype=torch.float32)
        oh.scatter_(2, a, 1.0)
        ov += (oh[1:] * oh[:-1]).sum(2).sum(0).double() / K; n_ov += T - 1
        for d in ds:
            if T <= d:
                continue
            win = torch.stack([oh[i:T - d + i] for i in range(d)]).amax(0)
            reuse[d] += (oh[d:] * win).sum(2).sum(0).double() / K; n_re[d] += T - d
        cnt = oh.sum(0)
        top = cnt.sort(1, descending=True).values
        skews.append((top[:, :max(E // 10, 1)].sum(1) / cnt.sum(1)).mean().item())
    out = {"overlap@1": ov / max(n_ov, 1), "skew_top10": sum(skews) / max(len(skews), 1)}
    out.update({f"reuse@{d}": reuse[d] / max(n_re[d], 1) for d in ds})
    out["random_overlap@1"] = K / E
    return out
