"""Synthetic checks (CPU, no model, no data): cache simulations agree with each other and with brute force, recall
is exact for a perfect predictor, locality of fixed routing."""
import numpy as np
import torch

from moe_trace import cachesim, metrics
from moe_trace.traces import Trace


def rand_trace(T=80, L=3, E=16, k=2, P=10, seed=0, name="s"):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(T, L, E, generator=g)
    z += torch.linspace(0, 3, E)[torch.randperm(E, generator=g)]  # skewed popularity -> non-trivial cache behaviour
    act = z.topk(k, -1).indices
    pred = z + 0.8 * torch.randn(T, L, E, generator=g)
    pred[:, 0] = 0
    return Trace(name=name, prompt_len=P, act=act, num_experts=E, z_logits=z.half(), pred_logits={"hop1": pred.half()})


def brute_lru(tr, cap):
    hits = n = 0
    for l in range(tr.L):
        od = []
        for t in range(tr.T):
            for e in tr.act[t, l].tolist():
                if t >= tr.prompt_len:
                    hits += e in od; n += 1
            for e in tr.act[t, l].tolist():
                if e in od:
                    od.remove(e)
                od.append(e)
                if len(od) > cap:
                    od.pop(0)
    return hits / n


def test_lru_curve_matches_brute_force_and_prefetch_b0():
    trs = [rand_trace(seed=s, name=f"s{s}") for s in range(3)]
    curve = cachesim.lru_curve(trs, workers=1)
    caps = [2, 4, 8, 12]
    pf = cachesim.prefetch_hit(trs, caps, Bs=(0,), workers=1)
    ig = cachesim.integrated(trs, caps, Bs=(0,), policies=("LRU",), workers=1)
    for ci, c in enumerate(caps):
        ref = np.mean([brute_lru(tr, c) for tr in trs])  # all traces have the same length -> plain mean
        assert abs(curve[:, c].mean() - ref) < 1e-9, (c, curve[:, c].mean(), ref)
        assert abs(pf[0, ci].mean() - ref) < 1e-9
        assert abs(ig["hit"][0, 0, ci].mean() - ref) < 1e-9


def test_prefetch_helps_and_per_layer_budgets():
    trs = [rand_trace(seed=s, name=f"s{s}") for s in range(3)]
    pf = cachesim.prefetch_hit(trs, [4], Bs=(0, 2, [0, 2, 2]), workers=1)
    assert pf[1, 0, 1:].mean() > pf[0, 0, 1:].mean()  # a good predictor raises the hit rate
    assert np.allclose(pf[1, 0], pf[2, 0])  # per-layer [0, 2, 2] == B=2 (layer 0 is never prefetched)


def test_recall_perfect_and_noisy():
    tr = rand_trace(T=100, E=16, k=8)
    tr.pred_logits = {"hop1": tr.z_logits.clone()}
    per, _ = metrics.recall([tr], kinds=("hop1",), ks=(8, 16))
    assert torch.allclose(per[("hop1", 8)][1:], torch.ones(tr.L - 1, dtype=torch.float64))
    tr = rand_trace(T=100, E=16, k=8)  # noisy predictor: 0 < recall@8 < recall@16 <= 1
    per, _ = metrics.recall([tr], kinds=("hop1",), ks=(8, 16))
    r8, r16 = per[("hop1", 8)][1:].mean(), per[("hop1", 16)][1:].mean()
    assert 0 < r8 < r16 <= 1, (r8, r16)


def test_locality_fixed_routing():
    tr = rand_trace()
    tr.act = tr.act[:1].expand(tr.T, -1, -1).clone()
    loc = metrics.locality([tr])
    assert torch.allclose(loc["overlap@1"], torch.ones(tr.L, dtype=torch.float64))
    assert torch.allclose(loc["reuse@16"], torch.ones(tr.L, dtype=torch.float64))
