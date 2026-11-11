"""Per-layer expert-cache simulations on traces (CPU; no timing). L / E are taken from the trace.

Common setup: each MoE layer has its own cache of ``cap`` expert slots (``cap >= E``: everything resident); the
prompt warms the cache; hit rate = routed experts already cached, over decode positions, mean over layers.

lru_curve     pure LRU, no prefetch, every capacity at once (stack distances; within a token, later router rank =
              more recent).
prefetch_hit  LRU + cross-layer prefetch: before token t reaches layer l >= 1, the first B uncached experts of the
              predictor's ranking for layer l are inserted (most-recent end, may evict); then the layer runs.
integrated    prefetch + replacement policy: LRU / LRFU (decay lam, prefetch adds no count), each optionally
              "+protect" (the current prediction's top-16 is evicted last: Cao et al. SIGMETRICS'95 "do no harm").
"""
from __future__ import annotations

import heapq
import math
import multiprocessing as mp
import os
from collections import OrderedDict

import numpy as np


# ---------- pure LRU: stack distances ----------
def stack_hist(tr) -> np.ndarray:
    """h[l, d] = decode accesses at LRU stack distance d (d = E: first touch)."""
    act, P, L, E, K = tr.act.numpy().astype(np.int64), tr.prompt_len, tr.L, tr.num_experts, tr.k
    last, pos = np.full((L, E), -1, np.int64), np.zeros((L, E), np.int64)
    h = np.zeros((L, E + 1), np.int64)
    li = np.arange(L)[:, None]
    for t in range(len(act)):
        a = act[t]
        le, pe = last[li, a], pos[li, a]
        newer = (last[:, None, :] > le[:, :, None]) | ((last[:, None, :] == le[:, :, None]) & (pos[:, None, :] > pe[:, :, None]))
        d = np.where(le >= 0, newer.sum(-1), E)
        if t >= P:
            np.add.at(h, (np.broadcast_to(li, d.shape), d), 1)
        last[li, a] = t
        pos[li, a] = np.arange(K)
    return h


def lru_curve(traces, workers=None) -> np.ndarray:
    """hit[l, c] for c = 0..E slots per layer (pooled over traces)."""
    with mp.Pool(workers or min(len(traces), os.cpu_count() or 1)) as pool:
        h = sum(pool.map(stack_hist, traces))
    E = traces[0].num_experts
    cum = np.concatenate([np.zeros((h.shape[0], 1)), np.cumsum(h[:, :E], 1)], 1)
    return cum / h.sum(1, keepdims=True)


def _b(Bspec, l):
    """Prefetch budget of target layer l: an int (all layers) or a per-layer sequence."""
    return Bspec if isinstance(Bspec, int) else int(Bspec[l])


# ---------- LRU + prefetch ----------
def _prefetch_one(args):
    tr, caps, Bs, kind = args
    act, rk, P, L, E = tr.act.tolist(), tr.rank(kind, 16).tolist(), tr.prompt_len, tr.L, tr.num_experts
    T = len(act)
    hits = np.zeros((len(Bs), len(caps), L))
    for bi, Bspec in enumerate(Bs):
        for ci, cap in enumerate(caps):
            for l in range(L):
                B = _b(Bspec, l)
                if cap >= E:
                    hits[bi, ci, l] = (T - P) * tr.k
                    continue
                od, h = OrderedDict(), 0
                for t in range(T):
                    if B and l:
                        n = 0
                        for e in rk[t][l]:
                            if n == B:
                                break
                            if e not in od:
                                od[e] = 1; n += 1
                                if len(od) > cap:
                                    od.popitem(last=False)
                    truth = act[t][l]
                    if t >= P:
                        h += sum(e in od for e in truth)
                    for e in truth:
                        if e in od:
                            od.move_to_end(e)
                        else:
                            od[e] = 1
                            if len(od) > cap:
                                od.popitem(last=False)
                hits[bi, ci, l] = h
    return hits, (T - P) * tr.k


def prefetch_hit(traces, caps, Bs=(0, 2), kind="hop1", workers=None) -> np.ndarray:
    """hit[B, cap, layer] (uniform per-layer capacity ``cap``). Each entry of ``Bs``: an int or per-layer budgets."""
    with mp.Pool(workers or min(len(traces), os.cpu_count() or 1)) as pool:
        res = pool.map(_prefetch_one, [(tr, list(caps), list(Bs), kind) for tr in traces])
    return sum(r[0] for r in res) / sum(r[1] for r in res)


# ---------- prefetch + replacement policies ----------
class _LRU:
    def __init__(self, cap):
        self.cap, self.d = cap, OrderedDict()

    def __contains__(self, e):
        return e in self.d

    def use(self, e, t):
        self.d.move_to_end(e)

    def insert(self, e, t, S, hard, used):
        if len(self.d) >= self.cap:
            victim = next((x for x in self.d if x not in S and x not in hard), None)
            if victim is None:
                victim = next((x for x in self.d if x not in hard), next(iter(self.d)))
            del self.d[victim]
        self.d[e] = 1


class _LRFU:
    def __init__(self, cap, lam):
        self.cap, self.cur, self.heap, self.st, self.lam, self.lg = cap, {}, [], {}, lam, -math.log(lam)

    def __contains__(self, e):
        return e in self.cur

    def _key(self, e, t, add):
        s, last = self.st.get(e, (0.0, t))
        s = s * self.lam ** (t - last) + add
        self.st[e] = (s, t)
        return math.log(max(s, 1e-12)) + t * self.lg

    def _push(self, e, k):
        self.cur[e] = k
        heapq.heappush(self.heap, (k, e))

    def use(self, e, t):
        self._push(e, self._key(e, t, 1.0))

    def insert(self, e, t, S, hard, used):
        if len(self.cur) >= self.cap:
            popped, victim = [], None
            while self.heap:
                k, x = heapq.heappop(self.heap)
                if self.cur.get(x) != k:
                    continue
                popped.append((k, x))
                if x not in S and x not in hard:
                    victim = x
                    break
            if victim is None:
                victim = next((x for _, x in popped if x not in hard), popped[0][1])
            for k, x in popped:
                if x != victim:
                    heapq.heappush(self.heap, (k, x))
            del self.cur[victim]
        self._push(e, self._key(e, t, 1.0 if used else 0.0))


POLICIES = ("LRU", "LRU+protect", "LRFU", "LRFU+protect")


def _run(z, hop, P, l, cap, B, pol, lam):
    c = _LRFU(cap, lam) if pol.startswith("LRFU") else _LRU(cap)
    protect = "protect" in pol
    hits = fetch = pf = pf_used = 0
    for t in range(len(z)):
        dec = t >= P
        truth = z[t]
        if B and l:
            S = set(hop[t]) if protect else set()
            new = []
            for e in hop[t]:
                if len(new) == B:
                    break
                if e not in c:
                    c.insert(e, t, S, set(new), used=False)
                    new.append(e)
            if dec:
                fetch += len(new); pf += len(new); pf_used += sum(e in truth for e in new)
        if dec:
            hits += sum(e in c for e in truth)
        S = set(hop[t]) if protect else set()
        hard = set(truth)
        for e in truth:
            if e in c:
                c.use(e, t)
            else:
                c.insert(e, t, S, hard, used=True)
                fetch += dec
    return hits, fetch, pf, pf_used


def _integrated_one(args):
    tr, caps, Bs, pols, kind, lam = args
    z, hop, P, L = tr.act.tolist(), tr.rank(kind, 16).tolist(), tr.prompt_len, tr.L
    out = np.zeros((4, len(pols), len(Bs), len(caps), L))
    for l in range(L):
        zl, hl = [r[l] for r in z], [r[l] for r in hop]
        for pi, pol in enumerate(pols):
            for bi, Bspec in enumerate(Bs):
                for ci, cap in enumerate(caps):
                    out[:, pi, bi, ci, l] = _run(zl, hl, P, l, cap, _b(Bspec, l), pol, lam)
    return out, tr.T - P


def integrated(traces, caps, Bs=(0, 2), policies=POLICIES, kind="hop1", lam=0.95, workers=None):
    """-> dict(hit [pol, B, cap, L], fetch_per_tok [pol, B, cap], pf_used [pol, B, cap])."""
    with mp.Pool(workers or min(len(traces), os.cpu_count() or 1)) as pool:
        res = pool.map(_integrated_one, [(tr, list(caps), list(Bs), list(policies), kind, lam) for tr in traces])
    tot, n = sum(r[0] for r in res), sum(r[1] for r in res)
    k = traces[0].k
    return {"hit": tot[0] / (n * k), "fetch_per_tok": tot[1].sum(-1) / n,
            "pf_used": tot[3].sum(-1) / np.maximum(tot[2].sum(-1), 1)}
