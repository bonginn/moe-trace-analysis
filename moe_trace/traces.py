"""Expert-routing traces: one record per sequence, model-agnostic.

Conventions (all analyses rely on them):
  * ``act[t, l]``  the routed experts (router top-k, in router order) of token position t at MoE layer l.
  * Predictions are TARGET-indexed: ``pred_rank[kind][t, l]`` / ``pred_logits[kind][t, l]`` predict layer l;
    layer 0 has no cross-layer prediction (rank row = -1, logits row = NaN-free zeros, never scored).
      hop1  W_l x_{l-1}                         (x = MoE input; issued after layer l-1's attention)
      pre   W_l norm_l(h_{l-1}), h = residual    (hop-1.5; issued before layer l-1's attention)
  * Positions ``t < prompt_len`` are the prompt (they warm caches); decode = ``t >= prompt_len``.

On disk (``save``/``load``): one ``<name>.pt`` per sequence with the fields of :class:`Trace`. ``load_legacy`` reads
the split layout (``seqs/`` top-16 rankings + optional ``logits/`` full logits).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass
class Trace:
    name: str
    prompt_len: int
    act: torch.Tensor                          # int64 [T, L, k] routed experts
    num_experts: int
    source: str = ""                           # dataset / domain tag
    thinking: bool | None = None
    model: str = ""
    z_logits: torch.Tensor | None = None       # fp16 [T, L, E] router logits (selection scores)
    pred_rank: dict = field(default_factory=dict)    # kind -> int64 [T, L, K] (layer 0 = -1)
    pred_logits: dict = field(default_factory=dict)  # kind -> fp16 [T, L, E] (layer 0 = 0)
    tokens: torch.Tensor | None = None

    @property
    def T(self) -> int:
        return self.act.shape[0]

    @property
    def L(self) -> int:
        return self.act.shape[1]

    @property
    def k(self) -> int:
        return self.act.shape[2]

    def rank(self, kind: str, K: int = 16) -> torch.Tensor:
        """Target-indexed top-K ranking of a predictor (from its logits if only those are stored)."""
        if kind in self.pred_rank and self.pred_rank[kind].shape[-1] >= K:
            return self.pred_rank[kind][..., :K]
        lg = self.pred_logits[kind]
        r = lg.float().topk(K, -1).indices
        r[:, 0] = -1
        return r


def save(tr: Trace, out_dir: str | Path) -> Path:
    p = Path(out_dir) / f"{tr.name}.pt"
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v for k, v in tr.__dict__.items()}, p)
    return p


def load(path: str | Path) -> Trace:
    return Trace(**torch.load(path))


def load_dir(d: str | Path, limit: int = 0) -> list[Trace]:
    """Native directory (``<name>.pt`` per sequence) or the split layout (a dir with ``seqs/``)."""
    d = Path(d)
    if (d / "seqs").is_dir():
        return load_legacy(d, limit=limit)
    files = sorted(d.glob("*.pt"))
    return [load(f) for f in (files[:limit] if limit else files)]


def _target_index(issuing: torch.Tensor) -> torch.Tensor:
    """[T, L, E] stored at the issuing layer (index l predicts l+1) -> target-indexed (index l predicts l)."""
    out = torch.zeros_like(issuing)
    out[:, 1:] = issuing[:, :-1]
    return out


def load_legacy(root: str | Path, limit: int = 0, logits: bool | None = None, seqs: str = "seqs",
                logits_dir: str = "logits", num_experts: int | None = None) -> list[Trace]:
    """Split layout:
    seqs/<id>.pt   z [T, L, 16] router top-16, hop1 [T, L, 16] (TARGET-indexed, layer 0 zeros), tokens, prompt_len,
                   source, thinking
    logits/<id>.pt z / hop1 / pre fp16 [T, L, E] (hop1 / pre ISSUING-indexed: index l predicts l + 1)
    logits=None: attach full logits when the logits dir exists."""
    root = Path(root)
    files = sorted((root / seqs).glob("*.pt"))
    files = files[:limit] if limit else files
    ld = root / logits_dir
    use_lg = ld.is_dir() if logits is None else logits
    out = []
    for f in files:
        d = torch.load(f)
        z16 = d["z"].long()
        hop = d["hop1"].long().clone()
        hop[:, 0] = -1
        tr = Trace(name=f.stem, prompt_len=int(d["prompt_len"]), act=z16[:, :, :8].contiguous(),
                   num_experts=num_experts or 0, source=d.get("source", ""), thinking=d.get("thinking"),
                   tokens=d.get("tokens"), pred_rank={"hop1": hop})
        if use_lg and (ld / f.name).exists():
            g = torch.load(ld / f.name)
            tr.z_logits = g["z"]
            tr.num_experts = g["z"].shape[-1]
            for k in ("hop1", "pre"):
                if k in g:
                    tr.pred_logits[k] = _target_index(g[k])
        if not tr.num_experts:
            tr.num_experts = 256 if z16.max() >= 128 else 128 if z16.max() >= 64 else int(z16.max()) + 1
        out.append(tr)
    return out
