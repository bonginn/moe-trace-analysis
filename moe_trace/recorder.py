"""Record expert routing and cross-layer predictions from a HuggingFace transformers MoE model (forward hooks).

Model-agnostic: decoder layers are found at ``model[.model][.language_model].layers``; a layer is an MoE layer if
its ``mlp`` has a router (``mlp.gate`` or ``mlp.router`` with a [E, H] ``weight`` and optional ``bias``). Per token
position and MoE layer m (decoder index d_m) it records, as fp16 logits:
  z     router logits of layer m:            W_m x_m (+ b_m),  x_m = MoE input of layer m
  hop1  prediction for m from layer d_m - 1:  W_m x_{d_m - 1}   (that layer's MLP input; issued after its attention)
  pre   prediction for m (hop-1.5):           W_m norm_m(h_{d_m - 1}),  h = residual stream entering layer d_m - 1,
                                              norm_m = layer d_m's post_attention_layernorm (issued before attention)
hop1 / pre exist only if d_m >= 1 (the first decoder layer has no predecessor); rows are zeros otherwise.

Notes: routers with a selection-only correction bias (DeepSeek-V3 ``e_score_correction_bias``, ERNIE-4.5) pick
experts by sigmoid/softmax(logits) + bias; ``act`` here is the top-k of the RAW logits -- pass ``selection_bias=True``
to add the bias to z before taking the top-k (the predictions stay raw logits).
Weights of modules offloaded to "meta" (accelerate CPU offload) are re-read from the checkpoint.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def decoder_layers(model):
    m = model
    for attr in ("model", "language_model"):
        if hasattr(m, attr) and not hasattr(m, "layers"):
            m = getattr(m, attr)
    if not hasattr(m, "layers"):
        raise ValueError("cannot find decoder layers")
    return m, m.layers


def router_of(layer):
    mlp = getattr(layer, "mlp", None) or getattr(layer, "block_sparse_moe", None)
    if mlp is None:
        return None
    for name in ("gate", "router"):
        r = getattr(mlp, name, None)
        w = getattr(r, "weight", None)
        if isinstance(w, torch.Tensor) and w.dim() == 2 and w.shape[0] < w.shape[1]:
            return r
    return None


class _Ckpt:
    """Lazy reader of checkpoint tensors by module-path suffix (for parameters on the meta device)."""

    def __init__(self, model):
        self.model, self.wm, self.snap = model, None, None

    def get(self, qualname: str) -> torch.Tensor:
        if self.wm is None:
            from huggingface_hub import snapshot_download

            self.snap = Path(snapshot_download(self.model.config._name_or_path, allow_patterns=["*.json"]))
            idx = self.snap / "model.safetensors.index.json"
            self.wm = json.load(open(idx))["weight_map"] if idx.exists() else {}
        tail = qualname[qualname.index("layers."):] if "layers." in qualname else qualname
        keys = [k for k in self.wm if k.endswith(tail)]
        if len(keys) != 1:
            raise KeyError(f"{qualname}: {len(keys)} checkpoint keys end with {tail}")
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open

        f = hf_hub_download(self.model.config._name_or_path, self.wm[keys[0]])
        with safe_open(f, "pt", device="cpu") as fh:
            return fh.get_tensor(keys[0])


class Recorder:
    def __init__(self, model, device=None, selection_bias: bool = False):
        self.base, self.layers = decoder_layers(model)
        self.dev = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        names = {mod: n for n, mod in model.named_modules()}
        ck = _Ckpt(model)

        def real(t, owner, pname):
            if t is None or t.device.type != "meta":
                return t
            return ck.get(f"{names[owner]}.{pname}")

        self.moe = []  # (decoder index, router W fp32, bias fp32 | None, selection bias | None)
        for d, layer in enumerate(self.layers):
            r = router_of(layer)
            if r is None:
                continue
            W = real(r.weight, r, "weight").float().to(self.dev)
            b = getattr(r, "bias", None)
            b = None if b is None else real(b, r, "bias").float().to(self.dev)
            sb = getattr(r, "e_score_correction_bias", None) if selection_bias else None
            sb = None if sb is None else real(sb, r, "e_score_correction_bias").float().to(self.dev)
            self.moe.append((d, W, b, sb))
        if not self.moe:
            raise ValueError("no MoE layer found")
        self.m_of_d = {d: i for i, (d, *_) in enumerate(self.moe)}
        self.E = self.moe[0][1].shape[0]
        self.norms = {}
        for d, *_ in self.moe:
            n = self.layers[d].post_attention_layernorm
            if any(p.device.type == "meta" for p in n.parameters()):
                n2 = copy.deepcopy(n).to_empty(device=self.dev)
                for pn, _ in n.named_parameters():
                    getattr(n2, pn).data.copy_(real(getattr(n, pn), n, pn))
                n = n2
            self.norms[d] = n
        self.cur = None
        self.handles = []
        for d, layer in enumerate(self.layers):
            self.handles.append(layer.register_forward_pre_hook(self._layer_hook(d), with_kwargs=True))
            mlp = getattr(layer, "mlp", None) or getattr(layer, "block_sparse_moe")
            self.handles.append(mlp.register_forward_pre_hook(self._mlp_hook(d)))

    def remove(self):
        for h in self.handles:
            h.remove()

    def _lg(self, x, i):
        _, W, b, _ = self.moe[i]
        return F.linear(x.reshape(-1, x.shape[-1]).to(self.dev).float(), W, b)

    def _layer_hook(self, d):
        def pre(mod, args, kwargs):
            h = args[0] if args else kwargs["hidden_states"]
            if d == 0:
                M = len(self.moe)
                self.cur = {k: [None] * M for k in ("z", "hop1", "pre")}
            i = self.m_of_d.get(d + 1)
            if i is not None:  # hop-1.5 for the MoE layer at decoder index d + 1
                n = self.norms[d + 1]
                x = n(h.to(next(n.parameters()).device))
                self.cur["pre"][i] = self._lg(x, i).half().cpu()
        return pre

    def _mlp_hook(self, d):
        def pre(mod, args, kwargs=None):
            x = args[0]
            i = self.m_of_d.get(d)
            if i is not None:
                z = self._lg(x, i)
                sb = self.moe[i][3]
                self.cur["z"][i] = (z if sb is None else z + sb).half().cpu()
            j = self.m_of_d.get(d + 1)
            if j is not None:
                self.cur["hop1"][j] = self._lg(x, j).half().cpu()
        return pre

    def pack(self):
        """-> dict z / hop1 / pre: fp16 [T, M, E] (target-indexed over MoE layers; missing rows = 0)."""
        T = next(v for v in self.cur["z"] if v is not None).shape[0]
        out = {}
        for k, v in self.cur.items():
            out[k] = torch.stack([x if x is not None else torch.zeros(T, self.E, dtype=torch.float16) for x in v], 1)
        return out
