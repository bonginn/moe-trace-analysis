# moe-trace-analysis

Record and analyze **expert routing** of Mixture-of-Experts models running in HuggingFace `transformers`.

- **Collect** per-token, per-layer router logits plus two *cross-layer predictions* of the next MoE layer's experts,
  with forward hooks. This works on most MoE models without any code changes.
- **Analyze** on CPU:
  - how well the next layer's experts can be predicted from earlier layers;
  - how bursty or skewed expert usage is;
  - what hit rate an expert cache would reach for a given number of GPU slots, with or without prefetching.

The tool is built to answer questions such as "how much GPU memory does an expert-offloading system need, and how
much does prefetching help?" before you build the system.

## Install

```bash
pip install -e .                 # analysis only (torch, numpy)
pip install -e ".[collect]"      # + trace collection (transformers, accelerate, safetensors)
```

## Quick start

```bash
# 1. Collect traces (GPU). prompts.jsonl: one {"id", "prompt", "source"?, "thinking"?} per line.
CUDA_VISIBLE_DEVICES=0 moe-trace collect generate --model allenai/OLMoE-1B-7B-0125-Instruct \
    --prompts prompts.jsonl --out traces/olmoe --max-new-tokens 1024
#    larger models: add --cpu-offload-gib 60 (offloaded weights are re-read from the checkpoint by the hooks)
#    existing token sequences only: moe-trace collect replay --model M --traces SRC --out DIR

# 2. Report (CPU)
moe-trace report traces/olmoe --cache 0.125 0.25 0.40 --B 0 2 --out report.txt
```

`collect` refuses to run on a GPU that another user's process is using (`CUDA_VISIBLE_DEVICES` must be set).

The report contains:
- expert usage: `overlap@1`, `reuse@1/4/16` and skew;
- prediction recall@8 and @16 for `hop1` and `pre`, by layer quarter and by source;
- per-layer cache hit rate at each cache size for:
  - pure LRU;
  - LRU with `hop1` prefetch of B experts;
  - LRU / LRFU, each with or without "protect", plus prefetch (fetch count per token included).

## Python API

```python
from moe_trace.traces import load_dir
from moe_trace import metrics, cachesim

trs = load_dir("traces/olmoe")
per_layer, by_source = metrics.recall(trs, kinds=("hop1", "pre"), ks=(8, 16))
print(per_layer[("pre", 8)][1:].mean())

curve = cachesim.lru_curve(trs)                    # hit[layer, slots] for every capacity at once
E, L = trs[0].num_experts, trs[0].L
fullB = [6 if (l + 1) % 4 == 0 else 4 for l in range(L)]      # per-layer prefetch budgets are allowed
hit = cachesim.prefetch_hit(trs, caps=[E // 8], Bs=(0, 2, fullB), kind="hop1")   # [B, cap, layer]
```

## What is recorded

For each sequence, one forward pass runs over the prompt plus the generated tokens. For every token position and
MoE layer *m* (decoder index *d*), the hooks store these as fp16 logits:

| name | definition | issued (in a real system) |
|---|---|---|
| `z` | router logits of layer *m*: `W_m x_m (+ b_m)`, where `x_m` is the MoE input | at layer *m* |
| `hop1` | `W_m x_{d-1}`: layer *m*'s router applied to the previous layer's MoE input | after layer *d-1*'s attention |
| `pre` (hop-1.5) | `W_m norm_m(h_{d-1})`, where `h` is the residual stream entering layer *d-1* | before layer *d-1*'s attention |

How the hooks find each part of the model:
- **Decoder layers** are found at `model[.model][.language_model].layers`.
- **Routers** are `mlp.gate` or `mlp.router`, with a `[E, H]` weight and an optional bias.
- **Dense layers** (no router) are skipped.

All predictions are stored *by target layer*: row *m* predicts layer *m*. The first decoder layer has no
predecessor, so its row is unused.

**Trace format.** Each sequence is saved as one `<id>.pt` file holding a `moe_trace.traces.Trace`:
- `act [T, L, k]`: the routed experts;
- `z_logits`;
- `pred_logits`: the `hop1` / `pre` predictions;
- `prompt_len`, `tokens`, `source`, `thinking`, `model`.

`load_legacy` also reads a *split layout*: a `seqs/` directory with top-16 rankings, plus an optional `logits/`
directory with full logits.

## Definitions

- **Only decode tokens are scored.** The prompt warms the caches. The first MoE layer has no
  cross-layer prediction, so it is not scored.
- **recall@K** = |top-K of the prediction ∩ routed experts| / topk, where topk is the model's experts per token.
  - The truth is the top-topk of `z` from the same forward pass as the prediction.
  - Results are pooled over tokens, per target layer.
- **overlap@1** = |S_t ∩ S_{t+1}| / topk.
- **reuse@d** = P(an expert routed at *t* was also routed within the previous *d* tokens).
- **skew** = the share of routed accesses that go to the layer's 10% most-used experts.
- **Cache simulations** (`cachesim`):
  - Each MoE layer has its own cache with the same number of slots, `round(fraction × E)`. The hit rate is the mean
    over layers.
  - **Prefetch:** before token *t* reaches layer *l ≥ 1*, the first *B* uncached experts in the prediction's ranking
    are inserted, at the most-recent end (they may evict).
  - **LRFU:** decayed use count `s ← λ s + 1[used]`, with λ = 0.95. A prefetch adds no count.
  - **"+protect":** the current prediction's top-16 experts are evicted last ("do no harm" integrated prefetching,
    Cao, Felten, Karlin & Li, SIGMETRICS '95).

## Tested models

The hook logic is tested on tiny randomly initialized models (CPU, no download) for these architectures:
- OLMoE;
- Qwen3-MoE;
- Mixtral;
- gpt-oss (router with bias);
- DeepSeek-V2 (first layer dense).

The test checks that `z`, `hop1` and `pre` equal their definitions, and that `z` equals the model's own
`router_logits` when the model returns them. The analysis functions (recall, locality, cache simulations) have also been
checked against independent earlier scripts on real Qwen3.6-35B-A3B traces (identical numbers). The recorder itself
has not yet been run on a full-size checkpoint in this repository's test suite.

```bash
python tests/run_all.py          # or: pytest tests/
```

## Limitations

- **Routers that pick experts with an extra correction bias** (DeepSeek-V3 / Moonlight `e_score_correction_bias`,
  ERNIE-4.5): construct `Recorder(..., selection_bias=True)` so that `act` is the selected set. The predictions
  stay raw logits.
- **A dense layer before the first MoE layer** (e.g. DeepSeek): that MoE layer does have a prediction, but it is
  currently left unscored.
- **No timing:** the cache simulations do not model transfer time. Hit rate is not the same as speed.
- **Memory:** full logits are `T × L × E` fp16 per sequence and kind. Keep that in mind for long sequences and
  models with many experts.
