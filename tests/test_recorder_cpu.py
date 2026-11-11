"""Recorder on tiny random MoE models (CPU, no download): hooked router logits == the model's own router logits,
and hop1 / pre equal their definitions (W_m x_{d-1}, W_m norm_m(h_{d-1}))."""
import torch
import transformers as T

from moe_trace.recorder import Recorder, decoder_layers, router_of


def tiny(kind):
    common = dict(vocab_size=97, hidden_size=32, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                  max_position_embeddings=64)
    if kind == "olmoe":
        cfg = T.OlmoeConfig(intermediate_size=16, num_experts=8, num_experts_per_tok=2, **common)
        return T.OlmoeForCausalLM(cfg)
    if kind == "qwen3_moe":
        cfg = T.Qwen3MoeConfig(intermediate_size=64, moe_intermediate_size=16, num_experts=8, num_experts_per_tok=2,
                               head_dim=8, **common)
        return T.Qwen3MoeForCausalLM(cfg)
    if kind == "mixtral":
        cfg = T.MixtralConfig(intermediate_size=16, num_local_experts=4, num_experts_per_tok=2, **common)
        return T.MixtralForCausalLM(cfg)
    if kind == "gpt_oss":
        cfg = T.GptOssConfig(intermediate_size=16, num_local_experts=4, num_experts_per_tok=2, head_dim=8,
                             layer_types=["full_attention"] * 4, **common)
        return T.GptOssForCausalLM(cfg)
    if kind == "deepseek_v2":  # layer 0 dense (first_k_dense_replace=1), shared experts, MLA attention
        cfg = T.DeepseekV2Config(intermediate_size=64, moe_intermediate_size=16, n_routed_experts=8,
                                 num_experts_per_tok=2, n_shared_experts=1, first_k_dense_replace=1, kv_lora_rank=16,
                                 q_lora_rank=None, qk_rope_head_dim=8, qk_nope_head_dim=8, v_head_dim=8, n_group=1,
                                 topk_group=1, **{**common, "num_key_value_heads": 4})
        return T.DeepseekV2ForCausalLM(cfg)
    raise KeyError(kind)


@torch.no_grad()
def check(kind):
    torch.manual_seed(0)
    m = tiny(kind).eval()
    rec = Recorder(m, device="cpu")
    base, layers = decoder_layers(m)
    # independent reference: capture each layer's input h and MLP input x
    H, X = {}, {}
    hs = [l.register_forward_pre_hook(lambda mod, a, kw, d=d: H.__setitem__(d, (a[0] if a else kw["hidden_states"]).detach()),
                                      with_kwargs=True) for d, l in enumerate(layers)]
    hs += [l.mlp.register_forward_pre_hook(lambda mod, a, d=d: X.__setitem__(d, a[0].detach())) for d, l in enumerate(layers)]
    ids = torch.randint(0, 97, (1, 11))
    m.model(input_ids=ids)
    out = rec.pack()
    for h in hs:
        h.remove()
    for i, (d, W, b, _) in enumerate(rec.moe):
        r = router_of(layers[d])
        ref_z = torch.nn.functional.linear(X[d].reshape(-1, X[d].shape[-1]).float(), r.weight.float(),
                                           None if getattr(r, "bias", None) is None else r.bias.float())
        assert torch.allclose(out["z"][:, i].float(), ref_z, atol=2e-3), (kind, "z", i)
        if d >= 1:
            ref_h1 = torch.nn.functional.linear(X[d - 1].reshape(-1, X[d - 1].shape[-1]).float(), W, b)
            assert torch.allclose(out["hop1"][:, i].float(), ref_h1, atol=2e-3), (kind, "hop1", i)
            u = layers[d].post_attention_layernorm(H[d - 1])
            ref_pre = torch.nn.functional.linear(u.reshape(-1, u.shape[-1]).float(), W, b)
            assert torch.allclose(out["pre"][:, i].float(), ref_pre, atol=2e-3), (kind, "pre", i)
    # the model's own router (when it reports router logits) picks the same top-k as our z
    rl = getattr(m.model(input_ids=ids, output_router_logits=True), "router_logits", None)
    if rl is not None and isinstance(rl, (tuple, list)) and len(rl) == len(rec.moe):
        for i, g in enumerate(rl):
            assert torch.allclose(g.float().reshape(-1, rec.E), out["z"][:, i].float(), atol=2e-3), (kind, "router", i)
    rec.remove()
    return len(rec.moe)


def test_models():
    for kind in ("olmoe", "qwen3_moe", "mixtral", "gpt_oss", "deepseek_v2"):
        print(kind, check(kind), "MoE layers OK")


if __name__ == "__main__":
    test_models()
