import time
import torch
import itertools as it
from tabulate import tabulate
from cs336_basics.transformer.scaled_dot_product_attention import scaled_dot_product_attention
from cs336_systems.flash_attention.flash_attention_pytorch import FlashAttention as FlashAttentionPytorch
from cs336_systems.flash_attention.flash_attention_triton import FlashAttention as FlashAttentionTriton


SEQ_LENs = [256, 1024, 4096, 8192, 16384]
D_MODELS = [16, 32, 64, 128]
DTYPES = [torch.bfloat16, torch.float32]


def make_inputs(seq_len, d_model, dtype):
    q, k, v, do = [torch.randn(8, seq_len, d_model, device='cuda', dtype=dtype, requires_grad=False) for _ in range(4)]
    return q, k, v, do


def baseline_attn(q, k, v, is_causal=True):
    seq_len = q.size(-2)
    if is_causal:
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device))
    else:
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device)
    return scaled_dot_product_attention(q, k, v, mask)


def profile_attention(attn_fn, q, k, v, grad_out,
                      *, warmup=5, measure=100, causal=True):
    torch.cuda.empty_cache()

    # ---------- forward warmup ----------
    for _ in range(warmup):
        with torch.no_grad():
            attn_fn(q, k, v, causal)
    torch.cuda.synchronize()

    # ---------- forward timing ----------
    start = time.perf_counter()
    for _ in range(measure):
        with torch.no_grad():
            attn_fn(q, k, v, causal)
    torch.cuda.synchronize()
    fwd_ms = (time.perf_counter() - start) / measure * 1e3

    # ---------- memory before / during backward ----------
    torch.cuda.reset_peak_memory_stats()
    q_f, k_f, v_f = [x.detach().clone().requires_grad_(True) for x in (q, k, v)]
    out = attn_fn(q_f, k_f, v_f, causal)
    torch.cuda.synchronize()
    peak_before_bwd = torch.cuda.max_memory_allocated()

    torch.cuda.reset_peak_memory_stats()
    out.backward(grad_out.clone())
    torch.cuda.synchronize()
    peak_during_bwd = torch.cuda.max_memory_allocated()
    del out, q_f, k_f, v_f  # 可选，帮助释放显存

    # ---------- backward warmup ----------
    for _ in range(warmup):
        q_w, k_w, v_w = [x.detach().clone().requires_grad_(True) for x in (q, k, v)]
        attn_fn(q_w, k_w, v_w, causal).backward(grad_out.clone())
    torch.cuda.synchronize()

    # ---------- backward timing ----------
    start = time.perf_counter()
    for _ in range(measure):
        q_b, k_b, v_b = [x.detach().clone().requires_grad_(True) for x in (q, k, v)]
        torch.cuda.synchronize()
        out = attn_fn(q_b, k_b, v_b, causal)
        out.backward(grad_out.clone())
        torch.cuda.synchronize()
    bwd_ms = (time.perf_counter() - start) / measure * 1e3

    return fwd_ms, bwd_ms, peak_before_bwd, peak_during_bwd

def main():
    results = []
    # compiled_baseline_attn = torch.compile(baseline_attn)
    # flash_attn_pytorch = lambda q, k, v, causal: FlashAttentionPytorch.apply(q, k, v, causal)
    # flash_attn_triton = lambda q, k, v, causal: FlashAttentionTriton.apply(q, k, v, causal)

    try:
        for seq_len, d_model, dtype in it.product(SEQ_LENs, D_MODELS, DTYPES):
            q, k, v, grad = make_inputs(seq_len, d_model, dtype)
            fwd_ms, bwd_ms, fwd_peak_bytes, bwd_peak_bytes = profile_attention(
                baseline_attn, q, k, v, grad,
                warmup=10, measure=100, causal=True,
            )
            results.append({
                "impl": "flash_attn_baseline",
                "dtype": dtype.__repr__().split(".")[-1],
                "seq_len": seq_len,
                "d_model": d_model,
                "forward_ms": fwd_ms,
                "backward_ms": bwd_ms,
                "peak_before_bwd_mb": fwd_peak_bytes / (1024 ** 2),
                "peak_after_bwd_mb": bwd_peak_bytes / (1024 ** 2),
            })

            print(f"seq_len={seq_len}, d_model={d_model}, dtype={dtype} done. fwd_ms={fwd_ms:.3f}, bwd_ms={bwd_ms:.3f}, fwd_mem_mb={fwd_peak_bytes / (1024 ** 2)}, bwd_mem_mb={bwd_peak_bytes / (1024 ** 2)}")
    finally:
        headers = [
            "impl", "dtype", "seq_len", "d_model",
            "forward_ms", "backward_ms", "peak_before_bwd_mb", "peak_after_bwd_mb",
        ]
        rows = [
            [r[h] for h in headers]
            for r in results
        ]
        print(tabulate(rows, headers=headers, floatfmt=".3f"))

if __name__ == "__main__":
    main()
