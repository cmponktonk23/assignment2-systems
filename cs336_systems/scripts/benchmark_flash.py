import torch
import argparse
import itertools as it
import triton.testing as tt
from tabulate import tabulate

from cs336_systems.flash_attention.flash_attention_pytorch import FlashAttention as FlashAttentionPytorch
from cs336_systems.flash_attention.flash_attention_triton import FlashAttention as FlashAttentionTriton
from cs336_basics.transformer.scaled_dot_product_attention import scaled_dot_product_attention


# SEQ_LENs = [2048, 4096, 8192, 16384, 32768, 65536]
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


def time_forward(fn, q, k, v, warmup_steps=5, measure_steps=10):
    def wrapper():
        torch.cuda.synchronize()
        with torch.no_grad():
            fn(q, k, v, True)
        torch.cuda.synchronize()
    return tt.do_bench(wrapper, warmup=warmup_steps, rep=measure_steps)


def time_backward(fn, q, k, v, do, warmup_steps=5, measure_steps=10):
    def wrapper():
        torch.cuda.synchronize()

        with torch.enable_grad():
            q_, k_, v_ = [x.clone().detach().requires_grad_(True) for x in (q, k, v)]
            do_ = do.clone().detach()
            out = fn(q_, k_, v_, True)
            out.backward(do_)
        
        torch.cuda.synchronize()
    return tt.do_bench(wrapper, warmup=warmup_steps, rep=measure_steps)


def time_end_to_end(fn, q, k, v, do, warmup_steps=5, measure_steps=10):
    def wrapper():
        q_, k_, v_ = (x.clone().detach().requires_grad_(True) for x in (q, k, v))
        do_ = do.clone().detach()
        torch.cuda.synchronize()
        with torch.enable_grad():
            out = fn(q_, k_, v_, True)
            out.backward(do_)
        torch.cuda.synchronize()
    return tt.do_bench(wrapper, warmup=warmup_steps, rep=measure_steps)


def benchmark(seq_len, d_model, dtype, warmup_steps=5, measure_steps=10, run_pytorch=False, run_triton=False):
    q, k, v, do = make_inputs(seq_len, d_model, dtype)

    flash_pytorch = lambda q, k, v, is_causal: FlashAttentionPytorch.apply(q, k, v, is_causal)
    flash_triton = lambda q, k, v, is_causal: FlashAttentionTriton.apply(q, k, v, is_causal)

    bsl_fwd = time_forward(baseline_attn, q, k, v, warmup_steps, measure_steps)
    bsl_bwd = time_backward(baseline_attn, q, k, v, do, warmup_steps, measure_steps)
    bsl_e2e = time_end_to_end(baseline_attn, q, k, v, do, warmup_steps, measure_steps)
    if run_pytorch:
        pyt_fwd = time_forward(flash_pytorch, q, k, v, warmup_steps, measure_steps)
        pyt_bwd = time_backward(flash_pytorch, q, k, v, do, warmup_steps, measure_steps)
        pyt_e2e = time_end_to_end(flash_pytorch, q, k, v, do, warmup_steps, measure_steps)
    if run_triton:
        tri_fwd = time_forward(flash_triton, q, k, v, warmup_steps, measure_steps)
        tri_bwd = time_backward(flash_triton, q, k, v, do, warmup_steps, measure_steps)
        tri_e2e = time_end_to_end(flash_triton, q, k, v, do, warmup_steps, measure_steps)

    ret = [{
        "seq_len": seq_len,
        "d_model": d_model,
        "dtype": dtype.__repr__().split(".")[-1],
        "impl": "baseline",
        "forward_ms": bsl_fwd * 1e3,
        "backward_ms": bsl_bwd * 1e3,
        "end_to_end_ms": bsl_e2e * 1e3,
    }]
    
    if run_pytorch:
        ret.append({
            "seq_len": seq_len,
            "d_model": d_model,
            "dtype": dtype.__repr__().split(".")[-1],
            "impl": "pytorch",
            "forward_ms": pyt_fwd * 1e3,
            "backward_ms": pyt_bwd * 1e3,
            "end_to_end_ms": pyt_e2e * 1e3,
        }) 
    
    if run_triton:
        ret.append({
            "seq_len": seq_len,
            "d_model": d_model,
            "dtype": dtype.__repr__().split(".")[-1],
            "impl": "triton",
            "forward_ms": tri_fwd * 1e3,
            "backward_ms": tri_bwd * 1e3,
            "end_to_end_ms": tri_e2e * 1e3,
        })

    return ret


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--measure_steps", type=int, default=10)
    parser.add_argument("--run_pytorch", action="store_true", default=False)
    parser.add_argument("--run_triton", action="store_true", default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    results = []
    for seq_len, d_model, dtype in it.product(SEQ_LENs, D_MODELS, DTYPES):
        torch.cuda.empty_cache()
        results.extend(benchmark(seq_len, d_model, dtype, **args))
        print(f"seq_len={seq_len}, d_model={d_model}, dtype={dtype} done")

    results.sort(key=lambda r: (r['dtype'], r['seq_len'], r['d_model'], r['impl']))
    print(tabulate(
        [[r['dtype'], r['seq_len'], r['d_model'], r['impl'],
        r['forward_ms'], r['backward_ms'], r['end_to_end_ms']] for r in results],
        headers=['dtype', 'seq_len', 'd_model', 'impl', 'forward_ms', 'backward_ms', 'end_to_end_ms'],
        floatfmt='.3f',
        tablefmt='github'
    ))

if __name__ == "__main__":
    main()