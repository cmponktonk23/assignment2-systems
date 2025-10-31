import torch
import itertools as it
import triton.testing as tt
from tabulate import tabulate

from cs336_systems.flash_attention.flash_attention_pytorch import FlashAttention as FlashAttentionPytorch
from cs336_systems.flash_attention.flash_attention_triton import FlashAttention as FlashAttentionTriton


SEQ_LENs = [128, 256,] #512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
D_MODELS = [16, 32, 64, 128]
DTYPES = [torch.bfloat16, torch.float32]


def make_inputs(seq_len, d_model, dtype):
    q, k, v, do = torch.randn(4, 1, seq_len, d_model, device='cuda', dtype=dtype, requires_grad=False)
    return q, k, v, do


def time_forward(fn, q, k, v):
    def wrapper():
        with torch.no_grad():
            fn.apply(q, k, v, True)
        torch.cuda.synchronize()
    return tt.do_bench(wrapper, warmup=5, rep=10)


def time_backward(fn, q, k, v, do):
    q_, k_, v_ = (x.clone().detach().requires_grad_(True) for x in (q, k, v))
    do_ = do.clone().detach()
    with torch.enable_grad():
        out = fn.apply(q_, k_, v_, True)
    torch.cuda.synchronize()

    def wrapper():
        if q_.grad is not None:
            q_.grad.zero_()
        if k_.grad is not None:
            k_.grad.zero_()
        if v_.grad is not None:
            v_.grad.zero_()
        out.backward(do_, retain_graph=True)
        torch.cuda.synchronize()
    return tt.do_bench(wrapper, warmup=5, rep=10)


def time_end_to_end(fn, q, k, v, do):
    def wrapper():
        q_, k_, v_ = (x.clone().detach().requires_grad_(True) for x in (q, k, v))
        do_ = do.clone().detach()
        with torch.enable_grad():
            out = fn.apply(q_, k_, v_, True)
            out.backward(do_)
        torch.cuda.synchronize()
    return tt.do_bench(wrapper, warmup=5, rep=10)


def benchmark(seq_len, d_model, dtype):
    q, k, v, do = make_inputs(seq_len, d_model, dtype)

    pyt_fwd = time_forward(FlashAttentionPytorch, q, k, v)
    tri_fwd = time_forward(FlashAttentionTriton, q, k, v)
    pyt_bwd = time_backward(FlashAttentionPytorch, q, k, v, do)
    tri_bwd = time_backward(FlashAttentionTriton, q, k, v, do)
    pyt_e2e = time_end_to_end(FlashAttentionPytorch, q, k, v, do)
    tri_e2e = time_end_to_end(FlashAttentionTriton, q, k, v, do)

    return {
        "seq_len": seq_len,
        "d_model": d_model,
        "dtype": dtype.__repr__().split(".")[-1],
        "impl": "pytorch",
        "forward_ms": pyt_fwd * 1e3,
        "backward_ms": pyt_bwd * 1e3,
        "end_to_end_ms": pyt_e2e * 1e3,
    }, {
        "seq_len": seq_len,
        "d_model": d_model,
        "dtype": dtype.__repr__().split(".")[-1],
        "impl": "triton",
        "forward_ms": tri_fwd * 1e3,
        "backward_ms": tri_bwd * 1e3,
        "end_to_end_ms": tri_e2e * 1e3,
    }


def main():
    results = []
    for seq_len, d_model, dtype in it.product(SEQ_LENs, D_MODELS, DTYPES):
        torch.cuda.empty_cache()
        results.extend(benchmark(seq_len, d_model, dtype))
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