import torch
import timeit
import argparse
import numpy as np
import contextlib
from tabulate import tabulate
import torch.cuda.nvtx as nvtx
from cs336_basics.transformer.transformer_lm import TransformerLM
from cs336_basics.train.cross_entropy_loss import cross_entropy_loss

from cs336_systems.flash_attention.flash_attention_pytorch import FlashAttention as FlashAttentionPT
from cs336_systems.flash_attention.flash_attention_triton import FlashAttention as FlashAttentionTri
from cs336_basics.transformer.scaled_dot_product_attention import scaled_dot_product_attention as baseline_sdp


MODEL_SPECS = {
    "small":  dict(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large":  dict(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl":     dict(d_model=1600, d_ff=6400, num_layers=48, num_heads=25),
    "2.7B":   dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
}


def flash_attention_wrapper(flash_attn_apply):
    def wrapped(Q, K, V, mask=None):
        # Q/K/V: (batch, n_head, seq_len, head_dim)
        B, H, L, D = Q.shape
        Q_flat = Q.reshape(B * H, L, D)
        K_flat = K.reshape(B * H, L, D)
        V_flat = V.reshape(B * H, L, D)

        is_causal = mask is not None

        out = flash_attn_apply(Q_flat, K_flat, V_flat, is_causal)
        return out.reshape(B, H, L, D)
    return wrapped


from cs336_basics.transformer import scaled_dot_product_attention as sdp_mod
from cs336_basics.transformer import transformer_lm as tlm_mod
from cs336_basics.transformer import multi_head_self_attention as mhsa_mod

def swap_attention(impl):
    if impl == "pytorch":
        attn = flash_attention_wrapper(FlashAttentionPT.apply)
    elif impl == "triton":
        attn = flash_attention_wrapper(FlashAttentionTri.apply)
    else:
        attn = baseline_sdp

    sdp_mod.scaled_dot_product_attention = attn
    tlm_mod.scaled_dot_product_attention = attn
    mhsa_mod.scaled_dot_product_attention = attn


def benchmark(
        model: str|None,
        mixed_precision: bool,
        warmup_steps: int,
        measure_steps: int,
        attn_impl: str,
        forward_only: bool,
        with_optimizer: bool,
        profile_memory: bool,
        batch_size: int,
        context_length: int,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float):
    
    if model:
        spec = MODEL_SPECS[model]
        d_model = spec['d_model']
        d_ff=spec['d_ff']
        num_layers=spec['num_layers']
        num_heads=spec['num_heads']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device.type}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    if forward_only and with_optimizer:
        raise ValueError("Optimizer step requires backward pass; set forward_only=False.")

    swap_attention(attn_impl)
    model = TransformerLM(
        vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta
    ).to(device)

    optimizer = None
    if with_optimizer:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # model parameter number
    param_cnt = sum(p.numel() for p in model.parameters())
    print(f"Model parameter number: {param_cnt}")

    fwd_times = []
    loss_times = []
    bwd_times = []
    optimizer_times = []
    times = []
    total_steps = warmup_steps + measure_steps

    ctx = torch.inference_mode if (forward_only and not with_optimizer) else contextlib.nullcontext
    
    for step in range(total_steps):
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        else:
            model.zero_grad(set_to_none=True)

        x = torch.randint(
            low=0,
            high=vocab_size,
            size=(batch_size, context_length),
            device=device,
            dtype=torch.long,
        )
        y = torch.randint(
            low=0,
            high=vocab_size,
            size=(batch_size, context_length),
            device=device,
            dtype=torch.long,
        )

        # Run model
        if device.type == "cuda":
            torch.cuda.synchronize()

        mixed_precision_ctx = torch.autocast(device_type=device.type, dtype=torch.bfloat16) if mixed_precision else contextlib.nullcontext()

        with nvtx.range(f"step_{step}"):
            start_time = timeit.default_timer()
        
            with ctx():
                with nvtx.range("forward"):
                    if profile_memory:
                        torch.cuda.memory._record_memory_history(max_entries=1000000)
                    fwd_start_time = timeit.default_timer()
                    with mixed_precision_ctx:
                        logits = model(x)
                    fwd_elapsed = timeit.default_timer() - fwd_start_time
                    if forward_only and profile_memory:
                        torch.cuda.memory._dump_snapshot("inference_mem_snapshot.pickle")
                        torch.cuda.memory._record_memory_history(enabled=None)

                with nvtx.range("loss_computation"):
                    loss_start_time = timeit.default_timer()
                    loss = cross_entropy_loss(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                    loss_elapsed = timeit.default_timer() - loss_start_time

                # Calculate loss and gradient
                if not forward_only:
                    bwd_start_time = timeit.default_timer()          
                    loss.backward()
                    bwd_elapsed = timeit.default_timer() - bwd_start_time

                    if optimizer is not None:
                        optimizer_start_time = timeit.default_timer()          
                        optimizer.step()
                        optimizer_elapsed = timeit.default_timer() - optimizer_start_time

                    if profile_memory:
                        torch.cuda.memory._dump_snapshot("train_mem_snapshot.pickle")
                        torch.cuda.memory._record_memory_history(enabled=None)

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = timeit.default_timer() - start_time

        if step >= warmup_steps:
            fwd_times.append(fwd_elapsed)
            loss_times.append(loss_elapsed)
            if not forward_only:
                bwd_times.append(bwd_elapsed)
                if with_optimizer:
                    optimizer_times.append(optimizer_elapsed)
            times.append(elapsed)

    fwd_mean = np.mean(fwd_times)
    fwd_std = np.std(fwd_times)
    loss_mean = np.mean(loss_times)
    loss_std = np.std(loss_times)
    if bwd_times:
        bwd_mean = np.mean(bwd_times)
        bwd_std = np.std(bwd_times)
    if optimizer_times:
        optimizer_mean = np.mean(optimizer_times)
        optimizer_std = np.std(optimizer_times)
    mean = np.mean(times)
    std = np.std(times)

    print(f"fwd - mean: {fwd_mean:.6f}s, std: {fwd_std:.6f}s")
    print(f"loss - mean: {loss_mean:.6f}s, std: {loss_std:.6f}s")
    if bwd_times:
        print(f"bwd - mean: {bwd_mean:.6f}s, std: {bwd_std:.6f}s")
    if optimizer_times:
        print(f"optimizer - mean: {optimizer_mean:.6f}s, std: {optimizer_std:.6f}s")
    print(f"step - mean: {mean:.6f}s, std: {std:.6f}s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=['small', 'medium', 'large', 'xl', '2.7B'])
    parser.add_argument("--mixed_precision", action="store_true", default=False)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--measure_steps", type=int, default=10)
    parser.add_argument("--attn_impl", choices=["baseline", "pytorch", "triton"], default="baseline")
    parser.add_argument("--forward_only", action="store_true", default=False)
    parser.add_argument("--forward_and_backward", dest="forward_only", action="store_false")
    parser.add_argument("--with_optimizer", action="store_true", default=False)
    parser.add_argument("--profile_memory", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--rope_theta", type=float, default=10000)
    return parser.parse_args()


def main():
    args = parse_args()
    benchmark(**vars(args))


if __name__ == "__main__":
    main()
