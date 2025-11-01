import torch
import timeit
import argparse
import numpy as np
import contextlib
import torch.cuda.nvtx as nvtx
from cs336_basics.transformer.transformer_lm import TransformerLM
from cs336_basics.train.cross_entropy_loss import cross_entropy_loss

from cs336_systems.flash_attention.flash_attention_pytorch import FlashAttention as FlashAttentionPT
from cs336_systems.flash_attention.flash_attention_triton import FlashAttention as FlashAttentionTri
from cs336_basics.transformer import scaled_dot_product_attention as sdp_module
from cs336_basics.transformer.scaled_dot_product_attention import scaled_dot_product_attention as baseline_sdp


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


def swap_attention(impl):
    if impl == "pytorch":
        sdp_module.scaled_dot_product_attention = flash_attention_wrapper(FlashAttentionPT.apply)
    elif impl == "triton":
        sdp_module.scaled_dot_product_attention = flash_attention_wrapper(FlashAttentionTri.apply)
    else:
        sdp_module.scaled_dot_product_attention = baseline_sdp


def benchmark(
        warmup_steps: int,
        measure_steps: int,
        attn_impl: str,
        forward_only: bool,
        batch_size: int,
        context_length: int,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device.type}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    swap_attention(attn_impl)
    model = TransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta).to(device)

    # model parameter number
    param_cnt = sum(p.numel() for p in model.parameters())
    print(f"Model parameter number: {param_cnt}")

    times = []
    total_steps = warmup_steps + measure_steps

    ctx = torch.inference_mode if forward_only else contextlib.nullcontext
    
    for step in range(total_steps):
        nvtx.range_push(f"step_{step}")

        x = torch.randint(low = 0, high = vocab_size, size = (batch_size, context_length), device = device, dtype = torch.long)
        y = torch.randint(low = 0, high = vocab_size, size = (batch_size, context_length), device = device, dtype = torch.long)

        # Run model
        if device.type == "cuda":
            torch.cuda.synchronize()

        start_time = timeit.default_timer()
    
        with ctx():
            nvtx.range_push("forward")
            logits = model(x)
            nvtx.range_pop()

            nvtx.range_push("loss_computation")
            loss = cross_entropy_loss(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            nvtx.range_pop()

            # Calculate loss and gradient
            if not forward_only:
                nvtx.range_push("backward")
                loss.backward()
                model.zero_grad(set_to_none=True)
                nvtx.range_pop()

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = timeit.default_timer() - start_time

        if step >= warmup_steps:
            times.append(elapsed)

        nvtx.range_pop()

    mean = np.mean(times)
    std = np.std(times)
    print(f"{'fwd' if forward_only else 'fwd+bwd'} - mean: {mean:.6f}s, std: {std:.6f}s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--measure_steps", type=int, default=10)
    parser.add_argument("--attn_impl", choices=["baseline", "pytorch", "triton"], default="baseline")
    parser.add_argument("--forward_only", action="store_true", default=False)
    parser.add_argument("--forward_and_backward", dest="forward_only", action="store_false")
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