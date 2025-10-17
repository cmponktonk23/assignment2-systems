import torch
import timeit
import argparse
import numpy as np
import contextlib
from cs336_basics.transformer.transformer_lm import TransformerLM
from cs336_basics.train.cross_entropy_loss import cross_entropy_loss


def benchmark(
        warmup_steps: int,
        measure_steps: int,
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

    model = TransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta).to(device)

    # model parameter number
    param_cnt = sum(p.numel() for p in model.parameters())
    print(f"Model parameter number: {param_cnt}")

    times = []
    total_steps = warmup_steps + measure_steps

    ctx = torch.inference_mode if forward_only else contextlib.nullcontext
    
    for step in range(total_steps):
        x = torch.randint(low = 0, high = vocab_size, size = (batch_size, context_length), device = device, dtype = torch.long)
        y = torch.randint(low = 0, high = vocab_size, size = (batch_size, context_length), device = device, dtype = torch.long)

        # Run model
        if device.type == "cuda":
            torch.cuda.synchronize()

        start_time = timeit.default_timer()
    
        with ctx():
            logits = model(x)
            loss = cross_entropy_loss(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

            # Calculate loss and gradient
            if not forward_only:
                loss.backward()
                model.zero_grad(set_to_none=True)

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = timeit.default_timer() - start_time

        if step >= warmup_steps:
            times.append(elapsed)

    mean = np.mean(times)
    std = np.std(times)
    print(f"{'fwd' if forward_only else 'fwd+bwd'} - mean: {mean:.6f}s, std: {std:.6f}s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--measure_steps", type=int, default=10)
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