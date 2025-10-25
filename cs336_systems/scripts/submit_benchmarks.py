import submitit
import re
import pandas as pd
from pathlib import Path


BENCH_SCRIPT = Path(__file__).resolve().parent / "benchmark.py"

MODEL_SPECS = {
    "small":  dict(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large":  dict(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl":     dict(d_model=1600, d_ff=6400, num_layers=48, num_heads=25),
    "2.7B":   dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
}

CONTEXT_LENGTHS = [64, 128, 256, 512, 1024]

COMMON_ARGS = [
    "--warmup_steps", "5",
    "--measure_steps", "10",
    "--batch_size", "4",
    "--vocab_size", "10000",
]


def main():
    out_dir = Path("slurm_logs/benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)

    executor = submitit.AutoExecutor(
        folder=out_dir, 
        cluster="local",
    )
    executor.update_parameters(
        timeout_min=30,
        cpus_per_task=8,
        gpus_per_node=1,
    )

    jobs = []
    rows = []

    # with executor.batch():
    for name, spec in MODEL_SPECS.items():
        for ctx_len in CONTEXT_LENGTHS:
            for forward_only in (True, False):
                args = [
                    "uv", "run", str(BENCH_SCRIPT),
                    *COMMON_ARGS,
                    "--d_model", str(spec["d_model"]),
                    "--context_length", str(ctx_len),
                    "--d_ff", str(spec["d_ff"]),
                    "--num_layers", str(spec["num_layers"]),
                    "--num_heads", str(spec["num_heads"]),
                ]
                if forward_only:
                    args.append("--forward_only")
                job = executor.submit(submitit.helpers.CommandFunction(args))
                job.spec_name = f"{name}-ctx{ctx_len}-{'fwd' if forward_only else 'fwd_bwd'}"
                jobs.append(job)
            
                print(job.job_id, job.spec_name)

                try:
                    job.wait()
                    text = Path(job.paths.stdout).read_text()
                    m = re.search(r"Model parameter number: (\d+)", text)
                    param_cnt = int(m.group(1)) if m else None
                    m = re.search(r"(fwd(?:\+bwd)?) - mean: ([\d.]+)s, std: ([\d.]+)s", text)
                    if m:
                        rows.append({
                            "spec": job.spec_name,
                            "context_length": ctx_len,
                            "forward_only": (m.group(1) == "fwd"),
                            "mean_s": float(m.group(2)),
                            "std_s": float(m.group(3)),
                            "params": param_cnt,
                        })
                except Exception as e:
                    print(f"[WARN] {job.spec_name} (job {job.job_id}) failed: {e}")
            
    df = pd.DataFrame(rows)
    print(df.to_markdown(index=False))


if __name__ == "__main__":
    main()
