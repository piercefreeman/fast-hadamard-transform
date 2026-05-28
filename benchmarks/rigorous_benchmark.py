import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from fast_hadamard_transform import hadamard_transform


DTYPES = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def parse_int_list(value):
    return [int(item) for item in value.split(",") if item]


def parse_str_list(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def run_command(args):
    try:
        proc = subprocess.run(args, check=False, capture_output=True, text=True)
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def percentile(values, pct):
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct / 100.0
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    weight = rank - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def summarize_us(times_us):
    q1 = percentile(times_us, 25)
    q3 = percentile(times_us, 75)
    return {
        "median_us": statistics.median(times_us),
        "mean_us": statistics.fmean(times_us),
        "min_us": min(times_us),
        "max_us": max(times_us),
        "p10_us": percentile(times_us, 10),
        "p90_us": percentile(times_us, 90),
        "iqr_us": q3 - q1,
    }


def element_size(dtype):
    return torch.empty((), dtype=dtype).element_size()


def choose_rows(dim, dtype, target_mib, min_rows, max_rows, align_rows):
    target_bytes = int(target_mib * 1024 * 1024)
    row_bytes = dim * element_size(dtype)
    rows = max(1, target_bytes // row_bytes)
    rows = min(max_rows, max(min_rows, rows))
    if rows >= align_rows:
        rows = max(align_rows, (rows // align_rows) * align_rows)
    return rows


def cuda_event_times_us(fn, x, scale, warmup, repeats):
    times_us = []
    with torch.no_grad():
        for _ in range(warmup):
            out = fn(x, scale)
            del out
        torch.cuda.synchronize()
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = fn(x, scale)
            end.record()
            end.synchronize()
            times_us.append(start.elapsed_time(end) * 1000.0)
            del out
    return times_us


def clone_fn(x, scale):
    return torch.clone(x)


def benchmark_case(dim, dtype_name, rows, warmup, repeats, seed, check_correctness, fast_low_precision):
    dtype = DTYPES[dtype_name]
    torch.manual_seed(seed)
    x = torch.randn(rows, dim, device="cuda", dtype=dtype)
    scale = 1.0 / math.sqrt(dim)

    if check_correctness:
        sample_rows = min(rows, 8)
        ref = hadamard_transform(x[:sample_rows].float(), scale)
        actual = hadamard_transform(x[:sample_rows], scale, fast_low_precision=fast_low_precision).float()
        max_abs = (actual - ref).abs().max().item()
    else:
        max_abs = None

    def fht_fn(x, scale):
        return hadamard_transform(x, scale, fast_low_precision=fast_low_precision)

    results = []
    for name, fn in (("hadamard_transform", fht_fn), ("torch.clone", clone_fn)):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        times_us = cuda_event_times_us(fn, x, scale, warmup, repeats)
        summary = summarize_us(times_us)
        bytes_moved = 2 * rows * dim * element_size(dtype)
        summary.update(
            {
                "name": name,
                "dtype": dtype_name,
                "dim": dim,
                "rows": rows,
                "input_mib": rows * dim * element_size(dtype) / 1024 / 1024,
                "bytes_moved": bytes_moved,
                "effective_gbps": bytes_moved / (summary["median_us"] * 1e-6) / 1e9,
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024 / 1024,
                "correctness_max_abs_vs_fp32": max_abs if name == "hadamard_transform" else None,
                "times_us": times_us,
            }
        )
        results.append(summary)

    fht = results[0]
    clone = results[1]
    fht["ratio_to_clone"] = fht["median_us"] / clone["median_us"]
    clone["ratio_to_clone"] = 1.0
    return results


def collect_metadata():
    device_index = torch.cuda.current_device()
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(device_index),
        "gpu_capability": torch.cuda.get_device_capability(device_index),
        "gpu_count": torch.cuda.device_count(),
        "git_commit": run_command(["git", "rev-parse", "HEAD"]),
        "git_status_short": run_command(["git", "status", "--short"]),
        "command": " ".join(sys.argv),
    }


def main():
    parser = argparse.ArgumentParser(description="Rigorous CUDA benchmark for fast_hadamard_transform.")
    parser.add_argument("--dims", default="256,512,1024,2048,4096,8192,16384,32768")
    parser.add_argument("--dtypes", default="fp16,bf16,fp32", help="Comma-separated: fp16,bf16,fp32")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-mib", type=int, default=256)
    parser.add_argument("--min-rows", type=int, default=1024)
    parser.add_argument("--max-rows", type=int, default=65536)
    parser.add_argument("--align-rows", type=int, default=128)
    parser.add_argument("--output", type=Path, default=Path("benchmark_results/fht_benchmark.json"))
    parser.add_argument("--check-correctness", action="store_true")
    parser.add_argument(
        "--fast-low-precision",
        action="store_true",
        help="Use native half2/bfloat162 arithmetic for supported fp16/bf16 power-of-two dimensions.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    dims = parse_int_list(args.dims)
    dtype_names = parse_str_list(args.dtypes)
    unknown = sorted(set(dtype_names) - set(DTYPES))
    if unknown:
        raise ValueError(f"Unknown dtypes: {unknown}")

    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    payload = {
        "metadata": collect_metadata(),
        "config": vars(args) | {"output": str(args.output)},
        "results": [],
    }

    print("name,dtype,dim,rows,median_us,iqr_us,effective_gbps,ratio_to_clone")
    for dtype_name in dtype_names:
        for dim in dims:
            rows = choose_rows(dim, DTYPES[dtype_name], args.target_mib, args.min_rows, args.max_rows, args.align_rows)
            case_results = benchmark_case(
                dim=dim,
                dtype_name=dtype_name,
                rows=rows,
                warmup=args.warmup,
                repeats=args.repeats,
                seed=args.seed,
                check_correctness=args.check_correctness,
                fast_low_precision=args.fast_low_precision,
            )
            payload["results"].extend(case_results)
            for result in case_results:
                print(
                    ",".join(
                        [
                            result["name"],
                            result["dtype"],
                            str(result["dim"]),
                            str(result["rows"]),
                            f"{result['median_us']:.3f}",
                            f"{result['iqr_us']:.3f}",
                            f"{result['effective_gbps']:.2f}",
                            f"{result['ratio_to_clone']:.3f}",
                        ]
                    ),
                    flush=True,
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
