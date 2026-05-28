import argparse
import json
import math
import random
import statistics
from pathlib import Path


def percentile(values, pct):
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


def geomean(values):
    return math.exp(statistics.fmean(math.log(value) for value in values))


def load_results(path, name):
    payload = json.loads(Path(path).read_text())
    results = {}
    for result in payload["results"]:
        if result["name"] != name:
            continue
        key = (result["dtype"], int(result["dim"]))
        results[key] = result
    return payload, results


def summarize(values):
    return {
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "p05": percentile(values, 5),
        "p95": percentile(values, 95),
    }


def bootstrap_case_speedup(reference_times, candidate_times, rng):
    reference_sample = [rng.choice(reference_times) for _ in reference_times]
    candidate_sample = [rng.choice(candidate_times) for _ in candidate_times]
    return statistics.median(reference_sample) / statistics.median(candidate_sample)


def build_comparison(reference, candidate, bootstrap, seed):
    common_keys = sorted(set(reference) & set(candidate), key=lambda item: (item[0], item[1]))
    if not common_keys:
        raise ValueError("No common benchmark cases found.")

    cases = []
    for key in common_keys:
        dtype, dim = key
        ref = reference[key]
        cand = candidate[key]
        speedup = ref["median_us"] / cand["median_us"]
        cases.append(
            {
                "dtype": dtype,
                "dim": dim,
                "reference_median_us": ref["median_us"],
                "candidate_median_us": cand["median_us"],
                "speedup": speedup,
                "delta_us": cand["median_us"] - ref["median_us"],
            }
        )

    overall_speedup = geomean(case["speedup"] for case in cases)
    dtype_speedups = {}
    for dtype in sorted({case["dtype"] for case in cases}):
        dtype_cases = [case for case in cases if case["dtype"] == dtype]
        dtype_speedups[dtype] = geomean(case["speedup"] for case in dtype_cases)

    payload = {
        "case_count": len(cases),
        "overall_speedup": overall_speedup,
        "dtype_speedups": dtype_speedups,
        "cases": cases,
    }

    if bootstrap > 0:
        rng = random.Random(seed)
        overall_samples = []
        dtype_samples = {dtype: [] for dtype in dtype_speedups}
        for _ in range(bootstrap):
            sampled = {}
            for key in common_keys:
                ref = reference[key]
                cand = candidate[key]
                sampled[key] = bootstrap_case_speedup(ref["times_us"], cand["times_us"], rng)
            overall_samples.append(geomean(sampled.values()))
            for dtype in dtype_samples:
                dtype_samples[dtype].append(
                    geomean(value for key, value in sampled.items() if key[0] == dtype)
                )
        payload["bootstrap"] = {
            "iterations": bootstrap,
            "seed": seed,
            "overall_speedup": summarize(overall_samples),
            "dtype_speedups": {
                dtype: summarize(samples) for dtype, samples in dtype_samples.items()
            },
        }

    return payload


def main():
    parser = argparse.ArgumentParser(
        description="Compare two rigorous_benchmark.py JSON artifacts."
    )
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--name", default="hadamard_transform")
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    reference_payload, reference = load_results(args.reference, args.name)
    candidate_payload, candidate = load_results(args.candidate, args.name)
    comparison = build_comparison(reference, candidate, args.bootstrap, args.seed)
    comparison["metadata"] = {
        "reference": str(args.reference),
        "candidate": str(args.candidate),
        "reference_git_commit": reference_payload.get("metadata", {}).get("git_commit"),
        "candidate_git_commit": candidate_payload.get("metadata", {}).get("git_commit"),
        "name": args.name,
    }

    print(f"cases: {comparison['case_count']}")
    print(f"overall speedup: {comparison['overall_speedup']:.6f}x")
    for dtype, speedup in comparison["dtype_speedups"].items():
        print(f"{dtype} speedup: {speedup:.6f}x")
    if "bootstrap" in comparison:
        overall = comparison["bootstrap"]["overall_speedup"]
        print(
            "bootstrap overall median/p05/p95: "
            f"{overall['median']:.6f}x / {overall['p05']:.6f}x / {overall['p95']:.6f}x"
        )
    print("slowest candidate-relative cases:")
    for case in sorted(comparison["cases"], key=lambda item: item["speedup"])[:8]:
        print(
            f"{case['dtype']:4s} {case['dim']:5d}: "
            f"{case['speedup']:.4f}x "
            f"ref {case['reference_median_us']:.3f} us "
            f"cand {case['candidate_median_us']:.3f} us"
        )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(comparison, indent=2) + "\n")
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
