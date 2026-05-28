# Fast Hadamard Transform Experiments

Goal: benchmark the current implementation rigorously, then compound optimizations until the implementation is 50% faster than the initial measured baseline.

## Benchmark Protocol

- Hardware and software metadata are captured in each JSON benchmark artifact.
- Timings use CUDA events around a single operation call after warmup iterations.
- Each case reports median, mean, min, max, p10, p90, and IQR in microseconds.
- The main comparison metric is `hadamard_transform` median time, with `torch.clone` as a memory-traffic lower-bound reference.
- Default benchmark cases cover fp16, bf16, and fp32 for dimensions 256 through 32768.
- Inputs are sized to target 256 MiB per tensor per case, bounded by 1024 to 65536 rows.

## Experiment 000: Benchmark Harness

Hypothesis: the existing benchmark script is not rigorous enough for optimization work because it depends on `flash_attn`, has a single fixed shape, and does not persist raw timing data or environment metadata.

Change:
- Added `benchmarks/rigorous_benchmark.py`.
- Added this experiment log.
- Installed `uv 0.11.16` for package management.

Result:
- Built the unmodified CUDA extension with `FAST_HADAMARD_TRANSFORM_FORCE_BUILD=TRUE uv pip install --system --no-build-isolation -v -e .`.
- Smoke benchmark passed: fp16 dim 256, 65536 rows, 3 repeats.
- Full baseline artifact: `benchmark_results/baseline_unmodified_h200_20260528.json`.
- Baseline environment: NVIDIA H200, CUDA 12.4 runtime in PyTorch 2.4.1, CUDA driver 12.8.
- Baseline aggregate: 24 FHT cases, geometric-mean median 151.586 us, geometric-mean ratio to `torch.clone` 1.649x.
- By dtype geometric-mean ratio to `torch.clone`: fp16 1.919x, bf16 1.912x, fp32 1.223x.
- Worst ratios: fp16 dim 32768 at 413.008 us / 3.091x clone; bf16 dim 32768 at 386.176 us / 2.895x clone; bf16 dim 16384 at 269.568 us / 2.022x clone; fp16 dim 16384 at 257.456 us / 1.928x clone.

Next:
- Target the large-dimension shared-memory exchange path first.

## Experiment 001: Larger Shared-Memory Exchange Tile

Hypothesis: for power-of-two dimensions 16384 and 32768, the main FHT kernel spends avoidable time in repeated shared-memory exchange rounds. Increasing the main-kernel exchange tile from 32 KiB to 64 KiB should reduce exchange loops and synchronizations for large dimensions, especially fp16/bf16, with limited occupancy cost on H200.

Change:
- Changed the main power-of-two kernel shared-memory exchange cap from 32 KiB to 64 KiB.

Result:
- Targeted artifact: `benchmark_results/exp001_64k_smem_h200_20260528.json`.
- Targeted cases: fp16/bf16/fp32 at dimensions 8192, 16384, and 32768.
- Targeted geometric-mean speedup over baseline: 1.013x.
- Improvements: fp16 dim 32768 413.008 us to 381.776 us, 1.082x; bf16 dim 16384 269.568 us to 253.216 us, 1.065x; fp16 dim 16384 257.456 us to 253.568 us, 1.015x.
- Regressions: bf16 dim 32768 386.176 us to 393.440 us, 0.982x; fp32 dim 32768 242.848 us to 248.368 us, 0.978x.
- Decision: do not accept the unconditional 64 KiB tile as-is. Revisit as a conditional optimization after larger wins.

## Experiment 002: One-Warp/Two-Warp Launches for Medium Dimensions

Hypothesis: dimensions 1024 and 2048 are slowed down by cross-warp shared-memory exchange from using 128 and 256 threads respectively. Using fewer threads should keep more of the transform inside a warp and reduce synchronization overhead, at the cost of more per-thread registers.

Change:
- Changed the main power-of-two launch for dim 1024 (`log_N == 10`) from 128 threads to 32 threads.
- Changed the main power-of-two launch for dim 2048 (`log_N == 11`) from 256 threads to 64 threads.

Result:
- Targeted artifact: `benchmark_results/exp002_medium_launch_h200_20260528.json`.
- Targeted cases: fp16/bf16/fp32 at dimensions 512, 1024, 2048, and 4096.
- Targeted geometric-mean speedup over baseline: 1.098x.
- Strong wins at dim 1024: fp16 135.904 us to 80.544 us, 1.687x; bf16 135.424 us to 81.072 us, 1.670x; fp32 149.696 us to 138.304 us, 1.082x.
- Dim 2048 was mostly neutral: fp16 241.040 us to 240.832 us, 1.001x; bf16 241.376 us to 236.992 us, 1.019x; fp32 144.864 us to 143.152 us, 1.012x.
- Guardrail dims 512 and 4096 were effectively neutral, with small noise-level movement.
- Decision: keep the dim 1024 one-warp launch. Keep the dim 2048 two-warp launch for now because it is non-regressing, but test a one-warp dim 2048 variant next.

## Experiment 003: One-Warp Launch for Dim 2048

Hypothesis: the dim 2048 kernel may also benefit from eliminating cross-warp exchange completely by using 32 threads instead of 64, despite higher per-thread register pressure.

Change:
- Changed the main power-of-two launch for dim 2048 (`log_N == 11`) from 64 threads to 32 threads.

Result:
- Targeted artifact: `benchmark_results/exp003_2048_one_warp_h200_20260528.json`.
- Targeted cases: fp16/bf16/fp32 at dimensions 1024, 2048, and 4096.
- Targeted geometric-mean speedup over baseline: 1.230x.
- Dim 2048 wins: fp16 241.040 us to 152.736 us, 1.578x; bf16 241.376 us to 177.808 us, 1.358x; fp32 144.864 us to 141.040 us, 1.027x.
- Dim 1024 remains a strong win over baseline: fp16 1.661x, bf16 1.644x, fp32 1.080x, though this run was slightly slower than Experiment 002 at dim 1024.
- Guardrail dim 4096 was effectively neutral.
- Decision: keep the dim 2048 one-warp launch.

## Experiment 004: Smaller Launch for Dim 4096

Hypothesis: dim 4096 may have the same cross-warp exchange overhead pattern as 1024 and 2048. Reducing the launch from 256 threads to 64 threads should reduce synchronization and exchange overhead while keeping register pressure lower than a 32-thread variant.

Change:
- Changed the main power-of-two launch for dim 4096 (`log_N == 12`) from 256 threads to 64 threads.

Result:
- Targeted artifact: `benchmark_results/exp004_4096_64_threads_h200_20260528.json`.
- Targeted cases: fp16/bf16/fp32 at dimensions 2048, 4096, and 8192.
- Targeted geometric-mean speedup over baseline: 1.097x, mostly from earlier dim 2048 changes in the window.
- Dim 4096 result was dtype-dependent: fp32 improved 155.808 us to 149.248 us, 1.044x; fp16 regressed 240.192 us to 244.288 us, 0.983x; bf16 regressed 237.408 us to 239.952 us, 0.989x.
- Decision: keep the 64-thread dim 4096 launch only for fp32; keep fp16/bf16 on the original 256-thread launch.

## Build Iteration Note

Added `FAST_HADAMARD_TRANSFORM_CUDA_ARCHS`, for example `FAST_HADAMARD_TRANSFORM_CUDA_ARCHS=90`, to restrict local CUDA architecture builds during benchmarking. Default package behavior is unchanged when the variable is unset.

## Experiment 005: Smaller Launch for Dim 8192

Hypothesis: dim 8192 may benefit from reducing the main launch from 256 threads to 128 threads, lowering cross-warp exchange overhead while keeping register pressure moderate.

Change:
- Changed the main power-of-two launch for dim 8192 (`log_N == 13`) from 256 threads to 128 threads.

Result:
- Targeted artifact: `benchmark_results/exp005_8192_128_threads_h200_20260528.json`.
- Targeted cases: fp16/bf16/fp32 at dimensions 4096, 8192, and 16384.
- Dim 8192 regressed across all dtypes: fp16 242.752 us to 259.232 us, 0.936x; bf16 241.472 us to 260.080 us, 0.929x; fp32 147.536 us to 154.688 us, 0.954x.
- Decision: reject and revert dim 8192 to the original 256-thread launch.

## Experiment 006: Larger Launch for Dim 32768

Hypothesis: dim 32768 is constrained by high per-thread register pressure. Increasing the launch from 256 to 512 threads should reduce per-thread chunks and may improve occupancy or instruction scheduling enough to offset the extra warps.

Change:
- Changed the main power-of-two launch for dim 32768 (`log_N == 15`) from 256 threads to 512 threads.

Result:
- Targeted artifact: `benchmark_results/exp006_32768_512_threads_h200_20260528.json`.
- Targeted cases: fp16/bf16/fp32 at dimensions 16384 and 32768.
- Targeted geometric-mean speedup over baseline: 1.079x.
- Dim 32768 wins: fp16 413.008 us to 371.200 us, 1.113x; bf16 386.176 us to 353.024 us, 1.094x; fp32 242.848 us to 202.880 us, 1.197x.
- Guardrail dim 16384 stayed non-regressing: fp16 257.456 us to 254.640 us, 1.011x; bf16 269.568 us to 253.312 us, 1.064x; fp32 167.360 us to 166.592 us, 1.005x.
- Decision: keep the dim 32768 512-thread launch.

## Current Compounded Result After Experiment 006

Result:
- Full-sweep artifact: `benchmark_results/current_optimized_h200_20260528.json`.
- Geometric-mean speedup over the original baseline: 1.108x across 24 cases.
- Dtype geometric-mean speedups: fp16 1.152x; bf16 1.129x; fp32 1.045x.
- Remaining slowest ratios to clone: fp16 dim 32768 2.756x; bf16 dim 32768 2.674x; fp16/bf16 dim 4096 through 16384 about 1.79x to 1.91x.

## Experiment 007: Wider Launches for Large Dimensions

Hypothesis: dimensions 4096 through 32768 may be limited by per-thread work and register pressure. Increasing launch width should reduce per-thread chunks and may improve occupancy or scheduling.

Change:
- Changed low-precision dim 4096 (`log_N == 12`) from 256 to 512 threads.
- Changed dim 8192 and 16384 (`log_N == 13` and `log_N == 14`) from 256 to 512 threads.
- Changed dim 32768 (`log_N == 15`) from 512 to 1024 threads.

Result:
- Targeted artifact: `benchmark_results/exp007_wide_launches_h200_20260528.json`.
- Targeted cases: fp16/bf16/fp32 at dimensions 4096, 8192, 16384, and 32768.
- Larger launches regressed dimensions 4096, 8192, and most 16384 cases. Examples: fp16 dim 4096 240.288 us to 256.432 us, 0.937x versus current; bf16 dim 8192 241.920 us to 261.328 us, 0.926x; fp32 dim 8192 147.904 us to 168.000 us, 0.880x.
- The 1024-thread dim 32768 launch improved low precision: fp16 367.648 us to 329.120 us, 1.117x versus current and 1.255x versus baseline; bf16 354.896 us to 327.376 us, 1.084x versus current and 1.180x versus baseline.
- The 1024-thread dim 32768 launch regressed fp32 versus current: 202.240 us to 211.312 us, 0.957x.
- Decision: reject wider launches for 4096, 8192, and 16384. Keep dim 32768 at 1024 threads only for fp16/bf16; keep fp32 dim 32768 at the prior 512-thread launch.

## Experiment 008: Group Multiple One-Warp Rows per Block

Hypothesis: dimensions 256 through 2048 currently use one CUDA block per row even though each row only needs one warp. Grouping multiple independent rows into one block should reduce block scheduling overhead while preserving the same per-row math.

Change:
- Added a grouped one-warp kernel path where each warp handles one row and each block handles 8 rows.
- Routed dimensions 256, 512, 1024, and 2048 (`log_N == 8..11`) through that grouped path.

Result:
- Targeted artifact: `benchmark_results/exp008_grouped_one_warp_h200_20260528.json`.
- Targeted cases: fp16/bf16/fp32 at dimensions 256, 512, 1024, 2048, plus guardrail 4096.
- Targeted geometric-mean speedup versus current accepted state for dimensions 256 through 2048: 1.154x.
- Strong wins: fp16 dim 256 51.520 us to 29.792 us, 1.729x; bf16 dim 256 50.752 us to 29.408 us, 1.726x; bf16 dim 2048 176.784 us to 158.624 us, 1.114x.
- Regressions: fp32 dim 512 74.656 us to 76.288 us, 0.979x; fp32 dim 1024 137.504 us to 138.672 us, 0.992x; fp16 dim 2048 150.912 us to 156.928 us, 0.962x.
- Decision: keep the grouped one-warp mechanism, but tune/select it per dtype and dimension.

## Experiment 009: Tune One-Warp Rows per Block

Hypothesis: lower dimensions can tolerate more rows per block, while dim 2048 may need fewer rows per block because register pressure is higher.

Change:
- Changed dimensions 256, 512, and 1024 from 8 to 16 rows per block.
- Changed dimension 2048 from 8 to 4 rows per block.

Result:
- Targeted artifact: `benchmark_results/exp009_grouped_one_warp_tuned_h200_20260528.json`.
- The 16-row grouping regressed dimensions 256 through 1024 compared with the 8-row grouping.
- The 4-row grouping improved fp16 dim 2048 versus both current and Experiment 008: 150.912 us current, 156.928 us with 8 rows, 149.552 us with 4 rows.
- The 4-row grouping regressed bf16 dim 2048 versus 8 rows: 158.624 us with 8 rows to 172.032 us with 4 rows.
- Decision: use 8 rows per block for dim 256 all dtypes; dim 512 fp16/bf16 only; dim 1024 fp16/bf16 only; dim 2048 bf16 only. Use 4 rows per block for dim 2048 fp16/fp32. Keep original single-row block for fp32 dims 512 and 1024.
- Simulated geometric-mean speedup of the selected one-warp choices over current accepted state for dimensions 256 through 2048: 1.161x.

## Current Compounded Result After Experiment 009

Result:
- Full-sweep artifact: `benchmark_results/current_optimized_after_exp009_h200_20260528.json`.
- Geometric-mean speedup over the original baseline: 1.202x across 24 cases.
- Dtype geometric-mean speedups: fp16 1.283x; bf16 1.272x; fp32 1.065x.
- Remaining slowest ratios to clone are still large low-precision dimensions: fp16 dim 32768 2.463x; bf16 dim 32768 2.462x; fp16/bf16 dim 4096 through 16384 about 1.79x to 1.91x.

## Experiment 010: Multi-Row Blocks for Large Low-Precision Dimensions

Hypothesis: low-precision dimensions 4096 through 16384 are still slow because each row uses one block. Grouping multiple independent rows into a larger block while preserving the same per-row 256-thread transform may reduce block scheduling overhead.

Change:
- Added a temporary multi-row block kernel path with local per-row shared-memory regions.
- Routed fp16/bf16 dim 4096 through 4 rows per block.
- Routed fp16/bf16 dims 8192 and 16384 through 2 rows per block.

Result:
- Targeted artifact: `benchmark_results/exp010_multirow_large_dims_h200_20260528.json`.
- Targeted geometric-mean speedup versus current accepted state for dimensions 4096 through 32768: 0.938x.
- Regressions: fp16 dim 4096 239.904 us to 253.104 us, 0.948x; bf16 dim 8192 242.144 us to 254.272 us, 0.952x; fp16 dim 16384 254.032 us to 335.504 us, 0.757x; bf16 dim 16384 253.504 us to 335.760 us, 0.755x.
- Guardrail dim 32768 was effectively neutral.
- Decision: reject and remove the multi-row block path. The extra block-level synchronization and shared-memory footprint outweigh any block scheduling savings.

## Experiment 011: Wider Vector Width for Large Low-Precision Dimensions

Hypothesis: fp16/bf16 dimensions 4096 and larger may benefit from loading and storing 16 elements per thread instead of 8, reducing loop overhead and memory instruction count.

Change:
- Temporarily changed low-precision `kNElts` from 8 to 16 for dimensions 4096 and larger.

Result:
- Build failed for the bf16 dim 32768 instantiation because the computed shared-memory exchange tiling became invalid (`kChunksPerExchange == 0`) with the existing 32 KiB exchange tile and 1024-thread launch.
- Decision: reject this broad change and restore `kNElts = 8` for fp16/bf16. A narrower vector-width experiment would need per-dimension/per-launch feasibility checks before benchmarking.

## Experiment 012: Low-Precision Shared-Memory Exchange

Hypothesis: large fp16/bf16 kernels are dominated by shared-memory exchange bandwidth. Exchanging fp16/bf16 values through shared memory, instead of exchanging float-expanded intermediates, should reduce shared-memory traffic and synchronization pressure.

Change:
- Added a low-precision exchange kernel path for fp16/bf16 power-of-two dimensions 4096 through 32768.
- Kept arithmetic in float registers, but cast values to fp16/bf16 for the cross-warp shared-memory exchange.

Result:
- Targeted artifact: `benchmark_results/exp012_lowp_smem_exchange_h200_20260528.json`.
- Targeted geometric-mean speedup versus Experiment 011 cleanup over the 12 large-dimension cases: 1.087x.
- Wins: fp16 dim 4096 240.288 us to 210.528 us, fp16 dim 32768 367.648 us to 284.336 us, bf16 dim 32768 354.896 us to 291.488 us.
- Decision: accept. This is the first large low-precision kernel win after the launch-shape work.

## Experiment 013: 512-Thread Low-Precision Exchange

Hypothesis: low-precision exchange kernels may benefit from wider launches because each thread would hold fewer chunks and use fewer registers.

Change:
- Changed large fp16/bf16 low-precision exchange launches to 512 threads.

Result:
- Targeted artifact: `benchmark_results/exp013_lowp_exchange_512_threads_h200_20260528.json`.
- Targeted geometric-mean speedup versus Experiment 012 over common cases: 0.940x.
- Regressions: fp16 dim 4096 210.528 us to 231.984 us; fp16 dim 32768 284.336 us to 359.424 us; bf16 dim 32768 291.488 us to 317.936 us.
- Decision: reject and restore the prior thread counts.

## Experiment 014: Narrower Low-Precision Exchange Launches

Hypothesis: dimensions 4096 through 16384 might prefer narrower low-precision exchange launches because they reduce cross-warp work and shared-memory pressure.

Change:
- Tested narrower low-precision exchange launch shapes for fp16/bf16 large dimensions.

Result:
- Targeted artifact: `benchmark_results/exp014_lowp_exchange_narrow_threads_h200_20260528.json`.
- Targeted geometric-mean speedup versus Experiment 012 over common cases: 0.942x.
- Mixed result: fp16 dim 4096 improved slightly to 206.880 us, but fp16 dim 16384 regressed to 257.360 us and fp16/bf16 dim 32768 regressed to about 362 us.
- Decision: reject. A small 4096 win is not worth the large-dimension losses.

## Experiment 015: Low-Precision Launch Bounds With Minimum Blocks

Hypothesis: `__launch_bounds__(..., 2)` may improve occupancy or scheduling for low-precision exchange kernels, especially 32768, where H200 appears sensitive to block residency.

Change:
- Added a two-block launch-bound variant broadly to the low-precision exchange kernels.

Result:
- Targeted artifact: `benchmark_results/exp015_lowp_exchange_launch_bounds2_h200_20260528.json`.
- Targeted geometric-mean speedup versus Experiment 012 over common cases: 0.975x.
- Helped bf16 dim 32768, 291.488 us to 279.456 us, but regressed dimensions 4096 through 16384.
- Decision: reject the global launch-bound change. Keep the 32768 observation for a narrower variant.

## Experiment 016: Conditional Launch Bounds for 1024-Thread Low-Precision 32768

Hypothesis: the launch-bound benefit is specific to the 1024-thread dim 32768 low-precision path, so applying it only there should keep the 32768 win without hurting smaller dimensions.

Change:
- Added a separate min-blocks kernel wrapper for low-precision exchange and routed only the 1024-thread 32768 path through it.

Result:
- Targeted artifact: `benchmark_results/exp016_lowp_exchange_launch_bounds2_1024_conditional_h200_20260528.json`.
- Targeted geometric-mean speedup versus Experiment 012 over common cases: 0.980x.
- 32768 improved, but several guardrail cases still moved down enough that this exact variant was not accepted.
- Decision: reject this form and split the wrapper more carefully in the next experiment.

## Experiment 017: Split Min-Blocks Wrapper Only for 32768

Hypothesis: using a distinct launch function for the min-blocks kernel will avoid unintended codegen or routing effects while preserving the 32768 low-precision benefit.

Change:
- Added a separate `fast_hadamard_transform_lowp_exchange_min_blocks_kernel`.
- Routed fp16/bf16 dim 32768 through `fast_hadamard_transform_lowp_exchange_launch<1024, 15, input_t, true>`.

Result:
- Targeted artifact: `benchmark_results/exp017_lowp_exchange_min_blocks_only_32768_h200_20260528.json`.
- Targeted geometric-mean speedup versus Experiment 012 over common cases: 1.001x, with the intended bf16 dim 32768 improvement.
- Full accepted artifact after this point: `benchmark_results/current_optimized_after_exp017_final_h200_20260528.json`.
- Full geometric-mean speedup over baseline: 1.230x.
- Decision: accept, mainly for bf16 dim 32768, while keeping the other accepted low-precision exchange routing.

## Experiment 018: Full Low-Precision Compute

Hypothesis: if shared-memory low-precision exchange helps by reducing bandwidth, then doing more of the transform directly in fp16/bf16 may further reduce register and conversion pressure.

Change:
- Added direct low-precision helper paths for load, thread Hadamard, warp Hadamard, shared-memory exchange, and store.

Result:
- Targeted artifact: `benchmark_results/exp018_lowp_compute_h200_20260528.json`.
- Targeted geometric-mean speedup versus Experiment 017 over common cases: 0.756x.
- Regressions were severe: bf16 dim 32768 278.752 us to 547.456 us; bf16 dim 16384 234.064 us to 461.984 us; fp16 dim 32768 284.400 us to 360.672 us.
- Decision: reject and remove the direct low-precision compute helpers. Float arithmetic with low-precision exchange remains the better tradeoff.

## Experiment 019: Conditional Add/Sub Warp Hadamard

Hypothesis: replacing `sign * x + other` with branch-selected add/sub inside the warp Hadamard may avoid multiply instructions and improve large kernels.

Change:
- Temporarily changed the warp Hadamard update globally to conditional add/sub.

Result:
- Targeted artifact: `benchmark_results/exp019_warp_conditional_addsub_h200_20260528.json`.
- Full geometric-mean speedup versus the accepted Experiment 017 state: 0.998x.
- Wins: fp16 dim 32768 284.400 us to 274.848 us; fp32 dim 16384 167.056 us to 158.048 us.
- Regressions: fp16 dim 8192 209.792 us to 216.960 us; bf16 dim 16384 234.064 us to 244.240 us; fp32 dim 32768 202.736 us to 208.352 us.
- Decision: reject the global change. A future selective variant could target the few winning dtype/dimension pairs, but the global form is not robust.

## Experiment 020: Wider FP32 Vector Width

Hypothesis: fp32 kernels may benefit from `kNElts = 8` by reducing loop and memory-instruction overhead.

Change:
- Temporarily changed the fp32 power-of-two kernel vector width from 4 to 8 elements.

Result:
- Targeted artifact: `benchmark_results/exp020_fp32_8elts_h200_20260528.json`.
- Targeted geometric-mean speedup over the fp32 baseline subset: 0.907x.
- Regressions: fp32 dim 4096 149.920 us to 176.928 us; dim 8192 148.336 us to 214.896 us; dim 16384 167.056 us to 222.480 us; dim 32768 202.736 us to 281.200 us.
- Decision: reject and restore fp32 `kNElts = 4`.

## Experiment 021: Python Inference Direct-Call Fast Path

Hypothesis: benchmark inference calls run under `torch.no_grad()`, so routing through a custom autograd function still pays Python autograd wrapper overhead that is avoidable.

Change:
- In each public Python wrapper, call the C++ extension directly when gradients are disabled or the input does not require gradients.
- Preserve the existing autograd path when gradients are required.

Result:
- Full artifact: `benchmark_results/exp021_inference_direct_call_h200_20260528.json`.
- Full geometric-mean speedup versus Experiment 017 accepted state: 1.046x.
- Full geometric-mean speedup versus baseline: 1.286x.
- Examples: fp16 dim 256 51.584 us baseline to 27.552 us; bf16 dim 256 51.120 us to 26.464 us; fp32 dim 32768 242.848 us to 200.512 us.
- Decision: accept. This is a low-risk inference fast path because autograd behavior is unchanged when gradients are active.

## Experiment 022: C++ 2D Row-Contiguous Fast Path

Hypothesis: the common benchmark shape is already 2D and row-contiguous, so the C++ extension can skip reshape and shape-restoration overhead before launching the kernel.

Change:
- Added a direct C++ fast path in `fast_hadamard_transform` for 2D row-contiguous tensors with supported power-of-two dimensions.
- Kept the original reshape-based fallback for other shapes.

Result:
- Full artifact: `benchmark_results/exp022_cpp_2d_fastpath_h200_20260528.json`.
- Full geometric-mean speedup versus Experiment 021: 1.013x.
- Full geometric-mean speedup versus baseline: 1.303x.
- Full geometric-mean ratio to `torch.clone`: 1.260x.
- Decision: accept. The change is narrow and preserves fallback behavior for non-2D or padded cases.

## Experiment 023: Low-Precision Exchange `kNElts = 16`

Hypothesis: with the low-precision exchange kernel now storing fp16/bf16 through shared memory, using 16 values per thread might reduce loop overhead and improve memory instruction efficiency.

Change:
- Temporarily changed low-precision exchange `kNElts` from 8 to 16.

Result:
- Targeted artifact: `benchmark_results/exp023_lowp_exchange_16elts_h200_20260528.json`.
- Targeted geometric-mean speedup versus Experiment 022 over common cases: 0.877x.
- Regressions across all targeted low-precision cases: fp16 dim 4096 205.504 us to 246.592 us; fp16 dim 32768 280.320 us to 380.784 us; bf16 dim 32768 275.200 us to 328.480 us.
- Decision: reject and restore low-precision exchange `kNElts = 8`.

## Experiment 024: FP16 32768 Without Min-Blocks Launch Bounds

Hypothesis: Experiment 017's min-blocks launch wrapper helped bf16 dim 32768, but fp16 dim 32768 might prefer the normal low-precision exchange wrapper.

Change:
- Temporarily routed fp16 dim 32768 to the non-min-blocks low-precision exchange launch while keeping bf16 on the min-blocks path.

Result:
- Targeted artifact: `benchmark_results/exp024_fp16_32768_no_min_blocks_h200_20260528.json`.
- Target case did not improve: fp16 dim 32768 changed from 280.320 us in Experiment 022 to 280.896 us.
- Other targeted case movements were noise-level and did not justify the change.
- Decision: reject and restore min-blocks routing for all low-precision dim 32768 cases.

## Checkpoint After Experiment 024

Result:
- Final full-sweep artifact: `benchmark_results/current_optimized_final_h200_20260528.json`.
- Final geometric-mean median time: 116.796 us versus 151.586 us baseline.
- Final geometric-mean speedup over baseline: 1.298x across 24 cases.
- Dtype geometric-mean speedups over baseline: fp16 1.425x, bf16 1.400x, fp32 1.096x.
- Final geometric-mean ratio to `torch.clone`: 1.260x.
- The requested 50% aggregate speedup was not reached. The accepted implementation is about 29.8% faster by the benchmark's geometric-mean metric, with several individual cases exceeding 50% but the full suite still limited by large low-precision kernels and the memory-traffic lower bound.

## Experiment 025: Native Half2 Large-Dimension Kernel

Hypothesis: doing the large fp16 power-of-two kernels directly in `half2` should reduce register pressure and instruction count compared with float arithmetic plus fp16 shared-memory exchange.

Change:
- Added a native `half2` large-dimension kernel for fp16 dimensions 4096 through 32768.

Result:
- Targeted artifact: `benchmark_results/exp025_fp16_half2_large_h200_20260528.json`.
- Large fp16 cases improved substantially: 4096 145.488 us, 8192 147.728 us, 16384 162.816 us, 32768 225.216 us.
- Decision: accept for fp16 large dimensions, while continuing to tune 32768.

## Experiment 026: Native Bfloat162 Large-Dimension Kernel

Hypothesis: the same vectorized low-precision compute path should help bf16 once implemented with `__nv_bfloat162`.

Change:
- Added a native `bfloat162` large-dimension kernel for bf16.

Result:
- Targeted artifact: `benchmark_results/exp026_fp16_bf16_vec2_large_h200_20260528.json`.
- bf16 improved to 138.176 us at 4096, 147.712 us at 8192, 166.288 us at 16384, and 230.576 us at 32768.
- fp16 remained near Experiment 025: 144.944 us, 147.504 us, 162.560 us, and 225.040 us.
- Decision: accept for bf16 dimensions 4096 through 16384; keep tuning 32768.

## Experiment 027: 512-Thread Vectorized 32768

Hypothesis: the vectorized 32768 low-precision kernels still carry too much per-thread work, and 512 threads may reduce latency without the earlier 1024-thread shared-memory penalty.

Change:
- Routed fp16/bf16 vectorized dim 32768 through 512-thread kernels.

Result:
- Targeted artifact: `benchmark_results/exp027_vec2_32768_512_threads_h200_20260528.json`.
- fp16 dim 32768 improved to 204.368 us and bf16 dim 32768 to 198.880 us.
- Decision: accept 512-thread vectorized 32768 for fp16 and bf16.

## Experiment 028: 256-Thread Vectorized 32768

Hypothesis: 256 threads may improve occupancy or scheduling for vectorized dim 32768.

Change:
- Routed vectorized fp16/bf16 dim 32768 through 256-thread kernels.

Result:
- Targeted artifact: `benchmark_results/exp028_vec2_32768_256_threads_h200_20260528.json`.
- The 256-thread variant was slower than the accepted 512-thread 32768 path.
- Decision: reject.

## Experiment 029: Vectorized One-Warp 2048

Hypothesis: dim 2048 fp16/bf16 can avoid float-expanded work by using the same native vectorized low-precision math inside a grouped one-warp kernel.

Change:
- Added vectorized one-warp kernels for fp16/bf16 dim 2048.
- Restored accepted 512-thread routing for vectorized dim 32768.

Result:
- Targeted artifact: `benchmark_results/exp029_vec2_2048_and_32768_512_h200_20260528.json`.
- fp16 dim 2048 improved to 139.184 us; bf16 dim 2048 improved to 138.144 us.
- Large low-precision results stayed strong, with fp16/bf16 dim 32768 at 203.760 us and 198.816 us.
- Decision: accept.

## Experiment 030: Direct Vectorized I/O

Hypothesis: loading and storing directly through vectorized `half2`/`bfloat162` storage should remove temporary scalar packing arrays.

Change:
- Reworked vectorized low-precision load/store helpers to use direct vectorized I/O.

Result:
- Targeted artifact: `benchmark_results/exp030_direct_vec2_io_h200_20260528.json`.
- Large low-precision dimensions regressed, with 4096 through 16384 moving to roughly 186-188 us.
- Decision: reject and restore the scalar-pack load/store helpers.

## Experiment 031: Half2 16-Element Variant

Hypothesis: fp16 might benefit from 16 values per thread by reducing loop overhead and memory instructions.

Change:
- Added an experimental `half2` kernel with `kNElts = 16`.

Result:
- Targeted artifact: `benchmark_results/exp031_fp16_half2_16elts_large_h200_20260528.json`.
- fp16 regressed: 4096 191.040 us, 8192 199.120 us, 16384 205.424 us, 32768 244.720 us.
- Decision: reject. The experimental code was later removed during cleanup.

## Experiment 032: 512-Thread Vectorized 4096-16384

Hypothesis: some lower large dimensions may also benefit from wider vectorized launches.

Change:
- Routed vectorized low-precision dimensions 4096 through 16384 through 512-thread kernels.

Result:
- Targeted artifact: `benchmark_results/exp032_vec2_512_threads_4096_16384_h200_20260528.json`.
- Broadly rejected, but fp16 dim 16384 improved to 158.880-159.536 us in repeated measurements.
- Decision: accept only fp16 dim 16384 at 512 threads; reject the broader change.

## Experiment 033: 128-Thread Vectorized 4096-16384

Hypothesis: narrower vectorized launches might reduce cross-warp overhead for 4096 through 16384.

Change:
- Tested 128-thread vectorized fp16/bf16 kernels for 4096 through 16384.

Result:
- Targeted artifact: `benchmark_results/exp033_vec2_128_threads_4096_16384_h200_20260528.json`.
- The narrower kernels regressed the targeted low-precision cases.
- Decision: reject.

## Experiment 034: 1024-Thread Vectorized 16384

Hypothesis: dim 16384 may prefer a still wider vectorized launch because it has enough work per row to amortize the extra warps.

Change:
- Tested 1024-thread vectorized low-precision kernels for dim 16384.

Result:
- Targeted artifact: `benchmark_results/exp034_vec2_1024_threads_16384_h200_20260528.json`.
- The 1024-thread dim 16384 variant did not improve the accepted route overall.
- Decision: reject.

## Experiment 035: Selective FP32 Conditional Add/Sub at 16384

Hypothesis: the conditional add/sub warp helper rejected globally in Experiment 019 may still help fp32 dim 16384 specifically.

Change:
- Routed fp32 dim 16384 through `hadamard_mult_warp_conditional`.

Result:
- Targeted artifact: `benchmark_results/exp035_selective_fp32_conditional_16384_h200_20260528.json`.
- fp32 dim 16384 improved to 153.712 us.
- Low-precision guardrails remained consistent with accepted vectorized paths.
- Decision: accept only for fp32 dim 16384.

## Experiment 036: Vectorized Small One-Warp Kernels

Hypothesis: fp16/bf16 dimensions 256 through 1024 should benefit from native vectorized low-precision math just like dim 2048.

Change:
- Routed low-precision dimensions 256, 512, and 1024 through grouped vectorized one-warp kernels with 8 rows per block.

Result:
- Targeted artifact: `benchmark_results/exp036_vec2_small_one_warp_h200_20260528.json`.
- fp16 improved to 24.208 us at 256, 40.160 us at 512, and 72.896 us at 1024.
- bf16 improved to 24.192 us at 256, 40.400 us at 512, and 72.352 us at 1024.
- Decision: accept the vectorized small-kernel path, then tune row grouping.

## Experiment 037: FP32 32768 64 KiB Exchange Tile

Hypothesis: the 64 KiB main-kernel shared-memory tile that was mixed globally in Experiment 001 may be useful for fp32 dim 32768 after the launch-shape changes.

Change:
- Enabled the 64 KiB exchange tile for main fp32 dimensions 16384 and 32768.

Result:
- Targeted artifact: `benchmark_results/exp037_fp32_32768_64k_exchange_h200_20260528.json`.
- fp32 dim 32768 improved to 196.608 us.
- fp32 dim 16384 stayed near the accepted conditional route.
- Decision: accept the selective large exchange tile for dimensions 16384 and 32768.

## Experiment 038: FP32 32768 Min-Blocks Launch Bounds

Hypothesis: `__launch_bounds__(..., 2)` may improve occupancy for fp32 dim 32768 as it once did for an older low-precision path.

Change:
- Added a min-blocks main-kernel wrapper and routed fp32 dim 32768 through it.

Result:
- Targeted artifact: `benchmark_results/exp038_fp32_32768_min_blocks2_h200_20260528.json`.
- fp32 dim 32768 regressed to 313.584 us.
- ptxas reported large spill traffic for the min-blocks variant.
- Decision: reject and remove the min-blocks wrapper during cleanup.

## Experiment 039: 32 KiB Vectorized Low-Precision 32768 Exchange

Hypothesis: vectorized low-precision dim 32768 may prefer a smaller 32 KiB exchange tile because the 64 KiB tile can reduce block residency.

Change:
- Temporarily capped vectorized low-precision shared-memory exchange at 32 KiB.

Result:
- Targeted artifact: `benchmark_results/exp039_lowp_32768_32k_vec2_exchange_h200_20260528.json`.
- fp16 dim 32768 regressed to 266.512 us and bf16 dim 32768 to 273.696 us.
- Decision: reject and restore the 64 KiB cap.

## Experiment 040: Small Vectorized One-Warp Rows Per Block = 4

Hypothesis: dimensions 256 through 2048 may prefer fewer rows per block once the math is vectorized, because register pressure is different from the float one-warp kernel.

Change:
- Tested 4 rows per block for small vectorized low-precision one-warp kernels.

Result:
- Targeted artifact: `benchmark_results/exp040_vec2_small_rows4_h200_20260528.json`.
- fp16 reached 23.616 us at 256, 39.168 us at 512, 71.968 us at 1024, and 139.216 us at 2048.
- bf16 reached 23.488 us at 256, 39.424 us at 512, 71.952 us at 1024, and 138.688 us at 2048.
- Decision: accept rows=4 for fp16/bf16 dimensions 256 and 512, and keep evaluating 1024/2048 individually.

## Experiment 041: Small Vectorized One-Warp Rows Per Block = 2

Hypothesis: dim 1024 may prefer fewer grouped rows because its per-warp register pressure is higher.

Change:
- Tested 2 rows per block for small vectorized low-precision one-warp kernels.

Result:
- Targeted artifacts: `benchmark_results/exp041_vec2_small_rows2_h200_20260528.json` and `benchmark_results/exp041_vec2_small_rows2_h200_20260528_repeat2.json`.
- bf16 dim 1024 improved to 71.136 us on repeat; fp16 dim 1024 did not beat the next experiment.
- Rows=2 regressed dim 256 and did not improve most other small cases.
- Decision: accept rows=2 only for bf16 dim 1024.

## Experiment 042: Small Vectorized One-Warp Rows Per Block = 1

Hypothesis: fp16 dim 1024 may be fastest with one row per block after vectorization.

Change:
- Routed fp16 dim 1024 through a one-row vectorized one-warp block.

Result:
- Targeted artifact: `benchmark_results/exp042_vec2_1024_rows1_h200_20260528.json`.
- fp16 dim 1024 improved to 71.344 us; bf16 was 71.248 us, slightly behind the best rows=2 repeat.
- Decision: accept rows=1 for fp16 dim 1024 and keep rows=2 for bf16 dim 1024.

## Current Accepted State After Experiment 042

Result:
- Full-sweep artifacts:
  - `benchmark_results/current_optimized_exp042_mix_full_h200_20260528.json`: 101.895 us geometric mean, 1.488x over baseline.
  - `benchmark_results/current_optimized_exp042_mix_full_h200_20260528_repeat2.json`: 100.952 us geometric mean, 1.502x over baseline.
  - `benchmark_results/current_optimized_exp042_mix_full_h200_20260528_repeat3.json`: 100.852 us geometric mean, 1.503x over baseline.
- Cleaned final full-sweep artifact: `benchmark_results/current_optimized_exp042_clean_full_h200_20260528.json`.
- Cleaned final geometric-mean median time: 100.871 us versus 151.586 us baseline.
- Cleaned final geometric-mean speedup over baseline: 1.503x across 24 cases.
- Correctness/unit verification after cleanup: `pytest -q tests/test_fast_hadamard_transform.py`, 51 passed.
- The final cleaned implementation reaches the requested 50% aggregate speedup by the benchmark's geometric-mean median metric.
- Accepted implementation highlights:
  - inference direct-call and C++ 2D row-contiguous fast path;
  - native `half2`/`bfloat162` kernels for fp16/bf16 power-of-two dimensions;
  - grouped vectorized one-warp kernels for low-precision dimensions 256 through 2048;
  - selective fp32 launch and warp-helper tuning for 4096, 16384, and 32768;
  - selective 64 KiB exchange tile for main-kernel dimensions 16384 and 32768.

## Experiment 043: Gate Native Low-Precision Arithmetic

Hypothesis: the native `half2`/`bfloat162` kernels improve speed but change fp16/bf16 rounding behavior versus the safer float-intermediate kernels. Upstream clients should get the conservative precision behavior by default and opt into the faster low-precision arithmetic explicitly.

Change:
- Added `fast_low_precision=False` to `hadamard_transform`.
- Threaded the flag through the autograd function, C++ binding, and `HadamardParamsBase`.
- Routed fp16/bf16 power-of-two dimensions through float-intermediate kernels by default.
- Kept native `half2`/`bfloat162` kernels behind `fast_low_precision=True`.
- Added `--fast-low-precision` to `benchmarks/rigorous_benchmark.py`.
- Added a focused unit test for the opt-in fast low-precision mode.

Result:
- Correctness/unit verification: `pytest -q tests/test_fast_hadamard_transform.py`, 55 passed.
- Direct precision check versus fp32 reference showed the default path back near baseline-like error levels:
  - fp16 default max abs over sampled dims: about 0.00097 to 0.00191.
  - fp16 fast max abs over sampled dims: about 0.00296 to 0.00560.
  - bf16 default max abs over sampled dims: about 0.00777 to 0.01546.
  - bf16 fast max abs over sampled dims: about 0.02387 to 0.04422.
- Post-gate fast-flag full-sweep artifacts:
  - `benchmark_results/current_optimized_exp043_fast_lowp_flag_full_h200_20260528.json`: 101.343 us geometric mean, 1.496x over baseline.
  - `benchmark_results/current_optimized_exp043_fast_lowp_flag_full_h200_20260528_repeat2.json`: 101.298 us geometric mean, 1.496x over baseline.
- Decision: accept the gate. The latest fast-flag speed runs are slightly below the previous 1.5x artifacts due to run-to-run movement, but the precision-risking arithmetic is now opt-in instead of default.

## Current Default Precision Baseline After Experiment 043

Result:
- Full default precision-preserving artifact: `benchmark_results/current_default_precision_full_h200_20260528.json`.
- Default geometric-mean median time: 121.634 us versus 151.586 us baseline.
- Default speedup over baseline: 1.246x across 24 cases.
- Dtype geometric-mean speedups: fp16 1.324x, bf16 1.313x, fp32 1.113x.
- The gap to 1.5x is mostly the fp16/bf16 large dimensions after moving the native `half2`/`bfloat162` kernels behind the opt-in flag. Many fp32 cases are already close to `torch.clone`, so a per-case 1.5x target is below the measured output-materialization lower bound for those shapes.

## Experiment 044: Precision-Preserving 32768 Launch Width

Hypothesis: the default fp16/bf16 dim 32768 path may still benefit from the wider 1024-thread launch found earlier, because it reduces per-thread chunks while keeping float-intermediate arithmetic.

Change:
- Tested 1024-thread default float-intermediate launches for fp16 and bf16 dim 32768.
- Split the route after the first measurement so only fp16 uses 1024 threads; bf16 returned to 512 threads.

Result:
- Broad artifact: `benchmark_results/exp044_default_lowp_32768_1024_threads_h200_20260528.json`.
- Broad result: fp16 dim 32768 improved to 326.048 us, but bf16 regressed to 359.744 us.
- Split artifact: `benchmark_results/exp044_default_lowp_32768_fp16_only_1024_threads_h200_20260528.json`.
- Split result: fp16 dim 32768 measured 324.480 us; bf16 stayed near the prior route at 350.928 us.
- Decision: accept 1024 threads only for default fp16 dim 32768. Reject the bf16 1024-thread route.

## Experiment 045: 128 KiB Float Exchange Tile for 32768

Hypothesis: dim 32768 still pays two shared-memory exchange rounds in the float-intermediate kernel. A 128 KiB exchange tile can reduce that to one round for 16-bit input while preserving float shared-memory storage.

Change:
- Temporarily enabled a 128 KiB main-kernel exchange tile for all dim 32768 dtypes.
- Narrowed the change to `sizeof(input_t) == 2` after fp32 regressed.

Result:
- Artifact: `benchmark_results/exp045_default_32768_128k_exchange_h200_20260528.json`.
- fp16 dim 32768 measured 323.584 us; bf16 measured 348.096 us.
- fp32 dim 32768 regressed to 199.488 us versus about 196 us in adjacent runs.
- Decision: accept the 128 KiB exchange tile only for 16-bit default dim 32768; keep fp32 on the prior 64 KiB tile.

## Experiment 046: Broad Conditional Add/Sub for Large 16-Bit Defaults

Hypothesis: replacing the sign-multiply form in warp Hadamard with branch-selected add/sub may reduce instruction work in the large default fp16/bf16 kernels without changing the intended float-intermediate arithmetic.

Change:
- Routed default fp16/bf16 dimensions 4096 through 32768 through the conditional add/sub warp helper.

Result:
- Artifact: `benchmark_results/exp046_default_lowp_large_conditional_addsub_h200_20260528.json`.
- Mixed result: bf16 dim 32768 improved to 342.304 us, but fp16 dim 32768 regressed to 359.936 us and bf16 dim 16384 regressed to 262.144 us.
- Decision: reject the broad route. Keep only the bf16 dim 32768 observation for a selective test.

## Experiment 047: Selective Precision-Preserving 32768 Defaults

Hypothesis: the useful default-path changes are specific to dim 32768: fp16 wants 1024 threads, bf16 wants conditional add/sub, and both 16-bit dtypes can use the 128 KiB float exchange tile.

Change:
- Kept fp16 dim 32768 on the 1024-thread float-intermediate launch.
- Kept bf16 dim 32768 on the 512-thread float-intermediate launch with conditional add/sub.
- Kept the 128 KiB exchange tile only for 16-bit dim 32768.

Result:
- Targeted artifact: `benchmark_results/exp047_default_selective_32768_precision_safe_h200_20260528.json`.
- Targeted dim 32768 results: fp16 323.520 us, bf16 343.408 us, fp32 197.664 us.
- Full-sweep artifacts:
  - `benchmark_results/current_default_precision_exp047_full_h200_20260528.json`: 121.921 us geometric mean, 1.243x over baseline.
  - `benchmark_results/current_default_precision_exp047_full_h200_20260528_repeat2.json`: 120.762 us geometric mean, 1.255x over baseline.
- Decision: accept the selective 32768 default-path changes. They improve the affected large 16-bit cases without using native low-precision arithmetic, but the default precision-preserving path remains far short of 1.5x aggregate speedup.

## Experiment 048: 16-Element Float-Exchange Shape for Large 16-Bit Defaults

Hypothesis: a wider per-thread vector shape may reduce loop and transpose overhead in large fp16/bf16 kernels while preserving float arithmetic and float shared-memory exchange.

Change:
- Temporarily changed the main kernel's 16-bit `kNElts` from 8 to 16 for dimensions 4096 and larger.

Result:
- Artifact: `benchmark_results/exp048_default_large_16elts_float_exchange_h200_20260528.json`.
- Regressed every targeted fp16/bf16 case. Examples: fp16 dim 4096 moved to 313.136 us and bf16 dim 32768 moved to 434.912 us.
- Decision: reject and restore `kNElts = 8` for the default 16-bit float-intermediate kernel.

## Experiment 049: Native Load/Store Boundary for Large Defaults

Hypothesis: keeping native fp16/bf16 vectors only at the global-memory boundary could reduce load/store overhead while retaining float intermediates for the transform math.

Change:
- Temporarily used native input/output vector types around the large default float-intermediate kernels.

Result:
- Artifact: `benchmark_results/exp049_native_load_store_default_large_h200_20260528.json`.
- Large fp16/bf16 cases were neutral to slower; fp16 dim 32768 regressed to 353.280 us and bf16 dim 32768 to 358.320 us.
- fp32 guard cases stayed near the existing route.
- Decision: reject. Boundary vectorization did not buy enough to offset codegen and conversion cost.

## Experiment 050: 32-Byte Float Exchange Vectors

Hypothesis: wider shared-memory exchange vectors might reduce transaction overhead in the float-intermediate default kernels.

Change:
- Temporarily widened the exchange vector shape for large default float-intermediate kernels.

Result:
- Artifact: `benchmark_results/exp050_float_exchange_32b_vec_default_large_h200_20260528.json`.
- Severe regressions across the targeted 16-bit cases: fp16 dim 4096 moved to 294.352 us and bf16 dim 32768 to 398.144 us.
- Decision: reject and keep the prior exchange vector shape.

## Experiment 051: Broad Initial Pre-Exchange Sync Skip

Hypothesis: the first shared-memory barrier in a one-round pre-exchange is redundant because the buffer has not been read by the current exchange sequence yet.

Change:
- Temporarily skipped the initial pre-exchange `__syncthreads()` for all one-round exchanges.

Result:
- Artifact: `benchmark_results/exp051_skip_initial_pre_exchange_sync_h200_20260528.json`.
- Large 16-bit cases improved, including fp16 dim 32768 at 313.200 us.
- fp32 dim 16384 regressed badly to 181.216 us.
- Decision: reject the broad route. The sync skip must be narrowed away from the affected fp32 path.

## Experiment 052: 16-Bit Initial Pre-Exchange Sync Skip

Hypothesis: the same one-round pre-exchange barrier skip can be limited to the 16-bit float-intermediate shape, avoiding the fp32 regression while preserving math precision.

Change:
- Skipped the initial pre-exchange barrier only for kernels with `kNElts == 8`.

Result:
- Artifact: `benchmark_results/exp052_skip_initial_pre_exchange_sync_16bit_only_h200_20260528.json`.
- fp32 returned to the expected range.
- fp16 large cases improved, but bf16 dim 32768 moved to 345.008 us, behind the accepted selective 32768 route.
- Decision: reject this still-broad 16-bit route and narrow the exception further.

## Experiment 053: Selective Initial Pre-Exchange Sync Skip

Hypothesis: skipping the initial pre-exchange barrier is useful for the one-round 16-bit float-intermediate kernels except the bf16 dim 32768 route, where prior measurements showed a small regression.

Change:
- Added a compile-time `kSkipInitialPreSync` parameter to `exchange_smem_pre`.
- Enabled the skip only for `kNElts == 8`, excluding bf16 dim 32768.
- Kept post-exchange barriers and multi-round exchange barriers intact.

Result:
- Artifact: `benchmark_results/exp053_selective_initial_pre_sync_skip_h200_20260528.json`.
- Targeted large-case results: fp16 4096 234.512 us, 8192 235.456 us, 16384 245.952 us, 32768 312.624 us; bf16 4096 233.440 us, 8192 235.680 us, 16384 248.720 us, 32768 341.840 us.
- fp32 guard cases stayed near the expected range.
- Decision: accept. This is precision-preserving because it changes synchronization only, not arithmetic dtype.

## Experiment 054: Explicit 8-Element Thread Hadamard

Hypothesis: spelling out the 8-element per-thread Hadamard butterfly may help the compiler reduce loop overhead in the large float-intermediate path.

Change:
- Temporarily replaced the generic 8-element thread helper with an explicit unrolled implementation.

Result:
- Artifact: `benchmark_results/exp054_explicit_thread8_full_h200_20260528.json`.
- Full-sweep geometric mean was 121.371 us, or 1.249x over baseline, behind the best Experiment 047 repeat.
- Decision: reject. The explicit helper did not improve the full suite.

## Experiment 055: Runtime Exact-Dimension Load/Store Fast Path

Hypothesis: rows whose dimension exactly matches the transform length can skip dynamic boundary checks in load/store.

Change:
- Temporarily added a runtime exact-dimension fast path for load/store.

Result:
- Artifact: `benchmark_results/exp055_full_dim_load_store_fastpath_h200_20260528.json`.
- Full-sweep geometric mean was 121.643 us, or 1.246x over baseline.
- bf16 dim 32768 improved to 328.832 us, but many other cases regressed, including fp32 dim 8192 and 16384.
- Decision: reject. The runtime branch and codegen cost outweighed the isolated bf16 win.

## Experiment 056: Compile-Time Exact-Dimension bf16 32768 Fast Path

Hypothesis: specializing only the observed bf16 dim 32768 exact-dimension win avoids the broader runtime-path regressions.

Change:
- Temporarily routed only bf16 dim 32768 through a compile-time exact-dimension load/store specialization.

Result:
- Artifact: `benchmark_results/exp056_selective_full_dim_bf16_32768_h200_20260528.json`.
- bf16 dim 32768 regressed to 359.040 us.
- Decision: reject and keep the Experiment 053 sync-only change.

## Current Default Precision State After Experiment 057

Result:
- Full default precision-preserving artifacts:
  - `benchmark_results/current_default_precision_exp057_sync_skip_full_h200_20260528.json`: 121.522 us geometric mean, 1.247x over baseline.
  - `benchmark_results/current_default_precision_exp057_sync_skip_full_h200_20260528_repeat2.json`: 121.371 us geometric mean, 1.249x over baseline.
- Dtype geometric-mean speedups in the repeat artifact: fp16 1.346x, bf16 1.311x, fp32 1.104x.
- Correctness/unit verification: `uv run pytest -q tests/test_fast_hadamard_transform.py`, 55 passed.
- Default-path precision spot check versus fp32 reference over dims 4096 through 32768:
  - fp16 max abs: 0.000976 to 0.001750.
  - bf16 max abs: 0.007806 to 0.015549.
  - fp32 max abs: 0.000000775 to 0.000004053.
- Decision: keep the selective sync skip. It does not change arithmetic precision, and native `half2`/`bfloat162` arithmetic remains behind `fast_low_precision=True`.
- The default precision-preserving path still does not meet the requested 50% aggregate speedup. The opt-in fast low-precision path remains the only measured route near 1.5x, at about 1.496x in the latest full repeat.

## Experiment 058: Vectorized Float-Intermediate I/O for Large 16-Bit Defaults

Hypothesis: fp16/bf16 default kernels may spend meaningful time unpacking input values to float and packing float results back to 16-bit scalars. Vectorized half2/bfloat162 conversion can reduce that boundary overhead while keeping all transform arithmetic in float.

Change:
- Added vectorized load/store conversion for both fp16 and bf16 `kNElts == 8` main kernels.
- The transform still stores intermediates as float and uses float add/sub math.

Result:
- Artifact: `benchmark_results/exp058_vectorized_float_io_default_large_h200_20260528.json`.
- fp16 large cases improved or stayed close; fp16 dim 32768 moved to 308.624 us.
- bf16 dim 32768 regressed badly to 365.344 us.
- Correctness max-abs values matched the prior float-intermediate benchmark samples.
- Decision: reject the broad bf16 route and narrow the vectorized conversion path.

## Experiment 059: fp16-Only Vectorized Main I/O

Hypothesis: the vectorized conversion path is useful for fp16 main kernels while bf16 large kernels need more selective handling.

Change:
- Kept vectorized conversion for fp16 `kNElts == 8` main kernels.
- Restored bf16 main kernels to the scalar conversion path.

Result:
- Artifact: `benchmark_results/exp059_fp16_vectorized_float_io_default_large_h200_20260528.json`.
- fp16 dim 32768 improved to 308.288 us; fp16 dims 4096 and 16384 also improved modestly.
- bf16 dim 32768 returned to the expected range at 343.104 us.
- Correctness max-abs values were unchanged versus Experiment 057 for the sampled cases.
- Decision: accept the fp16 main-kernel vectorized conversion path.

## Experiment 060: fp16 Vectorized One-Warp I/O

Hypothesis: the same fp16 conversion overhead matters in one-warp default kernels, especially dim 2048 where the current default path is still visibly above clone time.

Change:
- Added vectorized fp16 conversion for one-warp default kernels.

Result:
- Artifact: `benchmark_results/exp060_fp16_vectorized_float_io_all_default_h200_20260528.json`.
- fp16 dim 2048 improved from 145.216 us in the prior repeat to 134.624 us.
- fp16 dims 256, 512, 1024, 4096, 16384, and 32768 also improved; fp16 dim 8192 was effectively neutral.
- Correctness max-abs values were unchanged versus Experiment 057.
- Decision: accept.

## Experiment 061: Direct fp32 Float I/O

Hypothesis: fp32 kernels do not need scalar copy-through staging at load time because the input is already float. Direct vector load/store into the float register tile can reduce instruction overhead.

Change:
- Added direct vectorized fp32 load/store for main and one-warp float-intermediate kernels.

Result:
- Artifact: `benchmark_results/exp061_fp32_direct_float_io_h200_20260528.json`.
- fp32 dims 4096, 8192, and 16384 improved to 141.856 us, 142.752 us, and 149.456 us.
- fp32 dim 32768 regressed to 198.528 us.
- Decision: narrow the fp32 direct-I/O path away from dim 32768.

## Experiment 062: Direct fp32 I/O Except 32768

Hypothesis: keeping the generic fp32 I/O path for dim 32768 preserves that case while retaining direct-I/O gains for smaller fp32 dimensions.

Change:
- Excluded fp32 dim 32768 from the direct main-kernel I/O specialization.

Result:
- Artifact: `benchmark_results/exp062_fp32_direct_float_io_except_32768_h200_20260528.json`.
- fp32 dim 32768 returned to 196.816 us.
- fp32 dims 4096, 8192, and 16384 remained improved at 142.176 us, 143.040 us, and 149.744 us.
- Decision: accept the selective fp32 direct-I/O path.

## Experiments 063-065: fp32 32768 Launch and Sync Retests

Hypothesis: after direct I/O, fp32 dim 32768 or fp32 dim 4096 may respond differently to prior rejected launch and synchronization variants.

Change:
- Tested fp32 dim 32768 with 1024 threads.
- Tested fp32 dim 32768 with 256 threads.
- Tested an fp32 dim 4096 initial pre-exchange sync skip.

Result:
- Artifacts:
  - `benchmark_results/exp063_fp32_32768_1024_threads_h200_20260528.json`.
  - `benchmark_results/exp064_fp32_32768_256_threads_h200_20260528.json`.
  - `benchmark_results/exp065_fp32_4096_sync_skip_h200_20260528.json`.
- fp32 dim 32768 regressed to 198.864 us with 1024 threads and 245.920 us with 256 threads.
- fp32 dim 4096 regressed to 150.240 us with the sync skip.
- Decision: reject all three; keep fp32 dim 32768 at 512 threads and keep the fp32 initial barrier.

## Experiment 066: bf16 Vectorized One-Warp I/O

Hypothesis: bf16 one-warp kernels may benefit from vectorized bfloat162 conversion even though the broad bf16 main-kernel route regressed at dim 32768.

Change:
- Added vectorized bf16 conversion for one-warp default kernels only.

Result:
- Artifact: `benchmark_results/exp066_bf16_one_warp_vectorized_float_io_h200_20260528.json`.
- bf16 dim 2048 improved from 156.192 us in the prior repeat to 136.784 us.
- bf16 dims 256, 512, and 1024 also improved modestly.
- Correctness max-abs values were unchanged versus Experiment 057.
- Decision: accept.

## Experiment 067: Selective bf16 Main Vectorized I/O

Hypothesis: bf16 main-kernel vectorized conversion should only be used where Experiment 058 showed benefit, avoiding dims 8192 and 32768.

Change:
- Enabled vectorized bf16 main-kernel conversion only for dims 4096 and 16384.
- Kept bf16 dims 8192 and 32768 on generic scalar conversion.

Result:
- Artifact: `benchmark_results/exp067_bf16_selective_main_vectorized_float_io_h200_20260528.json`.
- bf16 dim 16384 improved to 247.504 us; bf16 dim 4096 was effectively neutral at 233.168 us.
- bf16 dim 32768 stayed in the expected range at 343.344 us.
- Decision: accept the selective route.

## Current Default Precision State After Experiment 068

Result:
- Full default precision-preserving artifacts:
  - `benchmark_results/current_default_precision_exp068_vectorized_io_full_h200_20260528.json`: 118.770 us geometric mean, 1.276x over baseline.
  - `benchmark_results/current_default_precision_exp068_vectorized_io_full_h200_20260528_repeat2.json`: 118.921 us geometric mean, 1.275x over baseline.
- Dtype geometric-mean speedups in the repeat artifact: fp16 1.374x, bf16 1.345x, fp32 1.121x.
- Correctness/unit verification after final rebuild: `uv run --no-project pytest -q tests/test_fast_hadamard_transform.py`, 55 passed.
- Default-path precision spot check versus fp32 reference over dims 256 through 32768:
  - fp16 max abs: 0.000968 to 0.001886.
  - bf16 max abs: 0.007407 to 0.015617.
  - fp32 max abs: 0.000000715 to 0.000003278.
- Decision: keep the vectorized I/O specializations. They change conversion packing only and preserve float-intermediate arithmetic.
- The default precision-preserving path is now about 27.5% faster than the original baseline by geometric mean, still well short of the requested 50% default target.

## Experiments 069-070: Conditional Add/Sub Retests

Hypothesis: after vectorized I/O, conditional add/sub may become useful for more fp32 large routes or one-warp default kernels.

Change:
- Tested conditional add/sub for fp32 dims 4096, 8192, and 32768.
- Tested conditional add/sub in the one-warp default kernel.

Result:
- Artifacts:
  - `benchmark_results/exp069_fp32_large_conditional_addsub_with_direct_io_h200_20260528.json`.
  - `benchmark_results/exp070_one_warp_conditional_addsub_h200_20260528.json`.
- fp32 dim 32768 regressed to 204.192 us, and fp32 dim 4096 regressed to 144.800 us.
- One-warp conditional add/sub regressed the important fp16/bf16 dim 2048 cases to 144.336 us and 144.608 us.
- Decision: reject both retests. Keep conditional add/sub only on the previously accepted fp32 dim 16384 and bf16 dim 32768 routes.
