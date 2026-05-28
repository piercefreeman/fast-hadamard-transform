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

## Experiment 071: Double-Buffered Exchange for Large 16-Bit Defaults

Hypothesis: in one-round shared-memory exchanges, the post-warp exchange can write to a fresh shared-memory buffer and skip its initial `__syncthreads()`. This preserves float arithmetic and only changes synchronization and shared-memory allocation.

Change:
- Added an optional second exchange buffer.
- Routed default fp16/bf16 dims 4096, 8192, and 16384 through the double-buffered post-exchange.

Result:
- Artifact: `benchmark_results/exp071_double_buffer_exchange_default_large_h200_20260528.json`.
- Dim 4096 improved modestly: fp16 232.064 us, bf16 231.472 us.
- Dim 8192 was neutral to worse, and dim 16384 regressed badly because doubling 64 KiB exchange storage to 128 KiB reduced occupancy.
- Decision: reject the broad route and narrow to dim 4096 only.

## Experiment 072: Double-Buffered Exchange Only for 4096

Hypothesis: dim 4096 is small enough that the extra 16 KiB post-exchange buffer does not meaningfully hurt occupancy, while still saving one CTA barrier.

Change:
- Kept double-buffered post-exchange only for default fp16/bf16 dim 4096.
- Restored dims 8192 and 16384 to the single-buffer route.

Result:
- Artifact: `benchmark_results/exp072_double_buffer_exchange_default_4096_only_h200_20260528.json`.
- fp16 dim 4096 measured 231.888 us and bf16 dim 4096 measured 231.424 us.
- Guardrail dims 8192 and 16384 returned near the prior accepted range.
- Decision: accept the 4096-only route.

## Experiment 073: fp32 Double-Buffered Exchange

Hypothesis: fp32 large kernels pay one more exchange barrier than the 16-bit paths, so double-buffered post-exchange may help fp32 dims 4096 through 16384.

Change:
- Temporarily enabled double-buffered post-exchange for fp32 dims 4096, 8192, and 16384.

Result:
- Artifact: `benchmark_results/exp073_fp32_double_buffer_exchange_large_h200_20260528.json`.
- fp32 dim 4096 regressed to 160.736 us and fp32 dim 16384 regressed to 210.464 us.
- fp32 dim 8192 was effectively neutral.
- Decision: reject. Keep fp32 on the single-buffer route.

## Experiment 074: Exact-Dimension Vectorized I/O

Hypothesis: benchmarked power-of-two cases have `dim == Ktraits::N`, so specialized vector I/O can skip boundary checks and zero initialization. This must not apply to padded non-power dimensions.

Change:
- Temporarily removed boundary checks from specialized vector I/O helpers to measure the upside on exact power-of-two cases.

Result:
- Artifact: `benchmark_results/exp074_exact_dim_vectorized_io_h200_20260528.json`.
- Follow-up safe-routing artifact: `benchmark_results/current_default_precision_exp076_double_buffer_exact_io_full_h200_20260528.json`, 118.640 us geometric mean and 1.278x over baseline.
- Some small fp16/bf16 cases improved, and fp16 dim 32768 improved to 306.896 us.
- Several larger and fp32 cases regressed, including fp32 dim 8192 at 159.152 us and bf16 dim 16384 at 259.856 us.
- Decision: reject the exact-I/O route. Even with safe runtime routing for selected exact dimensions, the full sweep did not beat the simpler 4096 double-buffer route.

## Experiment 075: 128-Thread 4096 Double-Buffer Launch

Hypothesis: dim 4096 may benefit from fewer warps per block when paired with the double-buffered exchange.

Change:
- Temporarily changed default fp16/bf16 dim 4096 from 256 threads to 128 threads, keeping double-buffered post-exchange.

Result:
- Artifact: `benchmark_results/exp075_default_4096_128_threads_double_buffer_h200_20260528.json`.
- fp16 dim 4096 regressed to 252.464 us and bf16 dim 4096 to 249.360 us.
- Decision: reject and restore the 256-thread dim 4096 route.

## Current Default Precision State After Experiment 076

Result:
- Full default precision-preserving artifacts:
  - `benchmark_results/current_default_precision_exp076_double_buffer_4096_full_h200_20260528.json`: 118.558 us geometric mean, 1.279x over baseline.
  - `benchmark_results/current_default_precision_exp076_double_buffer_4096_full_h200_20260528_repeat2.json`: 118.753 us geometric mean, 1.276x over baseline.
- Dtype geometric-mean speedups in the repeat artifact: fp16 1.376x, bf16 1.348x, fp32 1.122x.
- Correctness/unit verification after final rebuild: `uv run --no-project pytest -q tests/test_fast_hadamard_transform.py`, 55 passed.
- Decision: keep the 4096-only double-buffered exchange. It is synchronization-only and does not change arithmetic precision.
- The default precision-preserving path remains far short of the requested 50% aggregate speedup.

## Experiment 077: Retest 512-Thread Default Launches for 8192 and 16384

Hypothesis: the earlier 512-thread default launch rejection predates vectorized float-intermediate I/O, so dims 8192 and 16384 may now benefit from fewer chunks per thread.

Change:
- Temporarily routed default fp16/bf16 dims 8192 and 16384 through 512-thread launches.

Result:
- Artifact: `benchmark_results/exp077_default_8192_16384_512_threads_h200_20260528.json`.
- fp16 dim 8192 regressed to 254.720 us and fp16 dim 16384 to 261.056 us.
- bf16 dim 8192 regressed to 256.672 us and bf16 dim 16384 to 259.872 us.
- Decision: reject and restore the 256-thread launches.

## Experiment 078: fp16 32768 Exact I/O

Hypothesis: the dim 32768 default fp16 route always has `params.dim == Ktraits::N` in the benchmarked power-of-two case, so a guarded exact-I/O variant can remove boundary checks and zero initialization while preserving float arithmetic. Padded non-power inputs must stay on the guarded route.

Change:
- Added exact fp16 main-kernel load/store helpers.
- Routed default fp16 dim 32768 through the exact helpers only when `params.dim == 32768`.

Result:
- Artifact: `benchmark_results/exp078_fp16_32768_exact_io_h200_20260528.json`.
- fp16 dim 32768 improved to 307.024 us; bf16 dim 32768 remained on the guarded route at 343.056 us.
- Correctness max-abs versus fp32 reference matched the float-intermediate default range.
- Decision: accept.

## Experiment 079: Full Sweep With fp16 32768 Exact I/O

Result:
- Full artifact: `benchmark_results/current_default_precision_exp079_fp16_32768_exact_io_full_h200_20260528.json`.
- Full default precision-preserving geometric mean: 118.505 us, 1.279x over baseline.
- Decision: keep evaluating exact-I/O opportunities, but only with exact-dimension runtime guards.

## Experiment 080: bf16 2048 Exact One-Warp I/O

Hypothesis: bf16 dim 2048 showed an exact-I/O win in the earlier broad exact-dim test. A one-warp exact helper guarded by `params.dim == 2048` may capture that win without affecting padded inputs.

Change:
- Added exact bf16 one-warp load/store helpers.
- Routed default bf16 dim 2048 through them only when `params.dim == 2048`.

Result:
- Artifact: `benchmark_results/exp080_bf16_2048_and_fp16_32768_exact_io_h200_20260528.json`.
- bf16 dim 2048 improved to 136.080 us in the targeted run.
- fp16 dim 32768 stayed improved at 306.560 us.
- Decision: accept pending full-sweep repeat.

## Current Default Precision State After Experiment 081

Result:
- Full default precision-preserving artifacts:
  - `benchmark_results/current_default_precision_exp081_exact_io_selective_full_h200_20260528.json`: 118.538 us geometric mean, 1.279x over baseline.
  - `benchmark_results/current_default_precision_exp081_exact_io_selective_full_h200_20260528_repeat2.json`: 118.483 us geometric mean, 1.279x over baseline.
- Correctness/unit verification after final rebuild: `uv run --no-project pytest -q tests/test_fast_hadamard_transform.py`, 55 passed.
- Targeted precision spot check versus fp32 reference:
  - fp16 dim 2048 max abs 0.001915 and dim 32768 max abs 0.001814.
  - bf16 dim 2048 max abs 0.009661 and dim 32768 max abs 0.015195.
  - fp32 dim 2048 max abs 0.000000715 and dim 32768 max abs 0.000003099.
- Decision: keep the exact-I/O routes for fp16 dim 32768 and bf16 dim 2048. They remove boundary work only for exact power-of-two dimensions and preserve the same float-intermediate arithmetic.
- The default precision-preserving path remains about 27.9% faster than baseline, still well short of the requested 50% aggregate target.

## Experiment 082: Two-Min-Block Launch Bounds for 8192 and 16384

Hypothesis: default fp16/bf16 dims 8192 and 16384 may be occupancy-limited after the vectorized I/O changes, so adding `__launch_bounds__(threads, 2)` to those main-kernel routes may improve resident blocks without changing arithmetic.

Change:
- Temporarily routed default fp16/bf16 dims 8192 and 16384 through an otherwise identical main kernel annotated with `__launch_bounds__(Ktraits::kNThreads, 2)`.

Result:
- Artifact: `benchmark_results/exp082_default_8192_16384_min_blocks2_h200_20260528.json`.
- fp16 dim 8192 regressed to 244.128 us and fp16 dim 16384 regressed to 269.104 us.
- bf16 dim 8192 regressed to 241.728 us and bf16 dim 16384 regressed to 271.088 us.
- Decision: reject and restore the standard launches.

## Experiment 083: Main-Kernel Shared-Memory Carveout

Hypothesis: the remaining large default kernels are exchange-heavy enough that preferring maximum shared-memory carveout may improve the fp16/bf16 and fp32 main-kernel routes without changing arithmetic.

Change:
- Temporarily set `cudaFuncAttributePreferredSharedMemoryCarveout` to `cudaSharedmemCarveoutMaxShared` for main-kernel launches using at least 16 KiB of dynamic shared memory.

Result:
- Artifact: `benchmark_results/exp083_main_kernel_smem_carveout_h200_20260528.json`.
- The targeted large cases were neutral to slightly slower. Examples: fp16 dim 8192 was 237.296 us, bf16 dim 32768 was 342.368 us, fp32 dim 16384 was 148.336 us, and fp32 dim 32768 was 195.888 us.
- Decision: reject and restore the prior launch attributes.

## Current Fast-Low-Precision State After Experiment 083

Result:
- Full opt-in fast-low-precision artifact: `benchmark_results/current_fast_low_precision_after_exp083_restore_full_h200_20260528.json`.
- Geometric-mean speedup versus `benchmark_results/baseline_unmodified_h200_20260528.json`: 1.504x across all 24 benchmark cases.
- Dtype geometric-mean speedups: fp16 1.739x, bf16 1.738x, fp32 1.125x.
- This meets the aggregate 50% target only for the opt-in lower-precision arithmetic route. The default precision-preserving path remains around 1.279x.

## Experiment 084: Integer Log2 Parameter Setup

Hypothesis: the event-based benchmark may include CPU enqueue/setup delay before the CUDA kernel is submitted, so replacing the per-call floating-point `ceil(log2(...))` setup with an integer helper may improve all dtypes in small dimensions without changing GPU arithmetic.

Change:
- Temporarily replaced `int(ceil(std::log2(dim / multiple)))` with an integer ceil-log2 loop in `set_hadamard_params`.

Result:
- Artifact: `benchmark_results/exp084_integer_log2_setup_small_h200_20260528.json`.
- The small-dimension target was mixed and not a real improvement: fp16/bf16 were neutral to slightly slower, while fp32 moved only at noise scale.
- Decision: reject and restore the previous setup expression.

## Experiment 085: bf16 32768 Pre-Exchange Sync Skip Retest

Hypothesis: later default-kernel changes may have altered the earlier bf16 dim 32768 result, making it worth retesting the precision-preserving initial pre-exchange barrier skip for that route.

Change:
- Temporarily enabled the initial pre-exchange sync skip for all `kNElts == 8` main kernels, including bf16 dim 32768.

Result:
- Artifact: `benchmark_results/exp085_bf16_32768_pre_sync_skip_retest_h200_20260528.json`.
- bf16 dim 32768 regressed to 344.240 us; fp16 dim 32768 was neutral at 306.880 us.
- Decision: reject and keep the existing bf16 dim 32768 exception.

## Experiment 086: bf16 32768 Exact Main-Kernel I/O

Hypothesis: the accepted fp16 dim 32768 exact-I/O route may have a bf16 analog. Guarding bf16 dim 32768 on `params.dim == 32768` can remove boundary checks and zero initialization while still converting inputs to float before all Hadamard arithmetic.

Change:
- Temporarily added exact bf16 main-kernel load/store helpers.
- Routed default bf16 dim 32768 through those helpers only for exact 32768-wide rows.

Result:
- Artifact: `benchmark_results/exp086_bf16_32768_exact_main_io_h200_20260528.json`.
- bf16 dim 32768 regressed to 362.016 us. fp16 dim 32768 guardrail was neutral at 307.152 us.
- Decision: reject and keep the generic guarded bf16 dim 32768 I/O path.

## Experiment 087: Large 16-Bit Conditional Add/Sub Retest

Hypothesis: after vectorized float-intermediate I/O and selective exact I/O, the large fp16/bf16 default routes may benefit from the conditional add/sub warp helper that avoids the sign-multiply form.

Change:
- Temporarily routed default fp16/bf16 dims 8192 and 16384 through conditional add/sub.
- Temporarily routed default fp16 dim 32768 through conditional add/sub while preserving the exact-I/O guard. bf16 dim 32768 was already on the conditional route.

Result:
- Artifact: `benchmark_results/exp087_large_16bit_conditional_addsub_retest_h200_20260528.json`.
- Regressed every changed fp16/bf16 target: fp16 dim 8192 241.600 us, fp16 dim 16384 251.024 us, fp16 dim 32768 312.784 us, bf16 dim 8192 239.584 us, and bf16 dim 16384 255.040 us.
- Decision: reject and restore the prior selective conditional routing.

## Experiment 088: Direct Power-of-Two Chunk Stage

Hypothesis: the final per-thread chunk Hadamard for power-of-two chunk counts may spend avoidable register moves transposing `x_vals` into `x_vals_transposed` and back. Applying the same chunk butterflies directly in `x_vals` should preserve float arithmetic while reducing register-copy work.

Change:
- Temporarily added a direct power-of-two chunk-stage helper.
- Routed the main float-intermediate kernel and low-precision exchange kernel through the direct helper for power-of-two chunk counts.

Result:
- Artifact: `benchmark_results/exp088_direct_power2_chunk_stage_h200_20260528.json`.
- Geometric mean over the 18 common targeted cases was 0.9996x versus the accepted Experiment 081 route.
- Isolated small wins, such as fp32 dim 8192 at 140.576 us, did not offset regressions like bf16 dim 2048 at 135.984 us and fp32 dim 16384 at 148.608 us.
- Decision: reject and restore the transpose-based chunk stage.

## Experiment 089: bf16 32768 Vectorized Load Only

Hypothesis: broad vectorized bf16 main-kernel I/O regressed dim 32768, but the load side and store side may have different codegen effects. Using vectorized bfloat162 load conversion while keeping the generic scalar store may isolate a useful half of the change.

Change:
- Temporarily routed default bf16 dim 32768 through `FloatIntermediateIO` vectorized load conversion only.
- Kept all Hadamard arithmetic in float and kept the generic output path.

Result:
- Artifact: `benchmark_results/exp089_bf16_32768_vector_load_only_h200_20260528.json`.
- bf16 dim 32768 regressed to 360.336 us. fp16 dim 32768 guardrail was neutral at 306.880 us.
- Decision: reject and restore the generic bf16 load path.

## Experiment 090: bf16 32768 Vectorized Store Only

Hypothesis: vectorized bfloat162 output packing may still be useful if the bf16 dim 32768 regression is caused mainly by the vectorized load side.

Change:
- Temporarily routed default bf16 dim 32768 through `FloatIntermediateIO` vectorized store conversion only.
- Kept the generic scalar input path and float Hadamard arithmetic.

Result:
- Artifact: `benchmark_results/exp090_bf16_32768_vector_store_only_h200_20260528.json`.
- bf16 dim 32768 regressed to 349.568 us. fp16 dim 32768 guardrail was neutral at 306.656 us.
- Decision: reject and keep the generic guarded bf16 dim 32768 I/O path.

## Experiment 091: Restrict Row Pointers

Hypothesis: the CUDA kernels always write into a freshly allocated output tensor, so row input and output pointers do not alias. Marking row pointers `__restrict__` may help ptxas schedule load/store-heavy float-intermediate kernels without changing arithmetic.

Change:
- Temporarily marked row-local input and output pointers as `__restrict__` across the kernel family.

Result:
- Artifacts:
  - `benchmark_results/exp091_restrict_row_pointers_h200_20260528.json`.
  - `benchmark_results/exp091_restrict_row_pointers_bf16_32768_repeat_h200_20260528.json`.
- Global restrict was unusable: fp32 dim 4096 regressed to 150.432 us, fp32 dim 16384 to 171.312 us, and fp32 dim 32768 to 209.296 us.
- The bf16 dim 32768 case improved repeatably, measuring 334.352 us and 333.824 us in targeted runs.
- Decision: reject global restrict but isolate the bf16 dim 32768 signal in a dedicated route.

## Experiment 092: Specialized bf16 32768 Restrict Route

Hypothesis: the beneficial part of Experiment 091 is specific to the default bf16 dim 32768 kernel. A dedicated exact-dimension route can preserve the existing generic bf16 load/store and float arithmetic while applying `__restrict__` only to that case.

Change:
- Added a specialized default bf16 dim 32768 kernel with restricted row pointers.
- Kept the same generic bf16 input/output conversion, float Hadamard arithmetic, conditional cross-warp add/sub, 128 KiB exchange tile, and bf16 dim 32768 pre-exchange barrier behavior.
- Routed only exact `params.dim == 32768` bf16 default calls through the specialized kernel.

Result:
- Targeted artifact: `benchmark_results/exp092_bf16_32768_restrict_specialized_h200_20260528.json`.
- Full default artifact: `benchmark_results/current_default_precision_exp092_bf16_32768_restrict_full_h200_20260528.json`.
- bf16 dim 32768 improved from 342.304 us in the accepted Experiment 081 repeat to 334.896 us targeted and 331.024 us in the full sweep.
- Full default geometric speedup remained effectively unchanged because this affects one case: 1.279x versus 1.279x before, still far short of 1.5x.
- Decision: accept as a narrow precision-preserving compounding win.

## Experiment 093: bf16 32768 Restrict Route Sync Skip

Hypothesis: after specializing bf16 dim 32768 with restricted row pointers, the earlier pre-exchange sync-skip rejection may change because ptxas emits a different kernel. Skipping the initial pre-exchange barrier remains synchronization-only and should not affect arithmetic precision.

Change:
- Temporarily skipped the initial pre-exchange barrier inside the specialized default bf16 dim 32768 restrict kernel.

Result:
- Artifact: `benchmark_results/exp093_bf16_32768_restrict_sync_skip_h200_20260528.json`.
- bf16 dim 32768 regressed to 344.928 us. fp16 and fp32 guardrails were unaffected because they do not use the specialized route.
- Decision: reject and keep the initial pre-exchange barrier in the specialized bf16 dim 32768 route.

## Experiment 094: fp16 32768 Exact Specialized Kernel

Hypothesis: the accepted fp16 dim 32768 exact-I/O route still goes through the generic main-kernel body. A dedicated exact fp16 dim 32768 kernel may remove compile-time branches and give ptxas the same scheduling freedom that helped the bf16 32768 restrict route.

Change:
- Temporarily added a specialized default fp16 dim 32768 exact-I/O kernel.
- Kept float Hadamard arithmetic, exact fp16 boundary conversion, 1024-thread launch shape, 128 KiB exchange tile, and existing synchronization behavior.

Result:
- Artifact: `benchmark_results/exp094_fp16_32768_exact_specialized_h200_20260528.json`.
- fp16 dim 32768 was neutral to slower at 307.120 us. bf16 and fp32 guardrails stayed near expected ranges.
- Decision: reject and keep the generic exact-I/O route for fp16 dim 32768.

## Experiment 095: 32 KiB Exchange Tile for 16-Bit Dim 16384

Hypothesis: default fp16/bf16 dim 16384 may be occupancy-limited by the 64 KiB float exchange tile. Reducing the 16-bit dim 16384 exchange tile to 32 KiB adds an exchange round but may increase resident blocks enough to improve throughput while preserving float arithmetic.

Change:
- Temporarily changed only 16-bit dim 16384 main-kernel exchange storage from 64 KiB to 32 KiB.

Result:
- Artifact: `benchmark_results/exp095_16bit_16384_32k_exchange_h200_20260528.json`.
- fp16 dim 16384 regressed to 248.320 us and bf16 dim 16384 regressed to 249.632 us.
- fp32 guardrails were not directly affected by the conditional change and stayed in the expected range.
- Decision: reject and keep the 64 KiB exchange tile for dim 16384.

## Experiment 096: 16 KiB Exchange Tile for 16-Bit Dim 8192

Hypothesis: default fp16/bf16 dim 8192 may benefit from improved block residency if its one-round 32 KiB exchange tile is reduced to a two-round 16 KiB tile.

Change:
- Temporarily changed only 16-bit dim 8192 main-kernel exchange storage from 32 KiB to 16 KiB.

Result:
- Artifact: `benchmark_results/exp096_16bit_8192_16k_exchange_h200_20260528.json`.
- fp16 dim 8192 regressed to 240.160 us and bf16 dim 8192 regressed to 239.600 us.
- Decision: reject and keep the one-round 32 KiB exchange tile for dim 8192.

## Experiment 097: 16-Bit Main-Kernel Restrict for 4096-16384

Hypothesis: global row-pointer `__restrict__` was bad for fp32 but showed small 16-bit movements. Restricting only default fp16/bf16 main-kernel routes for dims 4096 through 16384 may keep any aliasing/codegen benefit while avoiding fp32 regressions.

Change:
- Temporarily added an optional restrict-pointer instantiation for the generic main kernel.
- Routed default fp16/bf16 dims 4096, 8192, and 16384 through the restrict-pointer instantiation.

Result:
- Artifact: `benchmark_results/exp097_16bit_main_restrict_4096_16384_h200_20260528.json`.
- The common targeted geometric mean was 0.9984x versus the accepted Experiment 092 full artifact.
- fp16 dims 8192 and 16384 improved slightly, but bf16 dim 8192 regressed and 4096 was neutral.
- Decision: reject the broad 16-bit route and test only the positive fp16 subset.

## Experiment 098: fp16 8192/16384 Restrict Subset

Hypothesis: the only useful signal in Experiment 097 was fp16 dims 8192 and 16384. Restricting only those routes may preserve the small fp16 gain without affecting bf16.

Change:
- Temporarily routed only default fp16 dims 8192 and 16384 through the restrict-pointer instantiation.

Result:
- Artifact: `benchmark_results/exp098_fp16_8192_16384_selective_restrict_h200_20260528.json`.
- fp16 dim 8192 was neutral/slightly worse at 237.696 us and fp16 dim 16384 was only noise-scale better at 245.840 us.
- Guardrails drifted worse in the same targeted run.
- Decision: reject and restore the generic non-restrict routes.

## Experiment 099: NVCC Extra Device Vectorization

Hypothesis: enabling NVCC's extra device vectorization pass may improve instruction selection across the default precision-preserving kernels, including fp32, without changing arithmetic precision.

Change:
- Temporarily added `--extra-device-vectorization` to the CUDA compile flags.
- Rebuilt the extension and ran the full default benchmark suite with correctness checks.

Result:
- Artifact: `benchmark_results/exp099_extra_device_vectorization_full_h200_20260528.json`.
- Full-suite geometric speedup versus the original baseline regressed from 1.2789x in the accepted Experiment 092 artifact to 1.2741x.
- The experiment was mostly neutral but regressed visible cases including fp16 dim 512 (40.384 us to 42.368 us) and bf16 dim 32768 (331.024 us to 333.312 us).
- Decision: reject and restore the prior CUDA compile flags.

## Experiment 100: One-Warp Launch Bounds With Minimum Blocks

Hypothesis: the grouped one-warp default kernels for dimensions 256 through 2048 may be limited by resident blocks or register allocation. Adding `__launch_bounds__(threads, 2)` could improve occupancy without changing float-intermediate arithmetic.

Change:
- Temporarily changed the generic precision-preserving one-warp kernel launch bounds to request at least two resident blocks per SM.
- Benchmarked dimensions 256, 512, 1024, and 2048 for fp16, bf16, and fp32.

Result:
- Artifact: `benchmark_results/exp100_one_warp_launch_bounds2_h200_20260528.json`.
- fp16 dim 2048 regressed from 134.896 us to 177.872 us and bf16 dim 2048 regressed from 134.912 us to 181.696 us.
- fp16/bf16 dim 1024 also regressed to about 77 us, while fp32 was only neutral.
- Decision: reject and restore the original one-warp launch bounds.

## Experiment 101: fp32 32768 Exact I/O Route

Hypothesis: the current fp32 dim 32768 route avoids the direct fp32 I/O specialization because the broad direct path regressed that case. A runtime exact-dimension guard may remove boundary checks for the power-of-two case without harming padded inputs.

Change:
- Temporarily added exact fp32 load/store helpers for `kNElts == 4`.
- Routed only `params.dim == 32768` fp32 through the exact I/O instantiation.

Result:
- Artifact: `benchmark_results/exp101_fp32_32768_exact_io_h200_20260528.json`.
- fp32 dim 32768 regressed sharply to 221.280 us versus about 196 us in the accepted artifact.
- fp32 dim 8192 and 16384 guardrails were also slightly slower in the same run.
- Decision: reject and keep the generic fp32 dim 32768 I/O path.

## Experiment 102: Read-Only Loads for Vectorized 16-Bit I/O

Hypothesis: the vectorized fp16/bf16 float-intermediate input paths may benefit from read-only cache loads via `__ldg` while preserving the same float arithmetic and output conversion.

Change:
- Temporarily changed vectorized fp16 and bf16 main-kernel input loads to use `__ldg`.
- Benchmarked dimensions 4096 through 32768 for fp16, bf16, and fp32.

Result:
- Artifact: `benchmark_results/exp102_16bit_vector_load_ldg_h200_20260528.json`.
- Common-case geometric speed was 0.9994x versus the accepted Experiment 092 full artifact.
- bf16 dim 16384 improved to 245.392 us and fp16 dim 8192 improved to 236.624 us, but fp16 dim 16384 regressed to 248.416 us and fp16 dim 4096 also moved slower.
- Decision: reject the broad vectorized-load `__ldg` route.

## Experiment 103: Read-Only Loads for Generic I/O

Hypothesis: the generic guarded load helper feeds fp32 dim 32768 and the specialized bf16 dim 32768 restrict route. Using `__ldg` there may improve large read-only input loads while preserving arithmetic precision.

Change:
- Temporarily changed the generic block and one-warp load helpers to use `__ldg`.
- Benchmarked selected medium and large dimensions for all dtypes.

Result:
- Artifact: `benchmark_results/exp103_generic_load_ldg_h200_20260528.json`.
- bf16 dim 32768 improved materially to 324.880 us, but fp32 dim 32768 regressed sharply to 238.880 us.
- Other cases were mostly neutral or noise-scale, with bf16 dim 8192 also slower.
- Decision: reject the broad generic-load route and isolate the bf16 dim 32768 signal.

## Experiment 104: bf16 32768 Restrict Route Read-Only Load

Hypothesis: the useful part of Experiment 103 is specific to the specialized bf16 dim 32768 restrict kernel. Applying `__ldg` only there may keep the bf16 win without the fp32 32768 regression.

Change:
- Added a bf16 `__ldg` load helper that preserves the generic bf16-to-float conversion path.
- Used it only in the specialized default bf16 dim 32768 restrict kernel.

Result:
- Targeted artifact: `benchmark_results/exp104_bf16_32768_restrict_ldg_h200_20260528.json`.
- Full default artifact: `benchmark_results/current_default_precision_exp104_bf16_32768_ldg_full_h200_20260528.json`.
- bf16 dim 32768 improved from 331.024 us in the accepted Experiment 092 full artifact to 326.048 us in the full Exp104 sweep.
- Full default precision-preserving speedup improved from 1.2789x to 1.2812x versus the original baseline.
- Decision: accept as a narrow precision-preserving codegen/load-path win.

## Experiment 105: Exact Read-Only Load for bf16 32768

Hypothesis: the accepted bf16 dim 32768 read-only load route is only used for exact `params.dim == 32768`, so removing the remaining boundary check may improve codegen without changing precision.

Change:
- Temporarily replaced the guarded bf16 `__ldg` load helper in the specialized restrict kernel with an exact helper that skips the boundary check and zero initialization.

Result:
- Artifact: `benchmark_results/exp105_bf16_32768_exact_ldg_load_h200_20260528.json`.
- bf16 dim 32768 measured 325.856 us versus 326.064 us in the Exp104 targeted run, which is inside the run's 4.832 us IQR and not a meaningful improvement.
- fp16 and fp32 guardrails were normal.
- Decision: reject and keep the accepted guarded `__ldg` helper from Experiment 104.

## Experiment 106: fp16 32768 Exact Read-Only Load

Hypothesis: the accepted fp16 dim 32768 exact-I/O route may benefit from using `__ldg` for its read-only vector input load while keeping float intermediates and the same output conversion.

Change:
- Temporarily changed `load_input_half_exact` to use `__ldg` for the exact fp16 input vector load.

Result:
- Artifact: `benchmark_results/exp106_fp16_32768_exact_ldg_load_h200_20260528.json`.
- fp16 dim 32768 regressed to 311.808 us versus the accepted roughly 306-307 us range.
- bf16 and fp32 guardrails were normal.
- Decision: reject and restore the normal exact fp16 load.

## Experiment 107: bf16 32768 Restrict Route Shared-Memory Carveout

Hypothesis: the specialized bf16 dim 32768 route uses a 128 KiB exchange buffer, so preferring maximum shared-memory carveout may help this kernel even though the earlier broad carveout experiment regressed.

Change:
- Temporarily set `cudaFuncAttributePreferredSharedMemoryCarveout` to `cudaSharedmemCarveoutMaxShared` only for the specialized bf16 dim 32768 restrict kernel.

Result:
- Artifact: `benchmark_results/exp107_bf16_32768_restrict_carveout_h200_20260528.json`.
- bf16 dim 32768 regressed to 346.336 us versus the accepted 326 us range.
- fp16 and fp32 guardrails were normal because they do not use this launch path.
- Decision: reject and keep only the dynamic shared-memory size attribute.

## Experiment 108: No XOR Shared-Memory Exchange Swizzle

Hypothesis: H200 shared-memory behavior may prefer a simpler exchange layout than the current XOR swizzle. Removing the swizzle tests whether the existing layout is still the right bank-conflict tradeoff.

Change:
- Temporarily removed the `^ warp_id` and `^ row_t` shared-memory index swizzles in the default float exchange helper.
- Benchmarked dimensions 4096 through 32768 for fp16, bf16, and fp32.

Result:
- Artifact: `benchmark_results/exp108_exchange_no_xor_swizzle_h200_20260528.json`.
- The swizzle is critical: fp16 dim 4096 regressed to 455.456 us, bf16 dim 4096 to 456.160 us, and fp32 dim 8192 to 252.752 us.
- Large fp16/bf16 cases were roughly 2x slower across the board.
- Decision: reject and keep the XOR exchange swizzle.

## Experiment 109: Shifted XOR Shared-Memory Exchange Swizzle

Hypothesis: a different XOR mask may improve H200 shared-memory bank behavior while preserving the exchange permutation. `col ^ (row << 1)` keeps the mask within the 32-column row for the current 8- and 16-warp exchange shapes.

Change:
- Temporarily replaced the exchange swizzle from `col ^ row` to `col ^ (row << 1)` in the default float exchange helper.
- Benchmarked dimensions 4096 through 32768 for fp16, bf16, and fp32.

Result:
- Artifact: `benchmark_results/exp109_exchange_xor_row_shift1_h200_20260528.json`.
- The shifted mask was worse than the accepted swizzle: fp16 dim 4096 regressed to 262.240 us, fp16 dim 8192 to 268.736 us, and bf16 dim 32768 to 356.128 us.
- fp32 large cases also regressed.
- Decision: reject and restore the original `col ^ row` swizzle.

## Experiment 110: 1024-Thread 4-Element 16-Bit Launch Shape

Hypothesis: default fp16/bf16 dimensions 4096 through 16384 may be limited by per-thread register and instruction work. Using 1024 threads with 4 float intermediates per thread changes the on-chip tile shape while preserving float arithmetic and guarded I/O.

Change:
- Temporarily allowed the main kernel traits to override `kNElts`.
- Routed default fp16/bf16 dims 4096, 8192, and 16384 through 1024-thread, 4-element launches.
- Kept the existing 4096 double-buffered exchange behavior in the 1024-thread variant.

Result:
- Artifact: `benchmark_results/exp110_16bit_1024t_4elts_4096_16384_h200_20260528.json`.
- The new layout regressed every targeted 16-bit case: fp16 dim 4096 measured 303.920 us, fp16 dim 8192 272.080 us, and bf16 dim 16384 281.728 us.
- fp32 guardrails were unaffected because they did not use the experimental route.
- Decision: reject and restore the standard 256-thread, 8-element launch shape.

## Experiment 111: 512-Thread 4-Element 16-Bit Launch Shape

Hypothesis: the 1024-thread 4-element shape may have been too wide, but a 512-thread 4-element shape could still reduce per-thread work versus the accepted 256-thread 8-element route without excessive warp count.

Change:
- Temporarily routed default fp16/bf16 dims 4096, 8192, and 16384 through 512-thread, 4-element launches.
- Kept the existing 4096 double-buffered exchange behavior in the experimental route.

Result:
- Artifact: `benchmark_results/exp111_16bit_512t_4elts_4096_16384_h200_20260528.json`.
- The intermediate shape was still slower than accepted: fp16 dim 4096 measured 250.384 us, fp16 dim 8192 257.104 us, and bf16 dim 16384 289.696 us.
- fp32 guardrails were unaffected.
- Decision: reject and restore the standard 256-thread, 8-element launch shape.

## Experiment 112: Broad 16-Bit Even-Dimension Pre-Scale

Hypothesis: for even power-of-two dimensions, the benchmark normalization scale is an exact power of two. Applying that scale after load and storing with scale 1 may be rounding-equivalent while moving work away from the output conversion path.

Change:
- Temporarily added a `kPreScale` main-kernel instantiation.
- Routed default fp16/bf16 dims 4096 and 16384 through the pre-scale variant.

Result:
- Artifact: `benchmark_results/exp112_16bit_even_dim_prescale_h200_20260528.json`.
- bf16 dim 16384 improved to 241.920 us, but fp16 dim 16384 regressed slightly to 246.688 us and dims 4096 were neutral.
- Decision: reject the broad route and narrow the useful signal to bf16 dim 16384.

## Experiment 113: Guarded bf16 16384 Pre-Scale

Hypothesis: the useful pre-scale signal is specific to bf16 dim 16384. Guarding it on exact `params.dim == 16384` and exact power-of-two scale `1/128` should preserve general API behavior while capturing the measured win.

Change:
- Kept the `kPreScale` kernel instantiation.
- Routed only default bf16 dim 16384 with `params.scale == 0.0078125f` through the pre-scale variant.
- All other dtypes, dimensions, and scale values stay on the accepted post-scale route.

Result:
- Targeted artifact: `benchmark_results/exp113_bf16_16384_prescale_guarded_h200_20260528.json`.
- Full default artifact: `benchmark_results/current_default_precision_exp113_bf16_16384_prescale_full_h200_20260528.json`.
- bf16 dim 16384 improved from 247.584 us in the accepted Exp104 full artifact to 242.112 us in the Exp113 full sweep.
- Full default precision-preserving speedup improved from 1.2812x to 1.2844x versus the original baseline.
- Precision check: for dim 16384, PyTorch reference post-scale and pre-scale forms were bitwise identical after bf16 rounding; kernel max abs versus the fp32 reference stayed in the established bf16 range.
- Decision: accept as a narrow precision-preserving scale-placement win.

## Experiment 114: fp32 Exact-Power Pre-Scale

Hypothesis: the exact-power pre-scale trick accepted for bf16 dim 16384 may also help fp32 dimensions 4096 and 16384, where the benchmark normalization scale is exactly representable.

Change:
- Temporarily routed fp32 dim 4096 with scale `1/64` and fp32 dim 16384 with scale `1/128` through the pre-scale main-kernel instantiation.

Result:
- Artifact: `benchmark_results/exp114_fp32_4096_16384_prescale_h200_20260528.json`.
- fp32 dim 4096 regressed to 147.024 us and fp32 dim 16384 regressed to 158.672 us.
- fp16/bf16 guardrails were normal.
- Decision: reject and keep fp32 on the accepted post-scale route.

## Experiment 115: bf16 16384 Pre-Scale With Conditional Add/Sub

Hypothesis: the newly accepted bf16 dim 16384 pre-scale route changes codegen enough that the previously rejected conditional add/sub helper may become useful for that specific kernel.

Change:
- Temporarily changed only the guarded bf16 dim 16384 pre-scale route to use conditional add/sub warp Hadamard.

Result:
- Artifact: `benchmark_results/exp115_bf16_16384_prescale_conditional_h200_20260528.json`.
- bf16 dim 16384 regressed to 246.208 us, losing the accepted pre-scale win.
- fp16 and fp32 guardrails were normal because they do not use this route.
- Decision: reject and keep the non-conditional pre-scale route.

## Experiment 116: 8192 Double-Buffered Exchange Retest

Hypothesis: after later vectorized I/O and pre-scale changes, default fp16/bf16 dim 8192 may respond differently to double-buffered post-exchange than it did in Experiment 071.

Change:
- Temporarily routed only default fp16/bf16 dim 8192 through the double-buffered post-exchange variant.

Result:
- Artifact: `benchmark_results/exp116_16bit_8192_double_buffer_retest_h200_20260528.json`.
- fp16 dim 8192 regressed to 242.640 us and bf16 dim 8192 regressed to 237.952 us.
- Neighboring dims and fp32 guardrails were normal.
- Decision: reject and keep dim 8192 on the accepted single-buffer route.

## Experiment 117: bf16 Vectorized Read-Only Load Retest

Hypothesis: Experiment 102 showed a bf16 dim 16384 improvement from read-only vector loads, but that broad test also changed fp16. Retesting only the bf16 vectorized load path under the current pre-scale route may isolate a useful load-path win.

Change:
- Temporarily changed only `FloatIntermediateIO<..., at::BFloat16>` vectorized main-kernel loads to use `__ldg`.
- This affects default bf16 dims 4096 and 16384, while bf16 8192 and 32768 stay on their guarded/specialized routes.

Result:
- Artifact: `benchmark_results/exp117_bf16_vector_load_ldg_current_h200_20260528.json`.
- bf16 dim 16384 regressed to 243.744 us versus the accepted 242.112 us full-sweep value.
- bf16 dim 4096 was only noise-scale better at 231.072 us.
- Decision: reject and keep the normal vectorized bf16 load path outside the specialized 32768 route.

## Experiment 118: Selective Direct Chunk Stage

Hypothesis: Experiment 088 had small isolated wins from avoiding the transpose-based final chunk stage, especially fp32 dim 8192 and fp16 dim 4096. Retesting only those routes under the current code may capture the useful part without broad regressions.

Change:
- Temporarily added an optional direct power-of-two chunk-stage helper.
- Routed only default fp16 dim 4096 and fp32 dim 8192 through it.

Result:
- Artifact: `benchmark_results/exp118_selective_direct_chunk_stage_h200_20260528.json`.
- fp32 dim 8192 measured 141.728 us, slower than the accepted full-sweep value of 141.376 us.
- fp16 dim 4096 measured 231.664 us, a noise-scale movement relative to the accepted 231.776 us.
- Decision: reject and keep the transpose-based chunk stage.

## Experiment 119: fp32 Large Launch-Width Retest

Hypothesis: after direct fp32 I/O and selective conditional add/sub, fp32 dim 8192 or 16384 may respond differently to launch-width changes that were rejected earlier.

Change:
- Temporarily routed fp32 dim 8192 through 128 threads.
- Temporarily routed fp32 dim 16384 through 512 threads while keeping conditional add/sub.

Result:
- Artifact: `benchmark_results/exp119_fp32_8192_128t_16384_512t_h200_20260528.json`.
- fp32 dim 8192 regressed to 159.872 us and fp32 dim 16384 regressed to 161.824 us.
- fp16/bf16 guardrails were normal.
- Decision: reject and keep the accepted fp32 launch widths.

## Experiment 120: Preferred-L1 Carveout for Large 16-Bit Defaults

Hypothesis: the max-shared carveout was rejected, but the opposite preference may help if default fp16/bf16 large kernels are sensitive to global input/output cache behavior. Preferring L1 is a launch attribute only and does not change arithmetic.

Change:
- Temporarily set `cudaFuncAttributePreferredSharedMemoryCarveout` to `cudaSharedmemCarveoutMaxL1` for default fp16/bf16 dims 4096, 8192, and 16384.

Result:
- Artifact: `benchmark_results/exp120_16bit_large_prefer_l1_carveout_h200_20260528.json`.
- The L1 preference badly regressed the shared-memory exchange path: fp16 dim 4096 measured 513.424 us, fp16 dim 8192 402.688 us, and bf16 dim 16384 380.048 us.
- fp32 guardrails were normal because they did not use the experimental route.
- Decision: reject and leave the launch carveout unset.

## Experiment 121: 4096 Two-Min-Block Launch Bounds

Hypothesis: default fp16/bf16 dim 4096 uses only 32 KiB of dynamic shared memory after double-buffering, so `__launch_bounds__(threads, 2)` may improve resident-block scheduling or register allocation without the larger-dimension occupancy penalty.

Change:
- Temporarily added a `__launch_bounds__(Ktraits::kNThreads, 2)` main-kernel wrapper.
- Routed only default fp16/bf16 dim 4096 through it, preserving the accepted double-buffered exchange.

Result:
- Artifact: `benchmark_results/exp121_16bit_4096_min_blocks2_h200_20260528.json`.
- fp16 dim 4096 regressed to 233.440 us and bf16 dim 4096 regressed to 233.728 us.
- Adjacent dims and fp32 guardrails were normal.
- Decision: reject and keep the standard launch bounds for dim 4096.

## Experiment 122: No Launch Bounds for 4096/8192

Hypothesis: the remaining slow default fp16/bf16 4096/8192 cells and fp32 8192 may be constrained by the generic `__launch_bounds__` annotation. Removing launch bounds for only those routes may let ptxas choose a better register/scheduling tradeoff without changing arithmetic precision.

Change:
- Temporarily added an otherwise identical main-kernel wrapper without `__launch_bounds__`.
- Routed default fp16/bf16 dims 4096 and 8192 through it, preserving the accepted 4096 double-buffered exchange.
- Routed fp32 dim 8192 through it, with fp32 dim 4096 included as a guardrail.

Result:
- Artifact: `benchmark_results/exp122_no_launch_bounds_4096_8192_h200_20260528.json`.
- fp32 dim 8192 regressed to 142.128 us versus 141.376 us in the accepted full artifact.
- fp16 dim 4096 regressed slightly to 231.984 us, while fp16 dim 8192 and bf16 dim 4096 moved only at noise scale.
- Decision: reject and keep the standard launch-bounds annotations.

## Experiment 123: fp32 8192 512-Thread Retest

Hypothesis: fp32 dim 8192 may respond differently to a 512-thread launch under the current direct-I/O code than it did in the early broad launch-width experiment. This changes only the launch shape and keeps fp32 arithmetic unchanged.

Change:
- Temporarily routed fp32 dim 8192 through the 512-thread main-kernel launch.
- Benchmarked fp32 dims 4096, 8192, and 16384 with correctness enabled.

Result:
- Artifact: `benchmark_results/exp123_fp32_8192_512_threads_current_h200_20260528.json`.
- fp32 dim 8192 regressed to 163.216 us versus 141.376 us in the accepted full artifact.
- fp32 dim 4096 and 16384 guardrails stayed near the expected range.
- Decision: reject and keep fp32 dim 8192 on the 256-thread launch.

## Experiment 124: Broad Grid-Constant Params

Hypothesis: CUDA `__grid_constant__` may reduce per-thread parameter traffic or local copies for the by-value `HadamardParamsBase` struct. This is a Hopper-friendly codegen change only; it does not change arithmetic precision, shared-memory value type, or output rounding.

Change:
- Temporarily marked the main default kernel params and specialized bf16 32768 params as `__grid_constant__`.
- Temporarily changed the main kernel body to take params by const reference so the grid-constant parameter would not be copied again.

Result:
- Artifact: `benchmark_results/exp124_grid_constant_params_large_h200_20260528.json`.
- Positive signals: fp16 dim 8192 improved to 231.808 us, bf16 dim 16384 improved to 234.000 us, and fp32 dim 32768 improved to 194.064 us.
- Broad regressions made the route unusable as-is: fp32 dim 4096 regressed to 146.496 us and fp32 dim 16384 to 165.584 us.
- Decision: reject the broad route and narrow the grid-constant entry to candidate cells only.

## Experiment 125: Selective Grid-Constant Params With Reference Body

Hypothesis: applying the grid-constant entry only to the positive cells from Experiment 124 may keep the wins while avoiding broad fp32 regressions.

Change:
- Added a separate grid-constant main-kernel wrapper.
- Routed only default fp16 dim 8192, guarded bf16 dim 16384 pre-scale, and fp32 dim 32768 through it.
- Kept the main kernel body on the temporary const-reference params form.

Result:
- Targeted artifact: `benchmark_results/exp125_selective_grid_constant_params_h200_20260528.json`.
- Full artifacts:
  - `benchmark_results/current_default_precision_exp125_selective_grid_constant_full_h200_20260528.json`: 1.2861x over baseline.
  - `benchmark_results/current_default_precision_exp125_selective_grid_constant_full_h200_20260528_repeat2.json`: 1.2836x over baseline.
- The candidate cells repeated well, but the full repeat fell behind the accepted 1.2844x artifact because unrelated fp32 guardrails drifted worse.
- Decision: reject this form and restore the regular body params ABI before retesting the selective grid-constant wrapper.

## Experiment 126: Selective Grid-Constant Params With By-Value Body

Hypothesis: keeping the original by-value body params preserves codegen for normal routes, while a separate grid-constant global entry can still improve the narrow cells identified in Experiment 124.

Change:
- Restored the regular main kernel body to by-value `HadamardParamsBase` params.
- Kept the separate grid-constant main-kernel wrapper.
- Routed only default fp16 dim 8192, guarded bf16 dim 16384 pre-scale, and fp32 dim 32768 through the grid-constant wrapper.

Result:
- Targeted artifact: `benchmark_results/exp126_grid_constant_byvalue_body_h200_20260528.json`.
- Full artifacts:
  - `benchmark_results/current_default_precision_exp126_grid_constant_byvalue_full_h200_20260528.json`: 1.2872x over baseline.
  - `benchmark_results/current_default_precision_exp126_grid_constant_byvalue_full_h200_20260528_repeat2.json`: 1.2871x over baseline.
- Repeat-2 dtype speedups over baseline: fp16 1.3823x, bf16 1.3702x, fp32 1.1259x.
- Repeat-2 candidate timings: fp16 dim 8192 231.264 us, bf16 dim 16384 233.568 us, fp32 dim 32768 194.560 us.
- Correctness/unit verification after final build: `uv run --no-project pytest -q tests/test_fast_hadamard_transform.py`, 55 passed.
- Decision: accept. This is precision-preserving because it changes only kernel parameter placement/codegen and keeps float-intermediate arithmetic and output dtype conversion unchanged.

## Experiment 127: bf16 8192 Grid-Constant Params

Hypothesis: the broad grid-constant experiment also showed a possible bf16 dim 8192 win. Adding that one route on top of Experiment 126 may improve another weak large bf16 cell without changing arithmetic precision.

Change:
- Temporarily routed default bf16 dim 8192 through the grid-constant main-kernel wrapper.
- Kept the accepted Experiment 126 routes unchanged.

Result:
- Targeted artifact: `benchmark_results/exp127_bf16_8192_grid_constant_h200_20260528.json`.
- Full artifacts:
  - `benchmark_results/current_default_precision_exp127_bf16_8192_grid_constant_full_h200_20260528.json`: 1.2859x over baseline.
  - `benchmark_results/current_default_precision_exp127_bf16_8192_grid_constant_full_h200_20260528_repeat2.json`: 1.2850x over baseline.
- The bf16 dim 8192 cell improved repeatably to about 233.5 us versus about 235.3 us in the accepted Exp126 artifacts.
- The full sweeps did not beat the accepted Exp126 full artifacts, likely because the isolated 0.8% cell win is below run-to-run movement across the full 24-case suite.
- Decision: reject for now and keep the accepted Exp126 source. Revisit only with a tighter paired protocol for single-cell changes.

## Experiment 128: bf16 8192 Vectorized Boundary I/O Retest

Hypothesis: bf16 dim 8192 currently stays on generic scalar boundary conversion, while bf16 dims 4096 and 16384 use vectorized bfloat162 conversion. Retesting vectorized conversion for only bf16 dim 8192 under the current grid-constant codegen may capture a boundary-packing win without changing float-intermediate arithmetic.

Change:
- Temporarily routed bf16 dim 8192 through `FloatIntermediateIO<..., at::BFloat16>` by excluding `N == 8192` from the generic bf16 I/O path.
- Kept all Hadamard arithmetic in float and output conversion unchanged.

Result:
- Artifact: `benchmark_results/exp128_bf16_8192_vector_io_current_h200_20260528.json`.
- bf16 dim 8192 regressed to 236.768 us versus 235.360 us in the accepted Exp126 repeat artifact.
- fp16/fp32 guardrails were normal.
- Decision: reject and keep bf16 dim 8192 on the generic scalar boundary-conversion path.

## Experiment 129: fp32 512/1024 Grouped One-Warp Retest

Hypothesis: fp32 dims 512 and 1024 still have weak baseline-relative speedups and use one 32-thread block per row. Routing them through the existing grouped one-warp kernel with 8 rows per block may reduce block scheduling overhead while preserving fp32 arithmetic.

Change:
- Temporarily routed fp32 dim 512 and 1024 through `fast_hadamard_transform_one_warp_launch<8, ...>`.
- Kept fp32 dims 256 and 2048 as guardrails.

Result:
- Artifact: `benchmark_results/exp129_fp32_512_1024_grouped_one_warp_h200_20260528.json`.
- fp32 dim 512 regressed to 72.960 us versus 71.472 us in the accepted Exp126 repeat artifact.
- fp32 dim 1024 regressed to 135.536 us versus 134.848 us in the accepted Exp126 repeat artifact.
- Decision: reject and keep fp32 dims 512 and 1024 on the standard single-row 32-thread launches.

## Experiment 130: fp32 1024/2048 Direct Chunk Stage Retest

Hypothesis: the final per-thread chunk Hadamard for fp32 dims 1024 and 2048 may spend more time copying through the transposed register tile than doing useful math. Replacing the transpose-based chunk stage with an equivalent direct in-register chunk butterfly for just those fp32 routes may reduce register movement without changing arithmetic precision.

Change:
- Temporarily added a direct power-of-two chunk-stage helper.
- Routed only fp32 dims 1024 and 2048 through it.

Result:
- Targeted artifact: `benchmark_results/exp130_fp32_1024_2048_direct_chunk_h200_20260528.json`.
- Full artifact: `benchmark_results/current_default_precision_exp130_fp32_direct_chunk_full_h200_20260528.json`.
- The targeted run showed only noise-scale gains: fp32 dim 1024 was 134.576 us and dim 2048 was 136.496 us.
- The full sweep regressed to 1.2855x over baseline versus 1.2871x in the accepted Exp126 repeat artifact.
- Decision: reject and keep the transpose-based chunk stage.

## Experiment 131: 16-Bit 4096 Exact-Power Pre-Scale Retest

Hypothesis: dim 4096 is one of the weakest remaining 16-bit cells, and the benchmark scale is exactly `1/64`. Applying the scale before the float Hadamard and storing with scale 1 may preserve output rounding while moving multiply work away from the output conversion path. This retests the earlier neutral 4096 pre-scale signal under the current grid-constant codegen.

Change:
- Temporarily routed default fp16 and bf16 dim 4096 through the existing `kPreScale` main-kernel instantiation when `params.scale == 0.015625f`.
- Kept the accepted double-buffered post-exchange route.

Result:
- Artifact: `benchmark_results/exp131_16bit_4096_prescale_current_h200_20260528.json`.
- fp16 dim 4096 regressed to 232.176 us versus 231.920 us in the accepted Exp126 repeat artifact.
- bf16 dim 4096 regressed to 232.112 us versus 231.024 us in the accepted Exp126 repeat artifact.
- Guardrails were normal.
- Decision: reject and keep fp16/bf16 dim 4096 on the accepted post-scale double-buffered route.

## Experiment 132: Specialized 4096 16-Bit Exchange Helper

Hypothesis: fp16/bf16 dim 4096 always uses the same one-round exchange shape: 2 chunks, 8 values per chunk, 8 warps, and two 16-byte exchange vectors per chunk. A specialized helper can remove the generic exchange loops and repeated index expressions while preserving the exact same shared-memory permutation and float arithmetic.

Change:
- Temporarily added `exchange_smem_pre_2chunks_8elts_8warps`.
- Routed only the matching fp16/bf16 dim 4096 main-kernel exchange shape through it.

Result:
- Artifact: `benchmark_results/exp132_16bit_4096_specialized_exchange_h200_20260528.json`.
- fp16 dim 4096 regressed to 232.096 us versus 231.920 us in the accepted Exp126 repeat artifact.
- bf16 dim 4096 was only noise-scale at 231.296 us versus 231.024 us accepted.
- fp32 guardrails were normal because they did not use the specialized helper.
- Decision: reject and keep the generic templated exchange helper.

## Experiment 133: Bootstrap Benchmark Artifact Comparator

Hypothesis: recent precision-preserving kernel changes are often below 1% and can look positive in targeted runs while losing in full-suite repeats. A small artifact-comparison utility with bootstrap intervals over raw CUDA event timings will make acceptance and rejection decisions more rigorous without changing package behavior.

Change:
- Added `benchmarks/compare_benchmark_artifacts.py`.
- The script compares two `rigorous_benchmark.py` JSON artifacts by common dtype/dim cases, reports geometric-mean speedups, dtype speedups, weakest cases, and optional bootstrap confidence intervals from the persisted `times_us` samples.

Result:
- Baseline comparison artifact: `benchmark_results/compare_exp126_vs_baseline_bootstrap_h200_20260528.json`.
- Exp126 versus original baseline: 1.287149x geometric speedup; bootstrap median 1.286968x with p05/p95 1.286235x/1.287566x.
- Rejection audit artifact: `benchmark_results/compare_exp127_vs_exp126_bootstrap_h200_20260528.json`.
- Exp127 versus Exp126: 0.998359x geometric speedup; bootstrap p95 was 0.999151x, confirming the rejected full-suite candidate did not beat accepted Exp126 despite its isolated bf16 8192 cell win.
- Decision: accept the comparator as part of the benchmark protocol for future narrow-route decisions.

## Experiment 134: Serial Two-Row CTA for 16-Bit 4096

Hypothesis: default fp16/bf16 dim 4096 is still block-scheduling and exchange-bound. Running two independent rows serially inside one CTA can reuse the existing float-intermediate body and shared tile while halving CTA count, without changing per-row arithmetic or output conversion.

Change:
- Temporarily refactored the main kernel body to accept an explicit batch id.
- Added a serial rows-per-CTA launch wrapper.
- Routed only default fp16/bf16 `log_N == 12` through the 2-row serial wrapper.

Result:
- Artifact: `benchmark_results/exp134_16bit_4096_serial_rows2_h200_20260528.json`.
- Correctness was enabled in the targeted benchmark.
- fp16 dim 4096 regressed to 233.680 us versus 231.920 us in the accepted Exp126 repeat artifact.
- bf16 dim 4096 regressed to 233.216 us versus 231.024 us in the accepted Exp126 repeat artifact.
- Decision: reject and restore the accepted one-row CTA route.

## Experiment 135: One-Warp 16-Bit 4096 Layout

Hypothesis: dim 4096 may be better served by one warp per row instead of the 256-thread shared-memory exchange layout. This removes both cross-warp exchanges and CTA barriers. It increases per-thread chunk state from 2 chunks to 16 chunks, but still uses the existing float-intermediate one-warp kernel and the same output dtype conversion.

Change:
- Temporarily routed default fp16/bf16 `log_N == 12` through `fast_hadamard_transform_one_warp_launch<1, 12, input_t>`.
- Kept fp32 dim 4096 as a guardrail.

Result:
- Artifact: `benchmark_results/exp135_16bit_4096_one_warp_h200_20260528.json`.
- Correctness was enabled in the targeted benchmark.
- ptxas reported 167 registers and no spills for the new fp16 dim 4096 one-warp entry.
- fp16 dim 4096 improved to 173.728 us versus 231.920 us in the accepted Exp126 repeat artifact.
- bf16 dim 4096 improved to 173.184 us versus 231.024 us in the accepted Exp126 repeat artifact.
- Decision: promising; tune rows per block and full-suite impact before accepting.

## Experiment 136: One-Warp 16-Bit 4096 Rows Per Block = 2

Hypothesis: once dim 4096 uses one warp per row, grouping two rows per CTA might recover some block-scheduling overhead without changing per-row math.

Change:
- Changed the default fp16/bf16 dim 4096 one-warp route from 1 row per CTA to 2 rows per CTA.

Result:
- Artifact: `benchmark_results/exp136_16bit_4096_one_warp_rows2_h200_20260528.json`.
- Correctness was enabled in the targeted benchmark.
- fp16 dim 4096 measured 174.496 us, slower than 173.728 us with 1 row per CTA.
- bf16 dim 4096 measured 173.520 us, slower than 173.184 us with 1 row per CTA.
- Decision: reject rows-per-block = 2 and keep 1 row per CTA for dim 4096.

## Experiment 137: One-Warp 16-Bit 8192 Layout

Hypothesis: the same no-shared-exchange one-warp layout may help dim 8192, another weak 16-bit cell, if avoiding CTA barriers outweighs the larger per-thread chunk state.

Change:
- Kept the default fp16/bf16 dim 4096 one-warp route from Experiment 135.
- Temporarily routed default fp16/bf16 dim 8192 through `fast_hadamard_transform_one_warp_launch<1, 13, input_t>`.

Result:
- Artifact: `benchmark_results/exp137_16bit_4096_8192_one_warp_h200_20260528.json`.
- Correctness was enabled in the targeted benchmark.
- ptxas reported heavy spilling for the fp16 dim 8192 one-warp entry: 255 registers, 304-byte stack, 844 bytes spill stores, and 884 bytes spill loads.
- fp16 dim 8192 regressed to 348.384 us versus 231.264 us in the accepted Exp126 repeat artifact.
- bf16 dim 8192 regressed to 365.696 us versus 235.360 us in the accepted Exp126 repeat artifact.
- Decision: reject the dim 8192 one-warp route and restore the accepted 8192 routes.

## Experiment 138: Accept One-Warp 16-Bit 4096

Hypothesis: the strong isolated fp16/bf16 4096 one-warp win from Experiment 135 is large enough to survive full-suite movement and improve the default precision-preserving path.

Change:
- Routed default fp16/bf16 `log_N == 12` through `fast_hadamard_transform_one_warp_launch<1, 12, input_t>`.
- Kept fast-low-precision native half2/bfloat162 routes gated behind `fast_low_precision=True`.
- Restored accepted 8192 and larger routes.

Result:
- Full artifacts:
  - `benchmark_results/current_default_precision_exp138_4096_one_warp_full_h200_20260528.json`: 115.039 us geometric mean, 1.317696x over baseline.
  - `benchmark_results/current_default_precision_exp138_4096_one_warp_full_h200_20260528_repeat2.json`: 115.182 us geometric mean, 1.316053x over baseline.
- Bootstrap comparison artifacts:
  - `benchmark_results/compare_exp138_vs_baseline_bootstrap_h200_20260528.json`: bootstrap median 1.317661x, p05/p95 1.316991x/1.318268x.
  - `benchmark_results/compare_exp138_vs_exp126_bootstrap_h200_20260528.json`: 1.023732x over Exp126, bootstrap p05/p95 1.023193x/1.024530x.
  - `benchmark_results/compare_exp138_repeat2_vs_baseline_bootstrap_h200_20260528.json`: bootstrap median 1.315909x, p05/p95 1.315079x/1.316594x.
  - `benchmark_results/compare_exp138_repeat2_vs_exp126_bootstrap_h200_20260528.json`: 1.022456x over Exp126, bootstrap p05/p95 1.021724x/1.023205x.
- Repeat2 dtype speedups versus the original baseline: fp16 1.430886x, bf16 1.417490x, fp32 1.123815x.
- Unit verification after final build: `uv run --no-project pytest -q tests/test_fast_hadamard_transform.py`, 55 passed in 158.06s.
- Precision note: this route keeps float intermediates and the standard dtype output conversion. It changes the on-chip layout and removes shared-memory exchange for dim 4096; it does not use native half2/bfloat162 arithmetic.
- Decision: accept. The default precision-preserving path improves from the Exp126 repeat2 speedup of 1.287149x to 1.316053x-1.317696x. The requested 1.5x default target remains unmet; from the repeat2 value, another roughly 13.98% speedup from here is still required.

## Experiment 139: Exact One-Warp I/O for 16-Bit 4096

Hypothesis: once default fp16/bf16 dim 4096 uses one warp per row, the remaining boundary-checked vector I/O work is unnecessary for the exact 4096 case. Exact half/bfloat16 one-warp load/store helpers can remove the guards and zero-fill path while preserving the same float-intermediate arithmetic and final dtype conversion.

Change:
- Added exact fp16 one-warp load/store helpers matching the existing exact bf16 one-warp I/O pattern.
- Extended the one-warp kernel and launch wrapper with a separate exact-half-I/O template flag.
- Routed only default fp16/bf16 dim 4096 through exact one-warp I/O. The `fast_low_precision=True` native half2/bfloat162 routes remain gated, and non-4096 default routes stay on the generic one-warp I/O path.

Result:
- Targeted artifact: `benchmark_results/exp139_16bit_4096_one_warp_exact_io_h200_20260528.json`.
- Targeted fp16 dim 4096 improved to 170.768 us versus 174.016 us in the accepted Exp138 repeat2 artifact.
- Targeted bf16 dim 4096 improved to 171.040 us versus 173.200 us in the accepted Exp138 repeat2 artifact.
- Full artifacts:
  - `benchmark_results/current_default_precision_exp139_exact_4096_io_full_h200_20260528.json`: 115.261 us geometric mean, 1.315150x over baseline.
  - `benchmark_results/current_default_precision_exp139_exact_4096_io_full_h200_20260528_repeat2.json`: 114.939 us geometric mean, 1.318843x over baseline.
- Bootstrap comparison artifacts:
  - `benchmark_results/compare_exp139_vs_baseline_bootstrap_h200_20260528.json`: bootstrap median 1.315060x, p05/p95 1.314290x/1.315767x.
  - `benchmark_results/compare_exp139_vs_exp138_repeat2_bootstrap_h200_20260528.json`: 0.999314x versus Exp138 repeat2, bootstrap p05/p95 0.998642x/1.000126x.
  - `benchmark_results/compare_exp139_repeat2_vs_baseline_bootstrap_h200_20260528.json`: bootstrap median 1.318786x, p05/p95 1.318131x/1.319420x.
  - `benchmark_results/compare_exp139_repeat2_vs_exp138_repeat2_bootstrap_h200_20260528.json`: 1.002120x versus Exp138 repeat2, bootstrap p05/p95 1.001536x/1.002924x.
- Repeat2 dtype speedups versus the original baseline: fp16 1.437042x, bf16 1.422245x, fp32 1.122368x.
- Precision note: the accepted path keeps float intermediates and standard fp16/bf16 output conversion. It removes exact-dimension I/O overhead only; it does not use native half2/bfloat162 arithmetic in the default path.
- Decision: accept. This is a small repeatable improvement over Exp138, raising the best default precision-preserving repeat from 1.316053x to 1.318843x over baseline. The requested 1.5x default target remains unmet; from the repeat2 value, another roughly 13.74% speedup from here is still required.

## Experiment 140: fp32 Exact I/O for Small Power-of-Two Routes

Hypothesis: fp32 dimensions 256 through 4096 still carry runtime boundary checks even for exact power-of-two benchmark dimensions. Exact fp32 vector load/store helpers may remove guard overhead while preserving fp32 arithmetic and output scaling.

Change:
- Temporarily added exact fp32 block and one-warp load/store helpers.
- First routed exact fp32 dims 256, 512, 1024, 2048, and 4096 through the exact-I/O instantiations.
- After broad regressions, narrowed the temporary route to exact fp32 dims 512 and 1024 only.

Result:
- Broad artifact: `benchmark_results/exp140_fp32_small_exact_io_h200_20260528.json`.
- Broad result: fp32 dim 512 had a small noise-scale improvement at 71.312 us, and fp32 dim 1024 was nearly neutral at 134.784 us. fp32 dim 2048 regressed to 138.272 us and fp32 dim 4096 regressed to 147.600 us versus the accepted Exp139 repeat2 values of 136.512 us and 141.904 us.
- Narrow artifact: `benchmark_results/exp140_fp32_512_1024_exact_io_narrow_h200_20260528.json`.
- Narrow result: fp32 dim 512 remained only noise-scale at 71.376 us, while fp32 dim 1024 regressed to 134.992 us versus 134.880 us in the accepted Exp139 repeat2 artifact. Guardrail fp32 dims 256 and 2048 were normal once their exact routes were removed.
- Precision note: this would have preserved fp32 arithmetic, but the codegen/timing result was not useful enough to keep.
- Decision: reject and restore the accepted Exp139 source. The best default precision-preserving repeat remains 1.318843x over baseline.

## Experiment 141: Narrow fp16 Exact I/O for 8192 and 16384

Hypothesis: under the accepted Exp139 codegen, routing only default fp16 dim 8192 and 16384 through exact-I/O instantiations could remove boundary checks and zero-fill work without changing arithmetic precision. The earlier broad exact-I/O attempts were too noisy, so this retested only the two weak fp16 cells.

Change:
- Temporarily routed default fp16 dim 8192 through `fast_hadamard_transform_grid_constant_launch<256, 13, input_t, false, false, true>`.
- Temporarily routed default fp16 dim 16384 through `fast_hadamard_transform_launch<256, 14, input_t, false, false, true>`.
- Left bf16, fp32, and `fast_low_precision=True` native half2/bfloat162 routes unchanged.

Result:
- Artifact: `benchmark_results/exp141_fp16_8192_16384_exact_io_h200_20260528.json`.
- Correctness was enabled in the targeted benchmark.
- fp16 dim 8192 regressed to 239.840 us versus 231.328 us in the accepted Exp139 repeat2 artifact.
- fp16 dim 16384 regressed to 252.528 us versus 245.792 us in the accepted Exp139 repeat2 artifact.
- Guardrail cells were neutral-to-normal: bf16 dim 8192 235.904 us, bf16 dim 16384 233.872 us, fp32 dim 8192 142.416 us, and fp32 dim 16384 149.296 us.
- Precision note: this experiment would have preserved float intermediates and the standard fp16 output conversion. It only changed exact-dimension I/O specialization, but the codegen/timing result was slower.
- Decision: reject and restore the accepted Exp139 source. The best default precision-preserving repeat remains 1.318843x over baseline.

## Experiment 142: 1024-Thread 16-Bit 8192/16384 Retest

Hypothesis: default fp16/bf16 dims 8192 and 16384 may be limited by per-thread register and instruction work. Routing only exact 8192 and 16384 16-bit cases through 1024-thread float-intermediate kernels could reduce per-thread chunks while preserving the same arithmetic and output conversion.

Change:
- Temporarily routed exact default fp16 dim 8192 through the grid-constant 1024-thread main kernel.
- Temporarily routed exact default bf16 dim 8192 through the 1024-thread main kernel.
- Temporarily routed exact default fp16 dim 16384 through the 1024-thread main kernel.
- Temporarily routed the accepted exact-power bf16 dim 16384 pre-scale path through the 1024-thread grid-constant main kernel.
- Left fp32 and `fast_low_precision=True` native half2/bfloat162 routes unchanged.

Result:
- Artifact: `benchmark_results/exp142_16bit_8192_16384_1024_threads_h200_20260528.json`.
- Correctness was enabled in the targeted benchmark.
- ptxas showed the intended lower register pressure for the new fp16 kernels, including 32 registers for fp16 dim 8192 and 46 registers for fp16 dim 16384, with no spills.
- The lower register pressure did not translate to speed: fp16 dim 8192 regressed to 271.840 us versus 231.328 us in the accepted Exp139 repeat2 artifact.
- fp16 dim 16384 regressed to 333.936 us versus 245.792 us.
- bf16 dim 8192 regressed to 270.880 us versus 235.840 us.
- bf16 dim 16384 regressed to 277.760 us versus 233.840 us.
- fp32 guardrails were normal: fp32 dim 8192 142.768 us and fp32 dim 16384 149.744 us.
- Precision note: this experiment preserved float intermediates and standard fp16/bf16 output conversion; it only changed launch shape for exact dimensions.
- Decision: reject and restore the accepted Exp139 source. The best default precision-preserving repeat remains 1.318843x over baseline.

## Experiment 143: fp16 32768 Restricted Exact Specialized Kernel

Hypothesis: the accepted fp16 dim 32768 exact-I/O route still uses the generic float-intermediate body. A dedicated exact fp16 32768 kernel with restricted row pointers and inline exact vector I/O may give ptxas better aliasing information than Experiment 094 while preserving the same arithmetic, scaling, and fp16 output conversion.

Change:
- Temporarily added a specialized default fp16 dim 32768 kernel.
- Kept 1024 threads, the 128 KiB float exchange tile, float Hadamard arithmetic, the accepted initial pre-exchange sync skip, and exact fp16 load/store conversion.
- Marked the row input and output pointers as `__restrict__` inside the specialized kernel.
- Left bf16, fp32, and `fast_low_precision=True` native half2/bfloat162 routes unchanged.

Result:
- Artifact: `benchmark_results/exp143_fp16_32768_restrict_specialized_h200_20260528.json`.
- Correctness was enabled in the targeted benchmark.
- ptxas reported 64 registers and no spills for the specialized fp16 32768 kernel, matching the accepted generic exact route's register count.
- fp16 dim 32768 measured 306.912 us versus 306.816 us in the accepted Exp139 repeat2 artifact, so the change was neutral to slightly slower.
- bf16 and fp32 guardrails stayed normal: bf16 dim 32768 326.112 us and fp32 dim 32768 194.336 us.
- Precision note: this experiment preserved float intermediates and standard fp16 output conversion; the only semantic-facing change was aliasing/codegen information in an exact-dimension kernel.
- Decision: reject and restore the accepted Exp139 source. The best default precision-preserving repeat remains 1.318843x over baseline.

## Experiment 144: bf16 8192 Grid-Constant Retest

Hypothesis: Experiment 127 showed an isolated positive signal for default bf16 dim 8192 through the existing grid-constant wrapper, but the full-suite result did not beat the then-accepted source. Retesting that single dispatch change under the accepted Exp139 codegen could recover a small precision-preserving win.

Change:
- Temporarily routed only default bf16 dim 8192 through `fast_hadamard_transform_grid_constant_launch<256, 13, input_t>`.
- Left fp16, fp32, other bf16 dimensions, and `fast_low_precision=True` native bfloat162 routes unchanged.

Result:
- Targeted artifacts:
  - `benchmark_results/exp144_bf16_8192_grid_constant_h200_20260528.json`.
  - `benchmark_results/exp144_bf16_8192_grid_constant_h200_20260528_repeat2.json`.
- The targeted bf16 dim 8192 cell improved on both focused repeats: 234.160 us and 233.760 us versus 235.840 us in the accepted Exp139 repeat2 artifact.
- Full artifact: `benchmark_results/exp144_bf16_8192_grid_constant_full_h200_20260528.json`.
- Full-suite bf16 dim 8192 measured 233.440 us, a 1.010281x cell speedup versus accepted Exp139 repeat2.
- Bootstrap comparison artifacts:
  - `benchmark_results/compare_exp144_vs_exp139_repeat2_bootstrap_h200_20260528.json`: 1.000210x geometric speedup versus accepted Exp139 repeat2, bootstrap median 1.000300x with p05/p95 0.999633x/1.000973x.
  - `benchmark_results/compare_exp144_vs_baseline_bootstrap_h200_20260528.json`: 1.319120x geometric speedup versus the original baseline, bootstrap median 1.319158x with p05/p95 1.318402x/1.319953x.
- Dtype speedups versus accepted Exp139 repeat2 were bf16 1.000877x, fp16 0.999467x, and fp32 1.000286x.
- Precision note: this experiment preserved float intermediates and standard bf16 output conversion. It only changed the launch wrapper for one exact dimension and did not use native bfloat162 arithmetic on the default path.
- Decision: reject and restore the accepted Exp139 source. The isolated bf16 dim 8192 improvement repeated, but the full-suite effect was only noise-level and the bootstrap interval versus accepted Exp139 crossed 1.0. The best default precision-preserving repeat remains 1.318843x over baseline, still about 13.74% short of the requested 1.5x target.

## Experiment 145: bf16 16384 Exact I/O With Pre-Scale

Hypothesis: the accepted bf16 dim 16384 pre-scale route still uses the guarded vectorized bf16 load/store path. Exact bf16 main-kernel I/O for the exact 16384 case could remove boundary checks and zero-fill work while preserving float intermediates, the accepted pre-scale condition, and standard bf16 output rounding.

Change:
- Temporarily added exact bf16 main-kernel load/store helpers.
- Routed only default bf16 dim 16384 with `params.scale == 0.0078125f` through exact bf16 I/O on the accepted grid-constant pre-scale route.
- Left fp16, fp32, other bf16 dimensions, and `fast_low_precision=True` native bfloat162 routes unchanged.

Result:
- Artifact: `benchmark_results/exp145_bf16_16384_exact_prescale_io_h200_20260528.json`.
- Correctness was enabled in the targeted benchmark.
- bf16 dim 16384 regressed badly to 257.024 us versus 233.840 us in the accepted Exp139 repeat2 artifact.
- Neighboring guardrails were normal: bf16 dim 8192 235.696 us, bf16 dim 32768 326.128 us, fp16 dim 32768 306.752 us, fp32 dim 16384 148.864 us, and fp32 dim 32768 194.432 us.
- Precision note: this experiment would have preserved float intermediates and standard bf16 output conversion; it only removed exact-dimension I/O guards in the already-accepted pre-scale route. The regression is a codegen/performance issue, not a precision tradeoff.
- Decision: reject and restore the accepted Exp139 source. The best default precision-preserving repeat remains 1.318843x over baseline, still about 13.74% short of the requested 1.5x target.

## Experiment 146: fp16 8192 Read-Only Vector Load

Hypothesis: Experiment 102 had a small fp16 dim 8192 signal from read-only vector loads before the later grid-constant and exact-I/O changes. Retesting only the current fp16 dim 8192 grid-constant route with `__ldg` input vector loads may improve the weak 8192 cell without affecting arithmetic precision or other dtypes.

Change:
- Temporarily added an fp16 read-only vector load helper using `__ldg`.
- Routed only default fp16 dim 8192 through that load helper on the accepted grid-constant route.
- Kept the same fp16-to-float conversion, float Hadamard arithmetic, standard fp16 output conversion, and `fast_low_precision=True` native half2 route.

Result:
- Artifact: `benchmark_results/exp146_fp16_8192_ldg_h200_20260528.json`.
- Correctness was enabled in the targeted benchmark.
- fp16 dim 8192 measured 231.776 us versus 231.328 us in the accepted Exp139 repeat2 artifact, so the intended cell moved slightly slower.
- Guardrails were normal: fp16 dim 16384 245.600 us, bf16 dim 8192 235.968 us, bf16 dim 16384 233.856 us, fp32 dim 8192 142.320 us, and fp32 dim 16384 149.248 us.
- Precision note: this experiment preserved float intermediates and standard fp16 output conversion. It changed only the global input load cache path for one exact route.
- Decision: reject and restore the accepted Exp139 source. The best default precision-preserving repeat remains 1.318843x over baseline.

## Experiment 147: fp16 16384 Pre-Scale Route

Hypothesis: default fp16 dim 16384 has exact scale `1 / 128` (`params.scale == 0.0078125f`), analogous to the accepted bf16 dim 16384 pre-scale route. A guarded fp16 16384 pre-scale path might reduce output conversion work while preserving float intermediates and standard fp16 output rounding.

Change:
- Temporarily routed only default fp16 dim 16384 with `params.scale == 0.0078125f` through `fast_hadamard_transform_launch<256, 14, input_t, false, false, false, true>`.
- Left fp16 dim 8192/32768, bf16, fp32, and `fast_low_precision=True` native half2 routes unchanged.

Result:
- Artifact: `benchmark_results/exp147_fp16_16384_prescale_h200_20260528.json`.
- Correctness was enabled in the targeted benchmark.
- fp16 dim 16384 measured 246.576 us versus 245.792 us in the accepted Exp139 repeat2 artifact, so the intended cell regressed.
- Neighboring guardrails stayed normal: fp16 dim 8192 231.440 us, fp16 dim 32768 306.848 us, bf16 dim 8192 235.680 us, bf16 dim 16384 234.112 us, bf16 dim 32768 325.552 us, fp32 dim 8192 141.888 us, fp32 dim 16384 147.936 us, and fp32 dim 32768 194.672 us.
- Precision note: this experiment preserved float intermediates and standard fp16 output conversion. It only moved the exact power-of-two scale into the float path for one guarded fp16 dimension, so the regression is performance/codegen-related rather than a precision tradeoff.
- Decision: reject and restore the accepted Exp139 source. The best default precision-preserving repeat remains 1.318843x over baseline, still about 13.74% short of the requested 1.5x target.

## Experiment 148: 16-Bit 8192/16384 16-Element Main-Kernel Shape

Hypothesis: default fp16/bf16 dims 8192 and 16384 still spend substantial time in the main-kernel chunk layout. Keeping the same 256-thread launches but widening the per-thread vector shape from 8 to 16 16-bit inputs could reduce chunk-loop and vector-I/O overhead while preserving float intermediates and standard fp16/bf16 output conversion.

Change:
- Temporarily added a kernel-traits element-count override and specialized 16-element fp16/bf16 vectorized float-intermediate I/O helpers.
- Routed default fp16/bf16 dims 8192 and 16384 through the 16-element shape.
- Preserved the existing grid-constant wrapper for fp16 dim 8192 and the accepted guarded bf16 dim 16384 pre-scale route.
- Left fp32 and `fast_low_precision=True` native half2/bfloat162 routes unchanged.

Result:
- Artifact: `benchmark_results/exp148_16bit_8192_16384_16elts_h200_20260528.json`.
- Correctness was enabled in the targeted benchmark.
- ptxas showed no spills for the new fp16 16-element 8192 and 16384 instantiations, but runtime performance was much worse.
- fp16 dim 8192 regressed to 320.112 us versus 231.328 us in the accepted Exp139 repeat2 artifact.
- fp16 dim 16384 regressed to 326.464 us versus 245.792 us.
- bf16 dim 8192 regressed to 336.224 us versus 235.840 us.
- bf16 dim 16384 regressed to 324.960 us versus 233.840 us.
- fp32 guardrails were normal: fp32 dim 8192 143.008 us and fp32 dim 16384 149.824 us.
- Precision note: this experiment preserved float intermediates and standard fp16/bf16 output conversion. It changed only the on-chip vector/chunk layout and exact I/O packing width for the tested 16-bit main-kernel routes.
- Decision: reject and restore the accepted Exp139 source. The best default precision-preserving repeat remains 1.318843x over baseline, still about 13.74% short of the requested 1.5x target.

## Experiment 149: 16-Bit 8192/16384 Wide Float Shared-Memory Exchange

Hypothesis: default fp16/bf16 dims 8192 and 16384 still spend a large fraction of time in the float register-exchange path. Storing and loading the full eight-float exchange vector as a single 32-byte shared-memory chunk could reduce shared-memory instruction overhead while preserving the same float intermediates and standard fp16/bf16 output conversion.

Change:
- Temporarily added a wide eight-float shared-memory exchange helper.
- Routed default fp16/bf16 dims 8192 and 16384 through the wide exchange path.
- Preserved the existing grid-constant wrapper for fp16 dim 8192 and the accepted guarded bf16 dim 16384 pre-scale route.
- Left fp32 and `fast_low_precision=True` native half2/bfloat162 routes unchanged.

Result:
- Artifact: `benchmark_results/exp149_16bit_8192_16384_wide_float_exchange_h200_20260528.json`.
- Correctness was enabled in the targeted benchmark.
- ptxas showed no spills and unchanged register counts for the routed fp16 kernels, but runtime performance was much worse.
- fp16 dim 8192 regressed to 292.992 us versus 231.328 us in the accepted Exp139 repeat2 artifact.
- fp16 dim 16384 regressed to 305.408 us versus 245.792 us.
- bf16 dim 8192 regressed to 293.008 us versus 235.840 us.
- bf16 dim 16384 regressed to 294.336 us versus 233.840 us.
- fp32 guardrails stayed normal enough for this targeted test: fp32 dim 8192 142.976 us and fp32 dim 16384 150.352 us.
- Precision note: this experiment preserved float intermediates and standard fp16/bf16 output conversion. It changed only the shared-memory exchange packing for the tested 16-bit main-kernel routes.
- Decision: reject and restore the accepted Exp139 source. The best default precision-preserving repeat remains 1.318843x over baseline, still about 13.74% short of the requested 1.5x target.
