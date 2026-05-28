/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

// #pragma once

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/cuda/CUDAException.h>  // For C10_CUDA_CHECK and C10_CUDA_KERNEL_LAUNCH_CHECK

#include "fast_hadamard_transform.h"
#include "fast_hadamard_transform_common.h"
#include "fast_hadamard_transform_special.h"
#include "static_switch.h"

#include <type_traits>

template<int kNThreads_, int kLogN_, typename input_t_>
struct fast_hadamard_transform_kernel_traits {
    using input_t = input_t_;
    static constexpr int kNThreads = kNThreads_;
    static constexpr int kLogN = kLogN_;
    static constexpr int N = 1 << kLogN;
    static constexpr int kNBytes = sizeof(input_t);
    static_assert(kNBytes == 2 || kNBytes == 4);
    static constexpr int kNElts = kNBytes == 4 ? 4 : 8;
    // It's possible that we need to do 2 rounds of exchange if input_t is 16 bits
    // (since then we'd have 8 values of float, and each round we can exchange 4 floats).
    static constexpr int kNExchangePerVec = sizeof(float) / sizeof(input_t);
    using vec_t = typename BytesToType<kNBytes * kNElts>::Type;
    static constexpr int kNChunks = N / (kNElts * kNThreads);
    // Large dimensions are exchange-bound on Hopper; use a larger tile where measured useful.
    static constexpr bool kUseLargeExchange =
        N == 16 * 1024 || N == 32 * 1024;
    static constexpr int kSmemExchangeSize = std::min(N * 4, (kUseLargeExchange ? 64 : 32) * 1024);
    static constexpr int kNExchangeRounds = N * 4 / kSmemExchangeSize;
    static_assert(kNExchangeRounds * kSmemExchangeSize == N * 4);
    static constexpr int kSmemSize = kSmemExchangeSize;
};

template<int kNThreads_, int kLogN_, typename input_t_>
struct fast_hadamard_transform_lowp_exchange_kernel_traits {
    using input_t = input_t_;
    static constexpr int kNThreads = kNThreads_;
    static constexpr int kLogN = kLogN_;
    static constexpr int N = 1 << kLogN;
    static constexpr int kNBytes = sizeof(input_t);
    static_assert(kNBytes == 2);
    static constexpr int kNElts = 8;
    using vec_t = typename BytesToType<kNBytes * kNElts>::Type;
    static constexpr int kNChunks = N / (kNElts * kNThreads);
    static constexpr int kSmemExchangeSize = std::min(N * kNBytes, 64 * 1024);
    static constexpr int kNExchangeRounds = N * kNBytes / kSmemExchangeSize;
    static_assert(kNExchangeRounds * kSmemExchangeSize == N * kNBytes);
    static constexpr int kSmemSize = kSmemExchangeSize;
};

template<int kNThreads_, int kLogN_, typename input_t_>
struct fast_hadamard_transform_12N_kernel_traits {
    using input_t = input_t_;
    static constexpr int kNThreads = kNThreads_;
    static constexpr int kLogN = kLogN_;
    static constexpr int N = (1 << kLogN) * 12;
    static_assert(N <= 12 * 1024, "fast_hadamard_transform_12 only supports dim <= 12288");
    static constexpr int kNBytes = sizeof(input_t);
    static_assert(kNBytes == 2 || kNBytes == 4);
    static constexpr int kNElts = 4;
    // It's possible that we need to do 2 rounds of exchange if input_t is 16 bits
    // (since then we'd have 8 values of float, and each round we can exchange 4 floats).
    static constexpr int kNExchangePerVec = sizeof(float) / sizeof(input_t);
    using vec_t = typename BytesToType<kNBytes * kNElts>::Type;
    static constexpr int kNChunks = N / (kNElts * kNThreads);
    static_assert(kNChunks == 12);
    // We don't want to use more than 24 KB of shared memory.
    static constexpr int kSmemExchangeSize = std::min(N * 4, 24 * 1024);
    static constexpr int kNExchangeRounds = N * 4 / kSmemExchangeSize;
    static_assert(kNExchangeRounds * kSmemExchangeSize == N * 4);
    static constexpr int kSmemSize = kSmemExchangeSize;
};

template<int kNThreads_, int kLogN_, typename input_t_>
struct fast_hadamard_transform_20N_kernel_traits {
    using input_t = input_t_;
    static constexpr int kNThreads = kNThreads_;
    static constexpr int kLogN = kLogN_;
    static constexpr int N = (1 << kLogN) * 20;
    static_assert(N <= 20 * 1024, "fast_hadamard_transform_20 only supports dim <= 20480");
    static constexpr int kNBytes = sizeof(input_t);
    static_assert(kNBytes == 2 || kNBytes == 4);
    static constexpr int kNElts = 4;
    // It's possible that we need to do 2 rounds of exchange if input_t is 16 bits
    // (since then we'd have 8 values of float, and each round we can exchange 4 floats).
    static constexpr int kNExchangePerVec = sizeof(float) / sizeof(input_t);
    using vec_t = typename BytesToType<kNBytes * kNElts>::Type;
    static constexpr int kNChunks = N / (kNElts * kNThreads);
    static_assert(kNChunks == 20);
    // We don't want to use more than 40 KB of shared memory.
    static constexpr int kSmemExchangeSize = std::min(N * 4, 40 * 1024);
    static constexpr int kNExchangeRounds = N * 4 / kSmemExchangeSize;
    static_assert(kNExchangeRounds * kSmemExchangeSize == N * 4);
    static constexpr int kSmemSize = kSmemExchangeSize;
};

template<int kNThreads_, int kLogN_, typename input_t_>
struct fast_hadamard_transform_28N_kernel_traits {
    using input_t = input_t_;
    static constexpr int kNThreads = kNThreads_;
    static constexpr int kLogN = kLogN_;
    static constexpr int N = (1 << kLogN) * 28;
    static_assert(N <= 28 * 1024, "fast_hadamard_transform_28 only supports dim <= 28672");
    static constexpr int kNBytes = sizeof(input_t);
    static_assert(kNBytes == 2 || kNBytes == 4);
    static constexpr int kNElts = 4;
    // It's possible that we need to do 2 rounds of exchange if input_t is 16 bits
    // (since then we'd have 8 values of float, and each round we can exchange 4 floats).
    static constexpr int kNExchangePerVec = sizeof(float) / sizeof(input_t);
    using vec_t = typename BytesToType<kNBytes * kNElts>::Type;
    static constexpr int kNChunks = N / (kNElts * kNThreads);
    static_assert(kNChunks == 28);
    // We don't want to use more than 28 KB of shared memory.
    static constexpr int kSmemExchangeSize = std::min(N * 4, 28 * 1024);
    static constexpr int kNExchangeRounds = N * 4 / kSmemExchangeSize;
    static_assert(kNExchangeRounds * kSmemExchangeSize == N * 4);
    static constexpr int kSmemSize = kSmemExchangeSize;
};

template<int kNThreads_, int kLogN_, typename input_t_>
struct fast_hadamard_transform_40N_kernel_traits {
    using input_t = input_t_;
    static constexpr int kNThreads = kNThreads_;
    static constexpr int kLogN = kLogN_;
    static constexpr int N = (1 << kLogN) * 40;
    static_assert(N <= 40 * 1024, "fast_hadamard_transform_40 only supports dim <= 40960");
    static constexpr int kNBytes = sizeof(input_t);
    static_assert(kNBytes == 2 || kNBytes == 4);
    static constexpr int kNElts = 4;
    // It's possible that we need to do 2 rounds of exchange if input_t is 16 bits
    // (since then we'd have 8 values of float, and each round we can exchange 4 floats).
    static constexpr int kNExchangePerVec = sizeof(float) / sizeof(input_t);
    using vec_t = typename BytesToType<kNBytes * kNElts>::Type;
    static constexpr int kNChunks = N / (kNElts * kNThreads);
    static_assert(kNChunks == 40);
    // We don't want to use more than 40 KB of shared memory.
    static constexpr int kSmemExchangeSize = std::min(N * 4, 40 * 1024);
    static constexpr int kNExchangeRounds = N * 4 / kSmemExchangeSize;
    static_assert(kNExchangeRounds * kSmemExchangeSize == N * 4);
    static constexpr int kSmemSize = kSmemExchangeSize;
};

template <int kNChunks>
__device__ __forceinline__ void hadamard_mult_thread_chunk_12(float x[kNChunks][12]) {
    #pragma unroll
    for (int c = 0; c < kNChunks; ++c) { hadamard_mult_thread_12(x[c]); }
}

template <int kNChunks>
__device__ __forceinline__ void hadamard_mult_thread_chunk_20(float x[kNChunks][20]) {
    #pragma unroll
    for (int c = 0; c < kNChunks; ++c) { hadamard_mult_thread_20(x[c]); }
}

template <int kNChunks>
__device__ __forceinline__ void hadamard_mult_thread_chunk_28(float x[kNChunks][28]) {
    #pragma unroll
    for (int c = 0; c < kNChunks; ++c) { hadamard_mult_thread_28(x[c]); }
}

template <int kNChunks>
__device__ __forceinline__ void hadamard_mult_thread_chunk_40(float x[kNChunks][40]) {
    #pragma unroll
    for (int c = 0; c < kNChunks; ++c) { hadamard_mult_thread_40(x[c]); }
}

template<typename Ktraits, bool kUseConditionalWarp = false>
__device__ __forceinline__ void fast_hadamard_transform_kernel_body(HadamardParamsBase params, char *smem_) {
    constexpr int kNThreads = Ktraits::kNThreads;
    constexpr int kNElts = Ktraits::kNElts;
    constexpr int kNExchangePerVec = Ktraits::kNExchangePerVec;
    constexpr int kNExchangeRounds = Ktraits::kNExchangeRounds;
    constexpr int kNChunks = Ktraits::kNChunks;
    using input_t = typename Ktraits::input_t;
    using vec_t = typename Ktraits::vec_t;

    constexpr int kLogNElts = cilog2(Ktraits::kNElts);
    static_assert(1 << kLogNElts == kNElts, "kNElts must be a power of 2");
    constexpr int kWarpSize = std::min(kNThreads, 32);
    constexpr int kLogWarpSize = cilog2(kWarpSize);
    static_assert(1 << kLogWarpSize == kWarpSize, "Warp size must be a power of 2");
    constexpr int kNWarps = kNThreads / kWarpSize;
    constexpr int kLogNWarps = cilog2(kNWarps);
    static_assert(1 << kLogNWarps == kNWarps, "kNWarps must be a power of 2");
    constexpr int kLoadsPerExchange = Ktraits::kSmemExchangeSize / (sizeof(vec_t) * kNThreads);
    static_assert(kLoadsPerExchange * sizeof(vec_t) * kNThreads == Ktraits::kSmemExchangeSize, "kSmemExchangeSize should be a power of 2");
    static_assert(kNExchangeRounds * kLoadsPerExchange * sizeof(vec_t) == kNChunks * kNElts * sizeof(float));

    constexpr int kChunksPerExchange = Ktraits::kSmemExchangeSize / (sizeof(vec_t) * kNExchangePerVec * kNThreads);
    static_assert(kChunksPerExchange * sizeof(vec_t) * kNExchangePerVec * kNThreads == Ktraits::kSmemExchangeSize);
    constexpr int kNExchanges = kNChunks / kChunksPerExchange;
    static_assert(kNExchanges * kChunksPerExchange == kNChunks);

    vec_t *smem_exchange = reinterpret_cast<vec_t *>(smem_);

    const int batch_id = blockIdx.x;
    input_t *x = reinterpret_cast<input_t *>(params.x_ptr) + batch_id * params.x_batch_stride;
    input_t *out = reinterpret_cast<input_t *>(params.out_ptr) + batch_id * params.out_batch_stride;

    float x_vals[kNChunks][kNElts];
    load_input<kNChunks, kNElts, input_t>(x, x_vals, params.dim);

    hadamard_mult_thread<kLogNElts, kNChunks>(x_vals);
    if constexpr (kUseConditionalWarp) {
        hadamard_mult_warp_conditional<kLogWarpSize, 0, kNChunks, kNElts>(x_vals);
    } else {
        hadamard_mult_warp<kLogWarpSize, 0, kNChunks, kNElts>(x_vals);
    }

    if constexpr (kNWarps > 1) {
        exchange_smem_pre<kNChunks, kChunksPerExchange, kNElts, kWarpSize, kNWarps, true, vec_t>(x_vals, smem_exchange);
        if constexpr (kUseConditionalWarp) {
            hadamard_mult_warp_conditional<kLogNWarps, 0, kNChunks, kNElts>(x_vals);
        } else {
            hadamard_mult_warp<kLogNWarps, 0, kNChunks, kNElts>(x_vals);
        }
        exchange_smem_pre<kNChunks, kChunksPerExchange, kNElts, kWarpSize, kNWarps, false, vec_t>(x_vals, smem_exchange);
    }

    if constexpr (kNChunks > 1) {
        float x_vals_transposed[kNElts][kNChunks];
        #pragma unroll
        for (int c = 0; c < kNChunks; ++c) {
            #pragma unroll
            for (int i = 0; i < kNElts; ++i) { x_vals_transposed[i][c] = x_vals[c][i]; }
        }
        if constexpr (kNChunks == 12) {
            hadamard_mult_thread_chunk_12<kNElts>(x_vals_transposed);
        } else if constexpr (kNChunks == 20) {
            hadamard_mult_thread_chunk_20<kNElts>(x_vals_transposed);
        } else if constexpr (kNChunks == 28) {
            hadamard_mult_thread_chunk_28<kNElts>(x_vals_transposed);
        } else if constexpr (kNChunks == 40) {
            hadamard_mult_thread_chunk_40<kNElts>(x_vals_transposed);
        } else {
            constexpr int kLogNChunks = cilog2(kNChunks);
            static_assert(1 << kLogNChunks == kNChunks, "kNChunks must be a power of 2");
            hadamard_mult_thread<kLogNChunks, kNElts>(x_vals_transposed);
        }
        #pragma unroll
        for (int c = 0; c < kNChunks; ++c) {
            #pragma unroll
            for (int i = 0; i < kNElts; ++i) { x_vals[c][i] = x_vals_transposed[i][c]; }
        }
    }

    store_output<kNChunks, kNElts, input_t>(out, x_vals, params.dim, params.scale);
}

template<typename Ktraits, bool kUseConditionalWarp = false>
__global__ __launch_bounds__(Ktraits::kNThreads)
void fast_hadamard_transform_kernel(HadamardParamsBase params) {
    extern __shared__ char smem_[];
    fast_hadamard_transform_kernel_body<Ktraits, kUseConditionalWarp>(params, smem_);
}

template<typename Ktraits>
__device__ __forceinline__ void fast_hadamard_transform_lowp_exchange_kernel_body(HadamardParamsBase params) {
    constexpr int kNThreads = Ktraits::kNThreads;
    constexpr int kNElts = Ktraits::kNElts;
    constexpr int kNExchangeRounds = Ktraits::kNExchangeRounds;
    constexpr int kNChunks = Ktraits::kNChunks;
    using input_t = typename Ktraits::input_t;
    using vec_t = typename Ktraits::vec_t;

    constexpr int kLogNElts = cilog2(Ktraits::kNElts);
    static_assert(1 << kLogNElts == kNElts, "kNElts must be a power of 2");
    constexpr int kWarpSize = std::min(kNThreads, 32);
    constexpr int kLogWarpSize = cilog2(kWarpSize);
    static_assert(1 << kLogWarpSize == kWarpSize, "Warp size must be a power of 2");
    constexpr int kNWarps = kNThreads / kWarpSize;
    constexpr int kLogNWarps = cilog2(kNWarps);
    static_assert(1 << kLogNWarps == kNWarps, "kNWarps must be a power of 2");
    constexpr int kLoadsPerExchange = Ktraits::kSmemExchangeSize / (sizeof(vec_t) * kNThreads);
    static_assert(kLoadsPerExchange * sizeof(vec_t) * kNThreads == Ktraits::kSmemExchangeSize, "kSmemExchangeSize should be a power of 2");
    static_assert(kNExchangeRounds * kLoadsPerExchange * sizeof(vec_t) == kNChunks * kNElts * sizeof(input_t));

    constexpr int kChunksPerExchange = Ktraits::kSmemExchangeSize / (sizeof(vec_t) * kNThreads);
    static_assert(kChunksPerExchange * sizeof(vec_t) * kNThreads == Ktraits::kSmemExchangeSize);
    constexpr int kNExchanges = kNChunks / kChunksPerExchange;
    static_assert(kNExchanges * kChunksPerExchange == kNChunks);

    extern __shared__ char smem_[];
    vec_t *smem_exchange = reinterpret_cast<vec_t *>(smem_);

    const int batch_id = blockIdx.x;
    input_t *x = reinterpret_cast<input_t *>(params.x_ptr) + batch_id * params.x_batch_stride;
    input_t *out = reinterpret_cast<input_t *>(params.out_ptr) + batch_id * params.out_batch_stride;

    float x_vals[kNChunks][kNElts];
    load_input<kNChunks, kNElts, input_t>(x, x_vals, params.dim);

    hadamard_mult_thread<kLogNElts, kNChunks>(x_vals);
    hadamard_mult_warp<kLogWarpSize, 0, kNChunks, kNElts>(x_vals);

    if constexpr (kNWarps > 1) {
        exchange_smem_pre_cast<kNChunks, kChunksPerExchange, kNElts, kWarpSize, kNWarps, true, input_t, vec_t>(x_vals, smem_exchange);
        hadamard_mult_warp<kLogNWarps, 0, kNChunks, kNElts>(x_vals);
        exchange_smem_pre_cast<kNChunks, kChunksPerExchange, kNElts, kWarpSize, kNWarps, false, input_t, vec_t>(x_vals, smem_exchange);
    }

    if constexpr (kNChunks > 1) {
        float x_vals_transposed[kNElts][kNChunks];
        #pragma unroll
        for (int c = 0; c < kNChunks; ++c) {
            #pragma unroll
            for (int i = 0; i < kNElts; ++i) { x_vals_transposed[i][c] = x_vals[c][i]; }
        }
        constexpr int kLogNChunks = cilog2(kNChunks);
        static_assert(1 << kLogNChunks == kNChunks, "kNChunks must be a power of 2");
        hadamard_mult_thread<kLogNChunks, kNElts>(x_vals_transposed);
        #pragma unroll
        for (int c = 0; c < kNChunks; ++c) {
            #pragma unroll
            for (int i = 0; i < kNElts; ++i) { x_vals[c][i] = x_vals_transposed[i][c]; }
        }
    }

    store_output<kNChunks, kNElts, input_t>(out, x_vals, params.dim, params.scale);
}

template<typename Ktraits>
__global__ __launch_bounds__(Ktraits::kNThreads)
void fast_hadamard_transform_lowp_exchange_kernel(HadamardParamsBase params) {
    fast_hadamard_transform_lowp_exchange_kernel_body<Ktraits>(params);
}

template <int kNChunks>
__device__ __forceinline__ void hadamard_mult_thread_half2_8(__half2 x[kNChunks][4]) {
    #pragma unroll
    for (int c = 0; c < kNChunks; ++c) {
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            const __half lo = __low2half(x[c][p]);
            const __half hi = __high2half(x[c][p]);
            x[c][p] = __halves2half2(__hadd(lo, hi), __hsub(lo, hi));
        }

        __half2 a = x[c][0];
        __half2 b = x[c][1];
        x[c][0] = __hadd2(a, b);
        x[c][1] = __hsub2(a, b);
        a = x[c][2];
        b = x[c][3];
        x[c][2] = __hadd2(a, b);
        x[c][3] = __hsub2(a, b);

        a = x[c][0];
        b = x[c][2];
        x[c][0] = __hadd2(a, b);
        x[c][2] = __hsub2(a, b);
        a = x[c][1];
        b = x[c][3];
        x[c][1] = __hadd2(a, b);
        x[c][3] = __hsub2(a, b);
    }
}

template<int kLogWarpSize, int kStepStart, int kNChunks>
__device__ __forceinline__ void hadamard_mult_warp_half2(__half2 x[kNChunks][4]) {
    constexpr int N = 1 << kLogWarpSize;
    const int lane_id = threadIdx.x % N;
    #pragma unroll
    for (int step = kStepStart; step < kLogWarpSize; ++step) {
        const int lane_mask = 1 << step;
        const bool subtract = lane_id & lane_mask;
        #pragma unroll
        for (int c = 0; c < kNChunks; ++c) {
            #pragma unroll
            for (int p = 0; p < 4; ++p) {
                const __half2 other = __shfl_xor_sync(FULL_MASK, x[c][p], lane_mask);
                x[c][p] = subtract ? __hsub2(other, x[c][p]) : __hadd2(x[c][p], other);
            }
        }
    }
}

template<int kNChunks>
__device__ __forceinline__ void hadamard_mult_chunks_half2(__half2 x[kNChunks][4]) {
    constexpr int kLogNChunks = cilog2(kNChunks);
    static_assert(1 << kLogNChunks == kNChunks, "kNChunks must be a power of 2");
    #pragma unroll
    for (int step = 0; step < kLogNChunks; ++step) {
        constexpr int kNPairs = 4;
        const int stride = 1 << step;
        #pragma unroll
        for (int j = 0; j < kNChunks / 2; ++j) {
            const int lo = j & (stride - 1);
            const int idx = (j - lo) * 2 + lo;
            #pragma unroll
            for (int p = 0; p < kNPairs; ++p) {
                const __half2 a = x[idx][p];
                const __half2 b = x[idx + stride][p];
                x[idx][p] = __hadd2(a, b);
                x[idx + stride][p] = __hsub2(a, b);
            }
        }
    }
}

template <int kNChunks, int kNElts>
inline __device__ void load_input_half2(__half *x, __half2 x_vals[kNChunks][4], int dim) {
    static_assert(kNElts == 8);
    using vec_t = typename BytesToType<sizeof(__half) * kNElts>::Type;
    __half x_vals_load[kNChunks][kNElts] = {__float2half(0.f)};
    #pragma unroll
    for (int c = 0; c < kNChunks; ++c) {
        if ((c * blockDim.x + threadIdx.x) * kNElts < dim) {
            reinterpret_cast<vec_t*>(x_vals_load)[c] = reinterpret_cast<const vec_t*>(x)[c * blockDim.x + threadIdx.x];
        }
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            x_vals[c][p] = __halves2half2(x_vals_load[c][2 * p], x_vals_load[c][2 * p + 1]);
        }
    }
}

template <int kNChunks, int kNElts>
inline __device__ void store_output_half2(__half *out, __half2 out_vals[kNChunks][4], int dim, float scale=1.f) {
    static_assert(kNElts == 8);
    using vec_t = typename BytesToType<sizeof(__half) * kNElts>::Type;
    const __half2 scale2 = __float2half2_rn(scale);
    __half out_vals_store[kNChunks][kNElts];
    #pragma unroll
    for (int c = 0; c < kNChunks; ++c) {
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            const __half2 val = __hmul2(out_vals[c][p], scale2);
            out_vals_store[c][2 * p] = __low2half(val);
            out_vals_store[c][2 * p + 1] = __high2half(val);
        }
    }
    #pragma unroll
    for (int c = 0; c < kNChunks; ++c) {
        if ((c * blockDim.x + threadIdx.x) * kNElts < dim) {
            reinterpret_cast<vec_t*>(out)[c * blockDim.x + threadIdx.x] = reinterpret_cast<const vec_t*>(out_vals_store)[c];
        }
    }
}

template <int kNChunks, int kNElts>
inline __device__ void load_input_half2_warp(__half *x, __half2 x_vals[kNChunks][4], int dim, int lane_id) {
    static_assert(kNElts == 8);
    using vec_t = typename BytesToType<sizeof(__half) * kNElts>::Type;
    __half x_vals_load[kNChunks][kNElts] = {__float2half(0.f)};
    #pragma unroll
    for (int c = 0; c < kNChunks; ++c) {
        if ((c * 32 + lane_id) * kNElts < dim) {
            reinterpret_cast<vec_t*>(x_vals_load)[c] = reinterpret_cast<const vec_t*>(x)[c * 32 + lane_id];
        }
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            x_vals[c][p] = __halves2half2(x_vals_load[c][2 * p], x_vals_load[c][2 * p + 1]);
        }
    }
}

template <int kNChunks, int kNElts>
inline __device__ void store_output_half2_warp(__half *out, __half2 out_vals[kNChunks][4], int dim, int lane_id, float scale=1.f) {
    static_assert(kNElts == 8);
    using vec_t = typename BytesToType<sizeof(__half) * kNElts>::Type;
    const __half2 scale2 = __float2half2_rn(scale);
    __half out_vals_store[kNChunks][kNElts];
    #pragma unroll
    for (int c = 0; c < kNChunks; ++c) {
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            const __half2 val = __hmul2(out_vals[c][p], scale2);
            out_vals_store[c][2 * p] = __low2half(val);
            out_vals_store[c][2 * p + 1] = __high2half(val);
        }
    }
    #pragma unroll
    for (int c = 0; c < kNChunks; ++c) {
        if ((c * 32 + lane_id) * kNElts < dim) {
            reinterpret_cast<vec_t*>(out)[c * 32 + lane_id] = reinterpret_cast<const vec_t*>(out_vals_store)[c];
        }
    }
}

template <int kNChunks, int kChunksPerExchange, int kWarpSize, int kNWarps, bool Pre, typename vec_t>
inline __device__ void exchange_smem_half2(__half2 x_vals[kNChunks][4], vec_t *smem) {
    constexpr int kNThreads = kWarpSize * kNWarps;
    const int warp_id = threadIdx.x / kWarpSize;
    const int lane_id = threadIdx.x % kWarpSize;
    const int row_t = threadIdx.x % kNWarps;
    const int col_t = threadIdx.x / kNWarps;
    #pragma unroll
    for (int c0 = 0; c0 < kNChunks / kChunksPerExchange; ++c0) {
        __syncthreads();
        #pragma unroll
        for (int c1 = 0; c1 < kChunksPerExchange; ++c1) {
            smem[c1 * kNThreads + (Pre ? warp_id * kWarpSize + lane_id ^ warp_id : row_t * kWarpSize + col_t ^ row_t)] =
                reinterpret_cast<vec_t*>(x_vals[c0 * kChunksPerExchange + c1])[0];
        }
        __syncthreads();
        #pragma unroll
        for (int c1 = 0; c1 < kChunksPerExchange; ++c1) {
            reinterpret_cast<vec_t*>(x_vals[c0 * kChunksPerExchange + c1])[0] =
                smem[c1 * kNThreads + (Pre ? row_t * kWarpSize + col_t ^ row_t : warp_id * kWarpSize + lane_id ^ warp_id)];
        }
    }
}

template<typename Ktraits>
__global__ __launch_bounds__(Ktraits::kNThreads)
void fast_hadamard_transform_half2_kernel(HadamardParamsBase params) {
    constexpr int kNThreads = Ktraits::kNThreads;
    constexpr int kNElts = Ktraits::kNElts;
    constexpr int kNChunks = Ktraits::kNChunks;
    using vec_t = typename Ktraits::vec_t;

    static_assert(kNElts == 8);
    constexpr int kWarpSize = std::min(kNThreads, 32);
    constexpr int kLogWarpSize = cilog2(kWarpSize);
    static_assert(1 << kLogWarpSize == kWarpSize, "Warp size must be a power of 2");
    constexpr int kNWarps = kNThreads / kWarpSize;
    constexpr int kLogNWarps = cilog2(kNWarps);
    static_assert(1 << kLogNWarps == kNWarps, "kNWarps must be a power of 2");

    constexpr int kChunksPerExchange = Ktraits::kSmemExchangeSize / (sizeof(vec_t) * kNThreads);
    static_assert(kChunksPerExchange * sizeof(vec_t) * kNThreads == Ktraits::kSmemExchangeSize);
    constexpr int kNExchanges = kNChunks / kChunksPerExchange;
    static_assert(kNExchanges * kChunksPerExchange == kNChunks);

    extern __shared__ char smem_[];
    vec_t *smem_exchange = reinterpret_cast<vec_t *>(smem_);

    const int batch_id = blockIdx.x;
    __half *x = reinterpret_cast<__half *>(params.x_ptr) + batch_id * params.x_batch_stride;
    __half *out = reinterpret_cast<__half *>(params.out_ptr) + batch_id * params.out_batch_stride;

    __half2 x_vals[kNChunks][4];
    load_input_half2<kNChunks, kNElts>(x, x_vals, params.dim);

    hadamard_mult_thread_half2_8<kNChunks>(x_vals);
    hadamard_mult_warp_half2<kLogWarpSize, 0, kNChunks>(x_vals);

    if constexpr (kNWarps > 1) {
        exchange_smem_half2<kNChunks, kChunksPerExchange, kWarpSize, kNWarps, true, vec_t>(x_vals, smem_exchange);
        hadamard_mult_warp_half2<kLogNWarps, 0, kNChunks>(x_vals);
        exchange_smem_half2<kNChunks, kChunksPerExchange, kWarpSize, kNWarps, false, vec_t>(x_vals, smem_exchange);
    }

    if constexpr (kNChunks > 1) {
        hadamard_mult_chunks_half2<kNChunks>(x_vals);
    }

    store_output_half2<kNChunks, kNElts>(out, x_vals, params.dim, params.scale);
}

template<int kNThreads, int kLogN>
void fast_hadamard_transform_half2_launch(HadamardParamsBase &params, cudaStream_t stream) {
    using Ktraits = fast_hadamard_transform_lowp_exchange_kernel_traits<kNThreads, kLogN, at::Half>;
    constexpr int kSmemSize = Ktraits::kSmemSize;
    dim3 grid(params.batch);
    auto kernel = &fast_hadamard_transform_half2_kernel<Ktraits>;
    if (kSmemSize >= 48 * 1024) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
    }
    kernel<<<grid, Ktraits::kNThreads, kSmemSize, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<typename Ktraits, int kRowsPerBlock>
__global__ __launch_bounds__(32 * kRowsPerBlock)
void fast_hadamard_transform_half2_one_warp_kernel(HadamardParamsBase params) {
    static_assert(Ktraits::kNThreads == 32);
    constexpr int kNElts = Ktraits::kNElts;
    constexpr int kNChunks = Ktraits::kNChunks;

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int batch_id = blockIdx.x * kRowsPerBlock + warp_id;
    if (batch_id >= params.batch) { return; }

    __half *x = reinterpret_cast<__half *>(params.x_ptr) + batch_id * params.x_batch_stride;
    __half *out = reinterpret_cast<__half *>(params.out_ptr) + batch_id * params.out_batch_stride;

    __half2 x_vals[kNChunks][4];
    load_input_half2_warp<kNChunks, kNElts>(x, x_vals, params.dim, lane_id);

    hadamard_mult_thread_half2_8<kNChunks>(x_vals);
    hadamard_mult_warp_half2<5, 0, kNChunks>(x_vals);

    if constexpr (kNChunks > 1) {
        hadamard_mult_chunks_half2<kNChunks>(x_vals);
    }

    store_output_half2_warp<kNChunks, kNElts>(out, x_vals, params.dim, lane_id, params.scale);
}

template<int kRowsPerBlock, int kLogN>
void fast_hadamard_transform_half2_one_warp_launch(HadamardParamsBase &params, cudaStream_t stream) {
    using Ktraits = fast_hadamard_transform_lowp_exchange_kernel_traits<32, kLogN, at::Half>;
    dim3 grid((params.batch + kRowsPerBlock - 1) / kRowsPerBlock);
    fast_hadamard_transform_half2_one_warp_kernel<Ktraits, kRowsPerBlock><<<grid, 32 * kRowsPerBlock, 0, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int kNChunks>
__device__ __forceinline__ void hadamard_mult_thread_bfloat162_8(__nv_bfloat162 x[kNChunks][4]) {
    #pragma unroll
    for (int c = 0; c < kNChunks; ++c) {
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            const __nv_bfloat16 lo = __low2bfloat16(x[c][p]);
            const __nv_bfloat16 hi = __high2bfloat16(x[c][p]);
            x[c][p] = __halves2bfloat162(__hadd(lo, hi), __hsub(lo, hi));
        }

        __nv_bfloat162 a = x[c][0];
        __nv_bfloat162 b = x[c][1];
        x[c][0] = __hadd2(a, b);
        x[c][1] = __hsub2(a, b);
        a = x[c][2];
        b = x[c][3];
        x[c][2] = __hadd2(a, b);
        x[c][3] = __hsub2(a, b);

        a = x[c][0];
        b = x[c][2];
        x[c][0] = __hadd2(a, b);
        x[c][2] = __hsub2(a, b);
        a = x[c][1];
        b = x[c][3];
        x[c][1] = __hadd2(a, b);
        x[c][3] = __hsub2(a, b);
    }
}

template<int kLogWarpSize, int kStepStart, int kNChunks>
__device__ __forceinline__ void hadamard_mult_warp_bfloat162(__nv_bfloat162 x[kNChunks][4]) {
    constexpr int N = 1 << kLogWarpSize;
    const int lane_id = threadIdx.x % N;
    #pragma unroll
    for (int step = kStepStart; step < kLogWarpSize; ++step) {
        const int lane_mask = 1 << step;
        const bool subtract = lane_id & lane_mask;
        #pragma unroll
        for (int c = 0; c < kNChunks; ++c) {
            #pragma unroll
            for (int p = 0; p < 4; ++p) {
                const __nv_bfloat162 other = __shfl_xor_sync(FULL_MASK, x[c][p], lane_mask);
                x[c][p] = subtract ? __hsub2(other, x[c][p]) : __hadd2(x[c][p], other);
            }
        }
    }
}

template<int kNChunks>
__device__ __forceinline__ void hadamard_mult_chunks_bfloat162(__nv_bfloat162 x[kNChunks][4]) {
    constexpr int kLogNChunks = cilog2(kNChunks);
    static_assert(1 << kLogNChunks == kNChunks, "kNChunks must be a power of 2");
    #pragma unroll
    for (int step = 0; step < kLogNChunks; ++step) {
        constexpr int kNPairs = 4;
        const int stride = 1 << step;
        #pragma unroll
        for (int j = 0; j < kNChunks / 2; ++j) {
            const int lo = j & (stride - 1);
            const int idx = (j - lo) * 2 + lo;
            #pragma unroll
            for (int p = 0; p < kNPairs; ++p) {
                const __nv_bfloat162 a = x[idx][p];
                const __nv_bfloat162 b = x[idx + stride][p];
                x[idx][p] = __hadd2(a, b);
                x[idx + stride][p] = __hsub2(a, b);
            }
        }
    }
}

template <int kNChunks, int kNElts>
inline __device__ void load_input_bfloat162(__nv_bfloat16 *x, __nv_bfloat162 x_vals[kNChunks][4], int dim) {
    static_assert(kNElts == 8);
    using vec_t = typename BytesToType<sizeof(__nv_bfloat16) * kNElts>::Type;
    __nv_bfloat16 x_vals_load[kNChunks][kNElts] = {__float2bfloat16_rn(0.f)};
    #pragma unroll
    for (int c = 0; c < kNChunks; ++c) {
        if ((c * blockDim.x + threadIdx.x) * kNElts < dim) {
            reinterpret_cast<vec_t*>(x_vals_load)[c] = reinterpret_cast<const vec_t*>(x)[c * blockDim.x + threadIdx.x];
        }
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            x_vals[c][p] = __halves2bfloat162(x_vals_load[c][2 * p], x_vals_load[c][2 * p + 1]);
        }
    }
}

template <int kNChunks, int kNElts>
inline __device__ void store_output_bfloat162(__nv_bfloat16 *out, __nv_bfloat162 out_vals[kNChunks][4], int dim, float scale=1.f) {
    static_assert(kNElts == 8);
    using vec_t = typename BytesToType<sizeof(__nv_bfloat16) * kNElts>::Type;
    const __nv_bfloat162 scale2 = __float2bfloat162_rn(scale);
    __nv_bfloat16 out_vals_store[kNChunks][kNElts];
    #pragma unroll
    for (int c = 0; c < kNChunks; ++c) {
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            const __nv_bfloat162 val = __hmul2(out_vals[c][p], scale2);
            out_vals_store[c][2 * p] = __low2bfloat16(val);
            out_vals_store[c][2 * p + 1] = __high2bfloat16(val);
        }
    }
    #pragma unroll
    for (int c = 0; c < kNChunks; ++c) {
        if ((c * blockDim.x + threadIdx.x) * kNElts < dim) {
            reinterpret_cast<vec_t*>(out)[c * blockDim.x + threadIdx.x] = reinterpret_cast<const vec_t*>(out_vals_store)[c];
        }
    }
}

template <int kNChunks, int kNElts>
inline __device__ void load_input_bfloat162_warp(__nv_bfloat16 *x, __nv_bfloat162 x_vals[kNChunks][4], int dim, int lane_id) {
    static_assert(kNElts == 8);
    using vec_t = typename BytesToType<sizeof(__nv_bfloat16) * kNElts>::Type;
    __nv_bfloat16 x_vals_load[kNChunks][kNElts] = {__float2bfloat16_rn(0.f)};
    #pragma unroll
    for (int c = 0; c < kNChunks; ++c) {
        if ((c * 32 + lane_id) * kNElts < dim) {
            reinterpret_cast<vec_t*>(x_vals_load)[c] = reinterpret_cast<const vec_t*>(x)[c * 32 + lane_id];
        }
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            x_vals[c][p] = __halves2bfloat162(x_vals_load[c][2 * p], x_vals_load[c][2 * p + 1]);
        }
    }
}

template <int kNChunks, int kNElts>
inline __device__ void store_output_bfloat162_warp(__nv_bfloat16 *out, __nv_bfloat162 out_vals[kNChunks][4], int dim, int lane_id, float scale=1.f) {
    static_assert(kNElts == 8);
    using vec_t = typename BytesToType<sizeof(__nv_bfloat16) * kNElts>::Type;
    const __nv_bfloat162 scale2 = __float2bfloat162_rn(scale);
    __nv_bfloat16 out_vals_store[kNChunks][kNElts];
    #pragma unroll
    for (int c = 0; c < kNChunks; ++c) {
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            const __nv_bfloat162 val = __hmul2(out_vals[c][p], scale2);
            out_vals_store[c][2 * p] = __low2bfloat16(val);
            out_vals_store[c][2 * p + 1] = __high2bfloat16(val);
        }
    }
    #pragma unroll
    for (int c = 0; c < kNChunks; ++c) {
        if ((c * 32 + lane_id) * kNElts < dim) {
            reinterpret_cast<vec_t*>(out)[c * 32 + lane_id] = reinterpret_cast<const vec_t*>(out_vals_store)[c];
        }
    }
}

template <int kNChunks, int kChunksPerExchange, int kWarpSize, int kNWarps, bool Pre, typename vec_t>
inline __device__ void exchange_smem_bfloat162(__nv_bfloat162 x_vals[kNChunks][4], vec_t *smem) {
    constexpr int kNThreads = kWarpSize * kNWarps;
    const int warp_id = threadIdx.x / kWarpSize;
    const int lane_id = threadIdx.x % kWarpSize;
    const int row_t = threadIdx.x % kNWarps;
    const int col_t = threadIdx.x / kNWarps;
    #pragma unroll
    for (int c0 = 0; c0 < kNChunks / kChunksPerExchange; ++c0) {
        __syncthreads();
        #pragma unroll
        for (int c1 = 0; c1 < kChunksPerExchange; ++c1) {
            smem[c1 * kNThreads + (Pre ? warp_id * kWarpSize + lane_id ^ warp_id : row_t * kWarpSize + col_t ^ row_t)] =
                reinterpret_cast<vec_t*>(x_vals[c0 * kChunksPerExchange + c1])[0];
        }
        __syncthreads();
        #pragma unroll
        for (int c1 = 0; c1 < kChunksPerExchange; ++c1) {
            reinterpret_cast<vec_t*>(x_vals[c0 * kChunksPerExchange + c1])[0] =
                smem[c1 * kNThreads + (Pre ? row_t * kWarpSize + col_t ^ row_t : warp_id * kWarpSize + lane_id ^ warp_id)];
        }
    }
}

template<typename Ktraits>
__global__ __launch_bounds__(Ktraits::kNThreads)
void fast_hadamard_transform_bfloat162_kernel(HadamardParamsBase params) {
    constexpr int kNThreads = Ktraits::kNThreads;
    constexpr int kNElts = Ktraits::kNElts;
    constexpr int kNChunks = Ktraits::kNChunks;
    using vec_t = typename Ktraits::vec_t;

    static_assert(kNElts == 8);
    constexpr int kWarpSize = std::min(kNThreads, 32);
    constexpr int kLogWarpSize = cilog2(kWarpSize);
    static_assert(1 << kLogWarpSize == kWarpSize, "Warp size must be a power of 2");
    constexpr int kNWarps = kNThreads / kWarpSize;
    constexpr int kLogNWarps = cilog2(kNWarps);
    static_assert(1 << kLogNWarps == kNWarps, "kNWarps must be a power of 2");

    constexpr int kChunksPerExchange = Ktraits::kSmemExchangeSize / (sizeof(vec_t) * kNThreads);
    static_assert(kChunksPerExchange * sizeof(vec_t) * kNThreads == Ktraits::kSmemExchangeSize);
    constexpr int kNExchanges = kNChunks / kChunksPerExchange;
    static_assert(kNExchanges * kChunksPerExchange == kNChunks);

    extern __shared__ char smem_[];
    vec_t *smem_exchange = reinterpret_cast<vec_t *>(smem_);

    const int batch_id = blockIdx.x;
    __nv_bfloat16 *x = reinterpret_cast<__nv_bfloat16 *>(params.x_ptr) + batch_id * params.x_batch_stride;
    __nv_bfloat16 *out = reinterpret_cast<__nv_bfloat16 *>(params.out_ptr) + batch_id * params.out_batch_stride;

    __nv_bfloat162 x_vals[kNChunks][4];
    load_input_bfloat162<kNChunks, kNElts>(x, x_vals, params.dim);

    hadamard_mult_thread_bfloat162_8<kNChunks>(x_vals);
    hadamard_mult_warp_bfloat162<kLogWarpSize, 0, kNChunks>(x_vals);

    if constexpr (kNWarps > 1) {
        exchange_smem_bfloat162<kNChunks, kChunksPerExchange, kWarpSize, kNWarps, true, vec_t>(x_vals, smem_exchange);
        hadamard_mult_warp_bfloat162<kLogNWarps, 0, kNChunks>(x_vals);
        exchange_smem_bfloat162<kNChunks, kChunksPerExchange, kWarpSize, kNWarps, false, vec_t>(x_vals, smem_exchange);
    }

    if constexpr (kNChunks > 1) {
        hadamard_mult_chunks_bfloat162<kNChunks>(x_vals);
    }

    store_output_bfloat162<kNChunks, kNElts>(out, x_vals, params.dim, params.scale);
}

template<int kNThreads, int kLogN>
void fast_hadamard_transform_bfloat162_launch(HadamardParamsBase &params, cudaStream_t stream) {
    using Ktraits = fast_hadamard_transform_lowp_exchange_kernel_traits<kNThreads, kLogN, at::BFloat16>;
    constexpr int kSmemSize = Ktraits::kSmemSize;
    dim3 grid(params.batch);
    auto kernel = &fast_hadamard_transform_bfloat162_kernel<Ktraits>;
    if (kSmemSize >= 48 * 1024) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
    }
    kernel<<<grid, Ktraits::kNThreads, kSmemSize, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<typename Ktraits, int kRowsPerBlock>
__global__ __launch_bounds__(32 * kRowsPerBlock)
void fast_hadamard_transform_bfloat162_one_warp_kernel(HadamardParamsBase params) {
    static_assert(Ktraits::kNThreads == 32);
    constexpr int kNElts = Ktraits::kNElts;
    constexpr int kNChunks = Ktraits::kNChunks;

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int batch_id = blockIdx.x * kRowsPerBlock + warp_id;
    if (batch_id >= params.batch) { return; }

    __nv_bfloat16 *x = reinterpret_cast<__nv_bfloat16 *>(params.x_ptr) + batch_id * params.x_batch_stride;
    __nv_bfloat16 *out = reinterpret_cast<__nv_bfloat16 *>(params.out_ptr) + batch_id * params.out_batch_stride;

    __nv_bfloat162 x_vals[kNChunks][4];
    load_input_bfloat162_warp<kNChunks, kNElts>(x, x_vals, params.dim, lane_id);

    hadamard_mult_thread_bfloat162_8<kNChunks>(x_vals);
    hadamard_mult_warp_bfloat162<5, 0, kNChunks>(x_vals);

    if constexpr (kNChunks > 1) {
        hadamard_mult_chunks_bfloat162<kNChunks>(x_vals);
    }

    store_output_bfloat162_warp<kNChunks, kNElts>(out, x_vals, params.dim, lane_id, params.scale);
}

template<int kRowsPerBlock, int kLogN>
void fast_hadamard_transform_bfloat162_one_warp_launch(HadamardParamsBase &params, cudaStream_t stream) {
    using Ktraits = fast_hadamard_transform_lowp_exchange_kernel_traits<32, kLogN, at::BFloat16>;
    dim3 grid((params.batch + kRowsPerBlock - 1) / kRowsPerBlock);
    fast_hadamard_transform_bfloat162_one_warp_kernel<Ktraits, kRowsPerBlock><<<grid, 32 * kRowsPerBlock, 0, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<typename Ktraits, int kRowsPerBlock>
__global__ __launch_bounds__(32 * kRowsPerBlock)
void fast_hadamard_transform_one_warp_kernel(HadamardParamsBase params) {
    static_assert(Ktraits::kNThreads == 32);
    constexpr int kNElts = Ktraits::kNElts;
    constexpr int kNChunks = Ktraits::kNChunks;
    using input_t = typename Ktraits::input_t;

    constexpr int kLogNElts = cilog2(Ktraits::kNElts);
    static_assert(1 << kLogNElts == kNElts, "kNElts must be a power of 2");

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int batch_id = blockIdx.x * kRowsPerBlock + warp_id;
    if (batch_id >= params.batch) { return; }

    input_t *x = reinterpret_cast<input_t *>(params.x_ptr) + batch_id * params.x_batch_stride;
    input_t *out = reinterpret_cast<input_t *>(params.out_ptr) + batch_id * params.out_batch_stride;

    float x_vals[kNChunks][kNElts];
    load_input_warp<kNChunks, kNElts, input_t>(x, x_vals, params.dim, lane_id);

    hadamard_mult_thread<kLogNElts, kNChunks>(x_vals);
    hadamard_mult_warp<5, 0, kNChunks, kNElts>(x_vals);

    if constexpr (kNChunks > 1) {
        float x_vals_transposed[kNElts][kNChunks];
        #pragma unroll
        for (int c = 0; c < kNChunks; ++c) {
            #pragma unroll
            for (int i = 0; i < kNElts; ++i) { x_vals_transposed[i][c] = x_vals[c][i]; }
        }
        constexpr int kLogNChunks = cilog2(kNChunks);
        static_assert(1 << kLogNChunks == kNChunks, "kNChunks must be a power of 2");
        hadamard_mult_thread<kLogNChunks, kNElts>(x_vals_transposed);
        #pragma unroll
        for (int c = 0; c < kNChunks; ++c) {
            #pragma unroll
            for (int i = 0; i < kNElts; ++i) { x_vals[c][i] = x_vals_transposed[i][c]; }
        }
    }

    store_output_warp<kNChunks, kNElts, input_t>(out, x_vals, params.dim, lane_id, params.scale);
}

template<int kNThreads, int kLogN, typename input_t, bool kUseConditionalWarp = false>
void fast_hadamard_transform_launch(HadamardParamsBase &params, cudaStream_t stream) {
    using Ktraits = fast_hadamard_transform_kernel_traits<kNThreads, kLogN, input_t>;
    constexpr int kSmemSize = Ktraits::kSmemSize;
    dim3 grid(params.batch);
    auto kernel = &fast_hadamard_transform_kernel<Ktraits, kUseConditionalWarp>;
    if (kSmemSize >= 48 * 1024) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
    }
    kernel<<<grid, Ktraits::kNThreads, kSmemSize, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int kNThreads, int kLogN, typename input_t>
void fast_hadamard_transform_lowp_exchange_launch(HadamardParamsBase &params, cudaStream_t stream) {
    using Ktraits = fast_hadamard_transform_lowp_exchange_kernel_traits<kNThreads, kLogN, input_t>;
    constexpr int kSmemSize = Ktraits::kSmemSize;
    dim3 grid(params.batch);
    auto kernel = &fast_hadamard_transform_lowp_exchange_kernel<Ktraits>;
    if (kSmemSize >= 48 * 1024) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
    }
    kernel<<<grid, Ktraits::kNThreads, kSmemSize, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int kRowsPerBlock, int kLogN, typename input_t>
void fast_hadamard_transform_one_warp_launch(HadamardParamsBase &params, cudaStream_t stream) {
    using Ktraits = fast_hadamard_transform_kernel_traits<32, kLogN, input_t>;
    dim3 grid((params.batch + kRowsPerBlock - 1) / kRowsPerBlock);
    fast_hadamard_transform_one_warp_kernel<Ktraits, kRowsPerBlock><<<grid, 32 * kRowsPerBlock, 0, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<typename input_t>
void fast_hadamard_transform_cuda(HadamardParamsBase &params, cudaStream_t stream) {
    if (params.log_N == 3) {
        fast_hadamard_transform_launch<1, 3, input_t>(params, stream);
    } else if (params.log_N == 4) {
        fast_hadamard_transform_launch<2, 4, input_t>(params, stream);
    } else if (params.log_N == 5) {
        fast_hadamard_transform_launch<4, 5, input_t>(params, stream);
    } else if (params.log_N == 6) {
        fast_hadamard_transform_launch<8, 6, input_t>(params, stream);
    } else if (params.log_N == 7) {
        fast_hadamard_transform_launch<16, 7, input_t>(params, stream);
    } else if (params.log_N == 8) {
        if constexpr (std::is_same_v<input_t, at::Half>) {
            if (params.fast_low_precision) {
                fast_hadamard_transform_half2_one_warp_launch<4, 8>(params, stream);
            } else {
                fast_hadamard_transform_one_warp_launch<8, 8, input_t>(params, stream);
            }
        } else if constexpr (std::is_same_v<input_t, at::BFloat16>) {
            if (params.fast_low_precision) {
                fast_hadamard_transform_bfloat162_one_warp_launch<4, 8>(params, stream);
            } else {
                fast_hadamard_transform_one_warp_launch<8, 8, input_t>(params, stream);
            }
        } else {
            fast_hadamard_transform_one_warp_launch<8, 8, input_t>(params, stream);
        }
    } else if (params.log_N == 9) {
        if constexpr (std::is_same_v<input_t, float>) {
            fast_hadamard_transform_launch<32, 9, input_t>(params, stream);
        } else if constexpr (std::is_same_v<input_t, at::Half>) {
            if (params.fast_low_precision) {
                fast_hadamard_transform_half2_one_warp_launch<4, 9>(params, stream);
            } else {
                fast_hadamard_transform_one_warp_launch<8, 9, input_t>(params, stream);
            }
        } else if constexpr (std::is_same_v<input_t, at::BFloat16>) {
            if (params.fast_low_precision) {
                fast_hadamard_transform_bfloat162_one_warp_launch<4, 9>(params, stream);
            } else {
                fast_hadamard_transform_one_warp_launch<8, 9, input_t>(params, stream);
            }
        } else {
            fast_hadamard_transform_one_warp_launch<8, 9, input_t>(params, stream);
        }
    } else if (params.log_N == 10) {
        if constexpr (std::is_same_v<input_t, float>) {
            fast_hadamard_transform_launch<32, 10, input_t>(params, stream);
        } else if constexpr (std::is_same_v<input_t, at::Half>) {
            if (params.fast_low_precision) {
                fast_hadamard_transform_half2_one_warp_launch<1, 10>(params, stream);
            } else {
                fast_hadamard_transform_one_warp_launch<8, 10, input_t>(params, stream);
            }
        } else if constexpr (std::is_same_v<input_t, at::BFloat16>) {
            if (params.fast_low_precision) {
                fast_hadamard_transform_bfloat162_one_warp_launch<2, 10>(params, stream);
            } else {
                fast_hadamard_transform_one_warp_launch<8, 10, input_t>(params, stream);
            }
        } else {
            fast_hadamard_transform_one_warp_launch<8, 10, input_t>(params, stream);
        }
    } else if (params.log_N == 11) {
        if constexpr (std::is_same_v<input_t, at::Half>) {
            if (params.fast_low_precision) {
                fast_hadamard_transform_half2_one_warp_launch<4, 11>(params, stream);
            } else {
                fast_hadamard_transform_one_warp_launch<4, 11, input_t>(params, stream);
            }
        } else if constexpr (std::is_same_v<input_t, at::BFloat16>) {
            if (params.fast_low_precision) {
                fast_hadamard_transform_bfloat162_one_warp_launch<8, 11>(params, stream);
            } else {
                fast_hadamard_transform_one_warp_launch<8, 11, input_t>(params, stream);
            }
        } else {
            fast_hadamard_transform_one_warp_launch<4, 11, input_t>(params, stream);
        }
    } else if (params.log_N == 12) {
        if constexpr (std::is_same_v<input_t, float>) {
            fast_hadamard_transform_launch<64, 12, input_t>(params, stream);
        } else if constexpr (std::is_same_v<input_t, at::Half>) {
            if (params.fast_low_precision) {
                fast_hadamard_transform_half2_launch<256, 12>(params, stream);
            } else {
                fast_hadamard_transform_launch<256, 12, input_t>(params, stream);
            }
        } else if constexpr (std::is_same_v<input_t, at::BFloat16>) {
            if (params.fast_low_precision) {
                fast_hadamard_transform_bfloat162_launch<256, 12>(params, stream);
            } else {
                fast_hadamard_transform_launch<256, 12, input_t>(params, stream);
            }
        } else {
            fast_hadamard_transform_lowp_exchange_launch<256, 12, input_t>(params, stream);
        }
    } else if (params.log_N == 13) {
        if constexpr (std::is_same_v<input_t, float>) {
            fast_hadamard_transform_launch<256, 13, input_t>(params, stream);
        } else if constexpr (std::is_same_v<input_t, at::Half>) {
            if (params.fast_low_precision) {
                fast_hadamard_transform_half2_launch<256, 13>(params, stream);
            } else {
                fast_hadamard_transform_launch<256, 13, input_t>(params, stream);
            }
        } else if constexpr (std::is_same_v<input_t, at::BFloat16>) {
            if (params.fast_low_precision) {
                fast_hadamard_transform_bfloat162_launch<256, 13>(params, stream);
            } else {
                fast_hadamard_transform_launch<256, 13, input_t>(params, stream);
            }
        } else {
            fast_hadamard_transform_lowp_exchange_launch<256, 13, input_t>(params, stream);
        }
    } else if (params.log_N == 14) {
        if constexpr (std::is_same_v<input_t, float>) {
            fast_hadamard_transform_launch<256, 14, input_t, true>(params, stream);
        } else if constexpr (std::is_same_v<input_t, at::Half>) {
            if (params.fast_low_precision) {
                fast_hadamard_transform_half2_launch<512, 14>(params, stream);
            } else {
                fast_hadamard_transform_launch<256, 14, input_t>(params, stream);
            }
        } else if constexpr (std::is_same_v<input_t, at::BFloat16>) {
            if (params.fast_low_precision) {
                fast_hadamard_transform_bfloat162_launch<256, 14>(params, stream);
            } else {
                fast_hadamard_transform_launch<256, 14, input_t>(params, stream);
            }
        } else {
            fast_hadamard_transform_lowp_exchange_launch<256, 14, input_t>(params, stream);
        }
    } else if (params.log_N == 15) {
        if constexpr (std::is_same_v<input_t, float>) {
            fast_hadamard_transform_launch<512, 15, input_t>(params, stream);
        } else if constexpr (std::is_same_v<input_t, at::Half>) {
            if (params.fast_low_precision) {
                fast_hadamard_transform_half2_launch<512, 15>(params, stream);
            } else {
                fast_hadamard_transform_launch<512, 15, input_t>(params, stream);
            }
        } else if constexpr (std::is_same_v<input_t, at::BFloat16>) {
            if (params.fast_low_precision) {
                fast_hadamard_transform_bfloat162_launch<512, 15>(params, stream);
            } else {
                fast_hadamard_transform_launch<512, 15, input_t>(params, stream);
            }
        } else {
            fast_hadamard_transform_lowp_exchange_launch<1024, 15, input_t>(params, stream);
        }
    }
}

template<int kNThreads, int kLogN, typename input_t>
void fast_hadamard_transform_12N_launch(HadamardParamsBase &params, cudaStream_t stream) {
    using Ktraits = fast_hadamard_transform_12N_kernel_traits<kNThreads, kLogN, input_t>;
    constexpr int kSmemSize = Ktraits::kSmemSize;
    dim3 grid(params.batch);
    auto kernel = &fast_hadamard_transform_kernel<Ktraits>;
    if (kSmemSize >= 48 * 1024) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
        }
    kernel<<<grid, Ktraits::kNThreads, kSmemSize, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<typename input_t>
void fast_hadamard_transform_12N_cuda(HadamardParamsBase &params, cudaStream_t stream) {
    if (params.log_N == 2) {
        fast_hadamard_transform_12N_launch<1, 2, input_t>(params, stream);
    } else if (params.log_N == 3) {
        fast_hadamard_transform_12N_launch<2, 3, input_t>(params, stream);
    } else if (params.log_N == 4) {
        fast_hadamard_transform_12N_launch<4, 4, input_t>(params, stream);
    } else if (params.log_N == 5) {
        fast_hadamard_transform_12N_launch<8, 5, input_t>(params, stream);
    } else if (params.log_N == 6) {
        fast_hadamard_transform_12N_launch<16, 6, input_t>(params, stream);
    } else if (params.log_N == 7) {
        fast_hadamard_transform_12N_launch<32, 7, input_t>(params, stream);
    } else if (params.log_N == 8) {
        fast_hadamard_transform_12N_launch<64, 8, input_t>(params, stream);
    } else if (params.log_N == 9) {
        fast_hadamard_transform_12N_launch<128, 9, input_t>(params, stream);
    } else if (params.log_N == 10) {
        fast_hadamard_transform_12N_launch<256, 10, input_t>(params, stream);
    } 
}

template<int kNThreads, int kLogN, typename input_t>
void fast_hadamard_transform_20N_launch(HadamardParamsBase &params, cudaStream_t stream) {
    using Ktraits = fast_hadamard_transform_20N_kernel_traits<kNThreads, kLogN, input_t>;
    constexpr int kSmemSize = Ktraits::kSmemSize;
    dim3 grid(params.batch);
    auto kernel = &fast_hadamard_transform_kernel<Ktraits>;
    if (kSmemSize >= 48 * 1024) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
        }
    kernel<<<grid, Ktraits::kNThreads, kSmemSize, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<typename input_t>
void fast_hadamard_transform_20N_cuda(HadamardParamsBase &params, cudaStream_t stream) {
    if (params.log_N == 2) {
        fast_hadamard_transform_20N_launch<1, 2, input_t>(params, stream);
    } else if (params.log_N == 3) {
        fast_hadamard_transform_20N_launch<2, 3, input_t>(params, stream);
    } else if (params.log_N == 4) {
        fast_hadamard_transform_20N_launch<4, 4, input_t>(params, stream);
    } else if (params.log_N == 5) {
        fast_hadamard_transform_20N_launch<8, 5, input_t>(params, stream);
    } else if (params.log_N == 6) {
        fast_hadamard_transform_20N_launch<16, 6, input_t>(params, stream);
    } else if (params.log_N == 7) {
        fast_hadamard_transform_20N_launch<32, 7, input_t>(params, stream);
    } else if (params.log_N == 8) {
        fast_hadamard_transform_20N_launch<64, 8, input_t>(params, stream);
    } else if (params.log_N == 9) {
        fast_hadamard_transform_20N_launch<128, 9, input_t>(params, stream);
    } else if (params.log_N == 10) {
        fast_hadamard_transform_20N_launch<256, 10, input_t>(params, stream);
    }
}

template<int kNThreads, int kLogN, typename input_t>
void fast_hadamard_transform_28N_launch(HadamardParamsBase &params, cudaStream_t stream) {
    using Ktraits = fast_hadamard_transform_28N_kernel_traits<kNThreads, kLogN, input_t>;
    constexpr int kSmemSize = Ktraits::kSmemSize;
    dim3 grid(params.batch);
    auto kernel = &fast_hadamard_transform_kernel<Ktraits>;
    if (kSmemSize >= 48 * 1024) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
        }
    kernel<<<grid, Ktraits::kNThreads, kSmemSize, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<typename input_t>
void fast_hadamard_transform_28N_cuda(HadamardParamsBase &params, cudaStream_t stream) {
    if (params.log_N == 2) {
        fast_hadamard_transform_28N_launch<1, 2, input_t>(params, stream);
    } else if (params.log_N == 3) {
        fast_hadamard_transform_28N_launch<2, 3, input_t>(params, stream);
    } else if (params.log_N == 4) {
        fast_hadamard_transform_28N_launch<4, 4, input_t>(params, stream);
    } else if (params.log_N == 5) {
        fast_hadamard_transform_28N_launch<8, 5, input_t>(params, stream);
    } else if (params.log_N == 6) {
        fast_hadamard_transform_28N_launch<16, 6, input_t>(params, stream);
    } else if (params.log_N == 7) {
        fast_hadamard_transform_28N_launch<32, 7, input_t>(params, stream);
    } else if (params.log_N == 8) {
        fast_hadamard_transform_28N_launch<64, 8, input_t>(params, stream);
    } else if (params.log_N == 9) {
        fast_hadamard_transform_28N_launch<128, 9, input_t>(params, stream);
    } else if (params.log_N == 10) {
        fast_hadamard_transform_28N_launch<256, 10, input_t>(params, stream);
    }
}

template<int kNThreads, int kLogN, typename input_t>
void fast_hadamard_transform_40N_launch(HadamardParamsBase &params, cudaStream_t stream) {
    using Ktraits = fast_hadamard_transform_40N_kernel_traits<kNThreads, kLogN, input_t>;
    constexpr int kSmemSize = Ktraits::kSmemSize;
    dim3 grid(params.batch);
    auto kernel = &fast_hadamard_transform_kernel<Ktraits>;
    if (kSmemSize >= 48 * 1024) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
        }
    kernel<<<grid, Ktraits::kNThreads, kSmemSize, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<typename input_t>
void fast_hadamard_transform_40N_cuda(HadamardParamsBase &params, cudaStream_t stream) {
    if (params.log_N == 2) {
        fast_hadamard_transform_40N_launch<1, 2, input_t>(params, stream);
    } else if (params.log_N == 3) {
        fast_hadamard_transform_40N_launch<2, 3, input_t>(params, stream);
    } else if (params.log_N == 4) {
        fast_hadamard_transform_40N_launch<4, 4, input_t>(params, stream);
    } else if (params.log_N == 5) {
        fast_hadamard_transform_40N_launch<8, 5, input_t>(params, stream);
    } else if (params.log_N == 6) {
        fast_hadamard_transform_40N_launch<16, 6, input_t>(params, stream);
    } else if (params.log_N == 7) {
        fast_hadamard_transform_40N_launch<32, 7, input_t>(params, stream);
    } else if (params.log_N == 8) {
        fast_hadamard_transform_40N_launch<64, 8, input_t>(params, stream);
    } else if (params.log_N == 9) {
        fast_hadamard_transform_40N_launch<128, 9, input_t>(params, stream);
    } else if (params.log_N == 10) {
        fast_hadamard_transform_40N_launch<256, 10, input_t>(params, stream);
    }
}

template void fast_hadamard_transform_cuda<float>(HadamardParamsBase &params, cudaStream_t stream);
template void fast_hadamard_transform_cuda<at::Half>(HadamardParamsBase &params, cudaStream_t stream);
template void fast_hadamard_transform_cuda<at::BFloat16>(HadamardParamsBase &params, cudaStream_t stream);

template void fast_hadamard_transform_12N_cuda<float>(HadamardParamsBase &params, cudaStream_t stream);
template void fast_hadamard_transform_12N_cuda<at::Half>(HadamardParamsBase &params, cudaStream_t stream);
template void fast_hadamard_transform_12N_cuda<at::BFloat16>(HadamardParamsBase &params, cudaStream_t stream);

template void fast_hadamard_transform_20N_cuda<float>(HadamardParamsBase &params, cudaStream_t stream);
template void fast_hadamard_transform_20N_cuda<at::Half>(HadamardParamsBase &params, cudaStream_t stream);
template void fast_hadamard_transform_20N_cuda<at::BFloat16>(HadamardParamsBase &params, cudaStream_t stream);

template void fast_hadamard_transform_28N_cuda<float>(HadamardParamsBase &params, cudaStream_t stream);
template void fast_hadamard_transform_28N_cuda<at::Half>(HadamardParamsBase &params, cudaStream_t stream);
template void fast_hadamard_transform_28N_cuda<at::BFloat16>(HadamardParamsBase &params, cudaStream_t stream);

template void fast_hadamard_transform_40N_cuda<float>(HadamardParamsBase &params, cudaStream_t stream);
template void fast_hadamard_transform_40N_cuda<at::Half>(HadamardParamsBase &params, cudaStream_t stream);
template void fast_hadamard_transform_40N_cuda<at::BFloat16>(HadamardParamsBase &params, cudaStream_t stream);
