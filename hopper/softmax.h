/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cmath>

#include <cute/tensor.hpp>

#include <cutlass/numeric_types.h>

#include "utils.h"

namespace flash {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void thread_reduce_(Tensor<Engine0, Layout0> const &tensor, Tensor<Engine1, Layout1> &summary, Operator &op) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(summary) == size<0>(tensor));
    #pragma unroll
    for (int ni = 0; ni < size<1>(tensor); ni++) {
        #pragma unroll
        for (int mi = 0; mi < size<0>(tensor); mi++) {
            summary(mi) = zero_init && ni == 0 ? tensor(mi, ni) : op(summary(mi), tensor(mi, ni));
        }
    }
}

template<typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void quad_allreduce_(Tensor<Engine0, Layout0> &dst, Tensor<Engine1, Layout1> &src, Operator &op) {
    CUTE_STATIC_ASSERT_V(size(dst) == size(src));
    #pragma unroll
    for (int i = 0; i < size(dst); i++) {
        dst(i) = Allreduce<4>::run(src(i), op);
    }
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void reduce_(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &summary, Operator &op) {
    thread_reduce_<zero_init>(tensor, summary, op);
    quad_allreduce_(summary, summary, op);
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ __forceinline__ void reduce_max(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &max){
    MaxOp<float> max_op;
    reduce_<zero_init>(tensor, max, max_op);
}

template<bool zero_init=true, bool warp_reduce=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ __forceinline__ void reduce_sum(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &sum){
    SumOp<float> sum_op;
    thread_reduce_<zero_init>(tensor, sum, sum_op);
    if constexpr (warp_reduce) { quad_allreduce_(sum, sum, sum_op); }
}

// Apply the exp to all the elements.
template <bool Scale_max=true, bool Check_inf=true, int Max_offset=0,
        typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void scale_apply_exp2(Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> const &max, const float scale) {
    // For FP8, we can subtract max by 8.0 so that the value after exp2 is in the range of [0, 256].
    // This lets us use more of the FP8 range (instead of just [0, 1]) to reduce underflow.
    static constexpr float max_offset = float(Max_offset);  // We can only template on int, not float
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        // If max is -inf, then all elements must have been -inf (possibly due to masking).
        // We don't want (-inf - (-inf)) since that would give NaN.
        const float max_scaled = Check_inf
            ? (max(mi) == -INFINITY ? 0.f : (!Scale_max ? max(mi) : max(mi) * scale) - max_offset)
            : (!Scale_max ? max(mi) : max(mi) * scale) - max_offset;
        #pragma unroll
        for (int ni = 0; ni < size<1>(tensor); ++ni)  {
            // Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
            // max * log_2(e)). This allows the compiler to use the ffma
            // instruction instead of fadd and fmul separately.
            tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled) * 448.0f;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int kNRows, int Max_offset=0>
struct Softmax {

    using TensorT = decltype(make_tensor<float>(Shape<Int<kNRows>>{}));
    TensorT row_max, row_sum;
    float const softmax_scale_log2;

    CUTLASS_DEVICE Softmax(float const softmax_scale_log2_) : softmax_scale_log2(softmax_scale_log2_) {};

    template<bool Is_first, bool Check_inf=false, typename Tensor0>
    __forceinline__ __device__ TensorT max_get_scale(Tensor0 &acc_s) {
        // Reshape acc_s from ((2, 2, V), MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, V, MMA_N))
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
        static_assert(CUTE_STATIC_V(size<0>(scores)) == kNRows);
        TensorT scores_scale;
        // return scores_scale;
        if constexpr (Is_first) {
            flash::template reduce_max</*zero_init=*/true>(scores, row_max);
            cute::fill(scores_scale, 1.f);
        } else {
            Tensor scores_max_prev = make_fragment_like(row_max);
            cute::copy(row_max, scores_max_prev);
            flash::template reduce_max</*zero_init=*/false>(scores, row_max);
            #pragma unroll
            for (int mi = 0; mi < size(row_max); ++mi) {
                float scores_max_cur = !Check_inf
                    ? row_max(mi)
                    : (row_max(mi) == -INFINITY ? 0.0f : row_max(mi));
                scores_scale(mi) = exp2f((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
                row_sum(mi) *= scores_scale(mi);
            }
        }
        return scores_scale;
    };


    __forceinline__ __device__ void print_row_max_row_sum() {
    
        for (int mi = 0; mi < size(row_max); ++mi) {
            const float row_max_val = row_max(mi);
            const float row_sum_val = row_sum(mi);
            if (row_max_val != 0.0f || row_sum_val != 0.0f) {
                CUTE_LOG("row_max_val: %f, row_sum_val: %f, mi: %d, softmax_scale_log2: %f\n", row_max_val, row_sum_val, mi, softmax_scale_log2);
            }
        }
    
    }

    template<typename Tensor0>
    __forceinline__ __device__ void _print(const Tensor0& scores, const int prefix) {
        for (int mi = 0; mi < size(row_max); ++mi) {
            const float row_max_val = row_max(mi);
            const float row_sum_val = row_sum(mi);
            if (row_max_val != 0.0f || row_sum_val != 0.0f) {
                CUTE_LOG("[%d] row_max_val: %f, row_sum_val: %f, mi: %d, softmax_scale_log2: %f\n", prefix, row_max_val, row_sum_val, mi, softmax_scale_log2);
            }
        }

        for (int mi = 0; mi < size(scores); ++mi) {
            for (int ni = 0; ni < size<1>(scores); ++ni) {
                const float score_val = scores(mi, ni);
                if (score_val != 0.0f && score_val != -INFINITY && score_val != 256.0f) {
                    CUTE_LOG("[%d] score_val: %f, mi: %d, ni: %d\n", prefix, score_val, mi, ni);
                }
            }
        }
    }


    template<bool Is_first, bool Check_inf=false, typename Tensor0>
    __forceinline__ __device__ void online_softmax(Tensor0 &acc_s) {
        // return;
        // Reshape acc_s from ((2, 2, V), MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, V, MMA_N))
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
        static_assert(CUTE_STATIC_V(size<0>(scores)) == kNRows);

        // _print(scores, 0);
        flash::template scale_apply_exp2</*Scale_max=*/true, Check_inf, Max_offset>(scores, row_max, softmax_scale_log2);
        // _print(scores, 1);
        // We don't do the reduce across threads here since we don't need to use the row_sum.
        // We do that reduce at the end when we need to normalize the softmax.
        flash::reduce_sum</*zero_init=*/Is_first, /*warp_reduce=*/false>(scores, row_sum);
        // _print(scores, 2);
    };

    __forceinline__ __device__ TensorT finalize(float const final_scale=1.f) {
        // return;
        SumOp<float> sum_op;
        quad_allreduce_(row_sum, row_sum, sum_op);
        TensorT scores_scale;
        #pragma unroll
        for (int mi = 0; mi < size(row_sum); ++mi) {
            float sum = row_sum(mi);
            float inv_sum = (sum == 0.f || sum != sum) ? 0.f : 1.f / sum;
            scores_scale(mi) = inv_sum * final_scale;
            // For FP8, we might have scaled the output of exp by 2**8 so we need to divide sum by that amount.
            if constexpr (Max_offset != 0) {
                static constexpr float sum_scale = 1.f / 448.0f;
                sum *= sum_scale;
            }
            row_sum(mi) = (sum == 0.f || sum != sum) ? -INFINITY : row_max(mi) * (softmax_scale_log2 * float(M_LN2)) + __logf(sum);
        }
        return scores_scale;
    };

    template<typename Tensor1>
    __forceinline__ __device__ void rescale_o(Tensor1 &acc_o, TensorT const &scores_scale) {
        // return;
        // Reshape acc_o from (MMA=4, MMA_M, MMA_K) to (nrow=(2, MMA_M), ncol=(2, MMA_K))
        Tensor acc_o_rowcol = make_tensor(acc_o.data(), flash::convert_layout_acc_rowcol(acc_o.layout()));
        static_assert(CUTE_STATIC_V(size<0>(acc_o_rowcol)) == kNRows);
        #pragma unroll
        for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
            #pragma unroll
            for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= scores_scale(mi); }
        }
    };

    template<typename Tensor0>
    CUTLASS_DEVICE TensorT get_row_scale(Tensor0 const &acc_p) {
        // Reshape acc_s from ((2, 2, V), MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, V, MMA_N))
        Tensor scores = make_tensor(acc_p.data(), flash::convert_layout_acc_rowcol(acc_p.layout()));
        static_assert(CUTE_STATIC_V(size<0>(scores)) == kNRows);

        constexpr float kMaxFP8E4M3 = 448.0f;
        constexpr float eps = 1e-5f;
        
        TensorT row_scale;
        // First get the local max for each thread
        #pragma unroll
        for (int mi = 0; mi < size<0>(scores); ++mi) {
            float max_val = 0.0f + eps;
            #pragma unroll
            for (int ni = 0; ni < size<1>(scores); ++ni) {
                const float s = scores(mi, ni);
                const float val = fabsf(s == -INFINITY ? 0.0f : s);
                const float s_fabsf = fabsf(s);

                if (s != 0.0f) {
                    CUTE_LOG("[get_row_scale]: max_val: %f, s: %f, val: %f, mi: %d, ni: %d\n", max_val, s, val, mi, ni);
                }
                max_val = max(max_val, val);
            }
        
            // Warp-level reduction using butterfly pattern
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                float other = __shfl_xor_sync(0xffffffff, max_val, offset);
                max_val = max(max_val, other);
            }
            row_scale(mi) = kMaxFP8E4M3 / max_val;

            // CUTE_LOG("[get_row_scale after sync]: max_val: %f, mi: %d\n", max_val, mi);

            // CUTE_LOG("[get_row_scale]: row_scale: %f, max_val: %f, mi: %d\n", row_scale(mi), max_val, mi);
            // row_scale(mi) = 1.75f;
        }
        
        return row_scale;
    }


    template<typename Tensor1>
    __forceinline__ __device__ void scale(Tensor1 &acc_p, TensorT const &row_scale) {
        // Reshape acc_o from (MMA=4, MMA_M, MMA_K) to (nrow=(2, MMA_M), ncol=(2, MMA_K))
        Tensor acc_p_rowcol = make_tensor(acc_p.data(), flash::convert_layout_acc_rowcol(acc_p.layout()));
        static_assert(CUTE_STATIC_V(size<0>(acc_p_rowcol)) == kNRows);
        #pragma unroll
        for (int mi = 0; mi < size<0>(acc_p_rowcol); ++mi) {
            #pragma unroll
            for (int ni = 0; ni < size<1>(acc_p_rowcol); ++ni) { acc_p_rowcol(mi, ni) *= row_scale(mi);
                const float p = acc_p_rowcol(mi, ni);
                const float scale = row_scale(mi);
                const float scaled_p = p * scale;
                acc_p_rowcol(mi, ni) = p * 1.0f;

                // CUTE_LOG("[scale]: p: %f, scaled_p: %f, scale: %f, mi: %d, ni: %d\n", p, scaled_p, scale, mi, ni);
            }
        }
    };

    template<typename Tensor1>
    __forceinline__ __device__ void unscale(Tensor1 &acc_p, TensorT const &row_scale) {
        // Reshape acc_o from (MMA=4, MMA_M, MMA_K) to (nrow=(2, MMA_M), ncol=(2, MMA_K))
        Tensor acc_p_rowcol = make_tensor(acc_p.data(), flash::convert_layout_acc_rowcol(acc_p.layout()));
        static_assert(CUTE_STATIC_V(size<0>(acc_p_rowcol)) == kNRows);
        #pragma unroll
        for (int mi = 0; mi < size<0>(acc_p_rowcol); ++mi) {
            #pragma unroll
            for (int ni = 0; ni < size<1>(acc_p_rowcol); ++ni) { 
                const float o = acc_p_rowcol(mi, ni);
                const float scale = row_scale(mi);
                const float unscaled_o = o / scale;
                acc_p_rowcol(mi, ni) = o / 1.0f;
                
                
                // CUTE_LOG("[unscale]: o: %f, unscaled_o: %f, scale: %f, mi: %d, ni: %d\n", o, unscaled_o, scale, mi, ni);

            }
        }
    };

};

}  // namespace flash
