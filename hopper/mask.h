/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cute/tensor.hpp>

#include "cutlass/fast_math.h"  // For cutlass::FastDivmod

#include "utils.h"

namespace flash {

using namespace cute;


CUTLASS_DEVICE bool isEqual(float a, float b, float epsilon = 1e-6f) {
    return fabs(a - b) <= epsilon;
}

template <int kBlockM, int kBlockN, bool PackGQA, typename TiledMma, typename SeqlenInfo_t, bool SwapAB=false>
struct Mask {

    static_assert(!(PackGQA && SwapAB), "Cannot be both PackGQA and SwapAB");

    int const thread_idx;
    int const seqlen_q, seqlen_k;
    int const window_size_left, window_size_right, sink_token_length;
    cutlass::FastDivmod const qhead_per_khead_divmod;
    
    float const * ptr_q_descale_base = nullptr;
    float const * ptr_k_descale_base = nullptr;
    int const batch_idx = 0;
    int64_t const bm_stride = 0;
    int64_t const bn_stride = 0;
    int const bidh = 0;
    int const bidh_kv = 0;

    SeqlenInfo_t const* seqlen_info = nullptr;

    CUTLASS_DEVICE
    Mask(const int thread_idx, const int seqlen_q, const int seqlen_k,
         const int window_size_left, const int window_size_right, const int sink_token_length,
         cutlass::FastDivmod const &qhead_per_khead_divmod)
        : thread_idx(thread_idx)
        , seqlen_q(seqlen_q)
        , seqlen_k(seqlen_k)
        , window_size_left(window_size_left)
        , window_size_right(window_size_right)
        , sink_token_length(sink_token_length)
        , qhead_per_khead_divmod(qhead_per_khead_divmod)
    {
    };

    CUTLASS_DEVICE
    Mask(const int thread_idx, const int seqlen_q, const int seqlen_k,
         const int window_size_left, const int window_size_right, const int sink_token_length,
         cutlass::FastDivmod const &qhead_per_khead_divmod, float const * ptr_q_descale_base, float const * ptr_k_descale_base, int const batch_idx, int64_t const bm_stride, int64_t const bn_stride, int const bidh, int const bidh_kv, const SeqlenInfo_t* seqlen_info)
        : thread_idx(thread_idx)
        , seqlen_q(seqlen_q)
        , seqlen_k(seqlen_k)
        , window_size_left(window_size_left)
        , window_size_right(window_size_right)
        , sink_token_length(sink_token_length)
        , qhead_per_khead_divmod(qhead_per_khead_divmod)
        , ptr_q_descale_base(ptr_q_descale_base)
        , ptr_k_descale_base(ptr_k_descale_base)
        , batch_idx(batch_idx)
        , bm_stride(bm_stride)
        , bn_stride(bn_stride)
        , bidh(bidh)
        , bidh_kv(bidh_kv)
        , seqlen_info(seqlen_info)
    {
        if (thread_idx == 0 or threadIdx.x == 128 or threadIdx.x == 129 or threadIdx.x == 130) {
            cute::print("[Mask - construct] BlockIdx: (%d, %d, %d), thread-idx: %d | seqlen_q: %d, seqlen_k: %d, bm_stride: %d, bn_stride: %d, batch_idx: %d\n\n",
             (int)blockIdx.x, (int)blockIdx.y, (int)blockIdx.z, (int)threadIdx.x, (int)seqlen_q, (int)seqlen_k, (int)bm_stride, (int)bn_stride, (int)batch_idx);
        }
    };

    template <bool Seqlenk_mask=false, bool Causal_mask=false, bool Local_mask=false, bool Scale_only=false,
        typename Engine, typename Layout>
    CUTLASS_DEVICE
    void apply(Tensor<Engine, Layout> &tSrS, const int m_block, const int n_block) const {
        static_assert(!(Causal_mask && Local_mask), "Cannot be both causal and local");
        static_assert(Layout::rank == 3, "Only support 3D Tensor");
        // if (!Seqlenk_mask && !Causal_mask && !Local_mask) { return; }

        auto thread_mma = TiledMma{}.get_thread_slice(thread_idx);
        auto thread0_mma = TiledMma{}.get_thread_slice(_0{});

        static constexpr int Row = !SwapAB ? 0 : 1, Col = !SwapAB ? 1 : 0;

        Tensor cS = cute::make_identity_tensor(Shape<Int<!SwapAB ? kBlockM : kBlockN>, Int<!SwapAB ? kBlockN : kBlockM>>{});
        Tensor tScS = thread_mma.partition_C(cS);
        Tensor tSrS_rowcol = make_tensor(tSrS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(tSrS.layout()));
        Tensor tScS_rowcol = make_tensor(tScS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(tScS.layout()));
        Tensor t0ScS = thread0_mma.partition_C(cS);
        Tensor t0ScS_rowcol = make_tensor(t0ScS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(t0ScS.layout()));
        // We want to use the col indices of thread0 to compare, since that is known at compile time.
        // So we subtract the limit by the first col index of this thread (get<Col>(tScS_rowcol(_0{}, _0{})))
        int const thread_col_offset = get<Col>(tScS_rowcol(_0{}, _0{}));
        int const seqlenk_col_limit = seqlen_k - n_block * kBlockN - thread_col_offset;



        // If PackGQA, we split the work of compute divmod among threads in the same row
        static constexpr int kMmaThreadsPerRow = size<0, 0>(typename TiledMma::AtomLayoutC_TV{}); // 4
        static_assert(cutlass::NumThreadsPerWarp % kMmaThreadsPerRow == 0);

        if (ptr_q_descale_base != nullptr && ptr_k_descale_base != nullptr) {
            #pragma unroll
            for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
                int const row_idx = get<Row>(tScS_rowcol(m, _0{})) + m_block * kBlockM;
                if (row_idx < seqlen_q) {
                    float qs = 1.0f;
                    if (seqlen_info != nullptr) {
                        qs = ptr_q_descale_base[(seqlen_info->offset_q + row_idx) * bm_stride];
                    }
                    else{
                        qs = ptr_q_descale_base[(batch_idx * seqlen_q + row_idx) * bm_stride];
                    }
                    const float ks = 1.0f;
                    #pragma unroll
                    for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                        int const col_idx = int(get<Col>(t0ScS_rowcol(m, n))) + n_block * kBlockN + thread_col_offset;
                        if (col_idx < seqlen_k) {
                            float ks = 1.0f;
                            if (seqlen_info != nullptr) {
                                ks = ptr_k_descale_base[(seqlen_info->offset_k + col_idx) * bn_stride];
                            } else {
                                ks = ptr_k_descale_base[(batch_idx * seqlen_k + col_idx) * bn_stride];
                            }
                            const float s = tSrS_rowcol(m, n);

                            const float s_scaled = s * qs * ks;
                            tSrS_rowcol(m, n) = s_scaled;  // Modifies all elements in row m

                            // CUTE_LOG(
                            //     "[Mask - Apply] [-1] seqlen_q: %d, seqlen_k: %d, bm_stride: %d, "
                            //     "bn_stride: %d, batch_idx: %d, bidh: %d, bidh_kv: %d, "
                            //     "row_idx: %d, col_idx: %d, q_descale: %f, k_descale: %f, tSrS-M: %d, tSrS-N: %d, "
                            //     "s: %f, s_scaled: %f,thread_col_offset: %d, n_block: %d, kBlockN: %d, m_block: %d, kBlockM: %d, scaleOnly: %d\n",
                            //     (int)seqlen_info->offset_q, (int)seqlen_k, (int)bm_stride, (int)bn_stride, (int)batch_idx, (int)bidh,
                            //     (int)bidh_kv, (int)row_idx, (int)col_idx, qs, ks, (int)size<0>(tSrS_rowcol), (int)size<1>(tSrS_rowcol),
                            //     (float)s, (float)s_scaled, (int)thread_col_offset, (int)n_block, (int)kBlockN, (int)m_block, (int)kBlockM, (int)Scale_only);
                        }
                    }
                }
            }
        }
    
            // #pragma unroll
            // for (int i = 0; i < size(tSrS); ++i) {
            //     tSrS(i) = tSrS(i) * 0.0022f;
            // }


            // #pragma unroll
            // for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
            //     int const row_idx = get<Row>(tScS_rowcol(m, _0{})) + m_block * kBlockM;
            //     const float qs = ptr_q_descale_base[(batch_idx * seqlen_q + row_idx) * bm_stride];
            //     tSrS_rowcol(m) = tSrS_rowcol(m) * qs;
            // }

                // tSrS_rowcol(m) = tSrS_rowcol(m) * 0.0022f;
                // continue;


                // const float s = 128.0;
                // const float qs = 1.0;
                // const float ks = 1.0;
                // const int col_idx = 0;
                // const int n = 0;

                // if (true) {
                //     CUTE_LOG(
                //         "[Mask - Apply] seqlen_q: %d, seqlen_k: %d, bm_stride: %d, "
                //         "bn_stride: %d, batch_idx: %d, bidh: %d, bidh_kv: %d, "
                //         "row_idx: %d, col_idx: %d, q_descale: %f, k_descale: %f, tSrS-M: %d, tSrS-N: %d, "
                //         "s_row: %f, thread_col_offset: %d, n_block: %d, kBlockN: %d, m_block: %d, kBlockM: %d \n",
                //         (int)seqlen_q, (int)seqlen_k, (int)bm_stride, (int)bn_stride, (int)batch_idx, (int)bidh,
                //         (int)bidh_kv, (int)row_idx, (int)col_idx, qs, ks, (int)size<0>(tSrS_rowcol), (int)size<1>(tSrS_rowcol),
                //         (float)s / 128.0, (int)thread_col_offset, (int)n_block, (int)kBlockN, (int)m_block, (int)kBlockM);
                // }


                // A: 16x4 (8x4, 8x4) => 32 threads => (A, B, C, D)
                // B: 4x16 (4x8, 4x8) 
                // CUTE_LOG(
                // "[Mask - Apply] seqlen_q: %d, seqlen_k: %d, bm_stride: %d, "
                // "bn_stride: %d, batch_idx: %d, bidh: %d, bidh_kv: %d, "
                // "row_idx: %d, tSrS-M: %d, tSrS-N: %d \n",
                // (int)seqlen_q, (int)seqlen_k, (int)bm_stride, (int)bn_stride, (int)batch_idx, (int)bidh, (int)bidh_kv, (int)row_idx, (int)size<0>(tSrS_rowcol), (int)size<1>(tSrS_rowcol));
                
                // #pragma unroll
                // for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                //     int const col_idx = int(get<Col>(t0ScS_rowcol(m, n))) + n_block * kBlockN + thread_col_offset;
                //     if (row_idx < seqlen_q && col_idx < seqlen_k) {
                //         const float qs = ptr_q_descale_base[(batch_idx * seqlen_q + row_idx) * bm_stride];
                //         // const float ks = ptr_k_descale_base[(batch_idx * seqlen_k + col_idx) * bn_stride];
                //         const float ks = 1.0;
                //         const float s = tSrS_rowcol(m, n);
                //         float s_scaled = 1.0;
                //         // if (s != -INFINITY) {
                //         //     s_scaled = s * qs * ks;
                //         //     tSrS_rowcol(m, n) = s_scaled;
                //         // }

                //     if (true) {
                //         CUTE_LOG(
                //         "[Mask - Apply] seqlen_q: %d, seqlen_k: %d, bm_stride: %d, "
                //         "bn_stride: %d, batch_idx: %d, bidh: %d, bidh_kv: %d, "
                //         "row_idx: %d, col_idx: %d, q_descale: %f, k_descale: %f, tSrS-M: %d, tSrS-N: %d, "
                //         "s: %f, s_scaled: %f,thread_col_offset: %d, n_block: %d, kBlockN: %d, m_block: %d, kBlockM: %d \n",
                //         (int)seqlen_q, (int)seqlen_k, (int)bm_stride, (int)bn_stride, (int)batch_idx, (int)bidh,
                //         (int)bidh_kv, (int)row_idx, (int)col_idx, qs, ks, (int)size<0>(tSrS_rowcol), (int)size<1>(tSrS_rowcol),
                //         (float)s, (float)s_scaled, (int)thread_col_offset, (int)n_block, (int)kBlockN, (int)m_block, (int)kBlockM);
                //     }

                //         // if (cute::thread0()) {
                //         //     cute::print("tScS_rowcol=");
                //         //     cute::print(tScS_rowcol);
                //         //     cute::print("\ntSrS_rowcol=");
                //         //     cute::print(tSrS_rowcol);
                //         //     printf("\nRow=%d, Col=%d\n", (int)Row, (int)Col);
                //         //     printf("size(TiledMma)=%d\n", (int)size(TiledMma{}));
                //         //     printf("blockDim.x=%d, blockDim.y=%d, blockDim.z=%d\n",
                //         //             (int)blockDim.x, (int)blockDim.y, (int)blockDim.z);
                //         //     printf("size<0>(tSrS_rowcol)=%d\n", (int)size<0>(tSrS_rowcol));
                //         //     printf("size<1>(tSrS_rowcol)=%d\n", (int)size<1>(tSrS_rowcol));
                //         // }
                //     }
                // }

            // __syncthreads();
        
        if (Scale_only) {
            return;
        }

        if constexpr (!Causal_mask && !Local_mask) {
            if constexpr (Seqlenk_mask) {  // Just masking based on col
                #pragma unroll
                for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                    if (int(get<Col>(t0ScS_rowcol(_0{}, n))) >= seqlenk_col_limit) {
                        #pragma unroll
                        for (int m = 0; m < size<0>(tSrS_rowcol); ++m) { tSrS_rowcol(m, n) = -INFINITY; }
                    }
                }
            }
        } else {  // mask based on both row and col
            if constexpr (!SwapAB) {
                // If PackGQA, we split the work of compute divmod among threads in the same row
                static constexpr int kMmaThreadsPerRow = size<0, 0>(typename TiledMma::AtomLayoutC_TV{}); // 4
                static_assert(cutlass::NumThreadsPerWarp % kMmaThreadsPerRow == 0);
                static_assert(!PackGQA);
                int mma_m_idx;
                // Might get OOB but it's ok since we'll check it later
                if constexpr (PackGQA) {
                    mma_m_idx = qhead_per_khead_divmod.divide(m_block * kBlockM + get<Row>(tScS_rowcol(thread_idx % kMmaThreadsPerRow, _0{})));
                }
                int const causal_row_offset = 1 + seqlen_k - n_block * kBlockN - seqlen_q - thread_col_offset;

                // if (ptr_q_descale_base != nullptr && ptr_k_descale_base != nullptr) {
                //     #pragma unroll
                //     for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
                //         int const row_idx = get<Row>(tScS_rowcol(m, _0{})) + m_block * kBlockM;

                //         const float s = 128.0;
                //         const float qs = 1.0;
                //         const float ks = 1.0;
                //         const int col_idx = 0;
                //         const int n = 0;

                //         // if (true) {
                //         //     CUTE_LOG(
                //         //         "[Mask - Apply] seqlen_q: %d, seqlen_k: %d, bm_stride: %d, "
                //         //         "bn_stride: %d, batch_idx: %d, bidh: %d, bidh_kv: %d, "
                //         //         "row_idx: %d, col_idx: %d, q_descale: %f, k_descale: %f, tSrS-M: %d, tSrS-N: %d, "
                //         //         "s_row: %f, thread_col_offset: %d, n_block: %d, kBlockN: %d, m_block: %d, kBlockM: %d \n",
                //         //         (int)seqlen_q, (int)seqlen_k, (int)bm_stride, (int)bn_stride, (int)batch_idx, (int)bidh,
                //         //         (int)bidh_kv, (int)row_idx, (int)col_idx, qs, ks, (int)size<0>(tSrS_rowcol), (int)size<1>(tSrS_rowcol),
                //         //         (float)s / 128.0, (int)thread_col_offset, (int)n_block, (int)kBlockN, (int)m_block, (int)kBlockM);
                //         // }


                //         // A: 16x4 (8x4, 8x4) => 32 threads => (A, B, C, D)
                //         // B: 4x16 (4x8, 4x8) 
                //         // CUTE_LOG(
                //         // "[Mask - Apply] seqlen_q: %d, seqlen_k: %d, bm_stride: %d, "
                //         // "bn_stride: %d, batch_idx: %d, bidh: %d, bidh_kv: %d, "
                //         // "row_idx: %d, tSrS-M: %d, tSrS-N: %d \n",
                //         // (int)seqlen_q, (int)seqlen_k, (int)bm_stride, (int)bn_stride, (int)batch_idx, (int)bidh, (int)bidh_kv, (int)row_idx, (int)size<0>(tSrS_rowcol), (int)size<1>(tSrS_rowcol));
                        
                //         #pragma unroll
                //         for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                //             int const col_idx = int(get<Col>(tScS_rowcol(m, n))) + n_block * kBlockN + thread_col_offset;
                //             if (row_idx < seqlen_q) {
                //                 const float qs = ptr_q_descale_base[(batch_idx * seqlen_q + row_idx) * bm_stride];
                //                 // const float ks = ptr_k_descale_base[(batch_idx * seqlen_k + col_idx) * bn_stride];
                //                 const float ks = 1.0;
                //                 const float s = tSrS_rowcol(m, n);
                //                 if (s != -INFINITY) {
                //                     tSrS_rowcol(m, n) = s * qs * ks;
                //                 }

                //             if (row_idx == 0 or row_idx ==1) {
                //                 CUTE_LOG(
                //                 "[Mask - Apply] seqlen_q: %d, seqlen_k: %d, bm_stride: %d, "
                //                 "bn_stride: %d, batch_idx: %d, bidh: %d, bidh_kv: %d, "
                //                 "row_idx: %d, col_idx: %d, q_descale: %f, k_descale: %f, tSrS-M: %d, tSrS-N: %d, "
                //                 "s_row: %f, thread_col_offset: %d, n_block: %d, kBlockN: %d, m_block: %d, kBlockM: %d \n",
                //                 (int)seqlen_q, (int)seqlen_k, (int)bm_stride, (int)bn_stride, (int)batch_idx, (int)bidh,
                //                 (int)bidh_kv, (int)row_idx, (int)col_idx, qs, ks, (int)size<0>(tSrS_rowcol), (int)size<1>(tSrS_rowcol),
                //                 (float)s / 128.0, (int)thread_col_offset, (int)n_block, (int)kBlockN, (int)m_block, (int)kBlockM);
                //             }

                //                 // if (cute::thread0()) {
                //                 //     cute::print("tScS_rowcol=");
                //                 //     cute::print(tScS_rowcol);
                //                 //     cute::print("\ntSrS_rowcol=");
                //                 //     cute::print(tSrS_rowcol);
                //                 //     printf("\nRow=%d, Col=%d\n", (int)Row, (int)Col);
                //                 //     printf("size(TiledMma)=%d\n", (int)size(TiledMma{}));
                //                 //     printf("blockDim.x=%d, blockDim.y=%d, blockDim.z=%d\n",
                //                 //             (int)blockDim.x, (int)blockDim.y, (int)blockDim.z);
                //                 //     printf("size<0>(tSrS_rowcol)=%d\n", (int)size<0>(tSrS_rowcol));
                //                 //     printf("size<1>(tSrS_rowcol)=%d\n", (int)size<1>(tSrS_rowcol));
                //                 // }
                //             }
                //         }
                //     }
                // }

                if constexpr (Causal_mask) {
                    #pragma unroll
                    for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
                        int const row_idx = !PackGQA
                            ? get<Row>(tScS_rowcol(m, _0{})) + m_block * kBlockM
                            :  __shfl_sync(0xffffffff, mma_m_idx, m % kMmaThreadsPerRow, kMmaThreadsPerRow);
                        int const col_limit_right = !Seqlenk_mask
                            ? row_idx + causal_row_offset
                            : __viaddmin_s32(row_idx, causal_row_offset, seqlenk_col_limit);
                        #pragma unroll
                        for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                            int const col_idx = int(get<Col>(t0ScS_rowcol(_0{}, n)));
                            if (int(get<Col>(t0ScS_rowcol(_0{}, n))) >= col_limit_right) { tSrS_rowcol(m, n) = -INFINITY; }
                            // else if (ptr_q_descale_base != nullptr && ptr_k_descale_base != nullptr) {
                            //     // Case #1: fixed length
                            //     if (seqlen_info == nullptr) {

                            //         const float qs = ptr_q_descale_base[(batch_idx * seqlen_q + row_idx) * bm_stride];
                            //         const float ks = ptr_k_descale_base[(batch_idx * seqlen_k + col_idx) * bn_stride];

                            //        CUTE_LOG(
                            //             "[Mask - Apply] seqlen_q: %d, seqlen_k: %d, bm_stride: %d, "
                            //             "bn_stride: %d, batch_idx: %d, bidh: %d, bidh_kv: %d, "
                            //             "row_idx: %d, col_idx: %d, q_descale: %f, k_descale: %f ",
                            //             (int)seqlen_q, (int)seqlen_k, (int)bm_stride, (int)bn_stride, (int)batch_idx, (int)bidh, (int)bidh_kv, (int)row_idx, (int)col_idx, qs, ks);
                            //         CUTE_LOG(
                            //             "[Mask - Apply] mblock: %d, kBlockM: %d, nblock: %d, kBlockN: %d, thread_col_offset: %d, causal_row_offset: %d\n",
                            //             (int)m_block, (int)kBlockM, (int)n_block, (int)kBlockN, (int)thread_col_offset, (int)causal_row_offset
                            //         );
                            //         // cute::print_tensor(tSrS_rowcol);
                            //         // cute::print_tensor(tScS_rowcol);
                            //         // cute::print_tensor(t0ScS_rowcol);


                            //         const auto qs_e = (float)(1.0 * (batch_idx * seqlen_q + row_idx) * bm_stride + bidh);
                            //         const auto ks_e = (float)(1.0 * (batch_idx * seqlen_k + col_idx) * bn_stride + bidh_kv);

                            //         // if (!isEqual(qs, qs_e)){
                            //         //     cute::print(
                            //         //     "[Mask - QS not Equal] BlockIdx: (%d, %d, %d), thread-idx: %d "
                            //         //     "| seqlen_q: %d, seqlen_k: %d, bm_stride: %d, "
                            //         //     "bn_stride: %d, batch_idx: %d, bidh: %d, bidh_kv: %d\n"
                            //         //     "row_idx: %d, col_idx: %d, q_descale: %f, k_descale: %f\n\n",
                            //         //     (int)blockIdx.x, (int)blockIdx.y, (int)blockIdx.z, (int)threadIdx.x,
                            //         //     (int)seqlen_q, (int)seqlen_k, (int)bm_stride, (int)bn_stride, (int)batch_idx, (int)bidh, (int)bidh_kv, (int)row_idx, (int)col_idx,
                            //         //     (float)ptr_q_descale_base[(batch_idx * seqlen_q + row_idx) * bm_stride], (float)ptr_k_descale_base[(batch_idx * seqlen_k + col_idx) * bn_stride]);
                            //         // }
                            //         if (!isEqual(ks, 1.0)){
                            //             cute::print(
                            //             "[Mask - KS not Equal] BlockIdx: (%d, %d, %d), thread-idx: %d "
                            //             "| seqlen_q: %d, seqlen_k: %d, bm_stride: %d, "
                            //             "bn_stride: %d, batch_idx: %d, bidh: %d, bidh_kv: %d\n"
                            //             "row_idx: %d, col_idx: %d, q_descale: %f, k_descale: %f\n\n",
                            //             (int)blockIdx.x, (int)blockIdx.y, (int)blockIdx.z, (int)threadIdx.x,
                            //             (int)seqlen_q, (int)seqlen_k, (int)bm_stride, (int)bn_stride, (int)batch_idx, (int)bidh, (int)bidh_kv, (int)row_idx, (int)col_idx,
                            //             (float)ptr_q_descale_base[(batch_idx * seqlen_q + row_idx) * bm_stride], (float)ptr_k_descale_base[(batch_idx * seqlen_k + col_idx) * bn_stride]);
                            //         }
                            //         tSrS_rowcol(m, n) = tSrS_rowcol(m, n) * qs * ks;
                            //         // tSrS_rowcol(m, n) = s * qs * ks;
                            //     }
                            //     // Case #2: variable length
                            //     else {
                            //         tSrS_rowcol(m, n) = tSrS_rowcol(m, n) * ptr_q_descale_base[(seqlen_info->offset_q + row_idx) * bm_stride] * ptr_k_descale_base[(seqlen_info->offset_k + col_idx) * bn_stride];
                            //     }
                            // }
                        }
                    }
                } else {
                    int const local_row_offset_right = causal_row_offset + window_size_right;
                    int const local_row_offset_left = causal_row_offset - 1 - window_size_left;
                    int const col_limit_sink = sink_token_length - n_block * kBlockN;
                    #pragma unroll
                    for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
                        int const row_idx = !PackGQA
                            ? get<Row>(tScS_rowcol(m, _0{})) + m_block * kBlockM
                            :  __shfl_sync(0xffffffff, mma_m_idx, m % kMmaThreadsPerRow, kMmaThreadsPerRow);
                        int const col_limit_right = !Seqlenk_mask
                            ? row_idx + local_row_offset_right
                            : __viaddmin_s32(row_idx, local_row_offset_right, seqlenk_col_limit);
                        int const col_limit_left = row_idx + local_row_offset_left;
                        #pragma unroll
                        for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                            int const col_idx = int(get<Col>(t0ScS_rowcol(m, n)));
                            if (col_idx >= col_limit_right || (col_idx < col_limit_left && col_idx >= col_limit_sink)) { tSrS_rowcol(m, n) = -INFINITY; }
                            // else if (row_idx < seqlen_q && col_idx < seqlen_k && ptr_q_descale_base != nullptr && ptr_k_descale_base != nullptr) {
                            //     // Case #1: fixed length
                            //     if (seqlen_info == nullptr) {
                            //         if (thread_idx == 0 or threadIdx.x == 128 or threadIdx.x == 129 or threadIdx.x == 130) {
                            //             cute::print(
                            //                 "[Mask - apply | no causal] BlockIdx: (%d, %d, %d), thread-idx: %d "
                            //                 "| seqlen_q: %d, seqlen_k: %d, bm_stride: %d, "
                            //                 "bn_stride: %d, batch_idx: %d, bidh: %d, bidh_kv: %d\n"
                            //                 "row_idx: %d, col_idx: %d, q_descale: %f, k_descale: %f\n\n",
                            //                 (int)blockIdx.x, (int)blockIdx.y, (int)blockIdx.z, (int)threadIdx.x,
                            //                 (int)seqlen_q, (int)seqlen_k, (int)bm_stride, (int)bn_stride, (int)batch_idx, (int)bidh, (int)bidh_kv, (int)col_idx, (int)row_idx,
                            //                 (float)ptr_q_descale_base[(batch_idx * seqlen_q + row_idx) * bm_stride], (float)ptr_k_descale_base[(batch_idx * seqlen_k + col_idx) * bn_stride]);
                            //         }
                            //         tSrS_rowcol(m, n) = tSrS_rowcol(m, n) * ptr_q_descale_base[(batch_idx * seqlen_q + row_idx) * bm_stride] * ptr_k_descale_base[(batch_idx * seqlen_k + col_idx) * bn_stride];
                            //     }
                            //     // Case #2: variable length
                            //     else {
                            //         tSrS_rowcol(m, n) = tSrS_rowcol(m, n) * ptr_q_descale_base[(seqlen_info->offset_q + row_idx) * bm_stride] * ptr_k_descale_base[(seqlen_info->offset_k + col_idx) * bn_stride];
                            //     }
                            // }
                        }
                    }
                }
            } else {
                int const thread_row_offset = get<Row>(tScS_rowcol(_0{}, _0{}));
                int const causal_row_offset = seqlenk_col_limit - seqlen_q + m_block * kBlockM + thread_row_offset;
                if constexpr (Causal_mask) {
                    #pragma unroll
                    for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                        int const col0 = int(get<Col>(t0ScS_rowcol(_0{}, n)));
                        // If col0 is beyond the column limit, we want to mask out the entire column, by setting
                        // row limit to be kBlockM.
                        int const row_limit_top = col0 >= seqlenk_col_limit ? kBlockM : col0 - causal_row_offset;
                        #pragma unroll
                        for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
                            if (int(get<Row>(t0ScS_rowcol(m, _0{}))) < row_limit_top) { tSrS_rowcol(m, n) = -INFINITY; }
                        }
                    }
                } else {
                    int const col_limit_sink = sink_token_length - n_block * kBlockN - thread_col_offset;
                    #pragma unroll
                    for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                        int const col0 = int(get<Col>(t0ScS_rowcol(_0{}, n)));
                        // If col0 is beyond the column limit, we want to mask out the entire column, by setting
                        // row limit to be kBlockM.
                        int const row_limit_top = col0 >= seqlenk_col_limit ? kBlockM : col0 - causal_row_offset - window_size_right;
                        int const row_limit_bot = col0 < col_limit_sink ? kBlockM : col0 - causal_row_offset + window_size_left;
                        #pragma unroll
                        for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
                            int const row_idx = int(get<Row>(t0ScS_rowcol(m, _0{})));
                            if (row_idx < row_limit_top || row_idx > row_limit_bot) { tSrS_rowcol(m, n) = -INFINITY; }
                        }
                    }
                }
            }
        }
    };


    template <bool Seqlenk_mask, typename Engine, typename Layout>
    CUTLASS_DEVICE
    void apply_scale(Tensor<Engine, Layout> &tSrS, const int m_block, const int n_block, const int logdix = 0) const {
        return;
        static_assert(Layout::rank == 3, "Only support 3D Tensor");

        auto thread_mma = TiledMma{}.get_thread_slice(thread_idx);
        auto thread0_mma = TiledMma{}.get_thread_slice(_0{});

        static constexpr int Row = !SwapAB ? 0 : 1, Col = !SwapAB ? 1 : 0;

        Tensor cS = cute::make_identity_tensor(Shape<Int<!SwapAB ? kBlockM : kBlockN>, Int<!SwapAB ? kBlockN : kBlockM>>{});
        Tensor tScS = thread_mma.partition_C(cS);
        Tensor tSrS_rowcol = make_tensor(tSrS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(tSrS.layout()));
        Tensor tScS_rowcol = make_tensor(tScS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(tScS.layout()));
        Tensor t0ScS = thread0_mma.partition_C(cS);
        Tensor t0ScS_rowcol = make_tensor(t0ScS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(t0ScS.layout()));
        // We want to use the col indices of thread0 to compare, since that is known at compile time.
        // So we subtract the limit by the first col index of this thread (get<Col>(tScS_rowcol(_0{}, _0{})))
        int const thread_col_offset = get<Col>(tScS_rowcol(_0{}, _0{}));
        int const seqlenk_col_limit = seqlen_k - n_block * kBlockN - thread_col_offset;

        static constexpr int kMmaThreadsPerRow = size<0, 0>(typename TiledMma::AtomLayoutC_TV{});
        static_assert(cutlass::NumThreadsPerWarp % kMmaThreadsPerRow == 0);
        int const causal_row_offset = 1 + seqlen_k - n_block * kBlockN - seqlen_q - thread_col_offset;
        int mma_m_idx;

        // bool Seqlenk_mask = false;

        #pragma unroll
        for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
            int const row_idx = get<Row>(tScS_rowcol(m, _0{})) + m_block * kBlockM;
            if (row_idx < seqlen_q) {
                float  qs = -1.0f;
                if (ptr_q_descale_base != nullptr) {
                    qs = ptr_q_descale_base[(batch_idx * seqlen_q + row_idx) * bm_stride];
                }
                #pragma unroll
                for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                    int const col_idx = int(get<Col>(t0ScS_rowcol(m, n))) + n_block * kBlockN + thread_col_offset;
                    const float ks = 1.0f;
                    const float s = tSrS_rowcol(m, n);
                    // tSrS_rowcol(m, n) = tSrS_rowcol(m, n) * qs;  // Modifies all elements in row m

                    if (row_idx < seqlen_q && col_idx < seqlen_k) {
                        CUTE_LOG(
                            "[Mask - Apply][%d] seqlen_q: %d, seqlen_k: %d, bm_stride: %d, "
                            "bn_stride: %d, batch_idx: %d, bidh: %d, bidh_kv: %d, "
                            "row_idx: %d, col_idx: %d, q_descale: %f, k_descale: %f, tSrS-M: %d, tSrS-N: %d, "
                            "s_row: %f, thread_col_offset: %d, n_block: %d, kBlockN: %d, m_block: %d, kBlockM: %d \n",
                            (int)logdix, (int)seqlen_q, (int)seqlen_k, (int)bm_stride, (int)bn_stride, (int)batch_idx, (int)bidh,
                            (int)bidh_kv, (int)row_idx, (int)col_idx, qs, ks, (int)size<0>(tSrS_rowcol), (int)size<1>(tSrS_rowcol),
                            (float)s, (int)thread_col_offset, (int)n_block, (int)kBlockN, (int)m_block, (int)kBlockM);
                    }

                }
            }
        }


        
    };

};

} // namespace flash
