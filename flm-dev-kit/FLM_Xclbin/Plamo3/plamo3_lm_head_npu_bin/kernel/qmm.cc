// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include <aie_api/aie.hpp>
#include "common.h"
#include "aie_kernel_utils.h"

typedef struct {
    bf16 scales[NPU_M_TILE * 8]; // per-group scales and mins
    bf16 mins[NPU_M_TILE * 8]; // per-group scales and mins
    uint4 qs[NPU_M_TILE * NPU_K_TILE];           // 4-bit quantized values
}q4k_block;

template<int m>
void copy_y(bf16 * y, const float * y_acc) {
    for (int i = 0; i < m / 16; i++){
        aie::vector<float, 16> y_vec = aie::load_v<16>(y_acc + i * 16);
        aie::accum<accfloat, 16> acc;
        acc.from_vector(y_vec);
        aie::store_v(y + i * 16, acc.template to_vector<bf16>());
    }
}

template<int m>
void zero_y(float * y) {
    zero_vectorized<float, m>(y);
}
template <int M, int N>
 void _qmm_q4k_bf16(
    const q4k_block * A,
    const bf16 * B,
    float * c,
    bf16* b_col_reduce_buffer
) { 

    constexpr int pr = 16;
    constexpr int pc = 8;
    const q4k_block * A_q4k = (q4k_block *)A;
    const uint4* qs_ptr = A_q4k->qs;
    // precompute chunk sum of B;

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(M/pr, M/pr)
    for (int row = 0; row < M; row += pr){
        aie::accum<accfloat, pr> c_accum;
        c_accum.from_vector(aie::load_v<pr>(c + row));
        const bf16* it_B =B;

        uint32_t scale_min_offset = row;
        bf16* cur_b_reduce_Buf = b_col_reduce_buffer;

        
        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_RANGE( N / 32, N / 32 )
        for (int sub_chunk = 0; sub_chunk < N / 32; sub_chunk++)AIE_LOOP_FLATTEN  {

            aie::vector<bf16, 32> b_col = aie::load_v<32>(it_B);
            it_B += 32;

            
            aie::vector<bf16, pr> a_scales = aie::load_v<pr>(A_q4k->scales +scale_min_offset);
            aie::vector<bf16, pr> a_mins = aie::load_v<pr>(A_q4k->mins + scale_min_offset);

            scale_min_offset+=M;
            bf16 reduce_b = *(cur_b_reduce_Buf);
            cur_b_reduce_Buf++;

            aie::accum<accfloat, pr> temp_acc0;
            aie::accum<accfloat, pr> temp_acc1;

            AIE_LOOP_UNROLL_FULL
            for (int l = 0; l < 4; l++) {
                aie::vector<uint4, pr * 8> a_cc = aie::load_v<pr * 8>(qs_ptr);
                qs_ptr += pr * 4;
                aie::accum<accfloat, pr * 8> a_cc_f32_accum;
                a_cc_f32_accum.from_vector(aie::to_float(a_cc, 0));
                aie::vector<bf16, pr * 8> a_cc_bf16 = a_cc_f32_accum.template to_vector<bf16>();

                AIE_LOOP_UNROLL_FULL
                for (int k = 0; k < 8; k += 2) {
                    const int col = l * 8 + k;
                    auto q0 = a_cc_bf16.template extract<pr>(k);
                    auto q1 = a_cc_bf16.template extract<pr>(k + 1);
                    if (col == 0) {
                        temp_acc0 = aie::mul(q0, b_col.get(col));
                        temp_acc1 = aie::mul(q1, b_col.get(col + 1));
                    } else {
                        temp_acc0 = aie::mac(temp_acc0, q0, b_col.get(col));
                        temp_acc1 = aie::mac(temp_acc1, q1, b_col.get(col + 1));
                    }
                }
            }

            c_accum = aie::mac(c_accum, temp_acc0.template to_vector<bf16>(), a_scales);
            c_accum = aie::mac(c_accum, temp_acc1.template to_vector<bf16>(), a_scales);
            c_accum = aie::mac(c_accum, a_mins, reduce_b);
            // 
        } // end of sub_chunk, 256 columns;
        aie::vector<float, pr> c_accum_f32 = c_accum.template to_vector<float>();
        aie::store_v(c + row, c_accum_f32);
    }
}

constexpr int x_prod_lock = 0;
constexpr int x_cons_lock = 1;
constexpr int w_prod_lock = 2;
constexpr int w_cons_lock = 3;
constexpr int y_prod_lock = 4;
constexpr int y_cons_lock = 5;

extern "C" {
    void qmm(
        const bf16 * x_ping,
        const bf16 * x_pong,
        const q4k_block * w_ping,
        const q4k_block * w_pong,
        bf16 * y_ping,
        bf16 * y_pong,
        const int M_div_m,
        const int K_div_k
    ){
        // Peano's CDO loader does not zero .bss; keep ping/pong selectors in .data.
#if !defined(__chess__)
        __attribute__((section(".data")))
#endif
        static bool is_y_ping = false;
#if !defined(__chess__)
        __attribute__((section(".data")))
#endif
        static bool is_x_ping = false;
#if !defined(__chess__)
        __attribute__((section(".data")))
#endif
        static bool is_w_ping = false;

        float y_acc[NPU_M_TILE] = {0};
        bfloat16 b_col_reduce_add[LM_K / 32]; 

        _lock_acquire(x_cons_lock);
        is_x_ping = !is_x_ping;
        const bf16 *x = reinterpret_cast<const bf16 *>(is_x_ping ? x_ping : x_pong);

        AIE_PREPARE_FOR_PIPELINING
        // Peano fully-unrolls this 80-iteration loop into ~7KB of code and
        // overflows the core's program memory; keep it rolled on peano (same
        // numerics). chess keeps the full unroll.
#if defined(__chess__)
        AIE_LOOP_UNROLL_FULL
#endif
        for (int k = 0; k < LM_K / 32; k++) {
            b_col_reduce_add[k] = bf16(aie::reduce_add(  aie::load_v<32>(x + 32*k) ));
        }
        for (int m = 0; m < M_div_m; m++) {
            zero_y<NPU_M_TILE>(y_acc);
            for (int k = 0; k < K_div_k; k++) {
                _lock_acquire(w_cons_lock);     
                is_w_ping = !is_w_ping;
                const q4k_block *w = reinterpret_cast<const q4k_block *>(is_w_ping ? w_ping : w_pong);
                bf16* b_col_reduce_add_ptr = b_col_reduce_add + k * 8;
                _qmm_q4k_bf16<32, 256>(w, x + 256 * k, y_acc, b_col_reduce_add_ptr);
                _lock_release(w_prod_lock);
            }
            _lock_acquire(y_prod_lock);
            is_y_ping = !is_y_ping;
            bf16 *y = reinterpret_cast<bf16 *>(is_y_ping ? y_ping : y_pong);
            copy_y<NPU_M_TILE>(y, y_acc);
            _lock_release(y_cons_lock);
        }
        _lock_release(x_prod_lock);
    }
}
