// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include "common.h"
#include "q4_k.h"
#include "aie_kernel_utils.h"

typedef struct {
    bf16 scales[NPU_M_TILE * 8]; // per-group scales and mins
    bf16 mins[NPU_M_TILE * 8]; // per-group scales and mins
    uint4 qs[NPU_M_TILE * NPU_K_TILE];           // 4-bit quantized values
}q4k_block;


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

            // Load one 128-nibble chunk at a time and interleave two independent
            // MAC chains (even/odd columns) to eliminate vector spills and
            // half-width accumulator dependency stalls on AIE2P.
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

template<int m>
void copy_y(bf16 * y, const float * y_acc) {
    for (int i = 0; i < m / 16; i++){
        aie::vector<float, 16> y_vec = aie::load_v<16>(y_acc + i * 16);
        aie::accum<accfloat, 16> acc;
        acc.from_vector(y_vec);
        aie::store_v(y + i * 16, acc.template to_vector<bf16>());
    }
}
constexpr int x_prod_lock = 0;
constexpr int x_cons_lock = 1;
constexpr int w_prod_lock = 2;
constexpr int w_cons_lock = 3;
constexpr int y_prod_lock = 4;
constexpr int y_cons_lock = 5;
constexpr int rtp_available_lock = 6;
// phase 1: qkv proj, D -> (D + D // 2) to RoPE
// phase 2: o_proj, D -> D to rms_residual
// phase 3: up/gate proj, D -> D * 8 to MT_GLU
// phase 4: down proj, D * 4 -> D, to rms_residual
// phase 4 is special, do not clear y to zero
// Peano's CDO loader does not zero .bss; keep ping/pong selectors in .data.
#if !defined(__chess__)
__attribute__((section(".data")))
#endif
bool is_y_ping = false;
#if !defined(__chess__)
__attribute__((section(".data")))
#endif
bool is_wx_ping = false;
float y_acc[32] = {0};

template<int M, int K, int m, int k>
void linear_proj_iD(
    bf16 * x_ping,
    bf16 * x_pong,
    q4k_block * w_ping,
    q4k_block * w_pong,
    bf16 * y_ping,
    bf16 * y_pong,
    const uint32 pkt_id,
    bool & is_wx_ping,
    bool & is_y_ping
) {
    uint32* pkt_id_ping = reinterpret_cast<uint32*>(y_ping + 14);
    uint32* pkt_id_pong = reinterpret_cast<uint32*>(y_pong + 14);
    bfloat16 b_col_reduce_add[ K/32];
    AIE_LOCK_LOOP_CTR i = 0;
    do {
        zero_y<m>(y_acc);
        AIE_LOCK_LOOP_CTR j = 0;
        do {
            is_wx_ping = !is_wx_ping;
            q4k_block* w_using = is_wx_ping ? w_ping : w_pong;
            bf16* x_using = is_wx_ping ? x_ping : x_pong;
            _lock_acquire(w_cons_lock);
            _lock_acquire(x_cons_lock);
            bf16* b_col_reduce_add_ptr = b_col_reduce_add + j *(k/32);
            // special logic for i==0:
            if(i == 0){
                // fill the b_col_reduce_add
                AIE_PREPARE_FOR_PIPELINING
                AIE_LOOP_UNROLL_FULL
                for(int l = 0; l < k / 32; l++)AIE_LOOP_FLATTEN{
                    *(b_col_reduce_add_ptr+ l) = bf16(aie::reduce_add(  aie::load_v<32>(x_using + 32*l) ));
                }
            }
            _qmm_q4k_bf16<m, k>(w_using, x_using, y_acc, b_col_reduce_add_ptr);
            _lock_release(w_prod_lock);
            _lock_release(x_prod_lock);
            j++;
        } while (j < K / k);
        _lock_acquire(y_prod_lock);
        is_y_ping = !is_y_ping;
        uint32* pkt_id_using = is_y_ping ? pkt_id_ping : pkt_id_pong;
        bf16* y_using = is_y_ping ? y_ping : y_pong;
        *pkt_id_using = pkt_id;
        copy_y<m>(y_using + 16, y_acc);
        _lock_release(y_cons_lock);
        i++;
    } while (i < M / m);
}
extern "C" {
constexpr int total_cores = 16;
void proj(
    bf16 * y_ping,
    q4k_block * w_ping,
    bf16 * x_ping,
    bf16 * y_pong,
    q4k_block * w_pong,
    bf16 * x_pong,
    int* IS_ATTN
)
{
    constexpr int m = 32;
    constexpr int k = 256;
    _lock_acquire(rtp_available_lock);
    // Peano miscompiles the RMW toggle+select on the persisted .data ping/pong
    // bools under proj's high register pressure (clobbers a control reg -> hang).
    // Localize to stack, pass by reference through every phase, write back once.
    bool wx_ping = is_wx_ping;
    bool y_ping_sel = is_y_ping;
    if (IS_ATTN[0] == 1) {
        // phase 1: qkv proj, D -> (D + D / 2) to RoPE
        linear_proj_iD<(DQ + DK + DV) / total_cores, D, m, k>(x_ping, x_pong, w_ping, w_pong, y_ping, y_pong, 0, wx_ping, y_ping_sel);

        // // phase 2: o proj, D -> D to  RMS
        linear_proj_iD<D / total_cores, DQ, m, k>(x_ping, x_pong, w_ping, w_pong, y_ping, y_pong, 1, wx_ping, y_ping_sel);

        // // phase 3: up/gate proj, D -> D * 8 to MT_GLU
        linear_proj_iD<2 * HIDDEN_SIZE / total_cores, D, m, k>(x_ping, x_pong, w_ping, w_pong, y_ping, y_pong, 2, wx_ping, y_ping_sel);

        // // phase 4.1 down-> 4D -> D, first D -> D
        linear_proj_iD<D / total_cores, HIDDEN_SIZE, m, k>(x_ping, x_pong, w_ping, w_pong, y_ping, y_pong, 1, wx_ping, y_ping_sel);
    }
    else{
        linear_proj_iD<vocab_size_padded / total_cores, D, m, k>(x_ping, x_pong, w_ping, w_pong, y_ping, y_pong, 1, wx_ping, y_ping_sel);
    }
    is_wx_ping = wx_ping;
    is_y_ping = y_ping_sel;

}


}