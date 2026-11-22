// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef COMMON_H
#define COMMON_H

// Peano can sink a per-iteration lock release out of a loop whose trip count it can
// prove, deadlocking the lock loops. A volatile counter keeps the trip count opaque.
// chess schedules correctly, so keep it a plain int there.
#if defined(__chess__)
#define AIE_LOCK_LOOP_CTR int
#else
#define AIE_LOCK_LOOP_CTR volatile int
#endif

#include <aie_api/aie.hpp>
// #include <aie2p_locks.h>
typedef bfloat16 bf16;

// Model type definitions
#define PLAMO3_0_8B 50
#define PLAMO3_2B 51
#define PLAMO3_8B 52

#ifndef MODEL_TYPE
#define MODEL_TYPE PLAMO3_2B
#endif

#include <aie_api/aie.hpp>
// #include <aie2p_locks.h>
typedef bfloat16 bf16;

#if MODEL_TYPE == PLAMO3_2B
// plamo-3-translate-2b; KV heads pre-replicated 4 -> 8 in Q4NX weights
constexpr int D = 2560;
constexpr int NUM_ATTN_HEADS = 24;
constexpr int NUM_KV_HEADS = 4;
constexpr int HEADS = 24;
constexpr int DQ = 3072;
constexpr int DK = 512;
constexpr int DV = DK;
const int DH = DQ / HEADS;
constexpr int HIDDEN_SIZE = 10240;
const float scale = 0.08838834764831845f;
constexpr int VOCAB_SIZE = 107520;
#elif MODEL_TYPE == PLAMO3_0_8B
// plamo-3-translate-0.8b; KV heads replicated 1 -> 8 (MQA expanded for kernel GQA)
constexpr int D = 1536;
constexpr int NUM_ATTN_HEADS = 16;
constexpr int NUM_KV_HEADS = 4;
constexpr int HEADS = 16;
constexpr int DQ = 2048;
constexpr int DK = 512;
constexpr int DV = DK;
const int DH = DQ / HEADS;
constexpr int HIDDEN_SIZE = 6144;
const float scale = 0.08838834764831845f;
constexpr int VOCAB_SIZE = 107520;
#elif MODEL_TYPE == PLAMO3_8B
// plamo-3-translate-8b; native 4 KV heads (GQA 8:1, 32 Q / 4 KV)
constexpr int D = 4096;
constexpr int NUM_ATTN_HEADS = 32;
constexpr int NUM_KV_HEADS = 4;
constexpr int HEADS = 32;
constexpr int DQ = 4096;
constexpr int DK = 512;
constexpr int DV = DK;
const int DH = DQ / HEADS;
constexpr int HIDDEN_SIZE = 16384;
const float scale = 0.08838834764831845f;
constexpr int VOCAB_SIZE = 107520;
#else
#error "Invalid model type. Please define MODEL_TYPE as PLAMO3_0_8B, PLAMO3_2B, or PLAMO3_8B"
#endif



constexpr int GELU_SLICE = 1024;
const bf16 exp_scale = 1.4426950408889634f;



constexpr int ATTN_GROUPS = NUM_ATTN_HEADS / NUM_KV_HEADS;


const int cores = 16;
const int qkv_repeats = (DQ + DK + DV) / cores / 32;
const int up_gate_repeats = (2 * HIDDEN_SIZE) / cores / 32;
constexpr int m_per_round = 32 * cores;
constexpr int vocab_size_padded = (VOCAB_SIZE + D - 1) / D * D;


inline void _lock_acquire(int lock_id, int num_locks = 1){
    acquire_greater_equal(lock_id + 48, num_locks);
}

inline void _lock_release(int lock_id, int num_locks = 1){
    release(lock_id + 48, num_locks);
}

inline void _left_lock_acquire(int lock_id, int num_locks = 1){
    acquire_greater_equal(lock_id + 16, num_locks);
}

inline void _left_lock_release(int lock_id, int num_locks = 1){
    release(lock_id + 16, num_locks);
}

inline void _up_lock_acquire(int lock_id, int num_locks = 1){
    acquire_greater_equal(lock_id + 32, num_locks);
}

inline void _up_lock_release(int lock_id, int num_locks = 1){
    release(lock_id + 32, num_locks);
}

inline void _down_lock_acquire(int lock_id, int num_locks = 1){
    acquire_greater_equal(lock_id, num_locks);
}

inline void _down_lock_release(int lock_id, int num_locks = 1){
    release(lock_id, num_locks);
}

template <typename T, int M>
void zero_vectorized(T *__restrict c) {
    constexpr int r = 256 / (sizeof(T) * 8); // one 256 bit store unit
    static_assert((M) % r == 0);
    const aie::vector<T, r> zeros = aie::zeros<T, r>();
    const T *__restrict c_end = c + M;
    for (; c < c_end; c += r) {
        aie::store_v(c, zeros);
    }
}



template <typename T_in, typename T_out, int N>
void passThrough_aie(T_in *restrict in0, T_out *restrict out) {
    event0();
    const int vec_factor = 16;

    aie::vector<T_out, vec_factor> Out;

    const int F = N / vec_factor;
    for (int i = 0; i < F; i++)
    chess_flatten_loop{
            aie::accum<accfloat, vec_factor> c_acc_in;
            c_acc_in.from_vector(aie::load_v<vec_factor>(in0));
            aie::store_v(out, c_acc_in.template to_vector<T_out>());
            in0 += vec_factor;
            out += vec_factor;
        }
    event1();
}
#endif
