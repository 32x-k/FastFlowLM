// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include "plamo3_npu_sequence_detail.hpp"

void plamo3_npu_sequence::Impl::_send_x(npu_sequence* seq){
    // send x
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        xy_arg_idx,
        MM2S,
        xyr_tile,
        bd_0,
        it_channel_0,
        {0, 0, 0, 0},
        {1, 1, 1, (uint32_t)D},
        {0, 0, 0, 1},
        0, 0, false
    );
}

void plamo3_npu_sequence::Impl::_send_rms_weights(npu_sequence* seq, uint32_t w_offset){
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        w_rms_arg_idx,
        MM2S,
        xyr_tile,
        bd_1,
        it_channel_1,
        {0, 0, 0, w_offset},
        {1, 1, 1, (uint32_t)D * 2},
        {0, 0, 0, 1},
        0, 0, false
    );
}

void plamo3_npu_sequence::Impl::_send_rope_rms_weights(npu_sequence* seq){
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        w_rope_rms_arg_idx,
        MM2S,
        xyr_tile,
        bd_2,
        it_channel_0,
        {0, 0, 0, 0},
        {1, 1, 1, (uint32_t)DH * 3},
        {0, 0, 0, 1},
        1, 0, false
    );
}

void plamo3_npu_sequence::Impl::_receive_kv_cache(npu_sequence* seq, const int L, bool is_sliding_window){
    int L_local = L - 1;
    if (is_sliding_window){
        L_local = L_local % sliding_window;
    }
    const uint32_t kv_cache_offset = L_local * DK / 2;
    uint32_t k01_offset = is_sliding_window ? K01_sliding_offset : K01_offset;
    uint32_t k23_offset = is_sliding_window ? K23_sliding_offset : K23_offset;
    uint32_t v01_offset = is_sliding_window ? V01_sliding_offset : V01_offset;
    uint32_t v23_offset = is_sliding_window ? V23_sliding_offset : V23_offset;
    // receive new k, v
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        kv_cache_arg_idx,
        S2MM,
        IT0,
        bd_0,
        it_channel_0,
        {0, 0, 0, kv_cache_offset},
        {1, 1, 1, (uint32_t)DK / 2},
        {0, 0, 0, 1},
        -1, 0, true
    );
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        kv_cache_arg_idx,
        S2MM,
        IT7,
        bd_0,
        it_channel_0,
        {0, 0, 0, kv_cache_offset + (uint32_t)v01_offset},
        {1, 1, 1, (uint32_t)DK / 2},
        {0, 0, 0, 1},
        -1, 0, true
    );
    _move_weights(seq, DQ + DK + DV, D, w_qkv_offset);
    seq->npu_dma_wait(
        IT0,
        S2MM,
        it_channel_0
    );
    seq->npu_dma_wait(
        IT7,
        S2MM,
        it_channel_0
    );
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        kv_cache_arg_idx,
        S2MM,
        IT0,
        bd_0,
        it_channel_0,
        {0, 0, 0, kv_cache_offset + (uint32_t)k23_offset},
        {1, 1, 1, (uint32_t)DK / 2},
        {0, 0, 0, 1},
        -1, 0, true
    );

    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        kv_cache_arg_idx,
        S2MM,
        IT7,
        bd_0,
        it_channel_0,
        {0, 0, 0, kv_cache_offset + (uint32_t)v23_offset},
        {1, 1, 1, (uint32_t)DK / 2},
        {0, 0, 0, 1},
        -1, 0, true
    );

    seq->npu_dma_wait(
        IT0,
        S2MM,
        it_channel_0
    );
    seq->npu_dma_wait(
        IT7,
        S2MM,
        it_channel_0
    );
}

void plamo3_npu_sequence::Impl::_move_weights(npu_sequence* seq, const size_t Dout, const size_t Din, const size_t w_offset){
    const uint32_t columns = 4;
    const uint32_t elems = m * k;
    const uint32_t a_block_size = elems * 5 / 8 / 2;
    const uint32_t blocks_per_row = Din / k;
    const uint32_t cores = columns * 4;
    assert(Din % k == 0);
    assert(blocks_per_row > 0);
    assert(Dout / m / cores > 0);
    assert(Dout % (m * cores) == 0);
    for (size_t round = 0; round < Dout / m / cores; round++){
        uint32_t bd_offset = (round % 2) * 8;
        for (uint32_t col = 0; col < columns; col++){
            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                w_proj_arg_idx,
                MM2S,
                mvm_tiles[col],
                npu_bd_id(bd_1 + bd_offset),
                it_channel_0,
                {0, 0, 0, (uint32_t)((round * cores + col * 4) * a_block_size * blocks_per_row + (uint32_t)w_offset)},
                {1, 1, 1, blocks_per_row * a_block_size * 2},
                {0, 0, 0, 1},
                -1, 0, true, aggressive_cache
            );
            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                w_proj_arg_idx,
                MM2S,
                mvm_tiles[col],
                npu_bd_id(bd_2 + bd_offset),
                it_channel_1,
                {0, 0, 0, (uint32_t)((round * cores + col * 4 + 2) * a_block_size * blocks_per_row + (uint32_t)w_offset)},
                {1, 1, 1, blocks_per_row * a_block_size * 2},
                {0, 0, 0, 1},
                -1, 0, true, aggressive_cache
            );
        }
        if (round > 0){
            for (uint32_t col = 0; col < columns; col++){
                seq->npu_dma_wait(
                    mvm_tiles[col],
                    MM2S,
                    it_channel_0
                );
                seq->npu_dma_wait(
                    mvm_tiles[col],
                    MM2S,
                    it_channel_1
                );
            }
        }
    }
    for (uint32_t col = 0; col < columns; col++){
        seq->npu_dma_wait(
            mvm_tiles[col],
            MM2S,
            it_channel_0
        );
        seq->npu_dma_wait(
            mvm_tiles[col],
            MM2S,
            it_channel_1
        );
    }
}

void plamo3_npu_sequence::Impl::_move_kv_cache(npu_sequence* seq, const size_t L, bool is_sliding_window){

    uint32_t k01_offset = is_sliding_window ? K01_sliding_offset : K01_offset;
    uint32_t k23_offset = is_sliding_window ? K23_sliding_offset : K23_offset;
    uint32_t v01_offset = is_sliding_window ? V01_sliding_offset : V01_offset;
    uint32_t v23_offset = is_sliding_window ? V23_sliding_offset : V23_offset;
    if (is_sliding_window){
        if (L % sliding_window == 0){ // corner case, just from zero to end
            const uint32_t data2move = sliding_window * DK / 2;
            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                kv_cache_arg_idx,
                MM2S,
                kv_tiles[0],
                bd_0,
                it_channel_0,
                {0, 0, 0, (uint32_t)k01_offset},
                {1, 1, 1, data2move},
                {0, 0, 0, 1},
                -1, 0, false, aggressive_cache
            );
            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                kv_cache_arg_idx,
                MM2S,
                kv_tiles[1],
                bd_0,
                it_channel_0,
                {0, 0, 0, (uint32_t)k23_offset},
                {1, 1, 1, data2move},
                {0, 0, 0, 1},
                -1, 0, false, aggressive_cache
            );
            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                kv_cache_arg_idx,
                MM2S,
                kv_tiles[0],
                bd_1,
                it_channel_1,
                {0, 0, 0, (uint32_t)v01_offset},
                {1, 1, 1, data2move},
                {0, 0, 0, 1},
                -1, 0, false, aggressive_cache
            );

            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                kv_cache_arg_idx,
                MM2S,
                kv_tiles[1],
                bd_1,
                it_channel_1,
                {0, 0, 0, (uint32_t)v23_offset},
                {1, 1, 1, data2move},
                {0, 0, 0, 1},
                -1, 0, false, aggressive_cache
            );
            return;
        }
        else if (L > sliding_window){ // still single phase, from zero to L
            uint32_t L_begin = L % sliding_window;
            uint32_t L_phase_1 = sliding_window - L_begin;
            uint32_t L_phase_2 = L_begin;
            uint32_t offset = L_begin * DK / 2;
            uint32_t data2move_1 = L_phase_1 * DK / 2;
            uint32_t data2move_2 = L_phase_2 * DK / 2;
            // phase 1
            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                kv_cache_arg_idx,
                MM2S,
                kv_tiles[0],
                bd_0,
                it_channel_0,
                {0, 0, 0, (uint32_t)k01_offset + offset},
                {1, 1, 1, data2move_1},
                {0, 0, 0, 1},
                -1, 0, false, aggressive_cache
            );
            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                kv_cache_arg_idx,
                MM2S,
                kv_tiles[1],
                bd_0,
                it_channel_0,
                {0, 0, 0, (uint32_t)k23_offset + offset},
                {1, 1, 1, data2move_1},
                {0, 0, 0, 1},
                -1, 0, false, aggressive_cache
            );
            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                kv_cache_arg_idx,
                MM2S,
                kv_tiles[0],
                bd_1,
                it_channel_1,
                {0, 0, 0, (uint32_t)v01_offset + offset},
                {1, 1, 1, data2move_1},
                {0, 0, 0, 1},
                -1, 0, false, aggressive_cache
            );

            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                kv_cache_arg_idx,
                MM2S,
                kv_tiles[1],
                bd_1,
                it_channel_1,
                {0, 0, 0, (uint32_t)v23_offset + offset},
                {1, 1, 1, data2move_1},
                {0, 0, 0, 1},
                -1, 0, false, aggressive_cache
            );
            // seq->npu_dma_wait(
            //     kv_tiles[0],
            //     MM2S,
            //     it_channel_0
            // );
            // seq->npu_dma_wait(
            //     kv_tiles[0],
            //     MM2S,
            //     it_channel_1
            // );
            // seq->npu_dma_wait(
            //     kv_tiles[1],
            //     MM2S,
            //     it_channel_0
            // );
            // seq->npu_dma_wait(
            //     kv_tiles[1],
            //     MM2S,
            //     it_channel_1
            // );
            // phase 2
            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                kv_cache_arg_idx,
                MM2S,
                kv_tiles[0],
                bd_2,
                it_channel_0,
                {0, 0, 0, (uint32_t)k01_offset},
                {1, 1, 1, data2move_2},
                {0, 0, 0, 1},
                -1, 0, false, aggressive_cache
            );
            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                kv_cache_arg_idx,
                MM2S,
                kv_tiles[1],
                bd_2,
                it_channel_0,
                {0, 0, 0, (uint32_t)k23_offset},
                {1, 1, 1, data2move_2},
                {0, 0, 0, 1},
                -1, 0, false, aggressive_cache
            );
            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                kv_cache_arg_idx,
                MM2S,
                kv_tiles[0],
                bd_3,
                it_channel_1,
                {0, 0, 0, (uint32_t)v01_offset},
                {1, 1, 1, data2move_2},
                {0, 0, 0, 1},
                -1, 0, false, aggressive_cache
            );

            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                kv_cache_arg_idx,
                MM2S,
                kv_tiles[1],
                bd_3,
                it_channel_1,
                {0, 0, 0, (uint32_t)v23_offset},
                {1, 1, 1, data2move_2},
                {0, 0, 0, 1},
                -1, 0, false, aggressive_cache
            );
            return;
        }
    }
    // fallback path, single phase, from zero to L
    const int L_padded = (L + L_CHUNK - 1) / L_CHUNK * L_CHUNK;
    const uint32_t data2move = L_padded * DK / 2;
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        kv_cache_arg_idx,
        MM2S,
        kv_tiles[0],
        bd_0,
        it_channel_0,
        {0, 0, 0, (uint32_t)k01_offset},
        {1, 1, 1, data2move},
        {0, 0, 0, 1},
        -1, 0, false, aggressive_cache
    );
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        kv_cache_arg_idx,
        MM2S,
        kv_tiles[1],
        bd_0,
        it_channel_0,
        {0, 0, 0, (uint32_t)k23_offset},
        {1, 1, 1, data2move},
        {0, 0, 0, 1},
        -1, 0, false, aggressive_cache
    );
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        kv_cache_arg_idx,
        MM2S,
        kv_tiles[0],
        bd_1,
        it_channel_1,
        {0, 0, 0, (uint32_t)v01_offset},
        {1, 1, 1, data2move},
        {0, 0, 0, 1},
        -1, 0, false, aggressive_cache
    );

    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        kv_cache_arg_idx,
        MM2S,
        kv_tiles[1],
        bd_1,
        it_channel_1,
        {0, 0, 0, (uint32_t)v23_offset},
        {1, 1, 1, data2move},
        {0, 0, 0, 1},
        -1, 0, false, aggressive_cache
    );
}

plamo3_npu_sequence::Impl::Impl(LM_Config config, uint32_t MAX_L){
    D = config.get("hidden_size");
    DH = config.get("head_dim");
    DQ = config.get("head_dim") * config.get("num_attention_heads");
    DK = config.get("head_dim") * config.get("num_key_value_heads");
    DV = config.get("head_dim") * config.get("num_key_value_heads");
    this->vocab_size = config.get("vocab_size");
    this->vocab_size_padded = (vocab_size + 1024 - 1) / 1024 * 1024;
    this->sliding_window = config.get("sliding_window");
    this->sliding_window_pattern = config.get("sliding_window_pattern");
    this->MAX_L = MAX_L;
    HIDDEN_SIZE = config.get("intermediate_size");
    kv_cache_size = MAX_L * (DK + DV);
    w_qkv_offset = 0;
    w_o_offset = (DQ + DK + DV) * D * 5 / 8 / sizeof(bf16);
    w_up_offset = w_o_offset + D * DQ * 5 / 8 / sizeof(bf16);
    w_down_offset = w_up_offset + (2 * HIDDEN_SIZE) * D * 5 / 8 / sizeof(bf16);
    uint32_t kv_cache_size_sliding = sliding_window * (DK + DV);
    K01_offset = 0;
    K23_offset = kv_cache_size / 4;
    V01_offset = kv_cache_size / 2;
    V23_offset = 3 * kv_cache_size / 4;
    // RTP addresses from the kernel builds (input_with_addresses.mlir).
    // 2B: layer.xclbin PLAMO3_2B + attn.xclbin attention_DH_128_GQA_1_3.
    // 0.8B: layer.xclbin PLAMO3_0_8B + attn.xclbin attention_DH_128_GQA_1_2.
    if (D == 2560){
        l_qk_address = 5120;
        l_kv_address = 34304;
        l_begin_mha_address = 15360;
        l_end_mha_address = 25088;
        window_size_address = 36992;
        proj_addr = 33280;
        rms_addr = 54272;
    }
    else if (D == 1536){
        l_qk_address = 5120;
        l_kv_address = 33792;
        l_begin_mha_address = 15360;
        l_end_mha_address = 25088;
        window_size_address = 36992;
        proj_addr = 33280;
        rms_addr = 52224;
    }
    else if (D == 4096){
        l_qk_address = 5120;
        l_kv_address = 34816;
        l_begin_mha_address = 15360;
        l_end_mha_address = 25088;
        window_size_address = 36992;
        proj_addr = 33280;
        rms_addr = 57344;
    }
    else {
        throw std::runtime_error("Unsupported plamo3 hidden size: " + std::to_string(D));
    }
    K01_sliding_offset = 0;
    K23_sliding_offset = kv_cache_size_sliding / 4;
    V01_sliding_offset = kv_cache_size_sliding / 2;
    V23_sliding_offset = 3 * kv_cache_size_sliding / 4;
}

void plamo3_npu_sequence::Impl::gen_layer_seq(npu_sequence* seq, const uint32_t L, bool is_sliding_window){
    assert(L <= MAX_L + 1);
    assert(L > 0);
    constexpr size_t CT_lock_address_base = 0x000001F000;
    int proj_rtp_lock_id = 6;
    int rms_rtp_lock_id = 6;
    seq->clear_cmds();

    seq->npu_preemption(0);

    if (is_sliding_window){
        uint32_t L_local;
        if (L <= sliding_window) {
            L_local = L;
        }
        else{
            L_local = sliding_window;
        }
        for (int i = 0; i < 4; i++){
            seq->rtp_write(attn_qk_tiles[i], l_qk_address, L_local);
            seq->rtp_write(attn_kv_tiles[i], l_kv_address, L_local);
        }
    }
    else{

        for (int i = 0; i < 4; i++){
            seq->rtp_write(attn_qk_tiles[i], l_qk_address, L);
            seq->rtp_write(attn_kv_tiles[i], l_kv_address, L);
        }
    }
    for (int i = 0; i < 16; i++){
        seq->rtp_write(proj_tiles[i], proj_addr, 1);
        seq->rtp_write(proj_tiles[i], CT_lock_address_base + 16 * proj_rtp_lock_id, 1); // set lock to 1
    }
    seq->rtp_write(rms_tile, rms_addr, 1);
    seq->rtp_write(rms_tile, CT_lock_address_base + 16 * rms_rtp_lock_id, 1); // set lock to 1

    _send_x(seq);
    // _send_rms_weights(seq, 0);
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        w_rms_arg_idx,
        MM2S,
        xyr_tile,
        bd_1,
        it_channel_1,
        {0, 0, 0, 0},
        {1, 1, 1, (uint32_t)D},
        {0, 0, 0, 1},
        0, 0, false
    );
    _send_rope_rms_weights(seq);

    // receive y
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        xy_arg_idx,
        S2MM,
        xyr_tile,
        bd_10,
        it_channel_1,
        {0, 0, 0, 0},
        {1, 1, 1, (uint32_t)D},
        {0, 0, 0, 1},
        -1, 0, true
    );
    _receive_kv_cache(seq, L, is_sliding_window); // also move qkv weights
    _move_kv_cache(seq, L, is_sliding_window);

    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        w_rms_arg_idx,
        MM2S,
        xyr_tile,
        bd_1,
        it_channel_1,
        {0, 0, 0, D},
        {1, 1, 1, (uint32_t)D * 2},
        {0, 0, 0, 1},
        0, 0, false
    );
    _move_weights(seq, D, DQ, w_o_offset);

    _move_weights(seq, 2 * HIDDEN_SIZE, D, w_up_offset);
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        w_rms_arg_idx,
        MM2S,
        xyr_tile,
        bd_1,
        it_channel_1,
        {0, 0, 0, D * 3},
        {1, 1, 1, (uint32_t)D},
        {0, 0, 0, 1},
        0, 0, false
    );
    _move_weights(seq, D, HIDDEN_SIZE, w_down_offset);

    // wait for receiving y
    seq->npu_dma_wait(
        xyr_tile,
        S2MM,
        it_channel_1
    );

    seq->cmds2seq();
}

void plamo3_npu_sequence::Impl::gen_mha_engine_seq(npu_sequence* seq, const uint32_t L_begin, const uint32_t L_end, bool is_sliding_window, int buffer_length){
    static constexpr npu_tiles IT[4][2] = {{IT0, IT1}, {IT2, IT3}, {IT4, IT5}, {IT6, IT7}};
    constexpr int lc = 16;
    constexpr int data_per_round = lc * 8; // 128
    const int Heads = DQ / DH;             // 24 (2B) or 32 (8B)
    const int num_kv_heads = DK / DH;      // 4
    const int Num_of_d_in_KV_cache = num_kv_heads / 2; // 2
    const size_t KV_CACHE_SIZE = (size_t)buffer_length * DH * num_kv_heads * 2;
    assert(L_begin % data_per_round == 0);
    assert(L_end % data_per_round == 0);
    assert((Heads == 24 || Heads == 32) && num_kv_heads == 4 && DH == 128);

    int using_window_size = is_sliding_window ? sliding_window : (int)L_end;
    seq->clear_cmds();

    for (int row = 2; row < 6; row++){
        for (int col = 0; col < 8; col++){
            npu_tiles tile = get_tile(row, col);
            seq->rtp_write(tile, l_begin_mha_address, L_begin);
            seq->rtp_write(tile, l_end_mha_address, L_end);
            seq->rtp_write(tile, window_size_address, using_window_size);
        }
    }

    int all_data_size = L_end - L_begin;
    const int down_rounds = (all_data_size + data_per_round - 1) / data_per_round;

    if (Heads == 32){
        // GQA 8:1 (32 Q heads, 4 KV heads, DH=128) mapped onto attention_DH_128_GQA_1_4
        // (4 CUs all share one KV stream via IT[1][1]; each KV head serves two 4-head batches).
        for (int head = 0; head < Heads / 4; head++){ // 0..7
            for (int round = 0; round < down_rounds; round++){
                int Lq_current = L_begin + round * data_per_round;
                int kv_begin = ((Lq_current - using_window_size) > 0) ? (Lq_current - using_window_size) : 0;
                int kv_length = Lq_current - kv_begin + data_per_round;
                int bd_offset = (round % 2) * 8;

                for (int cu = 0; cu < 4; cu++){
                    int head_offset = head * 4 + cu;
                    for (int col = 0; col < 2; col++){
                        size_t y_offset = (size_t)head_offset * DH + (size_t)round * data_per_round * DH * Heads + (size_t)col * lc * 4 * DH * Heads;
                        seq->npu_dma_memcpy_nd(
                            2, 0,
                            S2MM, IT[cu][col],
                            (npu_bd_id)(bd_offset + 0), it_channel_0,
                            {0, 0, 0, (uint32_t)y_offset},
                            {1, 1, (uint32_t)(4 * lc), (uint32_t)DH},
                            {0, 0, (uint32_t)(DH * Heads), 1},
                            -1, 0, true
                        );
                    }
                    size_t q_offset = (size_t)head_offset * DH + (size_t)round * data_per_round * DH * Heads;
                    seq->npu_dma_memcpy_nd(
                        2, 1,
                        MM2S, IT[cu][0],
                        (npu_bd_id)(bd_offset + 1), it_channel_0,
                        {0, 0, 0, (uint32_t)q_offset},
                        {1, 1, (uint32_t)(lc * 4), (uint32_t)DH},
                        {0, 0, (uint32_t)(DH * Heads), 1},
                        -1, 0, false
                    );
                    seq->npu_dma_memcpy_nd(
                        2, 1,
                        MM2S, IT[cu][0],
                        (npu_bd_id)(bd_offset + 2), it_channel_1,
                        {0, 0, 0, (uint32_t)(q_offset + (size_t)lc * 4 * DH * Heads)},
                        {1, 1, (uint32_t)(lc * 4), (uint32_t)DH},
                        {0, 0, (uint32_t)(DH * Heads), 1},
                        -1, 0, false
                    );
                }

                int kv_head = head / 2; // 0..3
                int kv_chunk_offset = kv_head / Num_of_d_in_KV_cache;
                int kv_head_offset = kv_head % Num_of_d_in_KV_cache;
                size_t k_offset = (size_t)kv_chunk_offset * buffer_length * Num_of_d_in_KV_cache * DH + (size_t)kv_head_offset * DH + (size_t)kv_begin * DH * Num_of_d_in_KV_cache;
                seq->npu_dma_memcpy_nd(
                    2, 2,
                    MM2S, IT[1][1],
                    (npu_bd_id)(bd_offset + 3), it_channel_0,
                    {0, 0, 0, (uint32_t)k_offset},
                    {1, (uint32_t)(kv_length / 128), (uint32_t)128, (uint32_t)DH},
                    {0, (uint32_t)(128 * DH * Num_of_d_in_KV_cache), (uint32_t)(DH * Num_of_d_in_KV_cache), 1},
                    -1, 0, false
                );
                size_t v_offset = k_offset + KV_CACHE_SIZE / 2;
                seq->npu_dma_memcpy_nd(
                    2, 2,
                    MM2S, IT[1][1],
                    (npu_bd_id)(bd_offset + 4), it_channel_1,
                    {0, 0, 0, (uint32_t)v_offset},
                    {1, (uint32_t)(kv_length / 128), (uint32_t)128, (uint32_t)DH},
                    {0, (uint32_t)(128 * DH * Num_of_d_in_KV_cache), (uint32_t)(DH * Num_of_d_in_KV_cache), 1},
                    -1, 0, false
                );

                if (round > 0){
                    for (int cu = 0; cu < 4; cu++){
                        for (int col = 0; col < 2; col++){
                            seq->npu_dma_wait(IT[cu][col], S2MM, it_channel_0);
                        }
                    }
                }
            }
            for (int cu = 0; cu < 4; cu++){
                for (int col = 0; col < 2; col++){
                    seq->npu_dma_wait(IT[cu][col], S2MM, it_channel_0);
                }
            }
        }
        seq->cmds2seq();
        return;
    }

    // GQA 6:1 (24 Q heads, 4 KV heads, DH=128) mapped onto attention_DH_128_GQA_1_3
    // (4 CUs: CUs 0..2 share one KV stream, CU 3 has its own KV stream).

    for (int head = 0; head < Heads / 4; head++){ // 0..5
        for (int round = 0; round < down_rounds; round++){
            int Lq_current = L_begin + round * data_per_round;
            int kv_begin = ((Lq_current - using_window_size) > 0) ? (Lq_current - using_window_size) : 0;
            int kv_length = Lq_current - kv_begin + data_per_round;
            int bd_offset = (round % 2) * 8;

            // CUs 0, 1, 2: Q heads [head*3 + 0..2] -> physical KV head (head / 2) in {0, 1, 2}
            for (int cu = 0; cu < 3; cu++){
                int head_offset = head * 3 + cu;
                for (int col = 0; col < 2; col++){
                    size_t y_offset = (size_t)head_offset * DH + (size_t)round * data_per_round * DH * Heads + (size_t)col * lc * 4 * DH * Heads;
                    seq->npu_dma_memcpy_nd(
                        2, 0,
                        S2MM, IT[cu][col],
                        (npu_bd_id)(bd_offset + 0), it_channel_0,
                        {0, 0, 0, (uint32_t)y_offset},
                        {1, 1, (uint32_t)(4 * lc), (uint32_t)DH},
                        {0, 0, (uint32_t)(DH * Heads), 1},
                        -1, 0, true
                    );
                }
                size_t q_offset = (size_t)head_offset * DH + (size_t)round * data_per_round * DH * Heads;
                seq->npu_dma_memcpy_nd(
                    2, 1,
                    MM2S, IT[cu][0],
                    (npu_bd_id)(bd_offset + 1), it_channel_0,
                    {0, 0, 0, (uint32_t)q_offset},
                    {1, 1, (uint32_t)(lc * 4), (uint32_t)DH},
                    {0, 0, (uint32_t)(DH * Heads), 1},
                    -1, 0, false
                );
                seq->npu_dma_memcpy_nd(
                    2, 1,
                    MM2S, IT[cu][0],
                    (npu_bd_id)(bd_offset + 2), it_channel_1,
                    {0, 0, 0, (uint32_t)(q_offset + (size_t)lc * 4 * DH * Heads)},
                    {1, 1, (uint32_t)(lc * 4), (uint32_t)DH},
                    {0, 0, (uint32_t)(DH * Heads), 1},
                    -1, 0, false
                );
            }
            int kv4_012 = head / 2;
            int kv_chunk_offset = kv4_012 / Num_of_d_in_KV_cache;
            int kv_head_offset = kv4_012 % Num_of_d_in_KV_cache;
            size_t k_offset = (size_t)kv_chunk_offset * buffer_length * Num_of_d_in_KV_cache * DH + (size_t)kv_head_offset * DH + (size_t)kv_begin * DH * Num_of_d_in_KV_cache;
            seq->npu_dma_memcpy_nd(
                2, 2,
                MM2S, IT[0][1],
                (npu_bd_id)(bd_offset + 3), it_channel_0,
                {0, 0, 0, (uint32_t)k_offset},
                {1, (uint32_t)(kv_length / 128), (uint32_t)128, (uint32_t)DH},
                {0, (uint32_t)(128 * DH * Num_of_d_in_KV_cache), (uint32_t)(DH * Num_of_d_in_KV_cache), 1},
                -1, 0, false
            );
            size_t v_offset = k_offset + KV_CACHE_SIZE / 2;
            seq->npu_dma_memcpy_nd(
                2, 2,
                MM2S, IT[0][1],
                (npu_bd_id)(bd_offset + 4), it_channel_1,
                {0, 0, 0, (uint32_t)v_offset},
                {1, (uint32_t)(kv_length / 128), (uint32_t)128, (uint32_t)DH},
                {0, (uint32_t)(128 * DH * Num_of_d_in_KV_cache), (uint32_t)(DH * Num_of_d_in_KV_cache), 1},
                -1, 0, false
            );

            // CU 3: Q heads [18 + head] -> all 6 (18..23) map to physical KV head 3
            int head_offset = head + 18;
            for (int col = 0; col < 2; col++){
                size_t y_offset = (size_t)head_offset * DH + (size_t)round * data_per_round * DH * Heads + (size_t)col * lc * 4 * DH * Heads;
                seq->npu_dma_memcpy_nd(
                    2, 0,
                    S2MM, IT[3][col],
                    (npu_bd_id)(bd_offset + 0), it_channel_0,
                    {0, 0, 0, (uint32_t)y_offset},
                    {1, 1, (uint32_t)(4 * lc), (uint32_t)DH},
                    {0, 0, (uint32_t)(DH * Heads), 1},
                    -1, 0, true
                );
            }
            size_t q_offset = (size_t)head_offset * DH + (size_t)round * data_per_round * DH * Heads;
            seq->npu_dma_memcpy_nd(
                2, 1,
                MM2S, IT[3][0],
                (npu_bd_id)(bd_offset + 1), it_channel_0,
                {0, 0, 0, (uint32_t)q_offset},
                {1, 1, (uint32_t)(lc * 4), (uint32_t)DH},
                {0, 0, (uint32_t)(DH * Heads), 1},
                -1, 0, false
            );
            seq->npu_dma_memcpy_nd(
                2, 1,
                MM2S, IT[3][0],
                (npu_bd_id)(bd_offset + 2), it_channel_1,
                {0, 0, 0, (uint32_t)(q_offset + (size_t)lc * 4 * DH * Heads)},
                {1, 1, (uint32_t)(lc * 4), (uint32_t)DH},
                {0, 0, (uint32_t)(DH * Heads), 1},
                -1, 0, false
            );
            kv_chunk_offset = 1;
            kv_head_offset = 1;
            k_offset = (size_t)kv_chunk_offset * buffer_length * Num_of_d_in_KV_cache * DH + (size_t)kv_head_offset * DH + (size_t)kv_begin * DH * Num_of_d_in_KV_cache;
            seq->npu_dma_memcpy_nd(
                2, 2,
                MM2S, IT[3][1],
                (npu_bd_id)(bd_offset + 3), it_channel_0,
                {0, 0, 0, (uint32_t)k_offset},
                {1, (uint32_t)(kv_length / 128), (uint32_t)128, (uint32_t)DH},
                {0, (uint32_t)(128 * DH * Num_of_d_in_KV_cache), (uint32_t)(DH * Num_of_d_in_KV_cache), 1},
                -1, 0, false
            );
            v_offset = k_offset + KV_CACHE_SIZE / 2;
            seq->npu_dma_memcpy_nd(
                2, 2,
                MM2S, IT[3][1],
                (npu_bd_id)(bd_offset + 4), it_channel_1,
                {0, 0, 0, (uint32_t)v_offset},
                {1, (uint32_t)(kv_length / 128), (uint32_t)128, (uint32_t)DH},
                {0, (uint32_t)(128 * DH * Num_of_d_in_KV_cache), (uint32_t)(DH * Num_of_d_in_KV_cache), 1},
                -1, 0, false
            );

            if (round > 0){
                for (int cu = 0; cu < 4; cu++){
                    for (int col = 0; col < 2; col++){
                        seq->npu_dma_wait(IT[cu][col], S2MM, it_channel_0);
                    }
                }
            }
        }
        for (int cu = 0; cu < 4; cu++){
            for (int col = 0; col < 2; col++){
                seq->npu_dma_wait(IT[cu][col], S2MM, it_channel_0);
            }
        }
    }
    seq->cmds2seq();
}

void plamo3_npu_sequence::Impl::gen_lm_head_seq(npu_sequence* seq){
    static constexpr npu_tiles lm_head_tiles[] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};
    static constexpr uint32_t columns = 8;
    static constexpr int lm_head_y_arg_idx = 0;
    static constexpr int lm_head_w_arg_idx = 1;
    static constexpr int lm_head_x_arg_idx = 2;

    constexpr int m_per_round = columns * 4 * 32;
    constexpr int a_block_size = 32 * 256 * 5 / 8 / 2;
    int blocks_per_row = D / 256;
    constexpr int row_blocks_per_round = columns * 4;

    int rounds = vocab_size_padded / m_per_round;

    seq->clear_cmds();
    seq->npu_preemption(0);

    for (int round = 0; round < rounds; round++){
        uint32_t round_offset = round * m_per_round;
        uint32_t bd_offset = (round % 2) * 8;
        // send x
        seq->npu_dma_memcpy_nd(
            sizeof(bf16), lm_head_x_arg_idx,
            MM2S, lm_head_tiles[0], npu_bd_id(bd_0 + bd_offset), it_channel_0,
            {0, 0, 0, 0},
            {1, 1, 1, (uint32_t)D},
            {0, 0, 0, 1},
            -1, 0, false
        );
        // y_out
        for (uint32_t col = 0; col < columns; col++){
            uint32_t y_offset = 4 * m * col + round_offset;
            seq->npu_dma_memcpy_nd(
                sizeof(bf16), lm_head_y_arg_idx,
                S2MM, lm_head_tiles[col], npu_bd_id(bd_2 + bd_offset), it_channel_0,
                {0, 0, 0, y_offset},
                {1, 1, 1, 4 * m},
                {0, 0, 0, 1},
                -1, 0, true
            );
        }

        for (uint32_t col = 0; col < columns; col++){
            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                lm_head_w_arg_idx,
                MM2S,
                lm_head_tiles[col],
                npu_bd_id(bd_1 + bd_offset),
                it_channel_1,
                {0, 0, 0, (round * row_blocks_per_round + col * 4) * a_block_size * blocks_per_row},
                {1, 1, 1, (uint32_t)(4 * a_block_size * blocks_per_row)},
                {0, 0, 0, 1},
                -1, 0, false
            );
        }
        if (round > 0){
            for (uint32_t col = 0; col < columns; col++){
                seq->npu_dma_wait(
                    lm_head_tiles[col],
                    S2MM,
                    it_channel_0
                );
            }
        }
    }
    for (uint32_t col = 0; col < columns; col++){
        seq->npu_dma_wait(
            lm_head_tiles[col],
            S2MM,
            it_channel_0
        );
    }
    seq->cmds2seq();
}
void plamo3_npu_sequence::Impl::set_max_length(const uint32_t MAX_L){
    this->MAX_L = MAX_L;
    this->kv_cache_size = MAX_L * (DK + DV);
    this->K01_offset = 0;

    this->K23_offset = kv_cache_size / 4;
    this->V01_offset = kv_cache_size / 2;
    this->V23_offset = 3 * kv_cache_size / 4;
}

// ===============================================
// Externals for plamo3_npu_sequence
// ===============================================
/// @brief
/// @param rtp_seq : npu_sequence ptr for layer rtp write (set L)
/// @param layer_seq : npu_sequence ptr for layer sequence
/// @param dequant_seq : npu_sequence ptr for dequant sequence
/// @param mm_engine_seq : npu_sequence ptr for mm engine sequence
/// @param mha_engine_seq : npu_sequence ptr for mha engine sequence
/// @param config : llama config
/// @param MAX_L : max length
plamo3_npu_sequence::plamo3_npu_sequence(LM_Config config, uint32_t MAX_L){
    this->_impl = new Impl(config, MAX_L);
}

void plamo3_npu_sequence::gen_layer_seq(npu_sequence* seq, const uint32_t L, bool is_sliding_window){
    this->_impl->gen_layer_seq(seq, L, is_sliding_window);
}

void plamo3_npu_sequence::gen_mha_engine_seq(npu_sequence* seq, const uint32_t L_begin, const uint32_t L_end, bool is_sliding_window, int buffer_length){
    this->_impl->gen_mha_engine_seq(seq, L_begin, L_end, is_sliding_window, buffer_length);
}

void plamo3_npu_sequence::gen_lm_head_seq(npu_sequence* seq){
    this->_impl->gen_lm_head_seq(seq);
}

void plamo3_npu_sequence::set_max_length(const uint32_t MAX_L){
    this->_impl->set_max_length(MAX_L);
}

size_t plamo3_npu_sequence::get_k01_offset() const{
    return this->_impl->K01_offset;
}

size_t plamo3_npu_sequence::get_k23_offset() const{
    return this->_impl->K23_offset;
}

size_t plamo3_npu_sequence::get_v01_offset() const{
    return this->_impl->V01_offset;
}

size_t plamo3_npu_sequence::get_v23_offset() const{
    return this->_impl->V23_offset;
}

plamo3_npu_sequence::~plamo3_npu_sequence() = default;
