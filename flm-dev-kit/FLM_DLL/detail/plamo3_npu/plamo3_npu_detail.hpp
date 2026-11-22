// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#pragma once
#include "models/plamo3/plamo3_npu.hpp"
#include "modules/gemm.hpp"
#include "modules/dequant.hpp"
#include "modules/mha.hpp"
#include "utils/kv_rows_state.hpp"

struct plamo3_npu::Impl{
    /// @brief RoPE tables; filled in the ctor from config (standard rope for
    /// sliding layers, YaRN for the every-8th global-attention layers).
    f32 inv_freq[64];
    f32 inv_freq_local[64];
    /// @brief YaRN mscale; multiplies cos/sin of the global-attention layers.
    f32 rope_concentration = 1.0f;


    /// @brief  initialize the plamo3_npu
    /// @param config
    /// @param npu_instance
    Impl(LM_Config config, npu_xclbin_manager *npu_instance, int MAX_L = 4096);
    ~Impl();

    /// @brief forward the plamo3_npu
    buffer<bf16> forward(int ids);
    buffer<bf16> prefill(std::vector<int>& ids, void* payload = nullptr);
    buffer<bf16> _prefill_with_mv(std::vector<int>& ids, void* payload = nullptr);
    buffer<bf16> _prefill_with_mm(std::vector<int>& ids, void* payload = nullptr);

    void set_context_length(int L);
    void load_weights(Q4NX& q4nx);
    void update_max_length(uint32_t MAX_L);
    void clear_context();

    buffer<bf16> get_k_cache(int layer_idx, int idx);
    buffer<bf16> get_v_cache(int layer_idx, int idx);
    buffer<bf16> get_logits(buffer<bf16>& x);

    int get_current_context_length();
    int checkpoint();
    int restore();
    void _allocate_checkpoint_buffers();

    kv_row_layout kv_layout() const;
    std::unique_ptr<hybrid_engine_state> new_state();
    int read_state(hybrid_engine_state& state, int kv_from);
    int write_state(const hybrid_engine_state& state);
    int write_state_kv(const hybrid_engine_state& state);
    /// @brief one flag per layer: global attention (keeps K/V rows) or sliding window (keeps a ring)
    std::vector<bool> attn_mask;
    /// @brief highest context length reached since the global rows past the length were last zero (see kv_rows_zero)
    size_t dirty_rows = 0;
    /// @brief sliding_temp_cache holds what an earlier prefill left there (or was never zeroed)
    bool temp_dirty = true;
    void _zero_temp_cache();
    bool is_checkpoint_valid;
    std::vector<buffer<bf16>> kv_checkpoint;

    size_t K01_offset;
    size_t K23_offset;
    size_t V01_offset;
    size_t V23_offset;
    size_t sliding_K01_offset;
    size_t sliding_K23_offset;
    size_t sliding_V01_offset;
    size_t sliding_V23_offset;

    LM_Config config;
    npu_xclbin_manager *npu;

    void _set_rope_rms_weights(int idx);

    inline void _rms_norm(buffer<bf16>& y, buffer<bf16>& x, buffer<bf16>& rms_weight);
    inline void _rope_rms(buffer<bf16>& x, int idx, buffer<bf16>& rms_weight, int layer_idx);
    inline void _residual_add(buffer<bf16>& dest, buffer<bf16>& src, buffer<bf16>& residual);
    inline void _elementwise_mul(buffer<bf16>& dest, buffer<bf16>& a, buffer<bf16>& b);
    void fill_kv_cache(buffer<bf16>& buffer_k, buffer<bf16>& buffer_v, int L_offset, int L_begin, int L_effective, int layer_idx);
    void fill_kv_cache_sliding(buffer<bf16>& local_kv_cache, buffer<bf16>& buffer_k, buffer<bf16>& buffer_v, int L_offset, int L_begin, int L_end, int L_effective, int sliding_l_begin, int slidding_l_end, int layer_idx);
    inline bool is_sliding_window(int layer_idx) {return (layer_idx + 1) % SLIDING_WINDOW_PATTERN != 0;}
    inline int linear2ring(int l) {return l % SLIDING_WINDOW;}

    npu_app_manager* layer_app_manager;
    npu_app_manager* gemm_app_manager;
    npu_app_manager* dequant_app_manager;
    npu_app_manager* mha_app_manager;
    npu_app_manager* lm_head_app_manager;

    npu_app layer;
    npu_app sliding_layer;
    npu_app qkv_dequant;
    npu_app o_dequant;
    npu_app gate_dequant;
    npu_app up_dequant;

    npu_app down_dequant;

    npu_app q_proj;
    npu_app k_proj;
    npu_app v_proj;
    npu_app o_proj;
    npu_app up_proj;
    npu_app gate_proj;
    npu_app down_proj;
    npu_app mha_engine;
    npu_app sliding_mha_engine;
    npu_app lm_head;

    std::unique_ptr<plamo3_npu_sequence> sequence;
    std::unique_ptr<Gemm> gemm_seq_generator;
    std::unique_ptr<Dequant> dequant_seq_generator;
    std::unique_ptr<MHA> mha_seq_generator;
    uint32_t D;
    uint32_t DH;
    uint32_t DQ;
    uint32_t DK;
    uint32_t DV;
    uint32_t MAX_L;
    uint32_t HIDDEN_SIZE;
    uint32_t INTERMEDIATE_SIZE;
    uint32_t num_hidden_layers;
    int SLIDING_WINDOW;
    int SLIDING_WINDOW_PATTERN;
    uint32_t vocab_size;
    uint32_t vocab_size_padded;
    // size
    std::vector<buffer<bf16>> rms_weights;
    std::vector<buffer<bf16>> rope_rms_weights;
    std::vector<buffer<u8>> proj_weights;
    std::vector<buffer<bf16>> kv_caches;

    buffer<bf16> dequanted_qkv_weights; // QKV
    buffer<bf16> dequanted_o_weights; // O
    buffer<bf16> dequanted_up_weights; // UP,
    buffer<bf16> dequanted_gate_weights; // Gate
    buffer<bf16> dequanted_down_weights; // Down

    buffer<bf16> sliding_temp_cache;

    buffer<bf16> x;
    std::unique_ptr<Embedding> embedding;

    buffer<bf16> logits;
    buffer<bf16> logits_valid;
    buffer<u8> lm_head_weights;
    buffer<bf16> final_rms_weight;
    
    flm_rt::runlist layers_run;
    flm_rt::run lm_head_run;
    flm_rt::runlist dequant_all;

    int current_context_length;
    int checkpoint_context_length;
};
