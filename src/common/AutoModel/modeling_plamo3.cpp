/// \file modeling_plamo3.cpp
/// \brief plamo3 class
/// \author FastFlowLM Team
/// \note This is a source file for the plamo3 class

#include "AutoModel/modeling_plamo3.hpp"
#include "models/plamo3/flm/aie2p/plamo3_npu.hpp"

/************              Plamo3_Translate family            **************/
Plamo3_Translate::Plamo3_Translate(flm_rt::device* npu_device_inst) : AutoModel(npu_device_inst, "Plamo3_Translate") {}

void Plamo3_Translate::load_model(std::string model_path, json model_info, int default_context_length, bool enable_preemption, const std::string& backend) {
    this->_shared_load_backend(model_path, model_info, default_context_length, enable_preemption, backend);
    this->setup_tokenizer(model_path);
    this->sampler.reset();

    sampler_config config;
    config.top_k = 1;
    config.top_p = 1.0f;
    config.min_p = 0.0f;
    config.temperature = 0.0f;

    this->set_sampler(config);
    for (size_t i = 0; i < PROFILER_TYPE_NUM; i++) {
        this->profiler_list[i].reset();
    }
    this->_prefix_attach(this->lm_engine, plamo3_npu_state_api(this->lm_engine));
}

void Plamo3_Translate::setup_tokenizer(std::string model_path) {
    auto tokenizer_config = this->_shared_setup_tokenizer(model_path);
}

std::string Plamo3_Translate::apply_chat_template(nlohmann::ordered_json& messages, nlohmann::ordered_json tools, bool continue_final_message) {
    minja::chat_template_inputs inputs;
    inputs.messages = messages;
    inputs.extra_context = this->extra_context;
    return this->_render_chat_template(inputs, continue_final_message);
}

void Plamo3_Translate::fail_inference(const std::string& detail) {
    const bool poisoned = this->backend_ && this->backend_->poisoned();
    this->_shared_after_inference_failure(poisoned);
    std::string message = poisoned
        ? "Inference failed; unload/reload is required because the model is poisoned"
        : "Inference failed; the current conversation was cleared";
    if (!detail.empty()) message += " (" + detail + ")";
    throw ModelRequestError(500, true, message);
}

bool Plamo3_Translate::insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input, std::function<bool()> is_cancelled) {
    this->_shared_guard_poisoned();
    // preprocess
    this->profiler_list[TKOEN_ENCODE_TIME].start();
    std::string templated_text;
    if (input.messages.empty() && input.prompt.empty()) {
        header_print("WARNING", "No messages or prompt provided");
        return false;
    }
    if (!input.messages.empty()) { // already a formated messages, usually from REST API
        templated_text = this->apply_chat_template(input.messages, nlohmann::ordered_json::object(), meta_info.continue_final_message);
    }
    else if (!input.prompt.empty()) { // a pure text, usually from the cli
        nlohmann::ordered_json messages;

        messages.push_back({ {"role", "user"}, {"content", input.prompt} });
        templated_text = this->apply_chat_template(messages);
    }

    std::vector<int> tokens = this->tokenizer->encode(templated_text);

    this->profiler_list[TKOEN_ENCODE_TIME].stop(tokens.size());
    // hardware

    this->_prefix_legacy_path();
    try {
        return this->_shared_insert(meta_info, tokens, is_cancelled, nullptr, 0, input.requested_max_new_tokens);
    } catch (const ModelRequestError&) {
        throw;
    } catch (const std::exception& error) {
        this->fail_inference(error.what());
    } catch (...) {
        this->fail_inference();
    }
}

std::string Plamo3_Translate::generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled) {
    this->_shared_guard_poisoned();
    try {
        return this->_shared_generate(meta_info, length_limit, os, is_cancelled);
    } catch (const ModelRequestError&) {
        throw;
    } catch (const std::exception& error) {
        this->fail_inference(error.what());
    } catch (...) {
        this->fail_inference();
    }
}

std::string Plamo3_Translate::generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input, int length_limit, std::ostream& os) {
    if (!this->insert(meta_info, input)) {
        return "";
    }
    return this->generate(meta_info, length_limit, os);
}
