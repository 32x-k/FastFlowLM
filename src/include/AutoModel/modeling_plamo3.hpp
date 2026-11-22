/// \file modeling_plamo3.hpp
/// \brief plamo3 class
/// \author FastFlowLM Team
/// \note This is a header file for the plamo3 class

#pragma once
#include "AutoModel/automodel.hpp"

/************              plamo3 family            **************/
class Plamo3_Translate : public AutoModel {
private:
    void setup_tokenizer(std::string model_path);
    /// \brief clear the conversation after a failed inference and report it
    /// \param detail the backend's own message, when the failure carried one
    /// \throws ModelRequestError 500, asking for a reload when the backend is poisoned
    [[noreturn]] void fail_inference(const std::string& detail = {});

public:
    Plamo3_Translate(flm_rt::device* npu_device_inst);

    void load_model(std::string model_path, json model_inf, int default_context_length = -1, bool enable_preemption = false, const std::string& backend = "") override;
    bool insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input, std::function<bool()> is_cancelled = [] { return false; }) override;
    std::string generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled = [] { return false; }) override;
    std::string generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input, int length_limit, std::ostream& os = std::cout) override;
    std::string apply_chat_template(nlohmann::ordered_json& messages, nlohmann::ordered_json tools = nlohmann::ordered_json::object(), bool continue_final_message = false) override;
    bool supports_continue_final_message() const override { return true; }
    bool check_using_checkpint() {
		return false;
	}
};
