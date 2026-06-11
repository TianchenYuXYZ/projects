// ONNX Runtime C++ API 的薄封装: 单输入/多输入 -> 单输出 float 张量。
#pragma once

#include <onnxruntime_cxx_api.h>

#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace s2r {

class OrtEngine {
public:
    explicit OrtEngine(const std::filesystem::path& onnx_path, int intra_threads = 2);

    // inputs[i] 与模型第 i 个输入对位; shapes[i] 为对应维度。返回第 0 个输出。
    std::vector<float> run(const std::vector<std::span<const float>>& inputs,
                           const std::vector<std::vector<int64_t>>& shapes);

    const std::vector<std::string>& input_names() const { return input_names_; }

private:
    static Ort::Env& env();
    Ort::Session session_{nullptr};
    Ort::MemoryInfo mem_info_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
};

}  // namespace s2r
