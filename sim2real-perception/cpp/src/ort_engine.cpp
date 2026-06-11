#include "s2r/ort_engine.hpp"

#include <stdexcept>

namespace s2r {

Ort::Env& OrtEngine::env() {
    static Ort::Env e(ORT_LOGGING_LEVEL_WARNING, "s2r");
    return e;
}

OrtEngine::OrtEngine(const std::filesystem::path& onnx_path, int intra_threads)
    : mem_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(intra_threads);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_ = Ort::Session(env(), onnx_path.wstring().c_str(), opts);

    Ort::AllocatorWithDefaultOptions alloc;
    for (size_t i = 0; i < session_.GetInputCount(); ++i) {
        input_names_.emplace_back(session_.GetInputNameAllocated(i, alloc).get());
    }
    for (size_t i = 0; i < session_.GetOutputCount(); ++i) {
        output_names_.emplace_back(session_.GetOutputNameAllocated(i, alloc).get());
    }
}

std::vector<float> OrtEngine::run(
    const std::vector<std::span<const float>>& inputs,
    const std::vector<std::vector<int64_t>>& shapes) {
    if (inputs.size() != input_names_.size() || inputs.size() != shapes.size()) {
        throw std::invalid_argument("OrtEngine::run input arity mismatch");
    }
    std::vector<Ort::Value> tensors;
    tensors.reserve(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        tensors.push_back(Ort::Value::CreateTensor<float>(
            mem_info_, const_cast<float*>(inputs[i].data()), inputs[i].size(),
            shapes[i].data(), shapes[i].size()));
    }
    std::vector<const char*> in_names, out_names;
    for (const auto& n : input_names_) in_names.push_back(n.c_str());
    for (const auto& n : output_names_) out_names.push_back(n.c_str());

    auto outputs = session_.Run(Ort::RunOptions{nullptr}, in_names.data(),
                                tensors.data(), tensors.size(),
                                out_names.data(), 1);
    const float* data = outputs[0].GetTensorData<float>();
    const size_t count = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
    return {data, data + count};
}

}  // namespace s2r
