// DeployRuntime: 部署主循环的单步推理。
//   图像 -> ImagePipeline -> perception.onnx -> [CosineGate] -> policy.onnx -> 动作
// 门控拒绝 (OOD) 时仍返回计算出的动作, 由 accepted=false 标记;
// 上层闭环可据此切换到安全保持策略。
#pragma once

#include <opencv2/core.hpp>

#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include "s2r/cosine_gate.hpp"
#include "s2r/image_pipeline.hpp"
#include "s2r/manifest.hpp"
#include "s2r/ort_engine.hpp"

namespace s2r {

struct StepResult {
    std::vector<float> action;     // (action_dim,)
    bool accepted = true;          // 门控判定
    float gate_score = 0.0f;
    // 分段延迟 (microseconds)
    long long t_preprocess_us = 0;
    long long t_perception_us = 0;
    long long t_policy_us = 0;
    long long t_total_us = 0;
};

class DeployRuntime {
public:
    explicit DeployRuntime(const std::filesystem::path& bundle_dir);

    StepResult step(const cv::Mat& bgr, std::span<const float> proprio);
    StepResult step_rgb(const unsigned char* rgb, int h, int w,
                        std::span<const float> proprio);

    const Manifest& manifest() const { return manifest_; }

private:
    StepResult infer(std::vector<float>&& chw, std::span<const float> proprio,
                     long long t_pre_us);

    Manifest manifest_;
    ImagePipeline pipeline_;
    std::unique_ptr<OrtEngine> perception_;
    std::unique_ptr<OrtEngine> policy_;
    std::optional<CosineGate> gate_;
};

}  // namespace s2r
