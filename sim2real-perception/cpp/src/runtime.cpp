#include "s2r/runtime.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>

namespace s2r {

namespace {
long long us_since(const std::chrono::steady_clock::time_point& t0) {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now() - t0).count();
}
}  // namespace

DeployRuntime::DeployRuntime(const std::filesystem::path& bundle_dir)
    : manifest_(Manifest::load(bundle_dir)),
      pipeline_(manifest_.image_size),
      perception_(std::make_unique<OrtEngine>(manifest_.perception_onnx)),
      policy_(std::make_unique<OrtEngine>(manifest_.policy_onnx)) {
    if (manifest_.n_anchors > 0) {
        gate_.emplace(manifest_.anchors_bin, manifest_.n_anchors,
                      manifest_.feature_dim, manifest_.tau);
    }
}

StepResult DeployRuntime::step(const cv::Mat& bgr, std::span<const float> proprio) {
    auto t0 = std::chrono::steady_clock::now();
    auto chw = pipeline_.preprocess(bgr);
    return infer(std::move(chw), proprio, us_since(t0));
}

StepResult DeployRuntime::step_rgb(const unsigned char* rgb, int h, int w,
                                   std::span<const float> proprio) {
    auto t0 = std::chrono::steady_clock::now();
    auto chw = pipeline_.preprocess_rgb(rgb, h, w);
    return infer(std::move(chw), proprio, us_since(t0));
}

StepResult DeployRuntime::infer(std::vector<float>&& chw,
                                std::span<const float> proprio,
                                long long t_pre_us) {
    if (static_cast<int>(proprio.size()) != manifest_.proprio_dim) {
        throw std::invalid_argument("proprio dim mismatch");
    }
    StepResult r;
    r.t_preprocess_us = t_pre_us;
    const auto s = static_cast<int64_t>(manifest_.image_size);

    auto t1 = std::chrono::steady_clock::now();
    auto feature = perception_->run({chw}, {{1, 3, s, s}});
    r.t_perception_us = us_since(t1);

    if (gate_) {
        r.gate_score = gate_->score(feature);
        r.accepted = r.gate_score >= gate_->tau();
    }

    auto t2 = std::chrono::steady_clock::now();
    r.action = policy_->run(
        {feature, proprio},
        {{1, manifest_.feature_dim}, {1, manifest_.proprio_dim}});
    // 策略头是线性输出, 动作契约为 [-1,1], 部署端负责裁剪
    for (auto& v : r.action) {
        v = std::clamp(v, -1.0f, 1.0f);
    }
    r.t_policy_us = us_since(t2);

    r.t_total_us = r.t_preprocess_us + r.t_perception_us + r.t_policy_us;
    return r;
}

}  // namespace s2r
