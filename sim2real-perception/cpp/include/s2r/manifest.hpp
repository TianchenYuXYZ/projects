// deploy_bundle/manifest.json 的解析与校验 —— C++ 端唯一的契约入口。
#pragma once

#include <filesystem>
#include <string>

namespace s2r {

struct Manifest {
    int version = 0;
    int image_size = 0;
    int feature_dim = 0;
    int proprio_dim = 0;
    int action_dim = 0;
    int n_anchors = 0;
    float tau = 0.0f;
    std::string backbone;
    std::filesystem::path perception_onnx;
    std::filesystem::path policy_onnx;
    std::filesystem::path anchors_bin;   // 可能为空 (无门控)

    static Manifest load(const std::filesystem::path& bundle_dir);
};

}  // namespace s2r
