// 部署侧 OOD 门控: 与训练端 CosineFilter 同一套 anchors + tau。
// 约定 (manifest.gate.note): anchors 已 L2 归一化; 本类负责对输入特征归一化。
#pragma once

#include <filesystem>
#include <span>
#include <vector>

namespace s2r {

class CosineGate {
public:
    // anchors.bin: n_anchors x dim 的 row-major float32
    CosineGate(const std::filesystem::path& anchors_bin,
               int n_anchors, int dim, float tau);

    // 特征与最近 anchor 的余弦相似度
    float score(std::span<const float> feature) const;
    bool accept(std::span<const float> feature) const { return score(feature) >= tau_; }

    float tau() const { return tau_; }
    int n_anchors() const { return n_; }

private:
    std::vector<float> anchors_;  // 扁平 n*dim
    int n_;
    int dim_;
    float tau_;
};

}  // namespace s2r
