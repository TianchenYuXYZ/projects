#include "s2r/cosine_gate.hpp"

#include <cmath>
#include <fstream>
#include <stdexcept>

namespace s2r {

CosineGate::CosineGate(const std::filesystem::path& anchors_bin,
                       int n_anchors, int dim, float tau)
    : anchors_(static_cast<size_t>(n_anchors) * dim), n_(n_anchors),
      dim_(dim), tau_(tau) {
    std::ifstream f(anchors_bin, std::ios::binary);
    if (!f) {
        throw std::runtime_error("cannot open " + anchors_bin.string());
    }
    f.read(reinterpret_cast<char*>(anchors_.data()),
           static_cast<std::streamsize>(anchors_.size() * sizeof(float)));
    if (f.gcount() != static_cast<std::streamsize>(anchors_.size() * sizeof(float))) {
        throw std::runtime_error("anchors.bin size mismatch (expect " +
                                 std::to_string(n_anchors) + "x" +
                                 std::to_string(dim) + " float32)");
    }
}

float CosineGate::score(std::span<const float> feature) const {
    if (static_cast<int>(feature.size()) != dim_) {
        throw std::invalid_argument("feature dim mismatch");
    }
    // anchors 已归一化 (训练端契约), 这里只归一化输入特征
    double sq = 0.0;
    for (float v : feature) sq += static_cast<double>(v) * v;
    const float inv_norm = 1.0f / static_cast<float>(std::sqrt(std::max(sq, 1e-16)));

    float best = -1.0f;
    for (int k = 0; k < n_; ++k) {
        const float* a = anchors_.data() + static_cast<size_t>(k) * dim_;
        float dot = 0.0f;
        for (int i = 0; i < dim_; ++i) {
            dot += a[i] * feature[i];
        }
        dot *= inv_norm;
        if (dot > best) best = dot;
    }
    return best;
}

}  // namespace s2r
