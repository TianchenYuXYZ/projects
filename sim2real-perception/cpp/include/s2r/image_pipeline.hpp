// 图像预处理: 解码后的图像 -> float32 CHW [0,1] RGB, 尺寸对齐到模型输入。
// 与 Python 训练端约定一致 (manifest.image): 归一化(mean/std)已烘焙进 ONNX 图,
// 这里只负责 布局/色序/缩放/[0,1]。
#pragma once

#include <opencv2/core.hpp>
#include <vector>

namespace s2r {

class ImagePipeline {
public:
    explicit ImagePipeline(int target_size) : size_(target_size) {}

    // bgr: cv::imread 的原生输出 (8UC3, BGR)。返回 3*size*size 的 CHW 缓冲。
    std::vector<float> preprocess(const cv::Mat& bgr) const;

    // rgb_u8: 来自 Python/相机的 HxWx3 RGB 紧凑缓冲 (pybind 路径)。
    std::vector<float> preprocess_rgb(const unsigned char* rgb, int h, int w) const;

    int size() const { return size_; }

private:
    std::vector<float> to_chw(const cv::Mat& rgb) const;
    int size_;
};

}  // namespace s2r
