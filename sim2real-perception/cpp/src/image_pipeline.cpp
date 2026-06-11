#include "s2r/image_pipeline.hpp"

#include <opencv2/imgproc.hpp>

#include <stdexcept>

namespace s2r {

std::vector<float> ImagePipeline::preprocess(const cv::Mat& bgr) const {
    if (bgr.empty() || bgr.type() != CV_8UC3) {
        throw std::invalid_argument("ImagePipeline expects non-empty 8UC3 image");
    }
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    return to_chw(rgb);
}

std::vector<float> ImagePipeline::preprocess_rgb(const unsigned char* rgb,
                                                 int h, int w) const {
    // 不拷贝地包一层 Mat 头 (上游缓冲在调用期内有效)
    cv::Mat view(h, w, CV_8UC3, const_cast<unsigned char*>(rgb));
    return to_chw(view);
}

std::vector<float> ImagePipeline::to_chw(const cv::Mat& rgb_in) const {
    cv::Mat rgb = rgb_in;
    if (rgb.cols != size_ || rgb.rows != size_) {
        cv::resize(rgb_in, rgb, cv::Size(size_, size_), 0, 0, cv::INTER_LINEAR);
    }
    std::vector<float> chw(static_cast<size_t>(3) * size_ * size_);
    const int plane = size_ * size_;
    for (int y = 0; y < size_; ++y) {
        const unsigned char* row = rgb.ptr<unsigned char>(y);
        float* r = chw.data() + 0 * plane + y * size_;
        float* g = chw.data() + 1 * plane + y * size_;
        float* b = chw.data() + 2 * plane + y * size_;
        for (int x = 0; x < size_; ++x) {
            r[x] = row[3 * x + 0] * (1.0f / 255.0f);
            g[x] = row[3 * x + 1] * (1.0f / 255.0f);
            b[x] = row[3 * x + 2] * (1.0f / 255.0f);
        }
    }
    return chw;
}

}  // namespace s2r
