// 轻量断言测试 (无框架依赖): 图像流水线布局/数值 + cosine 门控数学。
#include <opencv2/core.hpp>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <vector>

#include "s2r/cosine_gate.hpp"
#include "s2r/image_pipeline.hpp"

namespace {

bool near(float a, float b, float tol = 1e-5f) { return std::fabs(a - b) < tol; }

void test_image_pipeline_layout() {
    // 2x2 已知像素, 验证 BGR->RGB 与 CHW 排布
    cv::Mat bgr(2, 2, CV_8UC3);
    bgr.at<cv::Vec3b>(0, 0) = {255, 0, 0};      // B=255 -> 蓝色像素
    bgr.at<cv::Vec3b>(0, 1) = {0, 255, 0};      // 绿
    bgr.at<cv::Vec3b>(1, 0) = {0, 0, 255};      // 红
    bgr.at<cv::Vec3b>(1, 1) = {0, 0, 0};

    s2r::ImagePipeline pipe(2);  // 无缩放路径
    auto chw = pipe.preprocess(bgr);
    assert(chw.size() == 3 * 2 * 2);
    // R 平面: (0,0)=0, (1,0)=1
    assert(near(chw[0], 0.0f) && near(chw[2], 1.0f));
    // G 平面: (0,1)=1
    assert(near(chw[4 + 1], 1.0f));
    // B 平面: (0,0)=1
    assert(near(chw[8 + 0], 1.0f));
    std::puts("[ok] image_pipeline layout");
}

void test_image_pipeline_resize() {
    cv::Mat bgr(64, 48, CV_8UC3, cv::Scalar(10, 20, 30));
    s2r::ImagePipeline pipe(224);
    auto chw = pipe.preprocess(bgr);
    assert(chw.size() == 3u * 224 * 224);
    // 常数图像缩放后仍是常数
    assert(near(chw[0], 30.0f / 255.0f) && near(chw[224 * 224], 20.0f / 255.0f));
    std::puts("[ok] image_pipeline resize");
}

void test_cosine_gate() {
    namespace fs = std::filesystem;
    const fs::path tmp = fs::temp_directory_path() / "s2r_anchors_test.bin";
    // 2 个 4 维 anchor: e0, e1 (已归一化)
    std::vector<float> anchors = {1, 0, 0, 0,
                                  0, 1, 0, 0};
    std::ofstream(tmp, std::ios::binary)
        .write(reinterpret_cast<const char*>(anchors.data()),
               static_cast<std::streamsize>(anchors.size() * sizeof(float)));

    s2r::CosineGate gate(tmp, 2, 4, 0.9f);
    // 未归一化输入: (10,0,0,0) 与 e0 余弦 = 1
    std::vector<float> f1 = {10, 0, 0, 0};
    assert(near(gate.score(f1), 1.0f));
    assert(gate.accept(f1));
    // (1,1,0,0)/sqrt2 -> max cos = 0.7071 < 0.9
    std::vector<float> f2 = {1, 1, 0, 0};
    assert(near(gate.score(f2), 0.70710678f));
    assert(!gate.accept(f2));
    fs::remove(tmp);
    std::puts("[ok] cosine_gate");
}

}  // namespace

int main() {
    test_image_pipeline_layout();
    test_image_pipeline_resize();
    test_cosine_gate();
    std::puts("all C++ tests passed");
    return 0;
}
