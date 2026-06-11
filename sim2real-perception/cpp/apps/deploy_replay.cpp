// deploy_replay <bundle_dir> <parity_dir>
//
// 读取 parity_dir/proprio.csv (每行: 文件名, p0..p7), 逐帧推理,
// 输出 actions_cpp.csv (parity 比对用) 与 latency.json (延迟报告)。
// 注: parity 比对的是 *策略动作本身*, 门控 accepted 仅记录不替换。
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "s2r/runtime.hpp"

namespace {

struct Row {
    std::string file;
    std::vector<float> proprio;
};

std::vector<Row> read_csv(const std::filesystem::path& p) {
    std::ifstream f(p);
    if (!f) throw std::runtime_error("cannot open " + p.string());
    std::vector<Row> rows;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string cell;
        Row r;
        std::getline(ss, r.file, ',');
        while (std::getline(ss, cell, ',')) {
            r.proprio.push_back(std::stof(cell));
        }
        rows.push_back(std::move(r));
    }
    return rows;
}

double pct(std::vector<long long> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const auto idx = static_cast<size_t>(p * (v.size() - 1));
    return static_cast<double>(v[idx]);
}

double mean(const std::vector<long long>& v) {
    if (v.empty()) return 0.0;
    return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: deploy_replay <bundle_dir> <parity_dir>\n";
        return 2;
    }
    const std::filesystem::path bundle_dir = argv[1];
    const std::filesystem::path parity_dir = argv[2];

    try {
        s2r::DeployRuntime runtime(bundle_dir);
        const auto rows = read_csv(parity_dir / "proprio.csv");
        std::cout << "[deploy_replay] bundle=" << bundle_dir
                  << " backbone=" << runtime.manifest().backbone
                  << " frames=" << rows.size() << "\n";

        std::ofstream out(parity_dir / "actions_cpp.csv");
        out.setf(std::ios::fixed);
        out.precision(8);

        std::vector<long long> t_pre, t_per, t_pol, t_tot;
        int n_rejected = 0;
        // 预热一帧 (会话首跑包含图优化/内存分配, 不计入统计)
        if (!rows.empty()) {
            cv::Mat img = cv::imread((parity_dir / rows[0].file).string());
            (void)runtime.step(img, rows[0].proprio);
        }
        for (const auto& row : rows) {
            cv::Mat img = cv::imread((parity_dir / row.file).string());
            auto r = runtime.step(img, row.proprio);
            for (size_t i = 0; i < r.action.size(); ++i) {
                out << r.action[i] << (i + 1 < r.action.size() ? "," : "\n");
            }
            t_pre.push_back(r.t_preprocess_us);
            t_per.push_back(r.t_perception_us);
            t_pol.push_back(r.t_policy_us);
            t_tot.push_back(r.t_total_us);
            if (!r.accepted) ++n_rejected;
        }

        std::ofstream lat(parity_dir / "latency.json");
        lat.setf(std::ios::fixed);
        lat.precision(1);
        lat << "{\n"
            << "  \"n_frames\": " << rows.size() << ",\n"
            << "  \"gate_rejected\": " << n_rejected << ",\n"
            << "  \"preprocess_us\": {\"mean\": " << mean(t_pre)
            << ", \"p50\": " << pct(t_pre, 0.5) << ", \"p99\": " << pct(t_pre, 0.99) << "},\n"
            << "  \"perception_us\": {\"mean\": " << mean(t_per)
            << ", \"p50\": " << pct(t_per, 0.5) << ", \"p99\": " << pct(t_per, 0.99) << "},\n"
            << "  \"policy_us\": {\"mean\": " << mean(t_pol)
            << ", \"p50\": " << pct(t_pol, 0.5) << ", \"p99\": " << pct(t_pol, 0.99) << "},\n"
            << "  \"total_us\": {\"mean\": " << mean(t_tot)
            << ", \"p50\": " << pct(t_tot, 0.5) << ", \"p99\": " << pct(t_tot, 0.99) << "},\n"
            << "  \"fps_mean\": " << (mean(t_tot) > 0 ? 1e6 / mean(t_tot) : 0.0) << "\n"
            << "}\n";

        std::cout << "[deploy_replay] mean total " << mean(t_tot) / 1000.0
                  << " ms (" << (mean(t_tot) > 0 ? 1e6 / mean(t_tot) : 0.0)
                  << " fps), rejected " << n_rejected << "/" << rows.size() << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[deploy_replay] error: " << e.what() << "\n";
        return 1;
    }
}
