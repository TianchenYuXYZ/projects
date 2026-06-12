// 200Hz 控制环消费端: 打开命名共享内存, 轮询 seqlock 槽, 记录
// 每条指令首次可见的跨进程延迟 (写端 QPC -> 读端 QPC, 同源时钟)。
//
// 用法:
//   control_loop --shm Local\vla_safety_ring --hz 200 --duration 20
//                --out latency.csv [--wait 15]
//   --hz 0  = 自旋模式 (测纯 IPC 可见性延迟, 不含控制周期相位)
//
// 结束时输出 JSON 摘要到 stdout (脚本端解析)。
#include <windows.h>
#include <timeapi.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "vla/seqlock_ring.hpp"

namespace {

int64_t qpc_now() {
    LARGE_INTEGER v;
    QueryPerformanceCounter(&v);
    return v.QuadPart;
}

int64_t qpc_freq() {
    LARGE_INTEGER v;
    QueryPerformanceFrequency(&v);
    return v.QuadPart;
}

struct Sample {
    uint64_t cmd_id;
    int64_t  qpc_write;
    int64_t  qpc_read;
    uint32_t source;
};

double percentile(std::vector<double>& xs, double p) {
    if (xs.empty()) return -1.0;
    std::sort(xs.begin(), xs.end());
    const double idx = p / 100.0 * (static_cast<double>(xs.size()) - 1.0);
    const size_t lo = static_cast<size_t>(idx);
    const size_t hi = (lo + 1 < xs.size()) ? lo + 1 : lo;
    const double frac = idx - static_cast<double>(lo);
    return xs[lo] * (1.0 - frac) + xs[hi] * frac;
}

}  // namespace

int main(int argc, char** argv) {
    std::string shm_name = "Local\\vla_safety_ring";
    std::string out_path = "control_latency.csv";
    double hz = 200.0;
    double duration_s = 20.0;
    double wait_s = 15.0;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!std::strcmp(argv[i], "--shm")) shm_name = argv[i + 1];
        else if (!std::strcmp(argv[i], "--hz")) hz = std::atof(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--duration")) duration_s = std::atof(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--out")) out_path = argv[i + 1];
        else if (!std::strcmp(argv[i], "--wait")) wait_s = std::atof(argv[i + 1]);
    }

    timeBeginPeriod(1);
    const int64_t freq = qpc_freq();

    // 等待 Python 写端创建命名映射
    HANDLE hmap = nullptr;
    const int64_t wait_deadline = qpc_now() + static_cast<int64_t>(wait_s * freq);
    while (qpc_now() < wait_deadline) {
        hmap = OpenFileMappingA(FILE_MAP_READ, FALSE, shm_name.c_str());
        if (hmap) break;
        Sleep(20);
    }
    if (!hmap) {
        std::fprintf(stderr, "OpenFileMapping 失败: %s (写端未启动?)\n",
                     shm_name.c_str());
        return 2;
    }
    const auto* slot = static_cast<const volatile vla::SlotLayout*>(
        MapViewOfFile(hmap, FILE_MAP_READ, 0, 0, 0));
    if (!slot) {
        std::fprintf(stderr, "MapViewOfFile 失败\n");
        return 2;
    }
    // magic 校验 (允许写端稍后初始化)
    {
        const int64_t magic_deadline = qpc_now() + freq * 5;
        while (slot->magic != vla::kMagic && qpc_now() < magic_deadline) Sleep(5);
        if (slot->magic != vla::kMagic) {
            std::fprintf(stderr, "magic 不匹配: 0x%08X\n", slot->magic);
            return 2;
        }
    }

    std::vector<Sample> samples;
    samples.reserve(static_cast<size_t>(duration_s * (hz > 0 ? hz : 2.0e5)) + 1024);

    const int64_t t_end = qpc_now() + static_cast<int64_t>(duration_s * freq);
    const int64_t period = (hz > 0) ? static_cast<int64_t>(freq / hz) : 0;
    int64_t next_tick = qpc_now();
    uint64_t last_cmd = 0;
    uint64_t read_failures = 0;
    uint64_t loop_iters = 0;

    while (qpc_now() < t_end) {
        if (period > 0) {
            // 混合等待: 距下一拍 > 1ms 时让出 CPU, 否则自旋保精度
            next_tick += period;
            int64_t now = qpc_now();
            if (next_tick < now) next_tick = now;          // 落拍即重相位
            while ((now = qpc_now()) < next_tick) {
                if (next_tick - now > freq / 1000) Sleep(0);
                else YieldProcessor();
            }
        }
        ++loop_iters;
        vla::Command cmd;
        if (!vla::read_slot(slot, cmd)) {
            ++read_failures;
            continue;
        }
        if (cmd.cmd_id != 0 && cmd.cmd_id != last_cmd) {
            last_cmd = cmd.cmd_id;
            samples.push_back({cmd.cmd_id, cmd.qpc_write, qpc_now(), cmd.source});
        }
    }
    timeEndPeriod(1);

    // CSV 落盘
    if (FILE* f = std::fopen(out_path.c_str(), "w")) {
        std::fprintf(f, "cmd_id,qpc_write,qpc_read,latency_us,source\n");
        for (const auto& s : samples) {
            const double us =
                static_cast<double>(s.qpc_read - s.qpc_write) * 1e6 /
                static_cast<double>(freq);
            std::fprintf(f, "%llu,%lld,%lld,%.3f,%u\n",
                         static_cast<unsigned long long>(s.cmd_id),
                         static_cast<long long>(s.qpc_write),
                         static_cast<long long>(s.qpc_read), us, s.source);
        }
        std::fclose(f);
    } else {
        std::fprintf(stderr, "无法写出 %s\n", out_path.c_str());
        return 3;
    }

    std::vector<double> lat;
    lat.reserve(samples.size());
    for (const auto& s : samples)
        lat.push_back(static_cast<double>(s.qpc_read - s.qpc_write) * 1e6 /
                      static_cast<double>(freq));
    std::vector<double> tmp = lat;
    const double p50 = percentile(tmp, 50.0);
    const double p95 = percentile(tmp, 95.0);
    const double p99 = percentile(tmp, 99.0);
    double mean = 0.0;
    for (double x : lat) mean += x;
    if (!lat.empty()) mean /= static_cast<double>(lat.size());

    // JSON 摘要 (stdout, 由 05_bench_latency.py 解析)
    std::printf(
        "{\"n_commands\": %zu, \"loop_iters\": %llu, \"read_failures\": %llu, "
        "\"latency_us\": {\"mean\": %.3f, \"p50\": %.3f, \"p95\": %.3f, "
        "\"p99\": %.3f}, \"hz\": %.1f}\n",
        samples.size(), static_cast<unsigned long long>(loop_iters),
        static_cast<unsigned long long>(read_failures), mean, p50, p95, p99, hz);
    return 0;
}
