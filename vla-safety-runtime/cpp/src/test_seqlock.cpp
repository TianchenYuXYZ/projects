// seqlock 协议压力自测: 写线程 ~MHz 级更新, 读线程并发校验快照一致性。
// payload 全部字段由 cmd_id 推导 (v[i] = cmd_id * (i+1)), 任何撕裂读
// 都会破坏该不变量。期望: torn == 0。退出码非 0 即失败。
#include <windows.h>

#include <atomic>
#include <cstdio>
#include <thread>

#include "vla/seqlock_ring.hpp"

int main() {
    alignas(64) static vla::SlotLayout slot{};
    slot.magic = vla::kMagic;

    std::atomic<bool> stop{false};
    std::atomic<uint64_t> writes{0};

    std::thread writer([&] {
        uint32_t seq = 0;
        uint64_t id = 0;
        while (!stop.load(std::memory_order_relaxed)) {
            ++id;
            vla::Command c{};
            c.qpc_write = static_cast<int64_t>(id * 7);
            c.cmd_id = id;
            for (int i = 0; i < 4; ++i)
                c.v[i] = static_cast<float>(id % 100003) * static_cast<float>(i + 1);
            c.source = static_cast<uint32_t>(id % 3);
            c.flags = static_cast<uint32_t>(id & 0xFFFF);
            vla::write_slot(&slot, c, seq);
            writes.fetch_add(1, std::memory_order_relaxed);
        }
    });

    uint64_t reads = 0, torn = 0, retry_fail = 0;
    const auto t_end = GetTickCount64() + 2000;   // 2s
    while (GetTickCount64() < t_end) {
        vla::Command c;
        if (!vla::read_slot(&slot, c)) {
            ++retry_fail;
            continue;
        }
        ++reads;
        if (c.cmd_id == 0) continue;              // 写端尚未首写
        const float base = static_cast<float>(c.cmd_id % 100003);
        bool ok = c.qpc_write == static_cast<int64_t>(c.cmd_id * 7)
                  && c.source == static_cast<uint32_t>(c.cmd_id % 3)
                  && c.flags == static_cast<uint32_t>(c.cmd_id & 0xFFFF);
        for (int i = 0; i < 4; ++i)
            ok = ok && c.v[i] == base * static_cast<float>(i + 1);
        if (!ok) ++torn;
    }
    stop.store(true);
    writer.join();

    std::printf(
        "{\"writes\": %llu, \"reads\": %llu, \"torn\": %llu, "
        "\"retry_exhausted\": %llu}\n",
        static_cast<unsigned long long>(writes.load()),
        static_cast<unsigned long long>(reads),
        static_cast<unsigned long long>(torn),
        static_cast<unsigned long long>(retry_fail));
    if (torn != 0) {
        std::fprintf(stderr, "FAIL: 检测到 %llu 次撕裂读\n",
                     static_cast<unsigned long long>(torn));
        return 1;
    }
    return 0;
}
