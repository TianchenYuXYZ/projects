// seqlock 最新值槽 —— 跨进程恢复通道 (C++ 读端 / 自测写端)。
// 内存布局与 src/vla_safety/runtime/ring.py 严格一致 (64B, 单 cache line)。
//
// 协议: 写端 seq 先置奇 -> 写 payload -> seq 置偶; 读端读 seq(偶) ->
// 拷 payload -> 复核 seq 未变。x86-64 TSO 下配合 acquire/release fence
// 即保证无锁一致性; 读端无写副作用, 任意多读者。
#pragma once

#include <atomic>
#include <cstdint>

namespace vla {

constexpr uint32_t kMagic = 0x564C4131u;   // 'VLA1'
constexpr size_t kSlotSize = 64;

#pragma pack(push, 1)
struct SlotLayout {
    uint32_t seq;
    uint32_t magic;
    int64_t  qpc_write;
    uint64_t cmd_id;
    float    v[4];
    uint32_t source;     // 0=vla, 1=recovery, 2=qp
    uint32_t flags;
    uint8_t  pad[16];
};
#pragma pack(pop)
static_assert(sizeof(SlotLayout) == kSlotSize, "布局必须与 Python 写端一致");

struct Command {
    int64_t  qpc_write;
    uint64_t cmd_id;
    float    v[4];
    uint32_t source;
    uint32_t flags;
};

inline bool read_slot(const volatile SlotLayout* slot, Command& out) {
    for (int retry = 0; retry < 64; ++retry) {
        const uint32_t s0 = slot->seq;
        std::atomic_thread_fence(std::memory_order_acquire);
        if (s0 & 1u) continue;                       // 写入中
        Command tmp;
        tmp.qpc_write = slot->qpc_write;
        tmp.cmd_id    = slot->cmd_id;
        for (int i = 0; i < 4; ++i) tmp.v[i] = slot->v[i];
        tmp.source = slot->source;
        tmp.flags  = slot->flags;
        std::atomic_thread_fence(std::memory_order_acquire);
        const uint32_t s1 = slot->seq;
        if (s0 == s1) { out = tmp; return true; }    // 快照一致
    }
    return false;                                    // 写入风暴, 本轮放弃
}

// 自测用写端 (生产写端是 Python SeqlockWriter)
inline void write_slot(volatile SlotLayout* slot, const Command& cmd,
                       uint32_t& seq) {
    slot->seq = ++seq;                               // 奇数: 写入中
    std::atomic_thread_fence(std::memory_order_release);
    slot->qpc_write = cmd.qpc_write;
    slot->cmd_id    = cmd.cmd_id;
    for (int i = 0; i < 4; ++i) slot->v[i] = cmd.v[i];
    slot->source = cmd.source;
    slot->flags  = cmd.flags;
    std::atomic_thread_fence(std::memory_order_release);
    slot->seq = ++seq;                               // 偶数: 稳定
}

}  // namespace vla
