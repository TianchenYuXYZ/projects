# Sim-to-Real Robust Perception Pipeline

> Domain Randomization + CLIP-guided generation + R3M feature filtering + LoRA
> fine-tuning,解决机器人操作从仿真到部署的感知退化 (perception degradation)
> 问题。Python 负责训练链路,C++ 负责部署 runtime。

## 问题

单条专家 demo 在固定环境采集,部署环境一旦变化(光照/纹理/干扰物/相机),
visual encoder 输出的特征先于策略失效。本项目复现一条工程化解法:

1. **DR 增广**: 视觉随机化不改物理 → 回放 demo 动作序列,单 demo 放大为
   1500 个随机化场景的训练集,标签天然有效
2. **CLIP 引导**: 按表面语义 prompt 给候选纹理打分,剔除离谱的随机化
3. **R3M cosine 过滤**: demo 关键帧特征做 anchors,留一法标定阈值
   (95% recall),场景级剔除语义漂移的合成样本 —— R3M 既是表征器也是数据把关人
4. **LoRA 微调**: R3M backbone 完全冻结,只训 ResNet50 后两个 stage 1x1 conv
   的低秩旁路 (r=8) + BC 头,保住预训练泛化先验
5. **部署**: merge LoRA → ONNX → C++ runtime (ONNX Runtime + OpenCV),
   cosine 门控作为部署侧 OOD 检测

## 结果

<!-- RESULTS_TABLE -->
(运行 `scripts/05_eval.py` 后填充)

## 快速开始

```powershell
# Python >= 3.11, CUDA GPU 推荐
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -e .

python scripts/00_setup_assets.py   # Panda 模型 + 纹理库 + R3M 权重
python scripts/01_record_demo.py    # 标称场景采集专家 demo
python scripts/02_generate_data.py  # DR + CLIP 增广 (1500 场景)
python scripts/03_filter_data.py    # R3M cosine 过滤
python scripts/04_train.py --variant baseline   # 对照: frozen + 单demo BC
python scripts/04_train.py --variant ours       # DR+过滤+LoRA
python scripts/05_eval.py           # 三档 unseen 套件闭环评测
python scripts/06_export.py         # 导出 deploy_bundle/

# C++ runtime (见 cpp/README.md)
cmake -S cpp -B cpp/build -G "Visual Studio 17 2022" -A x64
cmake --build cpp/build --config Release -j 8
python scripts/07_bench_cpp.py      # parity + 延迟验收
```

## 架构

```
┌─────────────────────── Python 训练侧 ───────────────────────┐
│ sim/      MuJoCo + Franka Panda 抓取环境, 微分IK, 脚本化专家  │
│ datagen/  DomainRandomizer + CLIPGuide + 回放式数据集构建     │
│ perception/ R3M backbone + LoRA 注入/合并 + CosineFilter     │
│ policy/   BC 头 + 联合训练器 (只更新 LoRA + 头)              │
│ eval/     L1 视觉 / L2 杂物 / L3 全偏移 三档 unseen 套件     │
│ export/   merge LoRA → perception.onnx + policy.onnx        │
└──────────────────────────┬──────────────────────────────────┘
                  deploy_bundle/ (唯一契约)
                  manifest.json + 2×ONNX + anchors.bin
┌──────────────────────────┴─────────── C++ 部署侧 ───────────┐
│ ImagePipeline → OrtEngine(perception) → CosineGate(OOD门控) │
│              → OrtEngine(policy) → action + 分段延迟         │
│ deploy_replay CLI / pybind11 (s2r_cpp) / 单元测试            │
└─────────────────────────────────────────────────────────────┘
```

## 关键设计

- **回放式增广**: 干扰物纯视觉无碰撞 + 拒绝采样避开路径,保证回放轨迹
  物理严格不变,(image, proprio, action) 标签零成本有效
- **统一输入契约**: float32 RGB CHW [0,1];ImageNet 归一化烘焙进 ONNX 图,
  C++ 对 backbone 来源无感
- **parity 验收**: 同批帧 Python(onnxruntime) vs C++ runtime 动作最大误差
  < 1e-4;预处理在测试中逐位对齐
- **阈值标定**: cosine 门控阈值不拍脑袋 —— demo 关键帧留一交叉验证取
  95% recall 分位点,训练端过滤与部署端 OOD 门控同一套参数

## 目录

```
configs/    dr / train / eval / deploy 四份 YAML
scripts/    00-07 全流水线入口
src/sim2real/   Python 包 (sim, datagen, perception, policy, eval, export)
cpp/        C++ runtime (CMake, 见 cpp/README.md)
tests/      pytest (LoRA 数学 / 过滤逻辑 / 跨语言 parity)
```
