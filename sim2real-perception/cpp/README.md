# C++ 部署 Runtime

ONNX Runtime + OpenCV 的部署侧推理引擎。消费 `deploy_bundle/`(由
`scripts/06_export.py` 导出),对 Python 训练端零依赖。

## 构建 (Windows / MSVC)

```powershell
cmake -S cpp -B cpp/build -G "Visual Studio 17 2022" -A x64
cmake --build cpp/build --config Release -j 8
```

依赖全部由 CMake FetchContent 自动获取:
- **onnxruntime** 官方预编译包(不参与编译)
- **OpenCV** 源码最小构建(core/imgproc/imgcodecs,静态库,首次约 5-10 分钟)
- **nlohmann/json**、**pybind11** 头文件库

产物:
| 文件 | 说明 |
|---|---|
| `build/bin/Release/deploy_replay.exe` | 回放推理 CLI(parity + 延迟报告)|
| `build/bin/Release/s2r_tests.exe` | C++ 单元测试 |
| `build/Release/s2r_cpp.cp313-win_amd64.pyd` | pybind11 绑定(Python 评测复用 C++ 组件)|

## 运行

```powershell
# 单元测试
cpp/build/bin/Release/s2r_tests.exe

# 回放推理 (先跑完 python 侧 01→06)
cpp/build/bin/Release/deploy_replay.exe deploy_bundle data/parity

# 完整 parity + 延迟验收
python scripts/07_bench_cpp.py
```

## 架构

```
ImagePipeline   PNG/相机帧 -> RGB float CHW [0,1] (224x224)
OrtEngine       ONNX Runtime CPU 会话封装
CosineGate      特征 L2 归一化后与 anchors 点积, max-cos >= tau 判定 in-distribution
DeployRuntime   step(): 预处理 -> perception.onnx -> 门控 -> policy.onnx, 分段计时
```

门控拒绝 (OOD) 时 `StepResult.accepted=false`,动作仍返回,由上层闭环决定
是否切换到安全保持策略 —— 对应训练端 `CosineFilter` 的部署化身。
