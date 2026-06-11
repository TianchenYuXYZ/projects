# Sim-to-Real Robust Perception Pipeline

> **Domain Randomization + CLIP-Guided Generation + R3M Feature Filtering + LoRA Fine-Tuning**
>
> An engineered framework designed to mitigate perception degradation in robot manipulation tasks during sim-to-real transfer. Python handles the training pipeline, while C++ manages the high-performance deployment runtime.

---

## 核心痛点 (The Problem)

When an imitation learning agent is trained on expert demonstrations collected in a static simulation environment, the visual encoder's latent space often degrades before the policy itself fails when exposed to real-world deployment variances (e.g., shifts in lighting, unexpected textures, novel distractors, or minor camera misalignments). 

This repository implements a production-grade pipeline to robustify visual representations:

1. **Replay-Based DR Augmentation**: Applies aggressive visual domain randomization without altering scene physics. By replaying nominal expert action sequences, a single demonstration is scaled into a dataset of 1,500 highly randomized scenes with zero-cost, inherently valid labels.
2. **CLIP Semantic Guidance**: Scores randomized surface textures against target semantic prompts, filtering out catastrophic or unrealistic render artifacts.
3. **R3M Cosine Filtering**: Establishes keyframes from nominal demonstrations as feature anchors. By applying leave-one-out cross-validation, it calibrates a rigorous similarity threshold (targeting 95% recall) to eliminate synthetic scenes suffering from semantic drift. In this pipeline, R3M acts as both the primary feature extractor and the data quality gatekeeper.
4. **Parameter-Efficient LoRA Fine-Tuning**: Fully freezes the heavy R3M pre-trained backbone. It fine-tunes only the low-rank bypass ($r=8$) injected into the $1 \times 1$ convolutions of the final two ResNet50 stages alongside the Behavioral Cloning (BC) policy head, protecting the foundational pre-trained generalization priors.
5. **Production Deployment**: Merges the low-rank adapters directly into the base weights, exporting a clean graph structure: $\text{LoRA} \rightarrow \text{ONNX} \rightarrow \text{C++ Runtime}$ (powered by ONNX Runtime + OpenCV). The same calibrated cosine similarity metrics are reused downstream as a production-side Out-Of-Distribution (OOD) safety gate.

---

## 快速开始 (Quick Start)

### Prerequisites
* Python $\ge$ 3.11
* CUDA-capable GPU recommended
* C++ 17 compatible compiler, CMake $\ge$ 3.22

### Installation & Execution

```powershell
# Install dependencies and local editable package
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu126](https://download.pytorch.org/whl/cu126)
pip install -e .

# Run the end-to-end Python pipeline
python scripts/00_setup_assets.py    # Fetch Franka Panda meshes, texture assets, and R3M weights
python scripts/01_record_demo.py     # Collect expert demonstration in the nominal scene
python scripts/02_generate_data.py   # Execute DR + CLIP-guided dataset scaling (1500 scenes)
python scripts/03_filter_data.py     # Apply R3M cosine feature filtering
python scripts/04_train.py --variant baseline    # Train Baseline: frozen backbone + nominal demo BC
python scripts/04_train.py --variant ours        # Train Ours: DR + Cosine Filtering + LoRA Fine-tuning
python scripts/05_eval.py            # Run closed-loop evaluation across 3 tiers of unseen environments
python scripts/06_export.py          # Freeze weights and export deployment bundle

# Compile the C++ runtime (See cpp/README.md for detailed system setup)
cmake -S cpp -B cpp/build -G "Visual Studio 17 2022" -A x64
cmake --build cpp/build --config Release -j 8

# Verify cross-language parity and benchmark execution latency
python scripts/07_bench_cpp.py


## Architecture

```
┌────────────────────────────────── Python Training Pipeline ──────────────────────────────────┐
│ sim/        MuJoCo + Franka Panda gripping environment, differential IK, scripted expert      │
│ datagen/    DomainRandomizer + CLIPGuide + replay-based dataset construction generation     │
│ perception/ R3M backbone + LoRA injection/merging + Cosine Filter engine                     │
│ policy/     BC head + joint trainer (updates LoRA adapters + policy head exclusively)        │
│ eval/       Closed-loop validation suites: L1 Visual Shift / L2 Clutter / L3 Full OOD Shift  │
│ export/     Fuses LoRA layers → outputs perception.onnx + policy.onnx                       │
└──────────────────────────────────────────────┬───────────────────────────────────────────────┘
                                               │
                                               ▼ deploy_bundle/ (Unified Contract)
                                                 ├── manifest.json
                                                 ├── perception.onnx
                                                 ├── policy.onnx
                                                 └── anchors.bin
                                               │
┌───────────────────────────────── C++ Production Deployment ──────────────────────────────────┐
│ ImagePipeline → OrtEngine(perception) → CosineGate(OOD Safety Gate)                          │
│               → OrtEngine(policy)     → low-latency action generation + profiling            │
│ deploy_replay CLI / pybind11 bindings (s2r_cpp) / comprehensive unit tests                    │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
```

## Key Engineering Designs

- **Kinematically Strict Augmentation**: Distractors introduce purely visual variations without registering physical collisions in MuJoCo, and rejection sampling is applied to ensure no objects clip through the robot's pre-recorded path. This guarantees that replayed trajectories remain physically valid, keeping (image, proprioception, action) pairs accurate at zero annotation cost.
- ** Unified Input Contract**: The pipeline enforces a rigid tensor format: Float32 RGB CHW scaled to $[0, 1]$. ImageNet normalization transforms are baked directly into the computational graph of the exported ONNX model, leaving the C++ runtime entirely agnostic to upstream pre-processing quirks.
- **Cross-Language Parity Validation** : Rigorous unit tests ensure strict numerical parity between the Python (onnxruntime) and C++ runtimes. The maximum action token discrepancy on identical frame batches is mathematically guaranteed to be $< 10^{-4}$, achieved by aligning pre-processing interpolations bit-for-bit.
- **Analytical Threshold Calibration**: The OOD cosine gate threshold avoids arbitrary heuristics. It is derived analytically via leave-one-out cross-validation across nominal demonstration keyframes to extract the exact 95% recall quantile. This unifies data pruning during training and anomaly detection during deployment under a single mathematical profile.

## Directory Structure

```
configs/        # YAML configurations for domain randomization, training, evaluation, and deployment
scripts/        # Execution entrypoints for the end-to-end pipeline (00 through 07)
src/sim2real/   # Core Python package (sim, datagen, perception, policy, eval, export modules)
cpp/            # High-performance C++ inference runtime (CMake native, see cpp/README.md)
tests/          # Pytest suite validating LoRA math, filtering logic, and cross-language parity
```
