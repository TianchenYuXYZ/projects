"""资产准备: Panda 模型 (mujoco_menagerie) + 程序化纹理库 + R3M 权重。

R3M 权重经官方包下载 (Google Drive, 可能不稳定); 失败时打印警告,
训练配置可改用 imagenet_resnet50 回退, 接口不变。
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sim2real.common import PROJECT_ROOT
from sim2real.datagen import textures

MENAGERIE_URL = "https://github.com/google-deepmind/mujoco_menagerie"


def setup_panda() -> None:
    dst = PROJECT_ROOT / "assets" / "franka_emika_panda"
    if (dst / "panda.xml").exists():
        print(f"[panda] 已存在: {dst}")
        return
    tmp = PROJECT_ROOT / "assets" / "_menagerie_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    print("[panda] sparse clone mujoco_menagerie ...")
    subprocess.run(
        ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
         MENAGERIE_URL, str(tmp)], check=True)
    subprocess.run(["git", "-C", str(tmp), "sparse-checkout", "set",
                    "franka_emika_panda"], check=True)
    shutil.copytree(tmp / "franka_emika_panda", dst)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"[panda] OK -> {dst}")


def setup_textures() -> None:
    train_dir = PROJECT_ROOT / "assets" / "textures"
    test_dir = PROJECT_ROOT / "assets" / "textures_test"
    if not list(train_dir.glob("*.png")):
        n = len(textures.generate_library(train_dir, n=120, seed=7,
                                          styles=textures.TRAIN_STYLES))
        print(f"[textures] 训练库 {n} 张 -> {train_dir}")
    if not list(test_dir.glob("*.png")):
        n = len(textures.generate_library(test_dir, n=40, seed=4242,
                                          styles=textures.TEST_STYLES))
        print(f"[textures] 测试库 {n} 张 -> {test_dir}")


def setup_r3m() -> None:
    out = PROJECT_ROOT / "weights" / "r3m_resnet50.pth"
    if out.exists():
        print(f"[r3m] 已存在: {out}")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        try:
            import r3m  # noqa: F401
        except ImportError:
            print("[r3m] 安装官方 r3m 包 ...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q",
                 "git+https://github.com/facebookresearch/r3m.git"], check=True)
        import torch
        from r3m import load_r3m

        print("[r3m] 下载 R3M-ResNet50 权重 (Google Drive) ...")
        model = load_r3m("resnet50")          # DataParallel(R3M)
        trunk = model.module.convnet
        torch.save(trunk.state_dict(), out)
        print(f"[r3m] OK -> {out}")
    except Exception as e:  # noqa: BLE001
        print(f"[r3m] 下载失败: {e}\n"
              f"      回退方案: 把 configs/train.yaml 的 backbone.name 改为 "
              f"imagenet_resnet50 (接口一致)")


if __name__ == "__main__":
    setup_panda()
    setup_textures()
    setup_r3m()
