"""程序化纹理库生成 (numpy + PIL, 无外部素材依赖)。

训练库与测试库用不同风格参数 + 不同 seed, 保证评测纹理 unseen。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

SIZE = 256


def _norm(img: np.ndarray) -> np.ndarray:
    img = img - img.min()
    rng = img.max() if img.max() > 1e-8 else 1.0
    return img / rng


def _colorize(gray: np.ndarray, c0: np.ndarray, c1: np.ndarray) -> np.ndarray:
    return (c0[None, None] * (1 - gray[..., None]) + c1[None, None] * gray[..., None])


def _fractal_noise(rng: np.random.Generator, octaves: int = 4) -> np.ndarray:
    """多倍频值噪声: 低分辨率随机格 + 双线性放大叠加。"""
    out = np.zeros((SIZE, SIZE))
    amp = 1.0
    for o in range(octaves):
        n = 2 ** (o + 2)
        coarse = rng.random((n, n))
        img = np.array(Image.fromarray((coarse * 255).astype(np.uint8)).resize((SIZE, SIZE), Image.BILINEAR)) / 255.0
        out += amp * img
        amp *= 0.5
    return _norm(out)


def checker(rng: np.random.Generator) -> np.ndarray:
    n = int(rng.integers(4, 16))
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    g = (((xx * n // SIZE) + (yy * n // SIZE)) % 2).astype(float)
    return _colorize(g, rng.random(3), rng.random(3))


def stripes(rng: np.random.Generator) -> np.ndarray:
    n = float(rng.uniform(3, 20))
    theta = float(rng.uniform(0, np.pi))
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    proj = xx * np.cos(theta) + yy * np.sin(theta)
    g = (np.sin(proj / SIZE * 2 * np.pi * n) + 1) / 2
    return _colorize(g, rng.random(3), rng.random(3))


def noise(rng: np.random.Generator) -> np.ndarray:
    g = _fractal_noise(rng, octaves=int(rng.integers(3, 6)))
    return _colorize(g, rng.random(3), rng.random(3))


def wood(rng: np.random.Generator) -> np.ndarray:
    """同心环 + 噪声扰动, 配木色系。"""
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    cx, cy = rng.uniform(-0.5, 1.5, 2) * SIZE
    r = np.hypot(xx - cx, yy - cy) / SIZE
    warp = _fractal_noise(rng, 3) * rng.uniform(0.05, 0.2)
    rings = (np.sin((r + warp) * rng.uniform(20, 60)) + 1) / 2
    base = np.array([0.45, 0.30, 0.15]) * rng.uniform(0.7, 1.3)
    dark = base * rng.uniform(0.4, 0.7)
    return _colorize(rings, np.clip(dark, 0, 1), np.clip(base, 0, 1))


def marble(rng: np.random.Generator) -> np.ndarray:
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    warp = _fractal_noise(rng, 5)
    g = (np.sin((xx / SIZE + warp * rng.uniform(2, 6)) * np.pi * rng.uniform(2, 8)) + 1) / 2
    light = np.clip(rng.uniform(0.7, 1.0) * np.ones(3) + rng.normal(0, 0.05, 3), 0, 1)
    vein = rng.random(3) * 0.5
    return _colorize(g, vein, light)


def speckle(rng: np.random.Generator) -> np.ndarray:
    base = rng.random(3)
    img = np.tile(base, (SIZE, SIZE, 1))
    n_dots = int(rng.integers(200, 1500))
    ys = rng.integers(0, SIZE, n_dots)
    xs = rng.integers(0, SIZE, n_dots)
    img[ys, xs] = rng.random((n_dots, 3))
    return img


TRAIN_STYLES = [checker, stripes, noise, wood, marble, speckle]
# 测试库刻意只用部分风格 + 独立 seed, 形成分布偏移
TEST_STYLES = [noise, wood, marble, stripes]


def generate_library(out_dir: Path, n: int, seed: int, styles=None) -> list[Path]:
    styles = styles or TRAIN_STYLES
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    paths = []
    for i in range(n):
        fn = styles[i % len(styles)]
        img = np.clip(fn(rng) * 255, 0, 255).astype(np.uint8)
        p = out_dir / f"{fn.__name__}_{i:04d}.png"
        Image.fromarray(img).save(p)
        paths.append(p)
    return paths
