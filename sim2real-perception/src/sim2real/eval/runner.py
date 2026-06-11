"""闭环 zero-shot 评测: 策略驱动仿真, 统计成功率。"""
from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from sim2real.common import SceneConfig
from sim2real.perception.backbone import PerceptionBackbone
from sim2real.policy.bc import BCPolicy
from sim2real.sim.env import ManipEnv


class PolicyAgent:
    """Python 端推理 agent (评测用); C++ runtime 是它的部署版镜像。"""

    def __init__(self, backbone: PerceptionBackbone, policy: BCPolicy,
                 device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.backbone = backbone.to(self.device).eval()
        self.policy = policy.to(self.device).eval()

    @torch.no_grad()
    def act(self, image_u8: np.ndarray, proprio: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(image_u8.transpose(2, 0, 1)[None]).to(self.device)
        x = x.float() / 255.0
        p = torch.from_numpy(proprio[None].astype(np.float32)).to(self.device)
        a = self.policy(self.backbone(x), p)
        return a.squeeze(0).cpu().numpy()


def run_suite(agent: PolicyAgent, scenes: list[SceneConfig],
              max_steps: int, render_size: int = 224,
              desc: str = "eval") -> dict:
    env = ManipEnv(scenes[0], render_size=render_size)
    successes = []
    for scene in tqdm(scenes, desc=desc):
        obs = env.reset(scene)
        success = False
        for _ in range(max_steps):
            a = agent.act(obs.image, obs.proprio)
            obs, success, _ = env.step(a)
            if success:
                break
        successes.append(bool(success))
    env.close()
    rate = float(np.mean(successes)) if successes else 0.0
    return {"success_rate": rate, "n_episodes": len(successes),
            "successes": successes}
