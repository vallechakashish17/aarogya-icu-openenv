import gymnasium as gym
import numpy as np
from gymnasium import spaces

class OpenEnvHealthcare(gym.Env):
    def __init__(self, task_id=None):
        super().__init__()
        self.action_space = spaces.Discrete(4) # 0:Mon, 1:Med, 2:O2, 3:Doc
        self.observation_space = spaces.Box(low=0, high=200, shape=(3,), dtype=np.float32)
        self.state = np.array([120.0, 95.0, 0.0], dtype=np.float32)
        self.steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        start_val = options.get("custom_start", [130.0, 90.0, 0.0]) if options else [130.0, 90.0, 0.0]
        self.state = np.array(start_val, dtype=np.float32)
        self.steps = 0
        return self.state, {}

    def step(self, action):
        self.steps += 1
        hr, o2, tox = self.state
        doctor_called = False

        if action == 1: hr -= 15.0; tox += 0.2
        elif action == 2: o2 += 8.0; tox += 0.02
        elif action == 3: hr, o2, tox, doctor_called = 75.0, 98.0, tox*0.5, True

        if not doctor_called:
            hr += 1.5; o2 -= 0.8; tox = max(0, tox - 0.05)

        self.state = np.clip([hr, o2, tox], [30, 50, 0], [200, 100, 3.0]).astype(np.float32)
        terminated = bool(self.state[0] < 40 or self.state[0] > 180 or self.state[1] < 60 or tox > 2.0)
        truncated = self.steps >= 100

        # High-stability reward
        hr_s = max(0, 1.0 - abs(self.state[0] - 75) / 50)
        o2_s = max(0, (self.state[1] - 70) / 30)
        reward = (0.6 * hr_s) + (0.4 * o2_s)
        if doctor_called: reward = 0.1
        if tox > 0.8: reward -= 0.3

        return self.state, float(np.clip(reward, 0.01, 0.98)), terminated, truncated, {"doctor_called": doctor_called}