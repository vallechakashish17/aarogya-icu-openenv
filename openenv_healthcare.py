import gymnasium as gym
import numpy as np
from gymnasium import spaces

class OpenEnvHealthcare(gym.Env):
    def __init__(self, task_id=None):
        super().__init__()
        self.action_space = spaces.Discrete(3)
        # HR, O2, Tox
        self.observation_space = spaces.Box(low=0, high=200, shape=(3,), dtype=np.float32)
        self.state = np.array([120.0, 95.0, 0.0], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options and "custom_start" in options:
            self.state = np.array(options["custom_start"], dtype=np.float32)
        else:
            self.state = np.array([130.0, 90.0, 0.0], dtype=np.float32)
        self.steps = 0
        return self.state, {}

    def step(self, action):
        self.steps += 1
        hr, o2, tox = self.state
        
        if action == 1: hr -= 15.0; tox += 0.1 # Meds
        if action == 2: o2 += 10.0; tox += 0.05 # Oxygen
        
        # Natural drift
        hr += 1.0; o2 -= 0.5
        self.state = np.array([hr, o2, tox], dtype=np.float32)
        
        # Reward: +1 if stable, -1 if not
        reward = 1.0 if (60 <= hr <= 100 and o2 >= 92) else -1.0
        
        terminated = bool(hr < 40 or hr > 180 or o2 < 60 or tox > 2.0)
        truncated = self.steps >= 50
        return self.state, reward, terminated, truncated, {}