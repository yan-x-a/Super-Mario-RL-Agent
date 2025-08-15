from __future__ import annotations
import cv2, numpy as np, gym
from gym import spaces
from gym.wrappers import FrameStack

class SkipFrame(gym.Wrapper):
    """Repeat same action `skip` frames and sum rewards."""
    def __init__(self, env: gym.Env, skip: int = 4):
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        for _ in range(self.skip):
            obs, r, terminated, truncated, info = self.env.step(action)
            total_reward += r
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info

def _preprocess(obs: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return resized.astype(np.uint8)

class PreprocessFrame(gym.ObservationWrapper):
    """Convert to grayscale 84x84."""
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(84, 84), dtype=np.uint8
        )

    def observation(self, obs):
        return _preprocess(obs)

def apply_wrappers(env: gym.Env, skip: int = 4) -> gym.Env:
    env = SkipFrame(env, skip)
    env = PreprocessFrame(env)
    env = FrameStack(env, 4, lz4_compress=True)
    return env
