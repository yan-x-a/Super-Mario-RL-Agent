import time, random
from pathlib import Path

import numpy as np
import torch, gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from gym.wrappers.frame_stack import LazyFrames

from wrappers import apply_wrappers
from agent     import AgentNN

import logging, warnings, gym
gym.logger.set_level(logging.ERROR)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# runs a trained Mario DQN agent in real time (no learning)
# Replicates training loop logic: wrappers, reward shaping, termination.


#  user-configurable parameters
ROM             = "SuperMarioBros-1-1-v0"
SKIP_FRAMES     = 4            # must match training
EPSILON         = 0         # fixed evaluation ε
EPISODES        = 7
CHECKPOINT_DIR  = Path("checkpoints")
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED            = 10           # optional: reproducibility

# Reward shaping (same as train.py)
TIME_PENALTY    = -0.01
PROGRESS_BONUS  = 0.10


def latest_ckpt(dir_: Path) -> Path:
    ckpts = sorted(dir_.glob("mario_dqn_ep*.pt"),
                   key=lambda p: int(p.stem.split('ep')[-1]),
                   reverse=True)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {dir_}")
    return ckpts[0]

def make_env():
    # Seed base env before wrapping (mirrors training seeding order)
    base_env = gym_super_mario_bros.make(
        ROM, render_mode="human", apply_api_compatibility=True
    )
    try:
        base_env.reset(seed=SEED)
    except TypeError:
        base_env.seed(SEED)

    env = JoypadSpace(base_env, SIMPLE_MOVEMENT)
    env = apply_wrappers(env, skip=SKIP_FRAMES)
    return env

@torch.no_grad()
def play():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    ckpt = torch.load(latest_ckpt(CHECKPOINT_DIR), map_location=DEVICE, weights_only=False)
    state_dict = ckpt.get("policy", ckpt)

    env = make_env()
    net = AgentNN(4, env.action_space.n).to(DEVICE)
    net.load_state_dict(state_dict)
    net.eval()

    for ep in range(1, EPISODES + 1):
        state, _ = env.reset()
        ep_reward = 0.0
        max_x = 0
        done = False

        while not done:
            if isinstance(state, LazyFrames):
                obs = np.asarray(state, dtype=np.uint8)
            else:
                obs = state

            if EPSILON and np.random.rand() < EPSILON:
                action = env.action_space.sample()
            else:
                tensor = torch.from_numpy(obs).unsqueeze(0).float().to(DEVICE) / 255.0
                action = int(net(tensor).argmax(1).item())

            next_state, r, term, trunc, info = env.step(action)

            # reward shaping (identical to training) 
            shaped_r = r + TIME_PENALTY
            if info.get("x_pos", 0) > max_x:
                shaped_r += PROGRESS_BONUS * (info["x_pos"] - max_x)
                max_x = info["x_pos"]

            ep_reward += shaped_r
            done = term or trunc or (info.get("time", 400) <= 250)
            state = next_state

            # real-time rendering
            time.sleep(SKIP_FRAMES / 60.0)

        print(f"Episode {ep:2d} — shaped reward {ep_reward:.1f}")

    env.close()

if __name__ == "__main__":
    play()
