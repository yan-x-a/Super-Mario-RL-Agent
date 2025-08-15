#!/usr/bin/env python
import argparse, os, random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from gym.wrappers.frame_stack import LazyFrames

from agent import AgentNN          # same network architecture as training
from wrappers import apply_wrappers

# constants
DEFAULT_SEED = 42          # for deterministic evaluation
DEFAULT_SKIP = 4           # must match training



def _find_latest_checkpoint(dir_path: str) -> Optional[Path]:
    """Return newest *.pt file in *dir_path* or *None* if none exist."""
    ckpts = sorted(Path(dir_path).glob("*.pt"),
                   key=os.path.getmtime, reverse=True)
    return ckpts[0] if ckpts else None


def build_env(render: str = "human", skip: int = DEFAULT_SKIP, seed: int | None = None):
    base_env = gym_super_mario_bros.make(
        "SuperMarioBros-1-1-v0",
        render_mode=render,
        apply_api_compatibility=True,
    )
    if seed is not None:
        try:
            base_env.reset(seed=seed)
        except TypeError:
            base_env.seed(seed)

    env = JoypadSpace(base_env, SIMPLE_MOVEMENT)
    env = apply_wrappers(env, skip=skip)
    return env



def evaluate(checkpoint: Path,
             episodes: int,
             device: torch.device,
             seed: int,
             render_mode: str) -> None:
    # build environment
    env = build_env(render=render_mode, seed=seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # load model
    try:
        blob = torch.load(checkpoint, map_location=device, weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint {checkpoint}: {e}")

    if "policy" not in blob:
        raise KeyError(f"Checkpoint {checkpoint} does not contain 'policy' key.")

    model = AgentNN(4, env.action_space.n).to(device)
    model.load_state_dict(blob["policy"])
    model.eval()

    total_reward = 0.0
    clears = 0

    for ep in range(1, episodes + 1):
        state, _ = env.reset()
        ep_reward = 0.0
        terminated = truncated = False
        flag_cleared = False

        with torch.no_grad():
            while not (terminated or truncated):
                if isinstance(state, LazyFrames):
                    state = np.asarray(state, dtype=np.uint8)

                state_t = (torch.from_numpy(state)
                                .unsqueeze(0)
                                .float()
                                .to(device) / 255.0)
                q_vals = model(state_t)
                action = int(q_vals.argmax(1))

                state, r, terminated, truncated, info = env.step(action)
                ep_reward += r
                if info.get("flag_get"):
                    flag_cleared = True  # robust clear detection

        total_reward += ep_reward
        if flag_cleared:
            clears += 1
        print(f"Episode {ep:3d} — reward: {ep_reward:6.1f}  "
              f"{'CLEARED' if flag_cleared else ''}")

    env.close()
    print(f"\nAverage reward: {total_reward / episodes:.1f}  "
          f"Clears: {clears}/{episodes}")


# Main function to parse arguments and run evaluation
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Mario DQN agent."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to .pt file; if omitted, use newest file in ./checkpoints",
    )
    parser.add_argument(
        "--episodes", type=int, default=10,
        help="Number of episodes to run"
    )
    parser.add_argument(
        "--cuda", action="store_true",
        help="Use CUDA if available (default: CPU)"
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help="Random seed for evaluation"
    )
    parser.add_argument(
        "--no-render", action="store_true",
        help="Disable human render (off-screen evaluation)"
    )

    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    ckpt_path = Path(args.checkpoint) if args.checkpoint else _find_latest_checkpoint("checkpoints")
    if ckpt_path is None or not ckpt_path.exists():
        raise FileNotFoundError(
            "No checkpoint found. Specify --checkpoint or put files in ./checkpoints"
        )

    render_mode = None if args.no_render else "human"
    print(f"Evaluating {ckpt_path} on {device} for {args.episodes} episode(s) …")
    evaluate(ckpt_path, args.episodes, device, seed=args.seed, render_mode=render_mode)


if __name__ == "__main__":
    main()
