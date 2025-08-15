import csv, random
from pathlib import Path

import logging, warnings, gym
import numpy as np

# silence gym deprecation warnings
gym.logger.set_level(logging.ERROR)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

from wrappers import apply_wrappers
from agent     import DQNAgent

#hyperparameters
TIME_PENALTY    = -0.01          # per frame
PROGRESS_BONUS  =  0.10          
SAVE_EVERY      = 50             # episodes
NUM_EPISODES    = 100_000        # training episodes
SEED            = 42
REPLAY_SAVE_N   = 20_000        #  replay subset size


# progress logging
LOG_CSV = Path("progress.csv")
LOG_CSV.touch(exist_ok=True)
log_f  = LOG_CSV.open("a", newline="")
writer = csv.writer(log_f)
if LOG_CSV.stat().st_size == 0:
    writer.writerow(["episode", "reward"])

with LOG_CSV.open() as f:
    rows = list(csv.reader(f))
    next_ep = int(rows[-1][0]) + 1 if len(rows) > 1 else 0

print(f"▶ Resuming at episode {next_ep}")

# environment setup
# Global RNG seeding for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

base_env = gym_super_mario_bros.make(
    "SuperMarioBros-1-1-v0",
    render_mode=None,
    apply_api_compatibility=True,
)
try:
    base_env.reset(seed=SEED)   
except TypeError:
    base_env.seed(SEED)       

env = JoypadSpace(base_env, SIMPLE_MOVEMENT)
env = apply_wrappers(env, skip=4)
env.action_space.seed(SEED)

agent = DQNAgent(observation_shape=(4, 84, 84),
                 n_actions=env.action_space.n)

# checkpoint loading
ckpt_dir = Path("checkpoints"); ckpt_dir.mkdir(exist_ok=True)
ckpts = sorted(ckpt_dir.glob("mario_dqn_ep*.pt"),
               key=lambda p: int(p.stem.split('ep')[-1]), reverse=True)

if ckpts:
    latest = ckpts[0]
    blob   = torch.load(latest, map_location=agent.device, weights_only=False)

    agent.policy_net.load_state_dict(blob["policy"])
    agent.target_net.load_state_dict(blob.get("target", blob["policy"]))

    if "optim" in blob:
        agent.optimizer.load_state_dict(blob["optim"])
    else:
        print("checkpoint lacks optimizer state - optimiser re-initialised.")

    agent.load_replay_subset(blob.get("replay", []))
    agent.frame_idx = blob.get("frame", 0)
    agent.load_meta(blob.get("meta", {}))

    ep_from_ckpt = int(latest.stem.split("ep")[-1]) + 1
    next_ep = max(next_ep, ep_from_ckpt)

    print(f"Loaded {latest.name}  (ε={agent._epsilon():.3f}, frame={agent.frame_idx}, replay={agent.memory_size()})")
else:
    print(" No checkpoint found — starting fresh network.")

# training loop
while next_ep < NUM_EPISODES:
    obs, _  = env.reset()
    max_x   = 0
    ep_rwd  = 0.0
    done    = False

    while not done:
        act = agent.select_action(obs)
        nxt, r, term, trunc, info = env.step(act)

        r += TIME_PENALTY
        if info.get("x_pos", 0) > max_x:
            r += PROGRESS_BONUS * (info["x_pos"] - max_x)
            max_x = info["x_pos"]

        done = term or trunc or (info.get("time", 400) <= 250)
        agent.remember(obs, act, r, nxt, term, trunc)
        agent.train_step()
        obs  = nxt
        ep_rwd += r

    print(f"Ep {next_ep:5d}   reward = {ep_rwd:6.1f}   ε = {agent._epsilon():.3f}")
    writer.writerow([next_ep, ep_rwd]); log_f.flush()

    if next_ep % SAVE_EVERY == 0:
        replay_subset = agent.sample_replay_subset(REPLAY_SAVE_N)
        torch.save({
            "policy": agent.policy_net.state_dict(),
            "target": agent.target_net.state_dict(),
            "optim" : agent.optimizer.state_dict(),
            "frame" : agent.frame_idx,
            "replay": replay_subset,
            "meta"  : agent.export_meta(),
        }, ckpt_dir / f"mario_dqn_ep{next_ep}.pt", pickle_protocol=4)

    next_ep += 1

log_f.close(); env.close()
print("Training finished.")
