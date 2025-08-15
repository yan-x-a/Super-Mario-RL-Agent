import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, Tuple, List, Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gym.wrappers.frame_stack import LazyFrames

class AgentNN(nn.Module):
    """3-conv / 2-linear DQN."""

    def __init__(self, in_channels: int, n_actions: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, 4), nn.ReLU(),          # 20×20
            nn.Conv2d(32,           64, 4, 2), nn.ReLU(),          # 9×9
            nn.Conv2d(64,           64, 3, 1), nn.ReLU(),          # 7×7
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512), nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  
        return self.net(x)


# replay buffer
@dataclass
class Experience:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    terminated: bool
    truncated: bool


class ReplayBuffer:
    """Circular FIFO replay buffer."""

    def __init__(self, capacity: int) -> None:
        self.buffer: Deque[Experience] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def push(self, *exp) -> None:
        self.buffer.append(Experience(*exp))

    def sample(self, batch_size: int) -> List[Experience]:
        return random.sample(self.buffer, batch_size)



class DQNAgent:
    """Deep-Q agent with Double-DQN target and linear ε-greedy decay."""

    def __init__(
        self,
        observation_shape: Tuple[int, int, int],   
        n_actions: int,
        buffer_size: int = 200_000,
        batch_size: int = 32,
        gamma: float = 0.97,                      
        lr: float = 3e-4,
        eps_start: float = 1.0,
        eps_end: float = 0.02,                    
        eps_decay_frames: int = 400_000,
        target_sync_every: int = 1_000,             
        device: str | torch.device | None = None,
    ) -> None:
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        c, h, w = observation_shape
        self.policy_net = AgentNN(c, n_actions).to(self.device)
        self.target_net = AgentNN(c, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory    = ReplayBuffer(buffer_size)

        self.batch_size = batch_size
        self.gamma      = gamma
        self.n_actions  = n_actions

        # ε-greedy schedule tracking
        self.eps_start, self.eps_end = eps_start, eps_end
        self.eps_decay_frames        = eps_decay_frames
        self.frame_idx               = 0

        self.target_sync_every = target_sync_every

    # Select action using ε-greedy policy

    def select_action(self, state: np.ndarray) -> int:
        """Return an action following ε-greedy policy."""
        self.frame_idx += 1
        if isinstance(state, LazyFrames):
            state = np.asarray(state, dtype=np.uint8)

        if random.random() < self._epsilon():
            return random.randrange(self.n_actions)

        state_v = (
            torch.from_numpy(state)
            .unsqueeze(0).float().to(self.device) / 255.0
        )
        with torch.no_grad():
            return int(self.policy_net(state_v).argmax(1).item())

    def remember(self, s, a, r, s2, term, trunc) -> None:
        if isinstance(s,  LazyFrames): s  = np.asarray(s,  dtype=np.uint8).copy()
        if isinstance(s2, LazyFrames): s2 = np.asarray(s2, dtype=np.uint8).copy()
        self.memory.push(s, a, r, s2, term, trunc)

    def train_step(self) -> None:
        if len(self.memory) < self.batch_size:
            return
        batch = self.memory.sample(self.batch_size)
        self._optimize(batch)
        if self.frame_idx % self.target_sync_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def memory_size(self) -> int:
        return len(self.memory)

    def sample_replay_subset(self, n: int):
        return list(self.memory.buffer)[-n:]

    def load_replay_subset(self, subset: List[Experience]):
        for exp in subset:
            if isinstance(exp, Experience):
                self.memory.buffer.append(exp)
            else:
                self.memory.push(*exp)

    # Export/import schedule + RNG state 
    def export_meta(self) -> Dict[str, Any]:
        meta: Dict[str, Any] = {
            "eps_start": self.eps_start,
            "eps_end": self.eps_end,
            "eps_decay_frames": self.eps_decay_frames,
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all()
                               if torch.cuda.is_available() else None,
            },
        }
        return meta

    def load_meta(self, meta: Dict[str, Any]) -> None:
        if not meta:
            return
        # Schedule
        self.eps_start = meta.get("eps_start", self.eps_start)
        self.eps_end = meta.get("eps_end", self.eps_end)
        self.eps_decay_frames = meta.get("eps_decay_frames", self.eps_decay_frames)
        # RNG
        rng = meta.get("rng_state")
        if rng:
            try:
                random.setstate(rng["python"])
                np.random.set_state(rng["numpy"])
                torch.set_rng_state(rng["torch_cpu"])
                if torch.cuda.is_available() and rng.get("torch_cuda"):
                    for dev_id, state in enumerate(rng["torch_cuda"]):
                        torch.cuda.set_rng_state(state, dev_id)
            except Exception as e:
                print(f"Failed to restore RNG state: {e}")


    def _epsilon(self) -> float:
        progress = min(1.0, self.frame_idx / self.eps_decay_frames)
        return self.eps_start - (self.eps_start - self.eps_end) * progress

    def _optimize(self, batch: List[Experience]) -> None:
        # prepare tensors
        s  = torch.from_numpy(np.stack([e.state      for e in batch])).float().to(self.device) / 255.0
        a  = torch.tensor    ([e.action     for e in batch], dtype=torch.int64, device=self.device).unsqueeze(1)
        r  = torch.tensor    ([e.reward     for e in batch], dtype=torch.float32, device=self.device)
        s2 = torch.from_numpy(np.stack([e.next_state for e in batch])).float().to(self.device) / 255.0
        done = torch.tensor  ([e.terminated or e.truncated for e in batch],
                              dtype=torch.bool, device=self.device)

        # Q(s,a)
        q_sa = self.policy_net(s).gather(1, a).squeeze(1)

        # Double-DQN target
        with torch.no_grad():
            a_star = self.policy_net(s2).argmax(1, keepdim=True)
            q_s2_astar = self.target_net(s2).gather(1, a_star).squeeze(1)
            target = r + self.gamma * q_s2_astar * (~done)

        loss = nn.functional.smooth_l1_loss(q_sa, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()
