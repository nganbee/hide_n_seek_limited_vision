#!/usr/bin/env python3
# submissions/group_05/train_pacman_dqn.py
"""
Pacman DQN training for fast ghost capture.
Features 9-action space and belief tracking.
"""

import argparse
import math
import random
import sys
from collections import deque, namedtuple
from pathlib import Path
from typing import Optional, Tuple, List, Set

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

# Ensure project src is on path
ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if SRC_DIR.as_posix() not in sys.path:
    sys.path.append(SRC_DIR.as_posix())

from environment import Environment, Move

Transition = namedtuple("Transition", "state action reward next_state done")

ALL_MOVES = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# 9-action space:
# 0-3: 1-step moves
# 4-7: 2-step moves (ONLY legal if continuing same direction)
# 8: STAY
ACTIONS = [
    (Move.UP, 1), (Move.DOWN, 1), (Move.LEFT, 1), (Move.RIGHT, 1),
    (Move.UP, 2), (Move.DOWN, 2), (Move.LEFT, 2), (Move.RIGHT, 2),
    (Move.STAY, 1),
]
NUM_ACTIONS = len(ACTIONS)

MOVE_TO_1STEP_IDX = {Move.UP: 0, Move.DOWN: 1, Move.LEFT: 2, Move.RIGHT: 3}
MOVE_TO_2STEP_IDX = {Move.UP: 4, Move.DOWN: 5, Move.LEFT: 6, Move.RIGHT: 7}

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args, **kwargs):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int):
        n = min(batch_size, len(self.buffer))
        batch = random.sample(self.buffer, n)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)

class DuelingDQN(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 256, num_actions: int = NUM_ACTIONS):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.value = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )
        self.adv = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, num_actions),
        )

    def forward(self, x):
        f = self.features(x)
        v = self.value(f)
        a = self.adv(f)
        return v + a - a.mean(dim=1, keepdim=True)


def get_visible_cells_cross(map_state: np.ndarray, pos: Tuple[int, int], radius: int) -> Set[Tuple[int, int]]:
    visible = {pos}
    r, c = pos
    H, W = map_state.shape
    for dr, dc in DIR4:
        rr, cc = r, c
        for _ in range(radius):
            rr += dr
            cc += dc
            if not (0 <= rr < H and 0 <= cc < W):
                break
            if map_state[rr, cc] == 1:
                break
            visible.add((rr, cc))
    return visible

class SimpleBelief:
    def __init__(self, shape: Tuple[int, int]):
        self.shape = shape
        self.b = np.ones(shape, dtype=np.float32)
        self.b /= self.b.sum()
        self.steps_since_seen = 0

    def reset(self):
        self.b.fill(1.0)
        self.b /= self.b.sum()
        self.steps_since_seen = 0

    def update(self, memory_map: np.ndarray, visible_cells: Set[Tuple[int, int]], ghost_pos: Optional[Tuple[int, int]]):
        wall = (memory_map == 1)

        if ghost_pos is not None:
            self.b.fill(0.0)
            self.b[ghost_pos] = 1.0
            self.steps_since_seen = 0
            return

        self.steps_since_seen += 1

        # ghost not in visible cells
        for (r, c) in visible_cells:
            self.b[r, c] = 0.0

        # ghost not in walls
        self.b[wall] = 0.0

        # propagate (ghost moves 0/1 step)
        newb = np.zeros_like(self.b)
        H, W = self.shape
        rows, cols = np.where(self.b > 1e-8)
        for r, c in zip(rows, cols):
            p = float(self.b[r, c])
            if p <= 0:
                continue
            opts = [(r, c)]
            for dr, dc in DIR4:
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W and not wall[rr, cc]:
                    opts.append((rr, cc))
            share = p / len(opts)
            for rr, cc in opts:
                newb[rr, cc] += share

        s = float(newb.sum())
        if s <= 1e-9:
            newb = (~wall).astype(np.float32)
            newb /= (newb.sum() + 1e-9)
        else:
            newb /= s
        self.b = newb

    def map(self) -> np.ndarray:
        return self.b.copy()

    def best_guess(self) -> Optional[Tuple[int, int]]:
        mx = float(self.b.max())
        if mx < 1e-3:
            return None
        idx = np.argwhere(self.b >= 0.95 * mx)
        if idx.size == 0:
            return None
        r, c = idx[np.random.randint(len(idx))]
        return (int(r), int(c))

def build_compact_features(
    memory_map: np.ndarray,
    pac_pos: Tuple[int, int],
    ghost_pos_or_estimate: Optional[Tuple[int, int]],
    belief: np.ndarray,
    last_action: int,
    step: int,
    max_steps: int,
    pacman_speed: int = 2,
) -> np.ndarray:
    """Build compact feature vector for Pacman DQN."""
    H, W = memory_map.shape
    r, c = pac_pos
    feats = []
    
    for dr, dc in DIR4:
        rr, cc = r, c
        for _ in range(5):
            rr += dr
            cc += dc
            if 0 <= rr < H and 0 <= cc < W:
                cell = memory_map[rr, cc]
                if cell == 1:
                    feats.append(1.0)
                elif cell == -1:
                    feats.append(0.5)
                else:
                    feats.append(0.0)
            else:
                feats.append(1.0)
    
    for dr, dc in DIR4:
        rr, cc = r + dr, c + dc
        walkable = (0 <= rr < H and 0 <= cc < W and memory_map[rr, cc] != 1)
        feats.append(1.0 if walkable else 0.0)
    
    if ghost_pos_or_estimate is not None:
        gr, gc = ghost_pos_or_estimate
        dx = (gr - r) / 21.0
        dy = (gc - c) / 21.0
        dist = manhattan(pac_pos, ghost_pos_or_estimate) / 42.0
        
        dir_up = 1.0 if gr < r else 0.0
        dir_down = 1.0 if gr > r else 0.0
        dir_left = 1.0 if gc < c else 0.0
        dir_right = 1.0 if gc > c else 0.0
        
        in_los = 0.0
        if r == gr or c == gc:
            in_los = 1.0
            if r == gr:
                step_c = 1 if gc > c else -1
                for cc in range(c + step_c, gc, step_c):
                    if memory_map[r, cc] == 1:
                        in_los = 0.0
                        break
            else:
                step_r = 1 if gr > r else -1
                for rr in range(r + step_r, gr, step_r):
                    if memory_map[rr, c] == 1:
                        in_los = 0.0
                        break
        
        feats.extend([1.0, dx, dy, dist, dir_up, dir_down, dir_left, dir_right, in_los, 0.0])
    else:
        feats.extend([0.0] * 10)
    
    if belief is not None and belief.sum() > 1e-6:
        belief_norm = belief / (belief.sum() + 1e-9)
        rows, cols = np.indices(belief.shape)
        centroid_r = float((belief_norm * rows).sum())
        centroid_c = float((belief_norm * cols).sum())
        dx_b = (centroid_r - r) / 21.0
        dy_b = (centroid_c - c) / 21.0
        dist_b = (abs(centroid_r - r) + abs(centroid_c - c)) / 42.0
        max_belief = float(belief.max())
        entropy = float(-np.sum(belief_norm * np.log(belief_norm + 1e-9))) / 6.0
        feats.extend([dx_b, dy_b, dist_b, max_belief, entropy, 0.0, 0.0, 0.0])
    else:
        feats.extend([0.0] * 8)
    
    # 5. Position encoding (4 values)
    feats.append(r / 21.0)
    feats.append(c / 21.0)
    feats.append((r - 10.5) / 10.5)  # Distance from center
    feats.append((c - 10.5) / 10.5)
    
    # 6. Last action one-hot (9 values)
    action_onehot = [0.0] * NUM_ACTIONS
    if 0 <= last_action < NUM_ACTIONS:
        action_onehot[last_action] = 1.0
    feats.extend(action_onehot)
    
    # 7. Time features (3 values)
    feats.append(step / max_steps)
    feats.append(1.0 - step / max_steps)
    feats.append(1.0 if step < 40 else 0.0)  # Early game flag
    
    # 8. Game parameters (2 values)
    feats.append(pacman_speed / 2.0)
    feats.append(0.0)  # Reserved
    
    return np.array(feats, dtype=np.float32)

def build_state(
    memory_map: np.ndarray,
    pac_pos: Tuple[int, int],
    ghost_visible_pos: Optional[Tuple[int, int]],
    belief: np.ndarray,
    last_action: int,
    step: int,
    max_steps: int,
    device: torch.device,
    pacman_speed: int = 2,
) -> torch.Tensor:
    """Build state tensor for training/inference."""
    feats = build_compact_features(
        memory_map, pac_pos, ghost_visible_pos, belief,
        last_action, step, max_steps, pacman_speed
    )
    return torch.from_numpy(feats).unsqueeze(0).to(device)

# Legal action mask (rule: turn -> 1 step; straight -> 1 or 2)

def legal_actions(env: Environment, pac_pos: Tuple[int, int], last_move: Move, pacman_speed: int) -> List[int]:
    legal = []

    # STAY always allowed
    legal.append(8)

    for mv in ALL_MOVES:
        # 1-step always candidate
        dr, dc = mv.value
        r1, c1 = pac_pos[0] + dr, pac_pos[1] + dc
        if env._in_bounds(r1, c1) and env.map[r1, c1] == 0:
            legal.append(MOVE_TO_1STEP_IDX[mv])

        # 2-step only if continuing same direction (and speed>=2)
        if pacman_speed >= 2 and mv == last_move and mv != Move.STAY:
            r2, c2 = r1 + dr, c1 + dc
            if env._in_bounds(r2, c2) and env.map[r2, c2] == 0 and env.map[r1, c1] == 0:
                legal.append(MOVE_TO_2STEP_IDX[mv])

    # remove dup
    legal = sorted(set(legal))
    return legal

def select_action(policy_net: nn.Module, state: torch.Tensor, legal: List[int],
                  steps_done: int, eps_start: float, eps_end: float, eps_decay: int,
                  device: torch.device):
    eps = eps_end + (eps_start - eps_end) * math.exp(-steps_done / eps_decay)
    if random.random() < eps:
        return random.choice(legal), eps

    with torch.no_grad():
        q = policy_net(state).squeeze(0)
        mask = torch.full((NUM_ACTIONS,), float("-inf"), device=device)
        mask[legal] = 0.0
        q = q + mask
        return int(torch.argmax(q).item()), eps

# Ghost policy opponent (loads ghost_policy.pth if exists)
# Feature vector MUST match keys (your check showed net.0/net.2/net.4)

class GhostPolicyOpponent:
    IDX_TO_MOVE = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]
    MOVE_TO_IDX = {m: i for i, m in enumerate(IDX_TO_MOVE)}

    def __init__(self, policy_path: Path, pacman_speed: int, capture_distance: int):
        self.policy_path = policy_path
        self.pacman_speed = pacman_speed
        self.capture_distance = capture_distance
        self.reach1 = pacman_speed + capture_distance - 1

        self.device = torch.device("cpu")
        self.net = None
        self.input_dim = None

        # ghost internal memory for feature extraction
        self.memory_map = None
        self.visit = None
        self.prev_positions = deque(maxlen=10)
        self.last_action_idx = self.MOVE_TO_IDX[Move.STAY]
        self.last2_action_idx = self.MOVE_TO_IDX[Move.STAY]
        self.last_move = Move.STAY

        self._load()

    def _load(self):
        sd = torch.load(self.policy_path, map_location="cpu")
        w0 = sd.get("net.0.weight", None)
        if w0 is None:
            raise RuntimeError("ghost_policy.pth format not recognized (missing net.0.weight)")
        self.input_dim = int(w0.shape[1])
        h1 = int(w0.shape[0])
        w2 = sd["net.2.weight"]
        h2 = int(w2.shape[0])

        # Create network with named "net" sequential to match saved weights
        class GhostNet(nn.Module):
            def __init__(self, in_dim, h1, h2):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_dim, h1),
                    nn.ReLU(),
                    nn.Linear(h1, h2),
                    nn.ReLU(),
                    nn.Linear(h2, 5),
                )
            def forward(self, x):
                return self.net(x)
        
        self.net = GhostNet(self.input_dim, h1, h2)
        self.net.load_state_dict(sd, strict=True)
        self.net.eval()

    def _ensure(self, obs: np.ndarray):
        if self.memory_map is None or self.memory_map.shape != obs.shape:
            self.memory_map = np.full(obs.shape, -1, dtype=np.int8)
            self.visit = np.zeros(obs.shape, dtype=np.int16)

    def _update_memory(self, obs: np.ndarray):
        self._ensure(obs)
        vis = (obs != -1)
        self.memory_map[vis] = obs[vis]

    def _in_bounds(self, r, c):
        H, W = self.memory_map.shape
        return 0 <= r < H and 0 <= c < W

    def _walkable(self, r, c):
        return self._in_bounds(r, c) and int(self.memory_map[r, c]) != 1

    def _degree(self, pos):
        r, c = pos
        deg = 0
        for dr, dc in DIR4:
            rr, cc = r + dr, c + dc
            if self._walkable(rr, cc):
                deg += 1
        return deg

    def _los_flag(self, a: tuple, b: Optional[tuple], radius=5) -> float:
        if b is None:
            return 0.0
        ar, ac = a
        br, bc = b
        if ar != br and ac != bc:
            return 0.0
        dist = abs(ar - br) + abs(ac - bc)
        if dist > radius:
            return 0.0
        dr = 0 if ar == br else (1 if br > ar else -1)
        dc = 0 if ac == bc else (1 if bc > ac else -1)
        r, c = ar, ac
        for _ in range(dist):
            r += dr
            c += dc
            if not self._in_bounds(r, c):
                return 0.0
            if int(self.memory_map[r, c]) == 1:
                return 0.0
        return 1.0

    def _one_hot(self, idx: int, n: int) -> List[float]:
        v = [0.0] * n
        if 0 <= idx < n:
            v[idx] = 1.0
        return v

    def _legal_mask(self, pos):
        legal = [True, True, True, True, True]
        for mv in ALL_MOVES:
            dr, dc = mv.value
            rr, cc = pos[0] + dr, pos[1] + dc
            if not self._walkable(rr, cc):
                legal[self.MOVE_TO_IDX[mv]] = False
        legal[self.MOVE_TO_IDX[Move.STAY]] = True
        return legal

    def _feat(self, obs: np.ndarray, my_pos: tuple, pac_pos_visible: Optional[tuple], enemy_visible: bool) -> np.ndarray:
        # MUST match the 47-dim style you already used in agent.py earlier
        r, c = my_pos
        H, W = obs.shape

        feats = [float(obs[r, c])]

        # 4 rays * 5
        for dr, dc in DIR4:
            rr, cc = r, c
            for _ in range(5):
                rr += dr
                cc += dc
                if 0 <= rr < H and 0 <= cc < W:
                    feats.append(float(obs[rr, cc]))
                else:
                    feats.append(1.0)

        # neighbor walkable (memory)
        for dr, dc in DIR4:
            rr, cc = r + dr, c + dc
            feats.append(1.0 if (0 <= rr < H and 0 <= cc < W and int(self.memory_map[rr, cc]) != 1) else 0.0)

        feats.append(1.0 if enemy_visible else 0.0)

        if pac_pos_visible is None:
            feats.extend([0.0, 0.0, 0.0])  # dx dy dist
            feats.append(0.0)  # los
            feats.append(0.0)  # degree norm
        else:
            dx = float(pac_pos_visible[0] - r)
            dy = float(pac_pos_visible[1] - c)
            dist = float(abs(dx) + abs(dy))
            feats.extend([dx / 21.0, dy / 21.0, dist / 42.0])
            feats.append(self._los_flag(my_pos, pac_pos_visible, radius=5))
            feats.append(float(self._degree(my_pos)) / 4.0)

        feats.append(float(self.visit[r, c]) / 50.0)
        prev2 = self.prev_positions[-2] if len(self.prev_positions) >= 2 else None
        feats.append(1.0 if (prev2 is not None and prev2 == my_pos) else 0.0)
        feats.append(1.0 if int(self.memory_map[r, c]) == -1 else 0.0)

        feats.extend(self._one_hot(int(self.last_action_idx), 5))
        feats.extend(self._one_hot(int(self.last2_action_idx), 5))

        feats.append(float(self.pacman_speed) / 2.0)
        feats.append(float(self.capture_distance) / 2.0)
        feats.append(float(self.reach1) / 21.0)

        x = np.array(feats, dtype=np.float32)

        # Safety: if mismatch, pad/trim (so it never crashes training)
        if self.input_dim is not None:
            if x.shape[0] < self.input_dim:
                x = np.pad(x, (0, self.input_dim - x.shape[0]), mode="constant")
            elif x.shape[0] > self.input_dim:
                x = x[: self.input_dim]
        return x

    def step(self, ghost_obs_map: np.ndarray, ghost_pos: tuple, pac_pos_visible: Optional[tuple], step_number: int) -> Move:
        self._update_memory(ghost_obs_map)
        self.visit[ghost_pos[0], ghost_pos[1]] += 1
        self.prev_positions.append(ghost_pos)

        enemy_visible = pac_pos_visible is not None

        feat = self._feat(ghost_obs_map, ghost_pos, pac_pos_visible, enemy_visible)
        x = torch.from_numpy(feat).unsqueeze(0)

        with torch.no_grad():
            q = self.net(x).squeeze(0).cpu().numpy()

        legal = self._legal_mask(ghost_pos)
        for i in range(5):
            if not legal[i]:
                q[i] -= 1e9

        best = int(np.argmax(q))
        mv = self.IDX_TO_MOVE[best]

        self.last2_action_idx = int(self.last_action_idx)
        self.last_action_idx = int(best)
        self.last_move = mv
        return mv

# Fallback ghost heuristic (evade by BFS distance)
class SmartGhostHeuristic:
    def __init__(self, env: Environment):
        self.env = env
        self.last_move = Move.STAY

    def _apply(self, pos, mv):
        dr, dc = mv.value
        rr, cc = pos[0] + dr, pos[1] + dc
        if self.env._in_bounds(rr, cc) and self.env.map[rr, cc] == 0:
            return (rr, cc)
        return pos

    def _bfs_dist(self, start, goal):
        if start == goal:
            return 0
        q = deque([start])
        dist = {start: 0}
        while q:
            p = q.popleft()
            d = dist[p]
            for mv in ALL_MOVES:
                np2 = self._apply(p, mv)
                if np2 == p:
                    continue
                if np2 not in dist:
                    dist[np2] = d + 1
                    if np2 == goal:
                        return d + 1
                    q.append(np2)
        return 10**9

    def move(self, ghost_pos, pac_pos):
        best = Move.STAY
        bestd = -1
        for mv in ALL_MOVES:
            np2 = self._apply(ghost_pos, mv)
            if np2 == ghost_pos:
                continue
            d = self._bfs_dist(np2, pac_pos)
            if d > bestd:
                bestd = d
                best = mv
        return best

# Optimize (Simple DQN with MSE loss - more stable)
def optimize(policy_net, target_net, optimizer, replay: ReplayBuffer,
             batch_size: int, gamma: float, device: torch.device):
    if len(replay) < batch_size:
        return 0.0

    batch = replay.sample(batch_size)

    state = torch.cat(batch.state).to(device)
    action = torch.tensor(batch.action, dtype=torch.long, device=device).unsqueeze(1)
    reward = torch.tensor(batch.reward, dtype=torch.float32, device=device)
    done = torch.tensor(batch.done, dtype=torch.bool, device=device)

    # Current Q values
    q = policy_net(state).gather(1, action).squeeze(1)

    # Next Q values (standard DQN, not Double - simpler)
    next_q = torch.zeros(len(batch.state), device=device)
    non_final = ~done
    if non_final.any():
        ns = torch.cat([s for s, d in zip(batch.next_state, batch.done) if not d]).to(device)
        with torch.no_grad():
            next_q[non_final] = target_net(ns).max(dim=1)[0]

    # Target with clipping to prevent explosion
    target = reward + gamma * next_q
    target = target.clamp(-2.0, 2.0)  # Clamp targets

    # MSE loss (more stable than Huber for small rewards)
    loss = nn.functional.mse_loss(q, target)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
    optimizer.step()

    return float(loss.item())

# Reward shaping - AGGRESSIVE for FAST capture (<40 steps)
def compute_reward(done: bool, result: str, step: int, max_steps: int,
                   ghost_visible: bool, prev_dist: int, new_dist: int,
                   new_explored: int, steps_since_seen: int):
    """Reward shaping focused on FAST capture (<40 steps)."""
    if done:
        if result == "pacman_wins":
            # Stronger speed bonus: exponential decay with step
            if step <= 20:
                return 1.0  # Perfect!
            elif step <= 30:
                return 0.9
            elif step <= 40:
                return 0.8
            elif step <= 60:
                return 0.6
            elif step <= 100:
                return 0.4
            else:
                return 0.2  # Slow but won
        return -1.0  # timeout = worst case

    r = -0.01  # Stronger step penalty to encourage fast capture

    if ghost_visible:
        # Distance rewards (normalized)
        dd = prev_dist - new_dist
        r += 0.1 * dd  # Stronger bonus for closing distance
        
        # Proximity bonuses
        if new_dist <= 1:
            r += 0.2  # About to capture!
        elif new_dist <= 2:
            r += 0.1
        elif new_dist <= 3:
            r += 0.05
        elif new_dist <= 5:
            r += 0.02
        
        # Penalty for increasing distance when close
        if dd < 0 and prev_dist <= 5:
            r -= 0.05 * abs(dd)
    else:
        # Small exploration bonus
        r += 0.005 * min(float(new_explored), 5.0)
        
        # Penalty for losing ghost
        if steps_since_seen > 15:
            r -= 0.02

    # Clip to prevent any extreme values
    return max(-1.0, min(1.0, r))

# Training loop
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Training on device: {device}")

    env = Environment(
        max_steps=args.max_steps,
        deterministic_starts=not args.random_starts,
        pacman_speed=args.pacman_speed,
        capture_distance_threshold=args.capture_distance,
    )

    H, W = env.map.shape
    
    # Compute input dimension from compact features
    # 20 (rays) + 4 (neighbors) + 10 (ghost) + 8 (belief) + 4 (pos) + 9 (action) + 3 (time) + 2 (params) = 60
    input_dim = 60
    
    policy = DuelingDQN(input_dim=input_dim, hidden=args.hidden, num_actions=NUM_ACTIONS).to(device)
    target = DuelingDQN(input_dim=input_dim, hidden=args.hidden, num_actions=NUM_ACTIONS).to(device)
    target.load_state_dict(policy.state_dict())
    target.eval()

    optimizer = optim.Adam(policy.parameters(), lr=args.lr)
    replay = ReplayBuffer(args.buffer_size)

    # Curriculum: ghost opponent
    ghost_policy_path = Path(__file__).parent / "ghost_policy.pth"
    ghost_policy = None
    ghost_heur = SmartGhostHeuristic(env)

    if ghost_policy_path.exists():
        ghost_policy = GhostPolicyOpponent(ghost_policy_path, pacman_speed=args.pacman_speed, capture_distance=args.capture_distance)
        print(f"  [GhostRL] Loaded ghost policy from {ghost_policy_path}")
        print("Training against Ghost RL: True")
    else:
        print("  [GhostRL] ghost_policy.pth not found -> heuristic ghost")
        print("Training against Ghost RL: False")

    save_path = Path(args.output).expanduser().resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)

    recent_wins = deque(maxlen=100)
    recent_steps = deque(maxlen=100)
    best_score = -1e18

    global_step = 0
    started_optim = False

    beta_start, beta_end = 0.4, 1.0

    for ep in range(1, args.episodes + 1):
        full_map, pac_pos, ghost_pos = env.reset()

        # pacman memory starts unknown except walls (same as game)
        memory = np.full_like(full_map, -1, dtype=np.int8)
        memory[full_map == 1] = 1  # walls known

        belief = SimpleBelief((H, W))
        belief.reset()

        last_move = Move.STAY
        last_action = 8

        # Observability curriculum: first warmup episodes give true ghost pos
        reveal_true = (ep <= int(args.episodes * args.warmup_reveal))

        # initial obs - get_observation returns (obs, my_pos, enemy_pos_or_none)
        pac_obs, _, ghost_vis = env.get_observation("pacman", pacman_radius=args.obs_radius, ghost_radius=args.obs_radius)
        vis_cells = set(zip(*np.where(pac_obs != -1)))
        memory[pac_obs != -1] = pac_obs[pac_obs != -1]

        true_or_vis = ghost_pos if reveal_true else ghost_vis
        belief.update(memory, vis_cells, true_or_vis)

        state = build_state(memory, pac_pos, true_or_vis, belief.map(), last_action, 0, args.max_steps, device)

        done = False
        step = 0
        ep_reward = 0.0
        loss_sum = 0.0
        loss_cnt = 0

        explored = set(vis_cells)
        prev_dist = env.get_distance(pac_pos, ghost_pos)

        beta = beta_start + (beta_end - beta_start) * (ep / args.episodes)

        while not done:
            step += 1

            legals = legal_actions(env, pac_pos, last_move, args.pacman_speed)
            a_idx, eps = select_action(policy, state, legals, global_step, args.eps_start, args.eps_end, args.eps_decay, device)
            mv, st = ACTIONS[a_idx]

            # Enforce rule: if turn, force 1 step (even if network picked a 2-step somehow)
            if mv != Move.STAY and mv != last_move:
                st = 1
                a_idx = MOVE_TO_1STEP_IDX[mv]

            pac_action = (mv, st)

            # Ghost move - get_observation returns (obs, my_pos, enemy_pos_or_none)
            ghost_obs, _, pac_vis_for_ghost = env.get_observation("ghost", pacman_radius=args.obs_radius, ghost_radius=args.obs_radius)
            if ghost_policy is not None:
                gmv = ghost_policy.step(ghost_obs, ghost_pos, pac_vis_for_ghost, step)
            else:
                # heuristic uses true pac_pos (harder ghost), ok for training
                gmv = ghost_heur.move(ghost_pos, pac_pos)

            done, result, new_state = env.step(pac_action, gmv)
            full_map, pac_pos, ghost_pos = new_state

            # new obs
            pac_obs, _, ghost_vis = env.get_observation("pacman", pacman_radius=args.obs_radius, ghost_radius=args.obs_radius)
            vis_cells = set(zip(*np.where(pac_obs != -1)))
            memory[pac_obs != -1] = pac_obs[pac_obs != -1]

            new_explored = len(vis_cells - explored)
            explored |= vis_cells

            true_or_vis = ghost_pos if reveal_true else ghost_vis
            belief.update(memory, vis_cells, true_or_vis)

            new_dist = env.get_distance(pac_pos, ghost_pos)
            ghost_visible_flag = (true_or_vis is not None)

            r = compute_reward(
                done=done,
                result=result,
                step=step,
                max_steps=args.max_steps,
                ghost_visible=ghost_visible_flag,
                prev_dist=int(prev_dist),
                new_dist=int(new_dist),
                new_explored=new_explored,
                steps_since_seen=belief.steps_since_seen,
            )

            next_state = build_state(memory, pac_pos, true_or_vis, belief.map(), a_idx, step, args.max_steps, device)

            replay.push(state, a_idx, float(r), next_state, bool(done))

            state = next_state
            prev_dist = new_dist
            last_move = mv
            last_action = a_idx
            ep_reward += r
            global_step += 1

            if len(replay) >= args.min_replay_size:
                loss = optimize(policy, target, optimizer, replay, args.batch_size, args.gamma, device)
                started_optim = True
                loss_sum += loss
                loss_cnt += 1

            if global_step % args.target_update == 0:
                target.load_state_dict(policy.state_dict())

        won = (result == "pacman_wins")
        recent_wins.append(1 if won else 0)
        if won:
            recent_steps.append(step)

        win_rate = float(np.mean(recent_wins)) if recent_wins else 0.0
        avg_steps = float(np.mean(recent_steps)) if recent_steps else args.max_steps

        # score emphasizes win_rate, then speed
        score = 100.0 * win_rate - 0.5 * avg_steps  # Giảm penalty

        if len(recent_wins) >= 50 and score > best_score:
            best_score = score
            torch.save(policy.state_dict(), save_path)
            print(f"  [NEW BEST] win={win_rate:.1%} avg_steps={avg_steps:.1f} score={score:.2f}")

        if ep % args.log_every == 0:
            avg_loss = loss_sum / max(1, loss_cnt)
            print(f"Ep {ep:04d} | R {ep_reward:8.2f} | win {win_rate:5.1%} | "
                  f"avg_steps {avg_steps:6.1f} | eps {eps:.3f} | loss {avg_loss:.4f} | reveal {reveal_true}")

    print(f"\nTraining finished. Best model at: {save_path}")
    print(f"  Best score: {best_score:.2f} (win={win_rate:.1%}, steps={avg_steps:.1f})")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--buffer-size", type=int, default=100000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.1)
    p.add_argument("--eps-decay", type=int, default=30000)
    p.add_argument("--target-update", type=int, default=200)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--pacman-speed", type=int, default=2)
    p.add_argument("--capture-distance", type=int, default=2)
    p.add_argument("--obs-radius", type=int, default=5)
    p.add_argument("--min-replay-size", type=int, default=2000)  # Bắt đầu học sớm hơn
    p.add_argument("--random-starts", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--output", type=str, default="./submissions/group_05/pacman_dqn.pth")
    p.add_argument("--log-every", type=int, default=50)  # Log thường xuyên hơn
    p.add_argument("--warmup-reveal", type=float, default=1.0)  # Always reveal!
    return p.parse_args()

if __name__ == "__main__":
    train(parse_args())
