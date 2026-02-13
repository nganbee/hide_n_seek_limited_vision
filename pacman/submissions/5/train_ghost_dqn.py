"""
Train Ghost with DQN and save weights to ghost_policy.pth.
Includes stability fixes for robust training.
"""

from __future__ import annotations

import sys
import random
from dataclasses import dataclass
from pathlib import Path
from collections import deque
from typing import Deque, Tuple, Optional, List

import numpy as np

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]           
SRC_DIR = PROJECT_ROOT / "src"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
GROUP_DIR = THIS_FILE.parent               

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SUBMISSIONS_DIR))

from environment import Environment, Move 

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception as e:
    raise RuntimeError(
        "PyTorch is required to train. Please install torch.\n"
        f"Original error: {e}"
    )

try:
    torch.set_num_threads(1)
except Exception:
    pass


NUM_ACTIONS = 5
IDX_TO_MOVE = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]
MOVE_TO_IDX = {m: i for i, m in enumerate(IDX_TO_MOVE)}

DIR4 = [(-1,0),(1,0),(0,-1),(0,1)]


def one_hot(idx: int, n: int) -> List[float]:
    v = [0.0] * n
    if 0 <= idx < n:
        v[idx] = 1.0
    return v


class BeliefTracker:
    def __init__(self, decay: float = 0.95, visible_decay: float = 0.10, seed=None):
        self.decay = float(decay)
        self.visible_decay = float(visible_decay)
        self.belief = None
        self.rng = random.Random(seed)

    def ensure(self, shape):
        if self.belief is None or self.belief.shape != shape:
            self.belief = np.zeros(shape, dtype=np.float32)

    def reset_to(self, pos):
        self.belief.fill(0.0)
        self.belief[pos[0], pos[1]] = 1.0

    def init_uniform(self, possible_mask: np.ndarray):
        self.belief.fill(0.0)
        cnt = int(np.sum(possible_mask))
        if cnt <= 0:
            self.belief[:] = 1.0 / self.belief.size
        else:
            self.belief[possible_mask] = 1.0 / cnt

    def _propagate(self, possible_mask: np.ndarray):
        H, W = possible_mask.shape
        newb = np.zeros_like(self.belief, dtype=np.float32)

        rows, cols = np.where(self.belief > 1e-7)
        for r, c in zip(rows, cols):
            p = float(self.belief[r, c])
            if p <= 0 or not possible_mask[r, c]:
                continue

            opts = [(r, c)]
            for dr, dc in DIR4:
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W and possible_mask[rr, cc]:
                    opts.append((rr, cc))

            share = (p * self.decay) / len(opts)
            for rr, cc in opts:
                newb[rr, cc] += share

        s = float(newb.sum())
        if s > 1e-9:
            newb /= s
        self.belief = newb

    def update(self, obs_map: np.ndarray, memory_map: np.ndarray, enemy_pos):
        self.ensure(memory_map.shape)
        possible = (memory_map != 1)

        if enemy_pos is not None:
            self.reset_to(enemy_pos)
            return

        if float(self.belief.sum()) <= 1e-9:
            self.init_uniform(possible)
        else:
            self._propagate(possible)

        visible = (obs_map != -1)
        self.belief[visible] *= self.visible_decay
        self.belief[memory_map == 1] = 0.0

        s = float(self.belief.sum())
        if s <= 1e-9:
            self.init_uniform(possible)
        else:
            self.belief /= s

    def best_guess(self):
        if self.belief is None:
            return None
        mx = float(self.belief.max())
        if mx <= 0:
            return None
        coords = np.argwhere(self.belief >= 0.90 * mx)
        if coords.size == 0:
            return None
        r, c = coords[self.rng.randrange(len(coords))]
        return (int(r), int(c))



def load_pacman_agent_class(agent_py_path: Path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("student_agent_module", str(agent_py_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module spec from: {agent_py_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    if not hasattr(module, "PacmanAgent"):
        raise RuntimeError(f"PacmanAgent not found in {agent_py_path}")
    return module.PacmanAgent


class EnvAdapter:
    def __init__(
        self,
        max_steps: int = 200,
        deterministic_starts: bool = False,
        capture_distance_threshold: int = 1,
        pacman_speed: int = 1,
        pacman_obs_radius: int = 5,
        ghost_obs_radius: int = 5,
    ):
        self.env = Environment(
            max_steps=max_steps,
            deterministic_starts=deterministic_starts,
            capture_distance_threshold=max(1, int(capture_distance_threshold)),
            pacman_speed=max(1, int(pacman_speed)),
        )
        self.pacman_obs_radius = max(0, int(pacman_obs_radius))
        self.ghost_obs_radius = max(0, int(ghost_obs_radius))

    def reset(self):
        self.env.reset()
        pac_obs, pac_pos, pac_enemy = self.env.get_observation(
            "pacman", self.pacman_obs_radius, self.ghost_obs_radius
        )
        ghost_obs, ghost_pos, ghost_enemy = self.env.get_observation(
            "ghost", self.pacman_obs_radius, self.ghost_obs_radius
        )
        return pac_obs, pac_pos, pac_enemy, ghost_obs, ghost_pos, ghost_enemy

    def step(self, pacman_action, ghost_action_idx: int):
        ghost_move = IDX_TO_MOVE[int(ghost_action_idx)]
        game_over, result, _new_state = self.env.step(pacman_action, ghost_move)

        pac_obs, pac_pos, pac_enemy = self.env.get_observation(
            "pacman", self.pacman_obs_radius, self.ghost_obs_radius
        )
        ghost_obs, ghost_pos, ghost_enemy = self.env.get_observation(
            "ghost", self.pacman_obs_radius, self.ghost_obs_radius
        )

        info = {
            "result": result,
            "ghost_win": (result == "ghost_wins") if game_over else None,
            "ghost_caught": (result == "pacman_wins") if game_over else False,
        }
        return pac_obs, pac_pos, pac_enemy, ghost_obs, ghost_pos, ghost_enemy, game_over, info


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.buf: Deque = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buf.append((s, a, r, s2, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (
            np.stack(s),
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.stack(s2),
            np.array(d, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buf)


@dataclass
class DQNConfig:
    input_dim: int
    hidden1: int = 128
    hidden2: int = 128
    num_actions: int = NUM_ACTIONS


class MLPDQN(nn.Module):
    def __init__(self, cfg: DQNConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.hidden1),
            nn.ReLU(),
            nn.Linear(cfg.hidden1, cfg.hidden2),
            nn.ReLU(),
            nn.Linear(cfg.hidden2, cfg.num_actions),
        )

    def forward(self, x):
        return self.net(x)


def update_memory_from_obs(memory_map: np.ndarray, obs: np.ndarray):
    visible = (obs != -1)
    memory_map[visible] = obs[visible]

def degree(memory_map: np.ndarray, pos: Tuple[int,int]) -> int:
    r, c = pos
    deg = 0
    for dr, dc in DIR4:
        rr, cc = r+dr, c+dc
        if 0 <= rr < 21 and 0 <= cc < 21 and int(memory_map[rr, cc]) != 1:
            deg += 1
    return deg

def dist_to_nearest_corner(pos: Tuple[int,int]) -> int:
    """Calculate Manhattan distance to nearest corner of 21x21 map."""
    corners = [(1, 1), (1, 19), (19, 1), (19, 19)]
    return min(manhattan(pos, corner) for corner in corners)

def is_near_edge(pos: Tuple[int,int]) -> bool:
    """Check if position is near map edge (within 3 cells)."""
    r, c = pos
    return r < 3 or r > 17 or c < 3 or c > 17

def count_frontier_neighbors(memory_map: np.ndarray, pos: Tuple[int,int]) -> int:
    """Count number of unknown (-1) cells adjacent to position."""
    r, c = pos
    count = 0
    for dr, dc in DIR4:
        rr, cc = r + dr, c + dc
        if 0 <= rr < 21 and 0 <= cc < 21 and int(memory_map[rr, cc]) == -1:
            count += 1
    return count

def has_los(memory_map: np.ndarray, a: Tuple[int,int], b: Optional[Tuple[int,int]], radius: int = 5) -> bool:
    if b is None:
        return False
    ar, ac = a
    br, bc = b
    if ar != br and ac != bc:
        return False
    dist = abs(ar-br) + abs(ac-bc)
    if dist > radius:
        return False
    dr = 0 if ar == br else (1 if br > ar else -1)
    dc = 0 if ac == bc else (1 if bc > ac else -1)
    r, c = ar, ac
    for _ in range(dist):
        r += dr; c += dc
        if not (0 <= r < 21 and 0 <= c < 21):
            return False
        if int(memory_map[r, c]) == 1:
            return False
    return True

def legal_actions(memory_map: np.ndarray, pos: Tuple[int,int]) -> List[int]:
    r, c = pos
    legal = []
    for mv in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
        dr, dc = mv.value
        rr, cc = r+dr, c+dc
        if 0 <= rr < 21 and 0 <= cc < 21 and int(memory_map[rr, cc]) != 1:
            legal.append(MOVE_TO_IDX[mv])
    legal.append(MOVE_TO_IDX[Move.STAY])
    return legal


def extract_ghost_features(
    obs_ghost: np.ndarray,
    memory_map: np.ndarray,
    visit: np.ndarray,
    prev_positions: Deque[Tuple[int, int]],
    my_pos: Tuple[int, int],
    pac_est: Optional[Tuple[int, int]],
    enemy_visible: bool,
    last_action_idx: int,
    last2_action_idx: int,
    pacman_speed: int = 2,
    capture_distance: int = 2,
) -> np.ndarray:
    r, c = my_pos
    H, W = obs_ghost.shape

    feats: List[float] = [float(obs_ghost[r, c])]

    for dr, dc in DIR4:
        rr, cc = r, c
        for _ in range(5):
            rr += dr
            cc += dc
            if 0 <= rr < H and 0 <= cc < W:
                feats.append(float(obs_ghost[rr, cc]))
            else:
                feats.append(1.0)

    for dr, dc in DIR4:
        rr, cc = r + dr, c + dc
        ok = (0 <= rr < H and 0 <= cc < W and int(memory_map[rr, cc]) != 1)
        feats.append(1.0 if ok else 0.0)

    feats.append(1.0 if enemy_visible else 0.0)

    if pac_est is None:
        feats.extend([0.0, 0.0, 0.0, 0.0, 0.0])
    else:
        dx = float(pac_est[0] - r)
        dy = float(pac_est[1] - c)
        dist = float(abs(dx) + abs(dy))
        feats.extend([dx / 21.0, dy / 21.0, dist / 42.0])
        feats.append(1.0 if has_los(memory_map, my_pos, pac_est, radius=5) else 0.0)
        feats.append(float(degree(memory_map, my_pos)) / 4.0)

    feats.append(float(visit[r, c]) / 50.0)
    prev2 = prev_positions[-2] if len(prev_positions) >= 2 else None
    feats.append(1.0 if (prev2 is not None and prev2 == my_pos) else 0.0)
    feats.append(1.0 if int(memory_map[r, c]) == -1 else 0.0)

    feats.extend(one_hot(int(last_action_idx), 5))
    feats.extend(one_hot(int(last2_action_idx), 5))
    
    feats.append(float(pacman_speed) / 2.0)
    feats.append(float(capture_distance) / 2.0)
    reach1 = pacman_speed + capture_distance - 1
    feats.append(float(reach1) / 21.0)

    return np.array(feats, dtype=np.float32)


# Reward shaping (Strategy-aligned) - stronger anti-oscillation
def manhattan(a, b) -> int:
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def shaped_reward(
    prev_pos: Tuple[int,int],
    new_pos: Tuple[int,int],
    pac_est_prev: Optional[Tuple[int,int]],
    pac_est_new: Optional[Tuple[int,int]],
    memory_map: np.ndarray,
    visit: np.ndarray,
    prev_positions: Deque[Tuple[int,int]],
    caught: bool,
    done: bool,
    win: Optional[bool],
    was_unknown_before: bool,
    prev2_pos: Optional[Tuple[int,int]],
    cfg,
    step_in_episode: int = 0,  
    corner_dwell_info: Optional[Tuple[bool, int, Optional[Tuple], int]] = None, 
) -> float:
    if caught:
        return cfg.penalty_caught  
    if done and win is True:
        return cfg.reward_win     
    if done and win is False:
        return cfg.penalty_timeout 

    # ANTI-CORNER-TRAP: Continuous visit decay to prevent lock-in
    # Decay all visit counts slightly each step (0.2% per step)
    visit[:] = (visit * 0.998).astype(np.int16)

    r = -0.02  # time penalty

    # Compute reach1 (danger zone threshold)
    reach1 = cfg.pacman_speed + cfg.capture_distance_threshold - 1

    # distance progress (want distance to increase)
    if pac_est_prev is not None and pac_est_new is not None:
        d0 = manhattan(prev_pos, pac_est_prev)
        d1 = manhattan(new_pos, pac_est_new)
        r += cfg.w_dist * float(d1 - d0)
        
        # CRITICAL: Extra penalty for moving closer when in danger zone
        if d0 <= reach1 and d1 < d0:
            r -= 2.0  # Strong penalty for approaching in danger zone
        # Small reward for increasing distance in danger zone
        if d0 <= reach1 and d1 > d0:
            r += 0.3  # Bonus for escaping

    if pac_est_new is not None:
        dist_new = manhattan(new_pos, pac_est_new)
        if dist_new <= reach1:
            r -= cfg.penalty_close * float(reach1 + 2 - dist_new)
        elif dist_new <= reach1 + 2:
            r -= cfg.penalty_close_far

    if new_pos == prev_pos:
        if pac_est_new is None:
            r -= cfg.penalty_stay * 2.5
        else:
            r -= cfg.penalty_stay

    if prev2_pos is not None and new_pos == prev2_pos:
        if pac_est_new is None:
            r -= cfg.penalty_abab * 2.0
        else:
            r -= cfg.penalty_abab

    recent = list(prev_positions)[-cfg.recent_k:]
    if new_pos in recent:
        idx_from_end = recent[::-1].index(new_pos)
        r -= cfg.penalty_recent * float(cfg.recent_k - idx_from_end)

    if was_unknown_before:
        if pac_est_new is None:
            r += cfg.reward_explore * 1.5
        else:
            r += cfg.reward_explore
    
    frontier_count = count_frontier_neighbors(memory_map, new_pos)
    if pac_est_new is None:
        r += cfg.reward_frontier * frontier_count
    else:
        r += cfg.reward_frontier * 0.3 * frontier_count
    
    recent_4 = list(prev_positions)[-4:] if len(prev_positions) >= 4 else []
    if len(recent_4) >= 4:
        recent_set = set(recent_4)
        if len(recent_set) <= 2 and new_pos in recent_set:
            r -= cfg.penalty_stuck

    r += cfg.reward_novelty / (1.0 + float(visit[new_pos[0], new_pos[1]]))
    
    if visit[new_pos[0], new_pos[1]] > 50:
        visit[new_pos[0], new_pos[1]] = 50

    # LOS penalty
    if pac_est_new is not None and has_los(memory_map, new_pos, pac_est_new, radius=5):
        r -= cfg.penalty_los

    # dead-end / junction
    deg = degree(memory_map, new_pos)
    if deg <= 1:
        r -= cfg.penalty_deadend
    elif deg >= 3:
        r += cfg.reward_junction

    if pac_est_new is None or manhattan(new_pos, pac_est_new) > reach1 + 4:
        
        if step_in_episode <= 50:
            prev_corner_dist = dist_to_nearest_corner(prev_pos)
            new_corner_dist = dist_to_nearest_corner(new_pos)
            if new_corner_dist < prev_corner_dist:
                r += cfg.reward_corner_early
            
            nr, nc = new_pos
            if 8 <= nr <= 12 or 9 <= nc <= 11:
                r -= cfg.penalty_center_early
            
            if new_corner_dist <= 3:
                r += 0.5
        
        elif step_in_episode <= 120:
            if is_near_edge(new_pos) and not is_near_edge(prev_pos):
                r += cfg.reward_edge_mid
            
            if was_unknown_before:
                r += cfg.reward_coverage_mid
    
    if corner_dwell_info is not None:
        is_camping, dwell_counter, cluster_bounds, cooldown_until = corner_dwell_info
        
        if is_camping and cluster_bounds is not None:
            min_r, max_r, min_c, max_c = cluster_bounds
            stays_in_cluster = (min_r <= new_pos[0] < max_r and 
                               min_c <= new_pos[1] < max_c)
            
            if stays_in_cluster:
                dwell_penalty = 1.5 * max(0, dwell_counter - 8)
                r -= dwell_penalty
            else:
                if step_in_episode > cooldown_until:
                    r += 3.0
                else:
                    r -= 0.5

    return float(np.clip(r, -cfg.reward_clip, cfg.reward_clip))

# Training config
@dataclass
class TrainConfig:
    episodes: int = 4000            
    max_steps: int = 200
    gamma: float = 0.99
    lr: float = 1e-4                
    batch_size: int = 128
    start_learn: int = 4500
    train_every: int = 4
    target_sync: int = 500        
    buffer_size: int = 140_000
    grad_clip: float = 1.0          
    reward_clip: float = 10.0      

    eps_start: float = 0.9         
    eps_end: float = 0.10          
    eps_decay_steps: int = 300_000  

    pacman_obs_radius: int = 5
    ghost_obs_radius: int = 5
    deterministic_starts: bool = True
    capture_distance_threshold: int = 2
    
    pacman_speed: int = 2
    capture_distance: int = 2 
    
    w_dist: float = 0.15
    penalty_abab: float = 4.0       
    penalty_stay: float = 0.50   
    penalty_recent: float = 0.35   
    recent_k: int = 6
    penalty_stuck: float = 2.5     

    reward_win: float = 15.0
    penalty_caught: float = -15.0
    penalty_timeout: float = -8.0  

    # close_radius removed - now using dynamic reach1 = pacman_speed + capture_distance - 1
    penalty_close: float = 0.8      
    penalty_close_far: float = 0.2 

    reward_explore: float = 0.15   
    reward_novelty: float = 0.15   
    reward_frontier: float = 0.25  

    # PROGRESSIVE REWARD SHAPING (corner->explore->survive)
    # Early game (0-50 steps): Corner-seeking to avoid spawn area
    reward_corner_early: float = 0.8   # Reward for moving toward corners
    penalty_center_early: float = 0.5  # Penalty for center area (rows/cols 8-12)
    
    # Middle game (50-120 steps): Map exploration
    reward_edge_mid: float = 0.3       # Reward for edge exploration
    reward_coverage_mid: float = 0.4   # Bonus for visiting new quadrants
    
    # Late game (120+ steps): Survival focus (existing weights already tuned)

    penalty_los: float = 0.9
    penalty_deadend: float = 0.85
    reward_junction: float = 0.18   # stronger (was 0.12)

    weight_out: Path = GROUP_DIR / "ghost_policy.pth"


def epsilon_by_step(t: int, cfg: TrainConfig) -> float:
    if t >= cfg.eps_decay_steps:
        return cfg.eps_end
    frac = t / float(cfg.eps_decay_steps)
    return cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)


def main():
    cfg = TrainConfig()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    pacman_agent_path_own = GROUP_DIR / "agent.py"
    pacman_agent_path_old = SUBMISSIONS_DIR / "group_06" / "agent.py"
    pacman_agent_path_example = SUBMISSIONS_DIR / "example_student" / "agent.py"
    
    PacmanAgentCls_Own = load_pacman_agent_class(pacman_agent_path_own)
    PacmanAgentCls_Old = load_pacman_agent_class(pacman_agent_path_old)
    PacmanAgentCls_Example = load_pacman_agent_class(pacman_agent_path_example)

    memory_map = np.full((21, 21), -1, dtype=np.int8)
    visit = np.zeros((21, 21), dtype=np.int16)
    prev_positions: Deque[Tuple[int, int]] = deque(maxlen=10)

    bt = BeliefTracker(decay=0.95, visible_decay=0.10, seed=0)
    last_seen_pac: Optional[Tuple[int,int]] = None

    last_action_idx = MOVE_TO_IDX[Move.STAY]
    last2_action_idx = MOVE_TO_IDX[Move.STAY]

    def make_env(pac_speed: int) -> EnvAdapter:
        return EnvAdapter(
            max_steps=cfg.max_steps,
            deterministic_starts=cfg.deterministic_starts,
            capture_distance_threshold=cfg.capture_distance_threshold,
            pacman_speed=pac_speed,
            pacman_obs_radius=cfg.pacman_obs_radius,
            ghost_obs_radius=cfg.ghost_obs_radius,
        )

    # bootstrap to get input_dim
    env0 = make_env(pac_speed=1)
    obs_p, pos_p, en_p, obs_g, pos_g, en_g = env0.reset()
    update_memory_from_obs(memory_map, obs_g)
    bt.update(obs_g, memory_map, en_g)
    last_seen_pac = en_g if en_g is not None else None
    pac_est0 = en_g if en_g is not None else (bt.best_guess() or last_seen_pac)
    feat0 = extract_ghost_features(
        obs_g, memory_map, visit, prev_positions, pos_g, pac_est0, enemy_visible=(en_g is not None),
        last_action_idx=last_action_idx, last2_action_idx=last2_action_idx,
        pacman_speed=1, capture_distance=1  # Use minimum values for bootstrap
    )
    input_dim = int(feat0.shape[0])

    q = MLPDQN(DQNConfig(input_dim=input_dim))
    q_tgt = MLPDQN(DQNConfig(input_dim=input_dim))
    q_tgt.load_state_dict(q.state_dict())

    opt = optim.Adam(q.parameters(), lr=cfg.lr)
    rb = ReplayBuffer(capacity=cfg.buffer_size)

    global_step = 0

    for ep in range(cfg.episodes):
        # Prioritize hard scenarios (70% pac_speed=2, cap_dist=2) for worst-case training
        pac_speed = 2 if (random.random() < 0.7) else 1  # 70% hard mode
        cap_dist = 2 if (random.random() < 0.7) else 1   # 70% hard mode
        
        # Store in cfg for reward shaping access AND env creation
        cfg.pacman_speed = pac_speed
        cfg.capture_distance = cap_dist
        cfg.capture_distance_threshold = cap_dist  # CRITICAL: sync threshold for reach1 calculation
        
        env = make_env(pac_speed=pac_speed)  # Now uses updated cfg.capture_distance_threshold

        # Choose Pacman opponent: 50% own (hardest), 25% old, 25% example
        r = random.random()
        if r < 0.50:
            PacmanAgentCls = PacmanAgentCls_Own
            opponent_name = "Own"
        elif r < 0.75:
            PacmanAgentCls = PacmanAgentCls_Old
            opponent_name = "Group06"
        else:
            PacmanAgentCls = PacmanAgentCls_Example
            opponent_name = "Example"
        
        pacman_bot = PacmanAgentCls(pacman_speed=pac_speed)

        obs_p, pos_p, en_p, obs_g, pos_g, en_g = env.reset()

        memory_map[:] = -1
        visit[:] = 0
        prev_positions.clear()
        bt.belief = None
        last_seen_pac = None
        
        # CORNER DWELL TRACKING (TRAINING)
        corner_dwell_counter = 0
        current_corner_cluster = None
        corner_cooldown_until = 0  

        last_action_idx = MOVE_TO_IDX[Move.STAY]
        last2_action_idx = MOVE_TO_IDX[Move.STAY]

        update_memory_from_obs(memory_map, obs_g)
        prev_positions.append(pos_g)
        visit[pos_g[0], pos_g[1]] += 1

        bt.update(obs_g, memory_map, en_g)
        if en_g is not None:
            last_seen_pac = en_g
        pac_est = en_g if en_g is not None else (bt.best_guess() or last_seen_pac)
        pac_est_prev = pac_est

        feat = extract_ghost_features(
            obs_g, memory_map, visit, prev_positions, pos_g, pac_est, enemy_visible=(en_g is not None),
            last_action_idx=last_action_idx, last2_action_idx=last2_action_idx,
            pacman_speed=pac_speed, capture_distance=cap_dist
        )

        ghost_pos_prev2: Optional[Tuple[int,int]] = None
        ghost_pos_prev: Tuple[int,int] = pos_g

        ep_return = 0.0

        for t in range(cfg.max_steps):
            global_step += 1
            eps = epsilon_by_step(global_step, cfg)

            cand = legal_actions(memory_map, pos_g)

            if random.random() < eps:
                a = random.choice(cand)
            else:
                with torch.no_grad():
                    qv = q(torch.from_numpy(feat).unsqueeze(0)).squeeze(0).cpu().numpy()
                # legal mask
                mask = np.full(NUM_ACTIONS, -1e9, dtype=np.float32)
                for i in cand:
                    mask[i] = 0.0
                a = int(np.argmax(qv + mask))

            # pacman step
            pacman_action = pacman_bot.step(obs_p, pos_p, en_p, t + 1)

            # step env
            obs_p2, pos_p2, en_p2, obs_g2, pos_g2, en_g2, done, info = env.step(pacman_action, a)

            # was_unknown_before: check new cell BEFORE updating memory
            was_unknown_before = (int(memory_map[pos_g2[0], pos_g2[1]]) == -1)

            # update memory + trackers
            update_memory_from_obs(memory_map, obs_g2)
            prev_positions.append(pos_g2)
            visit[pos_g2[0], pos_g2[1]] += 1

            # update belief
            bt.update(obs_g2, memory_map, en_g2)
            if en_g2 is not None:
                last_seen_pac = en_g2
            pac_est_new = en_g2 if en_g2 is not None else (bt.best_guess() or last_seen_pac)

            # update action memory
            last2_action_idx = int(last_action_idx)
            last_action_idx = int(a)
            
            # UPDATE CORNER DWELL COUNTER (TRAINING)
            corners_3x3 = [
                (0, 3, 0, 3),    # Top-left: [0,1,2] x [0,1,2] 
                (0, 3, 18, 21),  # Top-right: [0,1,2] x [18,19,20]
                (18, 21, 0, 3),  # Bottom-left: [18,19,20] x [0,1,2] 
                (18, 21, 18, 21) # Bottom-right: [18,19,20] x [18,19,20] 
            ]
            
            is_in_corner = False
            current_cluster = None
            for min_r, max_r, min_c, max_c in corners_3x3:
                if (min_r <= pos_g2[0] < max_r and 
                    min_c <= pos_g2[1] < max_c):
                    is_in_corner = True
                    current_cluster = (min_r, max_r, min_c, max_c)
                    break

            if is_in_corner:
                if current_cluster == current_corner_cluster:
                    corner_dwell_counter += 1
                elif t < corner_cooldown_until:
                    # RE-ENTER TRONG COOLDOWN = VẪN CÒN CAMP!
                    corner_dwell_counter = max(9, corner_dwell_counter)
                    current_corner_cluster = current_cluster
                else:
                    # Chuyển sang cụm khác SAU cooldown
                    corner_dwell_counter = 1
                    current_corner_cluster = current_cluster
            else:
                # Ra khỏi góc
                if corner_dwell_counter > 0 and current_corner_cluster is not None:
                    # Đặt cooldown
                    corner_cooldown_until = t + 10
                
                # GIẢM DẦN thay vì reset
                corner_dwell_counter = max(0, corner_dwell_counter - 2)
                
                if corner_dwell_counter == 0:
                    current_corner_cluster = None
            
            is_corner_camping = (corner_dwell_counter > 8)
            corner_dwell_info = (is_corner_camping, corner_dwell_counter, 
                                current_corner_cluster, corner_cooldown_until)

            feat2 = extract_ghost_features(
                obs_g2, memory_map, visit, prev_positions, pos_g2, pac_est_new, enemy_visible=(en_g2 is not None),
                last_action_idx=last_action_idx, last2_action_idx=last2_action_idx,
                pacman_speed=pac_speed, capture_distance=cap_dist
            )

            caught = bool(info.get("ghost_caught", False))
            win = info.get("ghost_win", None)

            r = shaped_reward(
                ghost_pos_prev, pos_g2,
                pac_est_prev, pac_est_new,
                memory_map, visit, prev_positions,
                caught=caught, done=done, win=win,
                was_unknown_before=was_unknown_before,
                prev2_pos=ghost_pos_prev2,
                cfg=cfg,
                step_in_episode=t,  # Pass current step for phase-based rewards
                corner_dwell_info=corner_dwell_info
            )

            rb.push(feat, a, r, feat2, done)
            ep_return += r

            # learn
            if len(rb) >= cfg.start_learn and (global_step % cfg.train_every == 0):
                s, a_b, r_b, s2, d_b = rb.sample(cfg.batch_size)
                s = torch.from_numpy(s)
                a_b = torch.from_numpy(a_b)
                r_b = torch.from_numpy(r_b)
                s2 = torch.from_numpy(s2)
                d_b = torch.from_numpy(d_b)

                q_sa = q(s).gather(1, a_b.view(-1, 1)).squeeze(1)
                with torch.no_grad():
                    a2 = torch.argmax(q(s2), dim=1)
                    q_next = q_tgt(s2).gather(1, a2.view(-1, 1)).squeeze(1)
                    target = r_b + cfg.gamma * (1.0 - d_b) * q_next

                loss = nn.functional.smooth_l1_loss(q_sa, target)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q.parameters(), cfg.grad_clip)
                opt.step()

                # DIAGNOSTIC: Log training metrics every 1000 steps
                if global_step % 1000 == 0:
                    with torch.no_grad():
                        q_mean = q_sa.mean().item()
                        q_std = q_sa.std().item()
                        td_error = (target - q_sa).abs().mean().item()
                        grad_norm = sum(p.grad.norm().item() ** 2 for p in q.parameters() if p.grad is not None) ** 0.5
                    print(f"  [Step {global_step}] Q_mean={q_mean:.2f} Q_std={q_std:.2f} TD_err={td_error:.3f} grad_norm={grad_norm:.3f}")

            if global_step % cfg.target_sync == 0:
                q_tgt.load_state_dict(q.state_dict())

            # advance
            obs_p, pos_p, en_p = obs_p2, pos_p2, en_p2
            obs_g, pos_g, en_g = obs_g2, pos_g2, en_g2

            feat = feat2
            pac_est_prev = pac_est_new

            ghost_pos_prev2 = ghost_pos_prev
            ghost_pos_prev = pos_g2

            if done:
                break

        if (ep + 1) % 50 == 0:
            print(
                f"[EP {ep+1}/{cfg.episodes}] return={ep_return:.3f} "
                f"buffer={len(rb)} eps={epsilon_by_step(global_step, cfg):.3f} "
                f"pac_speed={pac_speed} cap_dist={cap_dist} opponent={opponent_name}"
            )

    cfg.weight_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(q.state_dict(), str(cfg.weight_out))
    print("Saved weights to:", cfg.weight_out)


if __name__ == "__main__":
    main()
