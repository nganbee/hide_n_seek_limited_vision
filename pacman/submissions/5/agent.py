# submissions/group_05/agent.py
"""
Pacman and Ghost agents with belief tracking.
Pacman: BFS-based pursuit with speed optimization.
Ghost: Hybrid RL + heuristic evasion.
"""

import os
import sys
from pathlib import Path
from collections import deque
import random
import numpy as np

try:
    from agent_interface import PacmanAgent as BasePacmanAgent
    from agent_interface import GhostAgent as BaseGhostAgent
    from environment import Move
except Exception:
    src_path = Path(__file__).parent.parent.parent / "src"
    sys.path.insert(0, str(src_path))
    from agent_interface import PacmanAgent as BasePacmanAgent
    from agent_interface import GhostAgent as BaseGhostAgent
    from environment import Move

try:
    import torch
    import torch.nn as nn
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
except Exception as e:
    torch = None
    nn = None

    _model_dir = Path(__file__).parent
    _has_pth = (_model_dir / "pacman_dqn.pth").exists() or (_model_dir / "ghost_policy.pth").exists()
    if _has_pth:
        raise ImportError(
            "This submission includes PyTorch model files (.pth) but 'torch' is not installed. "
            "Install it first, then re-run. For example: pip install torch\n"
            f"Original import error: {e!r}"
        )
    else:
        print(
            "Warning: 'torch' is not installed; RL/DQN will be disabled for this submission. "
            "(pip install torch)",
            file=sys.stderr,
        )


ALL_MOVES = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
DQN_MOVES = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]
DIRS = [
    (Move.UP, (-1, 0)),
    (Move.DOWN, (1, 0)),
    (Move.LEFT, (0, -1)),
    (Move.RIGHT, (0, 1)),
]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

OPPOSITE = {
    Move.UP: Move.DOWN,
    Move.DOWN: Move.UP,
    Move.LEFT: Move.RIGHT,
    Move.RIGHT: Move.LEFT,
    Move.STAY: Move.STAY,
}


IDX_TO_MOVE = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]
MOVE_TO_IDX = {m: i for i, m in enumerate(IDX_TO_MOVE)}


def manhattan(a, b) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

class PacmanDQN(nn.Module if nn is not None else object):
    """Dueling DQN for Pacman - matches training architecture."""
    def __init__(self, input_dim: int, hidden: int = 256, num_actions: int = 9):
        if nn is not None:
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
        else:
            self.features = None
            self.value = None
            self.adv = None

    def forward(self, x):
        if self.features is not None:
            f = self.features(x)
            v = self.value(f)
            a = self.adv(f)
            return v + a - a.mean(dim=1, keepdim=True)
        return None


PACMAN_ACTIONS = [
    (Move.UP, 1), (Move.DOWN, 1), (Move.LEFT, 1), (Move.RIGHT, 1),
    (Move.UP, 2), (Move.DOWN, 2), (Move.LEFT, 2), (Move.RIGHT, 2),
    (Move.STAY, 1),
]
NUM_PACMAN_ACTIONS = len(PACMAN_ACTIONS)
PACMAN_MOVE_TO_1STEP = {Move.UP: 0, Move.DOWN: 1, Move.LEFT: 2, Move.RIGHT: 3}
PACMAN_MOVE_TO_2STEP = {Move.UP: 4, Move.DOWN: 5, Move.LEFT: 6, Move.RIGHT: 7}



class BeliefTracker:
    def __init__(self, decay: float = 0.95, visible_decay: float = 0.05, seed=None):
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

    def _propagate(self, possible_mask: np.ndarray):
        H, W = possible_mask.shape
        newb = np.zeros_like(self.belief, dtype=np.float32)

        rows, cols = np.where(self.belief > 1e-7)
        for r, c in zip(rows, cols):
            p = float(self.belief[r, c])
            if p <= 0 or not possible_mask[r, c]:
                continue

            opts = [(r, c)]  # stay allowed
            for _, (dr, dc) in DIRS:
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

    def best_guess(self, last_known=None):
        if self.belief is None:
            return None
        mx = float(self.belief.max())
        if mx <= 0:
            return None
        coords = np.argwhere(self.belief >= 0.90 * mx)
        if coords.size == 0:
            return None
        cands = [tuple(x) for x in coords]
        if last_known is not None:
            cands.sort(key=lambda p: manhattan(p, last_known))
            cands = cands[:8]
        return self.rng.choice(cands)

    def get_map(self):
        return None if self.belief is None else self.belief.copy()


class PacmanAgent(BasePacmanAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 2)))
        self.capture_threshold = int(kwargs.get("capture_distance", 2))
        self.capture_ring = max(0, self.capture_threshold - 1)

        self.rng = random.Random(kwargs.get("seed", None))

        self.memory_map = None
        self.visit = None

        self.bt = BeliefTracker(decay=0.95, visible_decay=0.05, seed=kwargs.get("seed", None))

        self.last_move = Move.STAY
        self.prev_positions = deque(maxlen=6)

        self.last_seen_enemy = None
        self.last_seen_step = None

        self.name = "Pacman Agent v5"

    def _ensure(self, obs: np.ndarray):
        if self.memory_map is None or self.memory_map.shape != obs.shape:
            self.memory_map = np.full(obs.shape, -1, dtype=np.int8)
            self.visit = np.zeros(obs.shape, dtype=np.int16)

    def _update_memory(self, obs: np.ndarray):
        self._ensure(obs)
        visible = (obs != -1)
        self.memory_map[visible] = obs[visible]

    def _in_bounds(self, r, c):
        H, W = self.memory_map.shape
        return 0 <= r < H and 0 <= c < W

    def _walkable_known(self, r, c) -> bool:
        return self._in_bounds(r, c) and int(self.memory_map[r, c]) == 0

    def _walkable_possible(self, r, c) -> bool:
        return self._in_bounds(r, c) and int(self.memory_map[r, c]) != 1

    def _frontiers(self):
        if self.memory_map is None:
            return []
        H, W = self.memory_map.shape
        fr = []
        zeros = np.where(self.memory_map == 0)
        for r, c in zip(zeros[0], zeros[1]):
            for _, (dr, dc) in DIRS:
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W and int(self.memory_map[rr, cc]) == -1:
                    fr.append((int(r), int(c)))
                    break
        return fr

    def _unknown_gain(self, pos, radius=2):
        r, c = pos
        H, W = self.memory_map.shape
        r0, r1 = max(0, r - radius), min(H, r + radius + 1)
        c0, c1 = max(0, c - radius), min(W, c + radius + 1)
        return int(np.sum(self.memory_map[r0:r1, c0:c1] == -1))

    def _allowed_steps(self, mv: Move, last_mv: Move):
        if mv == Move.STAY:
            return [1]
        if mv != last_mv:
            return [1]
        if self.pacman_speed >= 2:
            return [2, 1]
        return [1]

    def _apply_action(self, pos, mv: Move, steps: int, strict_known: bool):
        if mv == Move.STAY:
            return pos, 0
        dr, dc = mv.value
        r, c = pos
        moved = 0
        for _ in range(steps):
            rr, cc = r + dr, c + dc
            ok = self._walkable_known(rr, cc) if strict_known else self._walkable_possible(rr, cc)
            if not ok:
                break
            r, c = rr, cc
            moved += 1
        return (int(r), int(c)), moved

    def _capture_cells(self, ghost_pos):
        """Get cells within capture range of ghost position."""
        if ghost_pos is None:
            return []
        r, c = ghost_pos
        cells = [(r, c)]
        if self.capture_ring >= 1:
            for _, (dr, dc) in DIRS:
                cells.append((r + dr, c + dc))
        out = []
        for rr, cc in cells:
            if self._in_bounds(rr, cc):
                out.append((int(rr), int(cc)))
        return out

    def _estimate_turns_single(self, start, last_mv, goal, max_turns=18):
        """BFS to estimate minimum turns to reach goal."""
        if start == goal:
            return 0
        q = deque([(start, last_mv)])
        dist = {(start, last_mv): 0}
        while q:
            pos, lm = q.popleft()
            d = dist[(pos, lm)]
            if d >= max_turns:
                continue
            for mv in ALL_MOVES:
                for st in self._allowed_steps(mv, lm):
                    nxt, moved = self._apply_action(pos, mv, st, strict_known=True)
                    if moved <= 0:
                        continue
                    ns = (nxt, mv)
                    if ns in dist:
                        continue
                    nd = d + 1
                    if nxt == goal:
                        return nd
                    dist[ns] = nd
                    q.append(ns)
        return None

    def _best_first_action_to_targets(
        self,
        start,
        start_last_move,
        target_set: set,
        strict_known: bool,
        max_turns=14,
        allow_one_unknown_finish=False
    ):
        """Find best first action to reach any target in target_set."""
        if start in target_set:
            return (Move.STAY, 1)

        q = deque([(start, start_last_move)])
        dist = {(start, start_last_move): 0}
        first_action = {(start, start_last_move): None}

        best_act = None
        best_turns = None
        best_tie = -1e18

        prev_cell = self.prev_positions[-2] if len(self.prev_positions) >= 2 else None

        def tie_value(curr_pos, curr_last_mv, mv, st_requested, moved_pos, moved_steps):
            val = 0.0
            if prev_cell is not None and moved_pos == prev_cell:
                val -= 120.0
            if mv == OPPOSITE.get(curr_last_mv, Move.STAY):
                val -= 60.0
            val -= 2.0 * float(self.visit[moved_pos[0], moved_pos[1]])

            if moved_steps == 2:
                near_target = any(manhattan(moved_pos, t) <= 2 for t in list(target_set)[:6])
                val += 70.0 if not near_target else 10.0
            return val

        while q:
            pos, last_mv = q.popleft()
            turns = dist[(pos, last_mv)]
            if best_turns is not None and turns > best_turns:
                continue
            if turns >= max_turns:
                continue

            for mv in ALL_MOVES:
                for st in self._allowed_steps(mv, last_mv):
                    nxt_pos, moved = self._apply_action(pos, mv, st, strict_known=strict_known)
                    if moved <= 0:
                        continue

                    ns = (nxt_pos, mv)
                    nd = turns + 1

                    fa = first_action[(pos, last_mv)]
                    if fa is None:
                        fa = (mv, moved)

                    if ns not in dist:
                        dist[ns] = nd
                        first_action[ns] = fa
                        q.append(ns)
                    elif nd < dist[ns]:
                        dist[ns] = nd
                        first_action[ns] = fa

                    if nxt_pos in target_set:
                        tv = tie_value(pos, last_mv, mv, st, nxt_pos, moved)
                        if best_turns is None or nd < best_turns or (nd == best_turns and tv > best_tie):
                            best_turns = nd
                            best_tie = tv
                            best_act = fa

            if allow_one_unknown_finish and strict_known:
                for mv in ALL_MOVES:
                    nxt_pos, moved = self._apply_action(pos, mv, 1, strict_known=False)
                    if moved <= 0:
                        continue
                    ns = (nxt_pos, mv)
                    nd = turns + 1
                    fa = first_action[(pos, last_mv)]
                    if fa is None:
                        fa = (mv, moved)
                    if ns not in dist:
                        dist[ns] = nd
                        first_action[ns] = fa
                        q.append(ns)
                    elif nd < dist[ns]:
                        dist[ns] = nd
                        first_action[ns] = fa

        return best_act

    def _axis_escape_predictions(self, ghost_pos, my_pos, steps_ahead=(2, 3, 4)):
        """Predict Ghost's likely escape positions based on current trajectory."""
        if ghost_pos is None:
            return []
        gr, gc = ghost_pos
        pr, pc = my_pos
        dr = gr - pr
        dc = gc - pc

        preds = []
        if abs(dr) >= abs(dc):
            step_r = int(np.sign(dr)) if dr != 0 else 0
            step_c = 0
        else:
            step_r = 0
            step_c = int(np.sign(dc)) if dc != 0 else 0

        if step_r == 0 and step_c == 0:
            return preds

        for k in steps_ahead:
            rr, cc = gr + step_r * k, gc + step_c * k
            if self._in_bounds(rr, cc):
                preds.append((int(rr), int(cc)))
        return preds

    def step(self, map_state: np.ndarray, my_position: tuple, enemy_position: tuple, step_number: int):
        self._update_memory(map_state)
        self.visit[my_position[0], my_position[1]] += 1
        self.prev_positions.append(my_position)

        if enemy_position is not None:
            self.last_seen_enemy = enemy_position
            self.last_seen_step = step_number

        self.bt.update(map_state, self.memory_map, enemy_position)

        if enemy_position is not None and manhattan(my_position, enemy_position) <= self.capture_ring:
            return (Move.STAY, 1)

        ghost_est = enemy_position if enemy_position is not None else self.bt.best_guess(self.last_seen_enemy)

        if ghost_est is not None:
            raw_targets = self._capture_cells(ghost_est)
            target_cells = {t for t in raw_targets if self._walkable_possible(t[0], t[1])}

            dist_to_est = manhattan(my_position, ghost_est)
            
            if enemy_position is not None and dist_to_est <= 8:
                for p in self._axis_escape_predictions(ghost_est, my_position):
                    if self._walkable_possible(p[0], p[1]):
                        target_cells.add(p)

            allow_unknown = (dist_to_est <= 4)

            act = self._best_first_action_to_targets(
                my_position,
                self.last_move,
                target_cells,
                strict_known=True,
                max_turns=16,
                allow_one_unknown_finish=allow_unknown
            )
            if act:
                mv, steps = act
                if mv != self.last_move:
                    steps = 1
                else:
                    steps = min(steps, 2, self.pacman_speed)
                steps = min(steps, self.pacman_speed)

                _, moved = self._apply_action(my_position, mv, steps, strict_known=True)
                if moved > 0:
                    self.last_move = mv
                    return (mv, moved)

        frontiers = self._frontiers()
        if frontiers:
            self.rng.shuffle(frontiers)
            sample = frontiers[:30]

            best_f = None
            best_val = -1e18
            bm = self.bt.get_map()

            for f in sample:
                turns = self._estimate_turns_single(my_position, self.last_move, f, max_turns=18)
                if turns is None:
                    continue

                gain = self._unknown_gain(f, radius=2)
                val = 10.0 * gain - 4.0 * turns - 3.0 * float(self.visit[f[0], f[1]])

                if bm is not None:
                    val += 40.0 * float(bm[f[0], f[1]])

                if len(self.prev_positions) >= 2 and f == self.prev_positions[-2]:
                    val -= 25.0

                if val > best_val:
                    best_val = val
                    best_f = f

            if best_f is not None:
                act = self._best_first_action_to_targets(
                    my_position, self.last_move, {best_f},
                    strict_known=True, max_turns=20
                )
                if act:
                    mv, steps = act
                    if mv != self.last_move:
                        steps = 1
                    else:
                        steps = min(steps, 2, self.pacman_speed)
                    steps = min(steps, self.pacman_speed)

                    _, moved = self._apply_action(my_position, mv, steps, strict_known=True)
                    if moved > 0:
                        self.last_move = mv
                        return (mv, moved)

        prev_cell = self.prev_positions[-2] if len(self.prev_positions) >= 2 else None
        candidates = []
        for mv in ALL_MOVES:
            for st in self._allowed_steps(mv, self.last_move):
                nxt, moved = self._apply_action(my_position, mv, st, strict_known=True)
                if moved <= 0:
                    continue
                score = 0.0
                if prev_cell is not None and nxt == prev_cell:
                    score -= 120.0
                if mv == OPPOSITE.get(self.last_move, Move.STAY):
                    score -= 40.0
                score -= 2.0 * float(self.visit[nxt[0], nxt[1]])
                if ghost_est is not None:
                    score -= 1.0 * manhattan(nxt, ghost_est)
                if moved == 2 and ghost_est is not None and manhattan(my_position, ghost_est) <= 3:
                    score -= 30.0
                candidates.append((score, mv, moved))

        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            _, mv, moved = candidates[0]
            self.last_move = mv
            return (mv, moved)

        return (Move.STAY, 1)


def _los_flag(memory_map: np.ndarray, a: tuple, b: tuple, radius=5) -> float:
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
        if not (0 <= r < memory_map.shape[0] and 0 <= c < memory_map.shape[1]):
            return 0.0
        if int(memory_map[r, c]) == 1:
            return 0.0
    return 1.0


def _one_hot(idx: int, n: int) -> list:
    v = [0.0] * n
    if 0 <= idx < n:
        v[idx] = 1.0
    return v


def extract_ghost_features_for_infer(
    obs_ghost: np.ndarray,
    memory_map: np.ndarray,
    visit: np.ndarray,
    prev_positions: deque,
    my_pos: tuple,
    pac_est,
    enemy_visible: bool,
    last_action_idx: int,
    last2_action_idx: int,
    **kwargs  # Accept pacman_speed, capture_distance
) -> np.ndarray:
    """
    MUST match train_ghost_dqn.py extract_ghost_features().

    Vector:
      - center cell (obs)
      - 4 rays * 5 (obs)
      - 4 neighbor walkable (memory)
      - enemy_visible
      - dx, dy, dist (norm)
      - los_flag (from memory)
      - degree_norm (from memory)
      - visit_norm (cell)
      - backtrack flag (pos == prev2)
      - unknown flag (current cell unknown in memory)
      - last_action one-hot (5)
      - last2_action one-hot (5)
      - pacman_speed (normalized)
      - capture_distance (normalized)
      - reach1 (normalized)
    """
    r, c = my_pos
    H, W = obs_ghost.shape

    feats = [float(obs_ghost[r, c])]

    # rays
    for dr, dc in DIR4:
        rr, cc = r, c
        for _ in range(5):
            rr += dr
            cc += dc
            if 0 <= rr < H and 0 <= cc < W:
                feats.append(float(obs_ghost[rr, cc]))
            else:
                feats.append(1.0)

    # neighbor walkable flags
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
        feats.append(_los_flag(memory_map, my_pos, pac_est, radius=5))
        # degree
        deg = 0
        for dr, dc in DIR4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W and int(memory_map[rr, cc]) != 1:
                deg += 1
        feats.append(float(deg) / 4.0)

    feats.append(float(visit[r, c]) / 50.0)
    prev2 = prev_positions[-2] if len(prev_positions) >= 2 else None
    feats.append(1.0 if (prev2 is not None and prev2 == my_pos) else 0.0)
    feats.append(1.0 if int(memory_map[r, c]) == -1 else 0.0)

    feats.extend(_one_hot(int(last_action_idx), 5))
    feats.extend(_one_hot(int(last2_action_idx), 5))

    # pacman_speed (1 or 2) -> 0.5 or 1.0
    feats.append(float(kwargs.get('pacman_speed', 2)) / 2.0)
    # capture_distance (1 or 2) -> 0.5 or 1.0
    feats.append(float(kwargs.get('capture_distance', 2)) / 2.0)
    # reach1 (max Pacman reach in 1 turn) normalized by map size
    reach1 = kwargs.get('pacman_speed', 2) + kwargs.get('capture_distance', 2) - 1
    feats.append(float(reach1) / 21.0)

    return np.array(feats, dtype=np.float32)


if nn is not None:
    class MLPDQN(nn.Module):
        """Must match train_ghost_dqn.py architecture."""

        def __init__(self, input_dim: int, hidden1: int = 128, hidden2: int = 128, num_actions: int = 5):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden1),
                nn.ReLU(),
                nn.Linear(hidden1, hidden2),
                nn.ReLU(),
                nn.Linear(hidden2, num_actions),
            )

        def forward(self, x):
            return self.net(x)
else:
    class MLPDQN(object):
        """Placeholder when torch isn't installed.

        RL is disabled in this case (see GhostAgent.__init__), but we still
        define this class so importing this file never crashes.
        """

        def __init__(self, *args, **kwargs):
            raise RuntimeError("torch is not available; Ghost RL is disabled")


class GhostAgent(BaseGhostAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rng = random.Random(kwargs.get("seed", None))
        self.memory_map = None
        self.visit = None
        self.prev_positions = deque(maxlen=10)

        self.bt = BeliefTracker(decay=0.95, visible_decay=0.10, seed=kwargs.get("seed", None))
        self.last_seen_enemy = None
        self.last_seen_step = None

        self.name = "Ghost (Hybrid RL + Safe Evade Fallback v3 - Reach-aware)"

        # Track last actions 
        self.last_action_idx = MOVE_TO_IDX[Move.STAY]
        self.last2_action_idx = MOVE_TO_IDX[Move.STAY]
        self.last_move = Move.STAY

        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 2)))
        self.capture_distance = max(1, int(kwargs.get("capture_distance", 2)))
        self.reach1 = self.pacman_speed + self.capture_distance - 1

        self.visit_decay_counter = 0
        self.corner_dwell_counter = 0
        self.current_corner_cluster = None
        self.trap_dwell_counter = 0
        self.current_trap_bounds = None
        self.corner_cooldown_until = 0  

        self.rl_enabled = bool(kwargs.get("use_rl", True))
        self.policy = None
        self._rl_input_dim = None
        self.weight_path = str(Path(__file__).parent / "ghost_policy.pth")
        self._load_warning_shown = False
        if torch is None or nn is None:
            self.rl_enabled = False

    def _ensure(self, obs: np.ndarray):
        if self.memory_map is None or self.memory_map.shape != obs.shape:
            self.memory_map = np.full(obs.shape, -1, dtype=np.int8)
            self.visit = np.zeros(obs.shape, dtype=np.int16)

    def _update_memory(self, obs: np.ndarray):
        self._ensure(obs)
        visible = (obs != -1)
        self.memory_map[visible] = obs[visible]

    def _in_bounds(self, r, c):
        H, W = self.memory_map.shape
        return 0 <= r < H and 0 <= c < W

    def _walkable(self, r, c):
        return self._in_bounds(r, c) and int(self.memory_map[r, c]) != 1

    def _degree(self, pos):
        deg = 0
        r, c = pos
        for dr, dc in DIR4:
            rr, cc = r + dr, c + dc
            if self._walkable(rr, cc):
                deg += 1
        return deg

    def _has_los(self, a, b, radius=5):
        if b is None:
            return False
        if a[0] != b[0] and a[1] != b[1]:
            return False
        dist = abs(a[0] - b[0]) + abs(a[1] - b[1])
        if dist > radius:
            return False
        dr = int(np.sign(b[0] - a[0]))
        dc = int(np.sign(b[1] - a[1]))
        r, c = a
        for _ in range(dist):
            r += dr
            c += dc
            if not self._in_bounds(r, c):
                return False
            if int(self.memory_map[r, c]) == 1:
                return False
        return True

    def _count_frontier_neighbors(self, pos):
        """Count number of unknown (-1) cells adjacent to position."""
        r, c = pos
        count = 0
        for dr, dc in DIR4:
            rr, cc = r + dr, c + dc
            if self._in_bounds(rr, cc) and int(self.memory_map[rr, cc]) == -1:
                count += 1
        return count
    
    def _has_escape_route(self, pos, max_depth=5):
        """Check if position has escape route (degree>2) within max_depth steps using BFS.
        Returns (has_escape, escape_distance)"""
        from collections import deque
        if self._degree(pos) > 2:
            return True, 0  # Already at junction
        
        visited = {pos}
        queue = deque([(pos, 0)])
        
        while queue:
            curr, depth = queue.popleft()
            if depth >= max_depth:
                continue
            
            for dr, dc in DIR4:
                nxt = (curr[0] + dr, curr[1] + dc)
                if not self._walkable(nxt[0], nxt[1]) or nxt in visited:
                    continue
                
                visited.add(nxt)
                deg = self._degree(nxt)
                
                if deg > 2:  # Found junction = escape route
                    return True, depth + 1
                
                queue.append((nxt, depth + 1))
        
        return False, max_depth  # No escape found

    def _find_nearest_unexplored_direction(self, pos):
        """Find direction toward nearest unexplored (-1) cell using BFS."""
        from collections import deque
        H, W = self.memory_map.shape
        visited = set()
        visited.add(pos)
        queue = deque()
        
        # Start BFS from neighbors
        for mv in ALL_MOVES:
            dr, dc = mv.value
            nxt = (pos[0] + dr, pos[1] + dc)
            if self._walkable(nxt[0], nxt[1]):
                queue.append((nxt, mv, 1))  # (position, first_move, distance)
                visited.add(nxt)
        
        while queue:
            curr, first_mv, dist = queue.popleft()
            if dist > 15:  # Limit search depth
                break
            
            # Check if current position is adjacent to unexplored
            for dr, dc in DIR4:
                rr, cc = curr[0] + dr, curr[1] + dc
                if self._in_bounds(rr, cc) and int(self.memory_map[rr, cc]) == -1:
                    return first_mv, dist  # Found! Return first move direction
            
            # Expand neighbors
            for mv in ALL_MOVES:
                dr, dc = mv.value
                nxt = (curr[0] + dr, curr[1] + dc)
                if nxt not in visited and self._walkable(nxt[0], nxt[1]):
                    visited.add(nxt)
                    queue.append((nxt, first_mv, dist + 1))
        
        return None, float('inf')  # No unexplored found

    def _count_unexplored_in_direction(self, pos, mv, depth=5):
        """Count unexplored cells reachable in a direction."""
        count = 0
        dr, dc = mv.value
        r, c = pos
        for step in range(1, depth + 1):
            r, c = r + dr, c + dc
            if not self._in_bounds(r, c):
                break
            cell = int(self.memory_map[r, c])
            if cell == 1:  # Wall
                break
            if cell == -1:  # Unexplored
                count += (depth - step + 1)  # Closer = more valuable
            # Also check sides
            for dr2, dc2 in DIR4:
                rr, cc = r + dr2, c + dc2
                if self._in_bounds(rr, cc) and int(self.memory_map[rr, cc]) == -1:
                    count += 0.5
        return count

    def _legal_mask(self, pos):
        legal = [True, True, True, True, True]
        for mv in ALL_MOVES:
            dr, dc = mv.value
            nxt = (pos[0] + dr, pos[1] + dc)
            if not self._walkable(nxt[0], nxt[1]):
                legal[MOVE_TO_IDX[mv]] = False
        legal[MOVE_TO_IDX[Move.STAY]] = True
        return legal

    def _maybe_load_policy(self, feat_dim: int):
        if not self.rl_enabled or self.policy is not None:
            return
        if not os.path.exists(self.weight_path):
            if not self._load_warning_shown:
                print(f"Warning: RL weights not found at {self.weight_path}, using heuristic fallback")
                self._load_warning_shown = True
            return
        try:
            self._rl_input_dim = int(feat_dim)
            model = MLPDQN(self._rl_input_dim, hidden1=128, hidden2=128, num_actions=5)
            state = torch.load(self.weight_path, map_location="cpu")
            model.load_state_dict(state)
            model.eval()
            self.policy = model
        except Exception as e:
            if not self._load_warning_shown:
                print(f"Warning: Failed to load RL weights: {e}, using heuristic fallback")
                self._load_warning_shown = True
            self.policy = None

    def _policy_q(self, feat: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.from_numpy(feat.astype(np.float32)).unsqueeze(0)
            q = self.policy(x).squeeze(0).cpu().numpy()
        return q

    def _update_action_memory(self, chosen_idx: int):
        self.last2_action_idx = int(self.last_action_idx)
        self.last_action_idx = int(chosen_idx)

    def step(self, map_state: np.ndarray, my_position: tuple, enemy_position: tuple, step_number: int) -> Move:
        self._update_memory(map_state)
        
        self.visit = (self.visit * 0.995).astype(np.int16)
        self.visit[my_position[0], my_position[1]] += 1
        
        if self.visit[my_position[0], my_position[1]] > 50:
            self.visit[my_position[0], my_position[1]] = 50
        
        self.prev_positions.append(my_position)
        
        corners_3x3 = [
            (0, 3, 0, 3),
            (0, 3, 18, 21),
            (18, 21, 0, 3),
            (18, 21, 18, 21)
        ]
        
        is_in_corner = False
        current_cluster = None
        for min_r, max_r, min_c, max_c in corners_3x3:
            if min_r <= my_position[0] < max_r and min_c <= my_position[1] < max_c:
                is_in_corner = True
                current_cluster = (min_r, max_r, min_c, max_c)
                break
        
        if is_in_corner:
            if current_cluster == self.current_corner_cluster:
                self.corner_dwell_counter += 1
            elif step_number < self.corner_cooldown_until:
                self.corner_dwell_counter = max(9, self.corner_dwell_counter)
                self.current_corner_cluster = current_cluster
            else:
                self.corner_dwell_counter = 1
                self.current_corner_cluster = current_cluster
        else:
            if self.corner_dwell_counter > 0 and self.current_corner_cluster is not None:
                self.corner_cooldown_until = step_number + 10
            
            self.corner_dwell_counter = max(0, self.corner_dwell_counter - 2)
            
            if self.corner_dwell_counter == 0:
                self.current_corner_cluster = None
        
        is_corner_camping = (self.corner_dwell_counter > 8)
        
        is_pocket_trapped = False
        pocket_bounds = None
        if len(self.prev_positions) >= 10:
            recent_10 = list(self.prev_positions)[-10:]
            rs = [p[0] for p in recent_10]
            cs = [p[1] for p in recent_10]
            r_range = max(rs) - min(rs)
            c_range = max(cs) - min(cs)
            
            if r_range <= 2 and c_range <= 2:
                is_pocket_trapped = True
                pocket_bounds = (min(rs), max(rs)+1, min(cs), max(cs)+1)
        
        is_trapped = is_corner_camping or is_pocket_trapped
        trap_bounds = self.current_corner_cluster if is_corner_camping else pocket_bounds
        
        self.visit_decay_counter += 1
        
        decay_threshold = 20 if is_trapped else 100
        decay_rate = 0.70 if is_trapped else 0.90
        
        if self.visit_decay_counter >= decay_threshold:
            self.visit = (self.visit * decay_rate).astype(np.int16)
            self.visit_decay_counter = 0
        
        if is_trapped:
            if trap_bounds == self.current_trap_bounds:
                self.trap_dwell_counter += 1
            else:
                self.trap_dwell_counter = 1
                self.current_trap_bounds = trap_bounds
        else:
            self.trap_dwell_counter = max(0, self.trap_dwell_counter - 2)
            if self.trap_dwell_counter == 0:
                self.current_trap_bounds = None

        if enemy_position is not None:
            self.last_seen_enemy = enemy_position
            self.last_seen_step = step_number

        self.bt.update(map_state, self.memory_map, enemy_position)
        pac_est = enemy_position if enemy_position is not None else self.bt.best_guess(self.last_seen_enemy)
        enemy_visible = (enemy_position is not None)

        is_looping = False
        loop_cells = set()
        if len(self.prev_positions) >= 8:
            recent_8 = list(self.prev_positions)[-8:]
            unique_cells = len(set(recent_8))
            if unique_cells <= 4:
                is_looping = True
                loop_cells = set(recent_8)
        
        if not is_looping and len(self.prev_positions) >= 6:
            recent_6 = list(self.prev_positions)[-6:]
            if len(set(recent_6)) <= 3:
                is_looping = True
                loop_cells = set(recent_6)

        if self.rl_enabled and torch is not None and nn is not None:
            feat = extract_ghost_features_for_infer(
                map_state, self.memory_map, self.visit, self.prev_positions,
                my_position, pac_est, enemy_visible,
                self.last_action_idx, self.last2_action_idx,
                pacman_speed=self.pacman_speed,
                capture_distance=self.capture_distance
            )
            self._maybe_load_policy(feat.shape[0])

            if self.policy is not None:
                if enemy_visible and pac_est is not None:
                    d0 = manhattan(my_position, pac_est)
                    if d0 <= self.reach1 + 1:  
                        legal = self._legal_mask(my_position)
                        best_escape = None
                        best_escape_score = -1e18
                        
                        for mv in ALL_MOVES + [Move.STAY]:
                            idx = MOVE_TO_IDX[mv]
                            if not legal[idx]:
                                continue
                            
                            if mv == Move.STAY:
                                nxt = my_position
                            else:
                                dr, dc = mv.value
                                nxt = (my_position[0] + dr, my_position[1] + dc)
                            
                            escape_score = 0.0
                            nxt_dist = manhattan(nxt, pac_est)
                            
                            if nxt_dist > d0:
                                escape_score += 200.0 * (nxt_dist - d0)
                            
                            if not self._has_los(nxt, pac_est, radius=5):
                                escape_score += 100.0
                            
                            if escape_score > best_escape_score:
                                best_escape_score = escape_score
                                best_escape = mv
                        
                        if best_escape is not None and best_escape_score > 0:
                            best_idx = MOVE_TO_IDX[best_escape]
                            self._update_action_memory(best_idx)
                            self.last_move = best_escape
                            return best_escape

                q = self._policy_q(feat)

                legal = self._legal_mask(my_position)
                for i in range(5):
                    if not legal[i]:
                        q[i] -= 1e9

                is_potentially_stuck = (len(set(list(self.prev_positions)[-6:])) <= 3 if len(self.prev_positions) >= 6 else False)
                
                is_corner_trapped = False
                if len(self.prev_positions) >= 10 and step_number > 40:
                    recent_10 = list(self.prev_positions)[-10:]
                    rs = [p[0] for p in recent_10]
                    cs = [p[1] for p in recent_10]
                    if max(rs) - min(rs) <= 3 and max(cs) - min(cs) <= 3:
                        is_corner_trapped = True
                
                epsilon_inference = 0.0
                
                if step_number <= 20:
                    epsilon_inference = 0.30
                elif step_number <= 40:
                    epsilon_inference = 0.20
                elif is_trapped:
                    epsilon_inference = 0.50
                elif is_looping:
                    epsilon_inference = 0.40
                elif pac_est is None:
                    epsilon_inference = 0.15
                elif is_corner_trapped:
                    epsilon_inference = 0.30
                elif is_potentially_stuck:
                    epsilon_inference = 0.25
                elif pac_est is not None:
                    d0_for_epsilon = manhattan(my_position, pac_est)
                    if d0_for_epsilon > self.reach1 + 3:
                        epsilon_inference = 0.05
                
                if epsilon_inference > 0 and random.random() < epsilon_inference:
                    legal_moves = [i for i in range(5) if legal[i]]
                    if legal_moves:
                        best_frontier_idx = None
                        best_frontier_count = -1
                        for idx in legal_moves:
                            mv = IDX_TO_MOVE[idx]
                            if mv == Move.STAY:
                                nxt = my_position
                            else:
                                dr, dc = mv.value
                                nxt = (my_position[0] + dr, my_position[1] + dc)
                            frontier = self._count_frontier_neighbors(nxt)
                            if frontier > best_frontier_count:
                                best_frontier_count = frontier
                                best_frontier_idx = idx
                        
                        chosen_idx = best_frontier_idx if best_frontier_idx is not None else random.choice(legal_moves)
                        best_mv = IDX_TO_MOVE[chosen_idx]
                        self._update_action_memory(chosen_idx)
                        self.last_move = best_mv
                        return best_mv

                if pac_est is not None:
                    d0 = manhattan(my_position, pac_est)

                    if d0 <= self.capture_distance:
                        has_safe_move = False
                        for mv in ALL_MOVES:
                            idx = MOVE_TO_IDX[mv]
                            if not legal[idx]:
                                continue
                            dr, dc = mv.value
                            nxt = (my_position[0] + dr, my_position[1] + dc)
                            if manhattan(nxt, pac_est) >= d0:
                                has_safe_move = True
                                break
                        
                        if has_safe_move:
                            for mv in ALL_MOVES:
                                idx = MOVE_TO_IDX[mv]
                                if not legal[idx]:
                                    continue
                                dr, dc = mv.value
                                nxt = (my_position[0] + dr, my_position[1] + dc)
                                if manhattan(nxt, pac_est) < d0:
                                    q[idx] -= 1e9

                    if d0 <= self.reach1 + 4:
                        if d0 <= self.reach1:
                            for mv in ALL_MOVES:
                                idx = MOVE_TO_IDX[mv]
                                if not legal[idx]:
                                    continue
                                dr, dc = mv.value
                                nxt = (my_position[0] + dr, my_position[1] + dc)
                                if manhattan(nxt, pac_est) < d0:
                                    q[idx] -= 120.0

                        for mv in ALL_MOVES:
                            idx = MOVE_TO_IDX[mv]
                            if not legal[idx]:
                                continue
                            dr, dc = mv.value
                            nxt = (my_position[0] + dr, my_position[1] + dc)
                            
                            if self._has_los(nxt, pac_est, radius=5):
                                if d0 <= self.reach1 + 1:
                                    q[idx] -= 180.0
                                elif d0 <= self.reach1 + 2:
                                    q[idx] -= 130.0
                                elif d0 <= self.reach1 + 3:
                                    q[idx] -= 110.0
                                elif d0 <= self.reach1 + 6:
                                    q[idx] -= 50.0
                                elif d0 <= 10:
                                    q[idx] -= 20.0
                                else:
                                    q[idx] -= 5.0
                            
                            if self._degree(nxt) <= 1:
                                if d0 <= self.reach1 + 2:
                                    q[idx] -= 150.0
                                elif d0 <= 6:
                                    q[idx] -= 65.0
                                else:
                                    q[idx] -= 20.0
                
                if step_number <= 40:
                    for mv in ALL_MOVES:
                        idx = MOVE_TO_IDX[mv]
                        if not legal[idx]:
                            continue
                        dr, dc = mv.value
                        nxt = (my_position[0] + dr, my_position[1] + dc)
                        
                        deg = self._degree(nxt)
                        if deg <= 1:
                            q[idx] -= 200.0
                        elif deg == 2:

                            has_escape, escape_dist = self._has_escape_route(nxt, max_depth=5)
                            if not has_escape:
                                q[idx] -= 180.0
                            elif escape_dist >= 4:
                                q[idx] -= 100.0
                            else:
                                q[idx] -= 50.0
                    else:
                        for mv in ALL_MOVES:
                            idx = MOVE_TO_IDX[mv]
                            if not legal[idx]:
                                continue
                            dr, dc = mv.value
                            nxt = (my_position[0] + dr, my_position[1] + dc)
                            
                            if self._has_los(nxt, pac_est, radius=5):
                                q[idx] -= 15.0
                            
                            if int(self.memory_map[nxt[0], nxt[1]]) == -1:
                                q[idx] += 15.0
                            
                            frontier = self._count_frontier_neighbors(nxt)
                            q[idx] += 10.0 * frontier
                else:
                    for mv in ALL_MOVES:
                        idx = MOVE_TO_IDX[mv]
                        if not legal[idx]:
                            continue
                        dr, dc = mv.value
                        nxt = (my_position[0] + dr, my_position[1] + dc)
                        
                        frontier_count = self._count_frontier_neighbors(nxt)
                        q[idx] += 25.0 * frontier_count
                        
                        if int(self.memory_map[nxt[0], nxt[1]]) == -1:
                            q[idx] += 20.0
                
                if step_number <= 40:
                    for mv in ALL_MOVES:
                        idx = MOVE_TO_IDX[mv]
                        if not legal[idx]:
                            continue
                        dr, dc = mv.value
                        nxt = (my_position[0] + dr, my_position[1] + dc)
                        r, c = nxt
                        
                        if step_number <= 30:
                            if 8 <= r <= 12:
                                q[idx] -= 150.0
                            
                            if 9 <= c <= 11:
                                q[idx] -= 120.0
                            
                            dist_to_spawn = abs(r - 15) + abs(c - 10)
                            if dist_to_spawn <= 6:
                                q[idx] -= 80.0
                            elif dist_to_spawn <= 8:
                                q[idx] -= 40.0
                            
                            row_dist = abs(r - 15)
                            if row_dist == 0:
                                q[idx] -= 200.0
                            elif row_dist == 1:
                                q[idx] -= 120.0
                            elif row_dist == 2:
                                q[idx] -= 60.0
                            if row_dist >= 5:
                                q[idx] += 30.0 * (row_dist - 4)
                        
                        corners = [(1, 1), (1, 19), (19, 1), (19, 19)]
                        min_corner_dist = min(abs(r - cr) + abs(c - cc) for cr, cc in corners)
                        corner_bonus = max(0, 200.0 - 8.0 * min_corner_dist)
                        q[idx] += corner_bonus
                        
                        if r < 3 or r > 17 or c < 3 or c > 17:
                            q[idx] += 35.0
                
                if is_trapped and trap_bounds is not None:
                    min_r, max_r, min_c, max_c = trap_bounds
                    for mv in ALL_MOVES:
                        idx = MOVE_TO_IDX[mv]
                        if not legal[idx]:
                            continue
                        dr, dc = mv.value
                        nxt = (my_position[0] + dr, my_position[1] + dc)
                        
                        stays_in_cluster = (min_r <= nxt[0] < max_r and min_c <= nxt[1] < max_c)
                        
                        if stays_in_cluster:
                            dwell_penalty = 300.0 * (self.trap_dwell_counter - 8)
                            q[idx] -= dwell_penalty
                        else:
                            q[idx] += 400.0
                            
                            if pac_est is not None and self._has_los(nxt, pac_est, radius=5):
                                d0 = manhattan(my_position, pac_est)
                                if d0 <= 3:
                                    q[idx] += 90.0
                                else:
                                    q[idx] += 60.0
                            
                            if self._degree(nxt) <= 1:
                                if pac_est and manhattan(nxt, pac_est) <= self.reach1 + 2:
                                    q[idx] += 150.0
                                else:
                                    q[idx] += 50.0

                prev2 = self.prev_positions[-2] if len(self.prev_positions) >= 2 else None
                if prev2 is not None:
                    for mv in ALL_MOVES:
                        idx = MOVE_TO_IDX[mv]
                        if not legal[idx]:
                            continue
                        dr, dc = mv.value
                        nxt = (my_position[0] + dr, my_position[1] + dc)
                        if nxt == prev2:
                            q[idx] -= 120.0

                if self.last_move in OPPOSITE:
                    rev = OPPOSITE[self.last_move]
                    if rev in MOVE_TO_IDX:
                        ridx = MOVE_TO_IDX[rev]
                        if legal[ridx]:
                            q[ridx] -= 45.0

                if step_number <= 40:
                    stay_idx = MOVE_TO_IDX[Move.STAY]
                    reach1_est = 3
                    
                    if pac_est is None:
                        q[stay_idx] -= 1e9
                    else:
                        d0 = manhattan(my_position, pac_est)
                        if d0 > reach1_est + 1:
                            q[stay_idx] -= 1e9
                        else:
                            has_escape = False
                            for mv in ALL_MOVES:
                                idx = MOVE_TO_IDX[mv]
                                if not legal[idx]:
                                    continue
                                dr, dc = mv.value
                                nxt = (my_position[0] + dr, my_position[1] + dc)
                                if manhattan(nxt, pac_est) > d0:
                                    has_escape = True
                                    break
                            if has_escape:
                                q[stay_idx] -= 1e9
                
                if is_trapped and self.trap_dwell_counter > 12:
                    if trap_bounds is not None:
                        min_r, max_r, min_c, max_c = trap_bounds
                        trap_center = ((min_r + max_r) / 2, (min_c + max_c) / 2)
                        
                        for mv in ALL_MOVES:
                            idx = MOVE_TO_IDX[mv]
                            if not legal[idx]:
                                continue
                            dr, dc = mv.value
                            nxt = (my_position[0] + dr, my_position[1] + dc)
                            
                            stays_in_cluster = (min_r <= nxt[0] < max_r and 
                                              min_c <= nxt[1] < max_c)
                            
                            if stays_in_cluster:
                                q[idx] = -1e12
                            else:
                                escape_dist = max(abs(nxt[0] - trap_center[0]), 
                                                abs(nxt[1] - trap_center[1]))
                                q[idx] += 150.0 * escape_dist
                        
                        q[MOVE_TO_IDX[Move.STAY]] = -1e12

                best_idx = int(np.argmax(q))
                best_mv = IDX_TO_MOVE[best_idx]

                if best_mv != Move.STAY:
                    dr, dc = best_mv.value
                    nxt = (my_position[0] + dr, my_position[1] + dc)
                    if not self._walkable(nxt[0], nxt[1]):
                        best_mv = Move.STAY
                        best_idx = MOVE_TO_IDX[Move.STAY]

                self._update_action_memory(best_idx)
                self.last_move = best_mv
                return best_mv

        moves = ALL_MOVES[:]
        
        if pac_est is None:
            move_frontier = []
            for mv in moves:
                dr, dc = mv.value
                nxt = (my_position[0] + dr, my_position[1] + dc)
                if self._walkable(nxt[0], nxt[1]):
                    frontier_count = self._count_frontier_neighbors(nxt)
                    move_frontier.append((frontier_count, self.rng.random(), mv))
            move_frontier.sort(reverse=True)
            moves = [mv for _, _, mv in move_frontier]
        else:
            self.rng.shuffle(moves)
        
        best_mv = Move.STAY
        best_score = -1e18
        bm = self.bt.get_map()
        prev2 = self.prev_positions[-2] if len(self.prev_positions) >= 2 else None
        
        no_pacman_multiplier = 5.0 if pac_est is None else 1.0
        
        is_stuck = False
        if len(self.prev_positions) >= 4:
            recent_set = set(list(self.prev_positions)[-4:])
            if len(recent_set) <= 2:
                is_stuck = True
        
        is_corner_trapped = False
        corner_cluster_bounds = None
        if len(self.prev_positions) >= 10 and step_number > 40:
            recent_10 = list(self.prev_positions)[-10:]
            rs = [p[0] for p in recent_10]
            cs = [p[1] for p in recent_10]
            if max(rs) - min(rs) <= 3 and max(cs) - min(cs) <= 3:
                is_corner_trapped = True
                corner_cluster_bounds = (min(rs), max(rs), min(cs), max(cs))

        my_frontier = self._count_frontier_neighbors(my_position)
        my_degree = self._degree(my_position)
        
        target_mv_for_unexplored, dist_to_unexplored = None, float('inf')
        if pac_est is None and (is_looping or my_frontier == 0):
            target_mv_for_unexplored, dist_to_unexplored = self._find_nearest_unexplored_direction(my_position)
        
        is_good_observation_point = (my_degree >= 3 and my_frontier >= 2)
        
        pacman_just_appeared = (enemy_visible and pac_est is not None and 
                                manhattan(my_position, pac_est) <= 3)

        for mv in moves:
            dr, dc = mv.value
            nxt = (my_position[0] + dr, my_position[1] + dc)
            if not self._walkable(nxt[0], nxt[1]):
                continue

            score = 0.0

            if prev2 is not None and nxt == prev2:
                score -= 200.0 * no_pacman_multiplier

            if mv == OPPOSITE.get(self.last_move, Move.STAY):
                score -= 70.0 * no_pacman_multiplier

            if pac_est is not None:
                d = manhattan(nxt, pac_est)
                score += 3.0 * d
                
                if d <= self.reach1:
                    score -= 250.0
                elif d <= self.reach1 + 1:
                    score -= 160.0
                elif d <= self.reach1 + 2:
                    score -= 90.0
                    
                if self._has_los(nxt, pac_est, radius=8):
                    if d <= self.reach1 + 1:
                        score -= 180.0
                    elif d <= self.reach1 + 2:
                        score -= 140.0
                    elif d <= self.reach1 + 3:
                        score -= 120.0
                    elif d <= self.reach1 + 6:
                        score -= 80.0
                    elif d <= 10:
                        score -= 60.0
                    else:
                        score -= 30.0
            else:
                if step_number <= 30 and self.last_seen_enemy is None:
                    if 8 <= nxt[0] <= 12:
                        score -= 150.0
                    dist_to_spawn = manhattan(nxt, (15, 10))
                    if dist_to_spawn <= 6:
                        score -= 80.0
                    elif dist_to_spawn <= 8:
                        score -= 40.0
                    
                    row_dist = abs(nxt[0] - 15)
                    if row_dist == 0:
                        score -= 200.0
                    elif row_dist == 1:
                        score -= 120.0
                    elif row_dist == 2:
                        score -= 60.0
                    if row_dist >= 5:
                        score += 30.0 * (row_dist - 4)
                
                frontier_count = self._count_frontier_neighbors(nxt)
                score += 40.0 * frontier_count
                
                exploration_waypoints = [
                    (3, 3),
                    (3, 17),
                    (17, 17),
                    (17, 3),
                ]
                
                min_waypoint_dist = float('inf')
                for wp in exploration_waypoints:
                    if int(self.memory_map[wp[0], wp[1]]) == -1:
                        wp_dist = manhattan(nxt, wp)
                        min_waypoint_dist = min(min_waypoint_dist, wp_dist)
                
                if min_waypoint_dist < float('inf'):
                    score += max(0, 100.0 - 5.0 * min_waypoint_dist)
                
                # Bonus for moving toward unexplored areas (only when safe)
                if len(self.prev_positions) >= 1:
                    start_pos = self.prev_positions[0]
                    dist_from_start = manhattan(nxt, start_pos)
                    score += 8.0 * dist_from_start
                
                # Extra bonus if next cell has higher frontier potential than current
                if frontier_count > my_frontier:
                    score += 30.0  # Reward improving frontier access
                
                # EARLY-GAME CORNER BIAS & AXIS AVOIDANCE
                # Disabled afterwards to avoid camping
                if step_number <= 40 and self.last_seen_enemy is None:
                    r, c = nxt
                    
                    # 1. CORNER BONUS: Pull ghost toward 4 corners (ONLY early game)
                    # Four corners of 21x21 map: (1,1), (1,19), (19,1), (19,19)
                    corners = [(1, 1), (1, 19), (19, 1), (19, 19)]
                    min_corner_dist = min(manhattan(nxt, corner) for corner in corners)
                    corner_bonus = max(0, 200.0 - 8.0 * min_corner_dist)
                    score += corner_bonus
                    
                    # 2. MAIN AXIS PENALTY: Avoid center rows/columns (highways)
                    # Center horizontal band (rows 8-12) - already penalized above but reinforce
                    # Center vertical band (cols 9-11)
                    if 9 <= c <= 11:  # Center columns
                        if step_number <= 30:  # Only early game
                            score -= 120.0  # Strong penalty for vertical axis
                        # After step 30, only penalize if Pacman nearby
                        elif pac_est is not None and manhattan(nxt, pac_est) <= self.reach1 + 2:
                            score -= 60.0  # Softer penalty when Pacman close
                    
                    # 3. EDGE PREFERENCE: Small bonus for staying near map edges (ONLY early game)
                    # Edges: r<3, r>17, c<3, c>17
                    if r < 3 or r > 17 or c < 3 or c > 17:
                        score += 35.0  # Edge bonus to pull toward perimeter
                
                #  CORNER DWELL PENALTY (ALL GAME)
                # If camping corner too long (>8 steps), heavy penalty + escape bonus
                # PHASE 3: STRENGTHEN penalties to force corner escape
                # Apply to both corner and pocket trapping
                if is_trapped and trap_bounds is not None:
                    min_r, max_r, min_c, max_c = trap_bounds
                    stays_in_cluster = (min_r <= nxt[0] < max_r and min_c <= nxt[1] < max_c)
                    
                    if stays_in_cluster:
                        # PENALTY increases with camping time (INCREASED 150 → 300)
                        dwell_penalty = 300.0 * (self.trap_dwell_counter - 8)
                        score -= dwell_penalty
                    else:
                        # HUGE ESCAPE BONUS (INCREASED 200 → 400)
                        score += 400.0
                        
                        # **NEW: EXEMPT other penalties for escape move**
                        if pac_est is not None and self._has_los(nxt, pac_est, radius=5):
                            score += 80.0  # Compensate LOS penalty
                        
                        deg = self._degree(nxt)
                        if deg <= 1:
                            score += 40.0  # Compensate dead-end penalty
                
                # MID-GAME EXPLORATION BOOST (step 40-120)
                # ANTI-CORNER-TRAP: Push ghost out of corners after early game
                if 40 < step_number <= 120 and pac_est is None:
                    # Reward moving away from starting corner
                    if len(self.prev_positions) >= 1:
                        start_area = self.prev_positions[0]
                        dist_from_start = manhattan(nxt, start_area)
                        score += 5.0 * dist_from_start  # Encourage wandering
                    
                    # Extra frontier bonus (stack with existing)
                    frontier_count = self._count_frontier_neighbors(nxt)
                    score += 20.0 * frontier_count
            
            # Disable frontier bonus when Pacman seen and close (focus on escape)
            if pac_est is not None:
                d_to_pac = manhattan(nxt, pac_est)
                if d_to_pac <= self.reach1 + 2:
                    # Within critical range: no exploration, just escape
                    frontier_count_bonus = self._count_frontier_neighbors(nxt)
                    score -= 40.0 * frontier_count_bonus  # Reverse the bonus added earlier
                
                # Prefer directions with more unexplored area ahead (deep lookahead)
                unexplored_in_dir = self._count_unexplored_in_direction(my_position, mv, depth=6)
                score += 12.0 * unexplored_in_dir
                
                # LOOP BREAKING: Strong attraction to unexplored
                if is_looping or my_frontier == 0:
                    # Heavy bonus for move toward unexplored
                    if target_mv_for_unexplored is not None and mv == target_mv_for_unexplored:
                        score += 150.0  # Very strong pull toward unexplored
                    
                    # Extra penalty for staying in the "loop zone"
                    if len(self.prev_positions) >= 8:
                        loop_zone = set(list(self.prev_positions)[-8:])
                        if nxt in loop_zone:
                            score -= 80.0  # Penalize cells in the loop
            
            # Stuck detection: heavily penalize returning to oscillation cells
            if is_stuck:
                recent_set = set(list(self.prev_positions)[-4:])
                if nxt in recent_set:
                    score -= 500.0  # Increased from 400 → 500 to break stronger
            
            # Loop detection penalty
            if is_looping and len(self.prev_positions) >= 8:
                loop_zone = set(list(self.prev_positions)[-8:])
                if nxt in loop_zone:
                    score -= 120.0  # Heavy penalty for cells in loop
            
            # ANTI-CORNER-TRAP: Huge bonus for exiting corner cluster
            if is_corner_trapped and corner_cluster_bounds is not None:
                min_r, max_r, min_c, max_c = corner_cluster_bounds
                # If next move exits the 3x3 cluster
                if not (min_r <= nxt[0] <= max_r and min_c <= nxt[1] <= max_c):
                    score += 200.0  # HUGE bonus for escaping corner trap

            deg = self._degree(nxt)
            if deg <= 1:
                score -= 40.0
            elif deg >= 3:
                score += 15.0  # Increased bonus for open areas

            if nxt in self.prev_positions:
                score -= 18.0 * no_pacman_multiplier
            
            # VISIT PENALTY (with decay)
            visit_count = float(self.visit[nxt[0], nxt[1]])
            
            # AGGRESSIVE penalty multiplier when looping or trapped
            loop_multiplier = 3.0 if (is_looping or is_trapped) else 1.0
            early_game_multiplier = 0.5 if step_number <= 40 else 1.0
            
            if pac_est is None:
                # Reduce penalty when no Pacman (already has decay)
                score -= 2.0 * visit_count * early_game_multiplier * loop_multiplier
            else:
                score -= 1.2 * visit_count * early_game_multiplier * loop_multiplier
            
            # Direct loop penalty - heavily penalize revisiting loop cells
            if is_looping and nxt in loop_cells:
                score -= 150.0 * no_pacman_multiplier  # HEAVY loop penalty

            if bm is not None:
                score -= 60.0 * float(bm[nxt[0], nxt[1]])

            if score > best_score:
                best_score = score
                best_mv = mv
        
        # STAY LOGIC: Only use in specific situations 
        stay_score = -1e18  # Default: NEVER STAY
        
        # EARLY GAME (step <= 40): SUPPRESS STAY completely
        if step_number <= 40:
            if pac_est is None:
                # No Pacman → NEVER STAY (must explore)
                stay_score = -1e12
            else:
                d0 = manhattan(my_position, pac_est)
                reach1_est = 3  # Assume pacman_speed=2, capture_distance=2
                
                if d0 <= reach1_est + 1:
                    # Pacman VERY CLOSE → check if all moves are worse
                    all_moves_bad = True
                    for mv in ALL_MOVES:
                        dr, dc = mv.value
                        nxt = (my_position[0] + dr, my_position[1] + dc)
                        if self._walkable(nxt[0], nxt[1]):
                            if manhattan(nxt, pac_est) >= d0:
                                all_moves_bad = False
                                break
                    if all_moves_bad:
                        # Last resort: STAY competitive
                        stay_score = best_score - 50.0
                    else:
                        stay_score = -1e12  # Has escape → don't stay
                else:
                    # Pacman far → NEVER STAY
                    stay_score = -1e12
        
        # MID/LATE GAME (step > 40): More flexible STAY conditions
        else:
            # Condition 1: Good observation point (junction with frontier access)
            if is_good_observation_point and pac_est is None:
                stay_score = -100.0  # Still negative but less severe
                # Add bonus if we just arrived here (don't stay multiple turns)
                if len(self.prev_positions) >= 2 and self.prev_positions[-2] != my_position:
                    stay_score += 30.0
            
            # Condition 2: Pacman just appeared nearby - pause to reassess
            if pacman_just_appeared:
                stay_score = max(stay_score, best_score - 20.0)  # Competitive with moving
            
            # Condition 3: All moves are terrible (stuck)
            if best_score < -300.0:
                stay_score = max(stay_score, -200.0)  # STAY as last resort
        
        # ALWAYS: Never STAY if stuck or looping
        
        # Apply STAY penalties
        if prev2 is not None and my_position == prev2:
            stay_score -= 200.0 * no_pacman_multiplier  # ABAB with STAY
        
        # Never STAY if stuck (oscillating)
        if is_stuck:
            stay_score -= 500.0
        
        # Choose STAY only if it's actually better
        if stay_score > best_score:
            best_mv = Move.STAY

        chosen_idx = MOVE_TO_IDX.get(best_mv, MOVE_TO_IDX[Move.STAY])
        self._update_action_memory(chosen_idx)
        self.last_move = best_mv
        return best_mv
