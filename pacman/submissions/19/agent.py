from __future__ import annotations

"""
agent.py (submission) — Improved Ghost (parametric minimax + learned weights)

What changed vs your current version:
- GhostAgent is now VERY strong even without a neural net:
  * 1-step minimax lookahead vs Pacman (worst-case chase) using Pacman speed=2 on straight lines.
  * Avoids "capture zone" (Manhattan <= 1).
  * Prefers junctions/corridors that force Pacman to turn (reduces Pacman speed to 1 on turns in YOUR Pacman).
  * Anti-loop memory + escape-potential scoring (exit count).
- Parameters are loaded from "ghost_params.json" (created by train.py).
- train.py uses Evolution Strategies (ES) to optimize these weights against your PacmanAgent.

This is designed for the Project 2 limited-vision setting (cross LOS), where walls are always visible.
"""

import heapq
import json
import math
import random
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))
from agent_interface import GhostAgent as BaseGhostAgent
from agent_interface import PacmanAgent as BasePacmanAgent
from environment import Move

ACTIONS = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]
DIRS = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
N_ACTIONS = 5  # required by some student test harnesses


# ==============================
# Helpers
# ==============================
def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def next_pos(pos: Tuple[int, int], mv: Move) -> Tuple[int, int]:
    dr, dc = mv.value
    return (pos[0] + dr, pos[1] + dc)


def in_bounds(p: Tuple[int, int], H: int = 21, W: int = 21) -> bool:
    return 0 <= p[0] < H and 0 <= p[1] < W


def wallness(map_obs: np.ndarray, p: Tuple[int, int]) -> int:
    """How corridor-like the cell is: 0..4 walls around."""
    cnt = 0
    for mv in DIRS:
        q = next_pos(p, mv)
        if (not in_bounds(q, *map_obs.shape)) or map_obs[q] == 1:
            cnt += 1
    return cnt


def exit_count(map_obs: np.ndarray, p: Tuple[int, int]) -> int:
    """Number of legal non-wall exits (0..4)."""
    if not in_bounds(p, *map_obs.shape) or map_obs[p] == 1:
        return 0
    c = 0
    for mv in DIRS:
        q = next_pos(p, mv)
        if in_bounds(q, *map_obs.shape) and map_obs[q] != 1:
            c += 1
    return c


def legal_ghost_moves(map_obs: np.ndarray, gpos: Tuple[int, int]) -> List[Move]:
    moves: List[Move] = []
    for mv in ACTIONS:
        if mv == Move.STAY:
            moves.append(mv)
            continue
        q = next_pos(gpos, mv)
        if in_bounds(q, *map_obs.shape) and map_obs[q] != 1:
            moves.append(mv)
    return moves


def legal_pacman_actions(
    map_obs: np.ndarray, ppos: Tuple[int, int], max_speed: int = 2
):
    """
    Worst-case set for minimax.
    We assume Pacman may take up to max_speed steps in a chosen direction
    if corridor is clear (this upper-bounds your Pacman that only takes 2
    on straight segments).
    """
    for mv in DIRS:
        cur = ppos
        # step 1 always possible if next isn't wall
        nxt = next_pos(cur, mv)
        if not (in_bounds(nxt, *map_obs.shape) and map_obs[nxt] != 1):
            continue
        # allow 1..max_speed steps along same direction while not blocked
        cur = nxt
        yield (mv, 1, cur)
        for s in range(2, max_speed + 1):
            nxt2 = next_pos(cur, mv)
            if not (in_bounds(nxt2, *map_obs.shape) and map_obs[nxt2] != 1):
                break
            cur = nxt2
            yield (mv, s, cur)
    # also allow STAY
    yield (Move.STAY, 1, ppos)


# ==============================
# PACMAN AGENT
# (Keep yours; unchanged placeholder here to satisfy interface if needed)
# ==============================
class PacmanAgent(BasePacmanAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Lấy speed từ kwargs, không hardcode
        self.pacman_speed = int(kwargs.get("pacman_speed", 1))
        self.name = "ML Predator Pacman"

        # 1. Bộ nhớ bản đồ
        self.known_map = None  # -1: Unknown, 0: Empty, 1: Wall

        # 2. Load trọng số từ file json
        self.weights = self.load_weights("pacman_params.json")

        # 3. Lịch sử để tránh đi lại (Loop penalty) & Speed logic
        self.visited_counts = {}
        self.last_action = None

    def load_weights(self, filename):
        """Đọc file weights.json."""
        try:
            weight_path = Path(__file__).parent / filename
            with open(weight_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            # Giá trị mặc định nếu chưa có file
            return {
                "bias": 10.0,
                "closest_frontier_dist": -5.0,
                "ghost_proximity": 50.0,
                "loop_penalty": -10.0,
            }

    def step(self, map_state, my_position, enemy_position, step_number):
        # --- A. CẬP NHẬT BẢN ĐỒ ---
        if self.known_map is None:
            self.known_map = map_state.copy()
        else:
            visible_mask = map_state != -1
            self.known_map[visible_mask] = map_state[visible_mask]

        # --- B. CHỌN HÀNH ĐỘNG TỐT NHẤT (MAX Q-VALUE) ---
        legal_actions = self.get_legal_actions(my_position)
        if not legal_actions:
            return Move.STAY

        # AGGRESSIVE MODE: Nếu thấy ghost GẦN -> ưu tiên tuyệt đối
        if enemy_position:
            dist_to_ghost = abs(my_position[0] - enemy_position[0]) + abs(
                my_position[1] - enemy_position[1]
            )
            # Nếu ghost trong tầm 8 ô -> full aggressive
            if dist_to_ghost <= 8:
                best_action = self._greedy_chase(
                    my_position, enemy_position, legal_actions
                )
            else:
                # Xa hơn thì dùng Q-value bình thường
                best_action = self._choose_best_q(
                    my_position, enemy_position, legal_actions
                )
        else:
            # Không thấy ghost -> explore
            best_action = self._choose_best_q(
                my_position, enemy_position, legal_actions
            )

        # Cập nhật lịch sử (decay để không phạt quá nặng)
        next_pos = self.get_successor_pos(my_position, best_action)
        self.visited_counts[next_pos] = self.visited_counts.get(next_pos, 0) + 1
        # Decay counts theo thời gian
        if step_number % 20 == 0:
            for k in list(self.visited_counts.keys()):
                self.visited_counts[k] = max(0, self.visited_counts[k] - 1)

        # --- C. SPEED OPTIMIZATION ---
        steps_to_take = 1

        # Nếu chase ghost hoặc đi thẳng -> thử speed 2
        if self.pacman_speed >= 2:
            pos_1 = self.get_successor_pos(my_position, best_action)
            pos_2 = self.get_successor_pos(pos_1, best_action)

            # Check ô 2 hợp lệ và có lợi
            if self._is_valid_pos(pos_2):
                # Nếu chase ghost -> luôn dùng speed 2
                if enemy_position and dist_to_ghost <= 10:
                    steps_to_take = 2
                # Hoặc nếu đi thẳng liên tục
                elif self.last_action == best_action:
                    steps_to_take = 2

        self.last_action = best_action

        if steps_to_take > 1 and steps_to_take <= self.pacman_speed:
            return (best_action, steps_to_take)
        return best_action

    def _greedy_chase(self, my_pos, ghost_pos, legal_actions):
        """Chase trực tiếp về phía ghost (greedy)"""
        best_action = legal_actions[0]
        min_dist = float("inf")

        for action in legal_actions:
            next_pos = self.get_successor_pos(my_pos, action)
            dist = abs(next_pos[0] - ghost_pos[0]) + abs(
                next_pos[1] - ghost_pos[1]
            )
            if dist < min_dist:
                min_dist = dist
                best_action = action
        return best_action

    def _choose_best_q(self, my_pos, enemy_pos, legal_actions):
        """Chọn action theo Q-value cao nhất"""
        best_action = legal_actions[0]
        max_q = float("-inf")

        for action in legal_actions:
            features = self.get_features(my_pos, action, enemy_pos)
            q_value = self.get_q_value(features)

            if q_value > max_q:
                max_q = q_value
                best_action = action

        return best_action

    # --- CÁC HÀM HỖ TRỢ ML ---
    def get_q_value(self, features):
        return sum(
            self.weights.get(feature, 0) * value
            for feature, value in features.items()
        )

    def get_features(self, my_pos, action, enemy_pos):
        features = {"bias": 1.0}
        next_pos = self.get_successor_pos(my_pos, action)

        # 1. Feature: AGGRESSIVE CHASE - càng gần ghost càng cao
        if enemy_pos:
            dist = abs(next_pos[0] - enemy_pos[0]) + abs(
                next_pos[1] - enemy_pos[1]
            )
            if dist == 0:
                features["ghost_proximity"] = 100.0
            else:
                # Reward càng mạnh khi càng gần
                features["ghost_proximity"] = 20.0 / (dist + 0.1)

            # 2. Chase direction - đi thẳng về phía ghost
            dx = enemy_pos[0] - next_pos[0]
            dy = enemy_pos[1] - next_pos[1]
            move_dx = action.value[0]
            move_dy = action.value[1]
            # Nếu hướng đi cùng chiều với ghost -> bonus
            if (dx * move_dx > 0) or (dy * move_dy > 0):
                features["chase_direction"] = 1.0
            else:
                features["chase_direction"] = 0.0

            # 3. Cutting path - dự đoán ghost sẽ chạy đâu
            predicted_ghost = (
                enemy_pos[0] + (1 if dx < 0 else -1 if dx > 0 else 0),
                enemy_pos[1] + (1 if dy < 0 else -1 if dy > 0 else 0),
            )
            pred_dist = abs(next_pos[0] - predicted_ghost[0]) + abs(
                next_pos[1] - predicted_ghost[1]
            )
            features["cut_path"] = 10.0 / (pred_dist + 1)
        else:
            features["ghost_proximity"] = 0.0
            features["chase_direction"] = 0.0
            features["cut_path"] = 0.0

        # 4. EXPLORATION - frontier distance
        dist_frontier = self.bfs_frontier(next_pos)
        if dist_frontier is not None:
            features["frontier_dist"] = dist_frontier / 15.0
        else:
            features["frontier_dist"] = 1.0

        # 5. UNKNOWN COVERAGE - số ô chưa biết xung quanh
        unknown_neighbors = self.count_unknown_neighbors(next_pos)
        features["unknown_nearby"] = unknown_neighbors / 8.0

        # 6. MOBILITY - số hướng đi hợp lệ từ next_pos
        legal_from_next = len(self.get_legal_actions(next_pos))
        features["mobility"] = legal_from_next / 4.0

        # 7. LOOP PENALTY - phạt đi lại
        visit_count = self.visited_counts.get(next_pos, 0)
        features["loop_penalty"] = min(visit_count, 5) / 5.0

        # 8. CENTER BIAS - ưu tiên ở trung tâm map
        h, w = self.known_map.shape
        center_dist = abs(next_pos[0] - h // 2) + abs(next_pos[1] - w // 2)
        features["center_bias"] = center_dist / 20.0

        return features

    # --- CÁC HÀM HELPER KHÁC ---
    def get_successor_pos(self, pos, action):
        dr, dc = action.value
        return (pos[0] + dr, pos[1] + dc)

    def _is_valid_pos(self, pos):
        # Đi được vào ô 0 (Đường) và -1 (Chưa biết). Tránh 1 (Tường)
        h, w = self.known_map.shape
        r, c = pos
        if r < 0 or r >= h or c < 0 or c >= w:
            return False
        return self.known_map[r, c] != 1

    def get_legal_actions(self, pos):
        actions = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            nr, nc = pos[0] + move.value[0], pos[1] + move.value[1]
            if self._is_valid_pos((nr, nc)):
                actions.append(move)
        return actions

    def count_unknown_neighbors(self, pos):
        """Đếm số ô -1 (unknown) xung quanh position (8 directions)"""
        count = 0
        h, w = self.known_map.shape
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = pos[0] + dr, pos[1] + dc
                if 0 <= nr < h and 0 <= nc < w:
                    if self.known_map[nr, nc] == -1:
                        count += 1
        return count

    def bfs_frontier(self, start):
        """Tìm khoảng cách tới ô -1 gần nhất"""
        queue = [(start, 0)]
        visited = {start}
        h, w = self.known_map.shape
        while queue:
            curr, dist = queue.pop(0)
            if self.known_map[curr[0], curr[1]] == -1:
                return dist
            if dist > 12:
                return 12  # Giảm search depth để nhanh hơn
            for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                nr, nc = curr[0] + m.value[0], curr[1] + m.value[1]
                if self._is_valid_pos((nr, nc)) and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append(((nr, nc), dist + 1))
        return None


# ==============================
# GHOST AGENT
# ==============================
class GhostAgent(BaseGhostAgent):
    """
    Strong planner + learned weights (from ghost_params.json).

    Core idea:
    - If Pacman is visible -> 1-step minimax against a worst-case Pacman chase.
    - If not visible -> move to "good hiding states":
        corridor/junctions with high escape options and low loopiness.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Minimax-EvadeGhost"

        # These kwargs are passed by arena when loading (see arena.py)
        self.capture_distance = int(
            kwargs.get("capture_distance", 2)
        )  # set to 2 to mean caught when manhattan<=1
        self.pacman_speed = int(kwargs.get("pacman_speed", 2))

        # Parameters (defaults, overwritten by json)
        self.params: Dict[str, float] = {
            "w_minimax": 6.0,
            "w_dist": 2.0,
            "w_corridor": 1.4,
            "w_exits": 0.9,
            "w_junction": 0.6,
            "w_stay": -2.0,
            "w_loop": -0.9,
            "danger_margin": 4.0,  # if visible pac within capture_distance + margin => minimax
        }
        self._load_params()

        # Memory for anti-loop and tracking
        self.step_idx = 0
        self.pos_hist = deque(maxlen=24)
        self.visit = defaultdict(int)
        self.last_seen_pac: Optional[Tuple[int, int]] = None
        self.last_seen_step: int = -10

    def reset(self):
        self.step_idx = 0
        self.pos_hist.clear()
        self.visit.clear()
        self.last_seen_pac = None
        self.last_seen_step = -10

    def _load_params(self):
        candidates = [
            Path("ghost_params.json"),
            Path(__file__).with_name("ghost_params.json"),
        ]
        p = None
        for c in candidates:
            if c.exists():
                p = c
                break
        if p is None:
            return
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                for k, v in data.items():
                    if k in self.params:
                        self.params[k] = float(v)
        except Exception:
            pass

    def _junction_bonus(
        self, map_obs: np.ndarray, pos: Tuple[int, int]
    ) -> float:
        """
        Junctions force Pacman to turn more often (reduces its 2-step advantage in your Pacman logic).
        Simple proxy: exits>=3.
        """
        ex = exit_count(map_obs, pos)
        return 1.0 if ex >= 3 else 0.0

    def _loop_penalty(self, pos: Tuple[int, int]) -> float:
        # penalize revisits; softened
        return float(self.visit[pos])

    def _is_capture(
        self, pac_pos: Tuple[int, int], gpos: Tuple[int, int]
    ) -> bool:
        # "ghost bị bắt khi manhattan <= 1" => capture_distance_threshold should be 2 in env/arena.
        return manhattan(pac_pos, gpos) <= 1

    def _minimax_move(
        self, map_obs: np.ndarray, gpos: Tuple[int, int], pac: Tuple[int, int]
    ) -> Move:
        best_mv = Move.STAY
        best_score = -1e18

        for mv in legal_ghost_moves(map_obs, gpos):
            g2 = next_pos(gpos, mv) if mv != Move.STAY else gpos

            # If this move walks into immediate capture zone, heavily penalize
            if self._is_capture(pac, g2):
                continue

            # Worst-case Pacman response: choose action that minimizes distance to ghost
            worst_d = 1e9
            for _, _, p2 in legal_pacman_actions(
                map_obs, pac, max_speed=self.pacman_speed
            ):
                d = manhattan(p2, g2)
                if d < worst_d:
                    worst_d = d

            # Build score
            sc = 0.0
            sc += self.params["w_minimax"] * float(worst_d)
            sc += self.params["w_corridor"] * float(wallness(map_obs, g2))
            sc += self.params["w_exits"] * float(exit_count(map_obs, g2))
            sc += self.params["w_junction"] * self._junction_bonus(map_obs, g2)
            sc += self.params["w_loop"] * self._loop_penalty(g2)
            if mv == Move.STAY:
                sc += self.params["w_stay"]

            # Extra safety: prefer increasing current distance
            sc += self.params["w_dist"] * float(
                manhattan(g2, pac) - manhattan(gpos, pac)
            )

            if sc > best_score:
                best_score = sc
                best_mv = mv

        # Fallback: if all moves were capture, just pick the move that maximizes distance
        if best_mv == Move.STAY and self._is_capture(pac, gpos):
            cand = max(
                legal_ghost_moves(map_obs, gpos),
                key=lambda m: manhattan(
                    next_pos(gpos, m) if m != Move.STAY else gpos, pac
                ),
            )
            return cand
        return best_mv

    def _hide_move(self, map_obs: np.ndarray, gpos: Tuple[int, int]) -> Move:
        """
        When Pacman not visible: move to "good hiding structure"
        - corridors/junctions (walls around)
        - many exits (escape potential)
        - avoid looping
        """
        best_mv = Move.STAY
        best = -1e18

        # decay effect of last seen pac position
        pac_hint = self.last_seen_pac
        hint_weight = (
            math.exp(-0.15 * max(0, self.step_idx - self.last_seen_step))
            if pac_hint is not None
            else 0.0
        )

        for mv in legal_ghost_moves(map_obs, gpos):
            g2 = next_pos(gpos, mv) if mv != Move.STAY else gpos

            sc = 0.0
            sc += 1.2 * float(wallness(map_obs, g2))
            sc += 0.7 * float(exit_count(map_obs, g2))
            sc += 0.6 * self._junction_bonus(map_obs, g2)
            sc += self.params["w_loop"] * self._loop_penalty(g2)
            if mv == Move.STAY:
                sc += self.params["w_stay"]

            # If we have a last-seen pac position, bias away from it.
            if pac_hint is not None:
                sc += 2.2 * hint_weight * float(manhattan(g2, pac_hint))

            if sc > best:
                best = sc
                best_mv = mv

        return best_mv

    def step(
        self,
        map_state: np.ndarray,
        my_position: Tuple[int, int],
        enemy_position: Optional[Tuple[int, int]],
        step_number: int,
    ):
        self.step_idx = int(step_number)
        gpos = (int(my_position[0]), int(my_position[1]))

        self.pos_hist.append(gpos)
        self.visit[gpos] += 1

        pac = (
            None
            if enemy_position is None
            else (int(enemy_position[0]), int(enemy_position[1]))
        )
        if pac is not None:
            self.last_seen_pac = pac
            self.last_seen_step = self.step_idx

            d = manhattan(gpos, pac)
            if d <= (
                self.capture_distance + float(self.params["danger_margin"])
            ):
                mv = self._minimax_move(map_state, gpos, pac)
            else:
                # not immediate danger: still keep distance and go to structure
                mv = self._hide_move(map_state, gpos)
                # nudge away
                if (
                    manhattan(
                        next_pos(gpos, mv) if mv != Move.STAY else gpos, pac
                    )
                    < d
                ):
                    mv = self._minimax_move(map_state, gpos, pac)
            return mv

        # Pacman not visible: hide / anti-loop
        mv = self._hide_move(map_state, gpos)
        return mv
