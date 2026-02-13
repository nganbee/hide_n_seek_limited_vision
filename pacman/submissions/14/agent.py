import sys
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move

Pos = Tuple[int, int]

DIRS_4: List[Tuple[Move, Tuple[int, int]]] = [
    (Move.UP, (-1, 0)),
    (Move.DOWN, (1, 0)),
    (Move.LEFT, (0, -1)),
    (Move.RIGHT, (0, 1)),
]
GHOST_MOVES: List[Move] = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]


def manhattan(a: Pos, b: Pos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


@dataclass
class _MapModel:
    walls: np.ndarray
    h: int
    w: int

    walkable: np.ndarray
    walkable_cells: List[Pos]
    degree: np.ndarray
    mobility: np.ndarray

    _vis_cache: Dict[int, List[List[Pos]]]
    _pac_actions_cache: Dict[int, List[List[Tuple[Pos, Tuple[Move, int]]]]]
    _pac_reach_cache: Dict[int, List[Set[Pos]]]
    _ghost_moves_from: List[List[Tuple[Pos, Move]]]

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.h and 0 <= c < self.w

    def idx(self, pos: Pos) -> int:
        return pos[0] * self.w + pos[1]

    def is_walkable(self, pos: Pos) -> bool:
        r, c = pos
        return self.in_bounds(r, c) and (not self.walls[r, c])

    def visible_from(self, pos: Pos, radius: int) -> List[Pos]:
        if radius not in self._vis_cache:
            self._build_visibility(radius)
        return self._vis_cache[radius][self.idx(pos)]

    def pac_actions(self, pos: Pos, pac_speed: int) -> List[Tuple[Pos, Tuple[Move, int]]]:
        pac_speed = max(1, int(pac_speed))
        if pac_speed not in self._pac_actions_cache:
            self._build_pac_actions(pac_speed)
        return self._pac_actions_cache[pac_speed][self.idx(pos)]

    def pac_reach_set(self, pos: Pos, pac_speed: int) -> Set[Pos]:
        pac_speed = max(1, int(pac_speed))
        if pac_speed not in self._pac_reach_cache:
            self._build_pac_actions(pac_speed)
        return self._pac_reach_cache[pac_speed][self.idx(pos)]

    def ghost_moves_from(self, pos: Pos) -> List[Tuple[Pos, Move]]:
        return self._ghost_moves_from[self.idx(pos)]

    def _build_visibility(self, radius: int) -> None:
        vis: List[List[Pos]] = [[] for _ in range(self.h * self.w)]
        for r in range(self.h):
            for c in range(self.w):
                p = (r, c)
                if not self.is_walkable(p):
                    continue
                seen: List[Pos] = [p]
                for _, (dr, dc) in DIRS_4:
                    for dist in range(1, radius + 1):
                        nr, nc = r + dr * dist, c + dc * dist
                        if not self.in_bounds(nr, nc):
                            break
                        if self.walls[nr, nc]:
                            break
                        seen.append((nr, nc))
                vis[self.idx(p)] = seen
        self._vis_cache[radius] = vis

    def _build_pac_actions(self, pac_speed: int) -> None:
        actions: List[List[Tuple[Pos, Tuple[Move, int]]]] = [[] for _ in range(self.h * self.w)]
        reach: List[Set[Pos]] = [set() for _ in range(self.h * self.w)]
        for r in range(self.h):
            for c in range(self.w):
                start = (r, c)
                if not self.is_walkable(start):
                    continue
                idx = self.idx(start)
                actions[idx].append((start, (Move.STAY, 1)))
                reach[idx].add(start)

                for mv, (dr, dc) in DIRS_4:
                    cur = start
                    for steps in range(1, pac_speed + 1):
                        nxt = (cur[0] + dr, cur[1] + dc)
                        if not self.is_walkable(nxt):
                            break
                        actions[idx].append((nxt, (mv, steps)))
                        reach[idx].add(nxt)
                        cur = nxt

                actions[idx].sort(key=lambda x: (-x[1][1], x[1][0].name))
        self._pac_actions_cache[pac_speed] = actions
        self._pac_reach_cache[pac_speed] = reach

    def _build_ghost_moves(self) -> None:
        gmf: List[List[Tuple[Pos, Move]]] = [[] for _ in range(self.h * self.w)]
        for r in range(self.h):
            for c in range(self.w):
                p = (r, c)
                if not self.is_walkable(p):
                    continue
                lst: List[Tuple[Pos, Move]] = [(p, Move.STAY)]
                for mv, (dr, dc) in DIRS_4:
                    nxt = (r + dr, c + dc)
                    if self.is_walkable(nxt):
                        lst.append((nxt, mv))
                gmf[self.idx(p)] = lst
        self._ghost_moves_from = gmf

    def _compute_degree(self) -> None:
        deg = np.zeros((self.h, self.w), dtype=np.int8)
        for r in range(self.h):
            for c in range(self.w):
                if self.walls[r, c]:
                    continue
                d = 0
                for _, (dr, dc) in DIRS_4:
                    nr, nc = r + dr, c + dc
                    if self.in_bounds(nr, nc) and (not self.walls[nr, nc]):
                        d += 1
                deg[r, c] = d
        self.degree = deg

    def _compute_mobility(self) -> None:
        mob = np.zeros((self.h, self.w), dtype=np.float32)
        for r in range(self.h):
            for c in range(self.w):
                if self.walls[r, c]:
                    continue
                score = float(self.degree[r, c])
                for _, (dr, dc) in DIRS_4:
                    nr, nc = r + dr, c + dc
                    if self.in_bounds(nr, nc) and (not self.walls[nr, nc]):
                        score += 0.5 * float(self.degree[nr, nc])
                mob[r, c] = score
        self.mobility = mob


_MAP: Optional[_MapModel] = None


def _ensure_map_model(map_state: np.ndarray) -> _MapModel:
    global _MAP
    walls = (map_state == 1)
    if _MAP is not None:
        if _MAP.walls.shape == walls.shape and np.array_equal(_MAP.walls, walls):
            return _MAP

    h, w = map_state.shape
    walkable = ~walls
    walkable_cells = [(r, c) for r in range(h) for c in range(w) if walkable[r, c]]

    model = _MapModel(
        walls=walls.astype(bool),
        h=h,
        w=w,
        walkable=walkable.astype(bool),
        walkable_cells=walkable_cells,
        degree=np.zeros((h, w), dtype=np.int8),
        mobility=np.zeros((h, w), dtype=np.float32),
        _vis_cache={},
        _pac_actions_cache={},
        _pac_reach_cache={},
        _ghost_moves_from=[],
    )
    model._compute_degree()
    model._compute_mobility()
    model._build_ghost_moves()
    _MAP = model
    return model


def bfs_turn_dist(model: _MapModel, start: Pos, pac_speed: int) -> Dict[Pos, int]:
    dist: Dict[Pos, int] = {start: 0}
    q = deque([start])
    while q:
        cur = q.popleft()
        d = dist[cur]
        for nxt, _action in model.pac_actions(cur, pac_speed):
            if nxt in dist:
                continue
            dist[nxt] = d + 1
            q.append(nxt)
    return dist


def normalize_inplace(belief: np.ndarray) -> None:
    s = float(np.sum(belief))
    if s > 0:
        belief /= s


def propagate_enemy_random_walk(
    model: _MapModel,
    belief: np.ndarray,
    decay: float = 0.98,
) -> np.ndarray:
    new_b = np.zeros_like(belief, dtype=np.float32)
    h, w = belief.shape
    for r in range(h):
        for c in range(w):
            m = float(belief[r, c])
            if m <= 0 or model.walls[r, c]:
                continue
            nxts = [(r, c)]
            for _, (dr, dc) in DIRS_4:
                nr, nc = r + dr, c + dc
                if model.in_bounds(nr, nc) and (not model.walls[nr, nc]):
                    nxts.append((nr, nc))
            share = (m * decay) / float(len(nxts))
            for (nr, nc) in nxts:
                new_b[nr, nc] += share
    new_b[model.walls] = 0.0
    normalize_inplace(new_b)
    return new_b


def propagate_pacman_one_turn(
    model: _MapModel,
    belief: np.ndarray,
    pac_speed: int,
    decay: float = 0.98,
) -> np.ndarray:
    speed = max(1, int(pac_speed))
    new_b = np.zeros_like(belief, dtype=np.float32)
    h, w = belief.shape
    for r in range(h):
        for c in range(w):
            m = float(belief[r, c])
            if m <= 0 or model.walls[r, c]:
                continue
            actions = model.pac_actions((r, c), speed)
            share = (m * decay) / float(len(actions))
            for nxt, _act in actions:
                new_b[nxt[0], nxt[1]] += share
    new_b[model.walls] = 0.0
    normalize_inplace(new_b)
    return new_b


def topk_positions_from_belief(belief: np.ndarray, k: int = 20) -> List[Tuple[Pos, float]]:
    flat = belief.flatten()
    if flat.size == 0:
        return []
    k = max(1, min(int(k), flat.size))
    idxs = np.argpartition(-flat, k - 1)[:k]
    idxs = idxs[np.argsort(-flat[idxs])]
    w = belief.shape[1]
    out: List[Tuple[Pos, float]] = []
    for idx in idxs:
        p = float(flat[idx])
        if p <= 0:
            break
        r, c = divmod(int(idx), w)
        out.append(((r, c), p))
    return out


class PacmanAgent(BasePacmanAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = int(kwargs.get("pacman_speed", 2))
        self.obs_radius = kwargs.get("pacman_obs_radius", kwargs.get("obs_radius", None))
        self.capture_threshold = int(kwargs.get("capture_distance_threshold", 2))
        self.rng = random.Random(kwargs.get("seed", None))

        self._model: Optional[_MapModel] = None
        self.belief: Optional[np.ndarray] = None
        self.seen: Optional[np.ndarray] = None
        self.visit_count: Optional[np.ndarray] = None

        self.last_seen: Optional[Pos] = None
        self.last_seen_step: int = -10**9
        self.last_pos: Optional[Pos] = None
        self.stuck_counter = 0

        self.recent: deque = deque(maxlen=10)
        self._dist_cache: Dict[Pos, Dict[Pos, int]] = {}
        self._dist_cache_order: deque = deque(maxlen=6)

        self.name = "VJP Hybrid Seeker"

    def step(
        self,
        map_state: np.ndarray,
        my_position: Pos,
        enemy_position: Optional[Pos],
        step_number: int,
    ):
        self._model = _ensure_map_model(map_state)
        model = self._model

        if self.obs_radius is None:
            self.obs_radius = 0 if not np.any(map_state == -1) else 5
        obs_radius = int(self.obs_radius)

        if self.seen is None or self.seen.shape != map_state.shape:
            self.seen = np.zeros(map_state.shape, dtype=bool)
            self.visit_count = np.zeros(map_state.shape, dtype=np.int16)

        self.seen[map_state != -1] = True
        self.visit_count[my_position[0], my_position[1]] += 1
        self.recent.append(my_position)

        if self.last_pos == my_position:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        self.last_pos = my_position

        if self.belief is None or self.belief.shape != map_state.shape:
            self.belief = np.ones((model.h, model.w), dtype=np.float32)
            self.belief[model.walls] = 0.0
            normalize_inplace(self.belief)

        self._belief_measurement_update(map_state, enemy_position)
        posterior = self.belief
        belief_pred = propagate_enemy_random_walk(model, posterior, decay=0.98)

        if enemy_position is not None:
            self.last_seen = enemy_position
            self.last_seen_step = step_number
            action = self._minimax_chase(my_position, enemy_position)
            if action is not None:
                return action

        action = self._belief_hunt(my_position, belief_pred, step_number, obs_radius)
        if action is not None:
            return action

        return self._explore_fallback(my_position, obs_radius)

    def _belief_measurement_update(
        self,
        map_state: np.ndarray,
        visible_enemy: Optional[Pos],
    ) -> None:
        model = self._model
        assert model is not None and self.belief is not None

        if visible_enemy is not None:
            self.belief.fill(0.0)
            self.belief[visible_enemy[0], visible_enemy[1]] = 1.0
            return

        visible_mask = (map_state != -1)
        self.belief[visible_mask] = 0.0
        self.belief[model.walls] = 0.0

        if float(np.sum(self.belief)) <= 1e-9:
            self.belief.fill(0.0)
            self.belief[model.walkable] = 1.0
            self.belief[visible_mask] = 0.0

        normalize_inplace(self.belief)

    def _get_turn_dist_map(self, start: Pos) -> Dict[Pos, int]:
        if start in self._dist_cache:
            return self._dist_cache[start]
        model = self._model
        assert model is not None
        dmap = bfs_turn_dist(model, start, self.pacman_speed)
        self._dist_cache[start] = dmap
        self._dist_cache_order.append(start)
        while len(self._dist_cache) > self._dist_cache_order.maxlen:
            old = self._dist_cache_order.popleft()
            self._dist_cache.pop(old, None)
        return dmap

    def _minimax_chase(
        self,
        my_pos: Pos,
        ghost_pos: Pos,
    ) -> Optional[Tuple[Move, int]]:
        model = self._model
        assert model is not None

        ghost_nexts = [p for (p, _mv) in model.ghost_moves_from(ghost_pos)]

        for p1, action in model.pac_actions(my_pos, self.pacman_speed):
            guaranteed = True
            for g1 in ghost_nexts:
                if manhattan(p1, g1) >= self.capture_threshold:
                    guaranteed = False
                    break
            if guaranteed:
                return action

        best_action: Optional[Tuple[Move, int]] = None
        best_value: Optional[float] = None

        for p1, action in model.pac_actions(my_pos, self.pacman_speed):
            dist_map = self._get_turn_dist_map(p1)
            reach_set = model.pac_reach_set(p1, self.pacman_speed)
            worst_pair_score = -1e9

            for g1 in ghost_nexts:
                d = manhattan(p1, g1)
                if d < self.capture_threshold:
                    pair_score = -1000.0
                elif g1 in reach_set:
                    pair_score = -500.0
                else:
                    turn_d = dist_map.get(g1, 50)
                    pair_score = float(turn_d) + 0.35 * float(d)

                worst_pair_score = max(worst_pair_score, pair_score)

            loop_pen = 1.0 if p1 in self.recent else 0.0
            visit_pen = 0.15 * float(self.visit_count[p1[0], p1[1]])
            step_bonus = -0.05 * float(action[1])
            value = worst_pair_score + loop_pen + visit_pen + step_bonus

            if best_value is None or value < best_value:
                best_value = value
                best_action = action

        return best_action

    def _belief_hunt(
        self,
        my_pos: Pos,
        belief_pred: np.ndarray,
        step_number: int,
        obs_radius: int,
    ) -> Optional[Tuple[Move, int]]:
        model = self._model
        assert model is not None
        top = topk_positions_from_belief(belief_pred, k=30)
        if not top:
            return None

        def approx_exp_turns(p: Pos) -> float:
            sp = max(1, int(self.pacman_speed))
            acc = 0.0
            for (q, prob) in top[:20]:
                acc += prob * (manhattan(p, q) / float(sp))
            return acc

        has_recent = (self.last_seen is not None) and ((step_number - self.last_seen_step) <= 6)
        last = self.last_seen

        best_action = None
        best_score = None

        for p1, action in model.pac_actions(my_pos, self.pacman_speed):
            if obs_radius <= 0:
                vis = model.walkable_cells
            else:
                vis = model.visible_from(p1, obs_radius)

            belief_gain = float(sum(belief_pred[r, c] for (r, c) in vis))
            info_gain = float(sum(1 for (r, c) in vis if not self.seen[r, c]))
            expd = approx_exp_turns(p1)

            loop_pen = 1.0 if p1 in self.recent else 0.0
            visit_pen = 0.10 * float(self.visit_count[p1[0], p1[1]])
            vista_bonus = float(len(vis)) / 80.0
            recent_bias = 0.0
            if has_recent and last is not None:
                recent_bias = -0.12 * (manhattan(p1, last) / max(1.0, float(self.pacman_speed)))

            stuck_bonus = 0.0
            if self.stuck_counter >= 3 and p1 != my_pos:
                stuck_bonus = 0.5

            step_bonus = 0.03 * float(action[1])
            score = (
                (4.0 * belief_gain)
                + (0.08 * info_gain)
                - (0.65 * expd)
                - (0.25 * loop_pen)
                - visit_pen
                + (0.10 * vista_bonus)
                + recent_bias
                + stuck_bonus
                + step_bonus
            )

            if best_score is None or score > best_score:
                best_score = score
                best_action = action

        return best_action

    def _explore_fallback(self, my_pos: Pos, obs_radius: int) -> Tuple[Move, int]:
        model = self._model
        assert model is not None

        best_action = None
        best_score = None

        for p1, action in model.pac_actions(my_pos, self.pacman_speed):
            if obs_radius <= 0:
                vis = model.walkable_cells
            else:
                vis = model.visible_from(p1, obs_radius)

            info_gain = float(sum(1 for (r, c) in vis if not self.seen[r, c]))
            loop_pen = 1.0 if p1 in self.recent else 0.0
            visit_pen = 0.15 * float(self.visit_count[p1[0], p1[1]])
            score = (0.10 * info_gain) - (0.5 * loop_pen) - visit_pen + (0.02 * action[1])

            if best_score is None or score > best_score:
                best_score = score
                best_action = action

        return best_action if best_action is not None else (Move.STAY, 1)
FIXED_MAP_LAYOUT = [
    "#####################",
    "#.........#.........#",
    "#.###.###.#.###.###.#",
    "#...................#",
    "#.###.#.#####.#.###.#",
    "#.....#...#...#.....#",
    "#####.### # ###.#####",
    "    #.#       #.#    ",
    "#####.# ##-## #.#####",
    "     .  . G .  .     ",
    "#####.# ##### #.#####",
    "    #.#       #.#    ",
    "#####.# ##### #.#####",
    "#.........#.........#",
    "#.###.###.#.###.###.#",
    "#...#.....P.....#...#",
    "###.#.#.#####.#.#.###",
    "#.....#...#...#.....#",
    "#.#######.#.#######.#",
    "#...................#",
    "#####################"
]
GLOBAL_GRID = np.array([[1 if c in '#-' else 0 for c in row] for row in FIXED_MAP_LAYOUT])
HEIGHT, WIDTH = GLOBAL_GRID.shape

# --- 2. HELPER FUNCTIONS ---

def get_valid_neighbors(pos):
    """Trả về danh sách các ô đi được xung quanh pos"""
    r, c = pos
    valid = []
    for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
        nr, nc = r + m.value[0], c + m.value[1]
        if 0 <= nr < HEIGHT and 0 <= nc < WIDTH and GLOBAL_GRID[nr, nc] == 0:
            valid.append((nr, nc))
    return valid

def has_line_of_sight(p1, p2):
    """Kiểm tra tầm nhìn thẳng giữa 2 điểm"""
    r1, c1 = p1
    r2, c2 = p2
    if r1 == r2: # Cùng hàng ngang
        step = 1 if c2 > c1 else -1
        for c in range(c1 + step, c2, step):
            if GLOBAL_GRID[r1, c] == 1: return False
        return True
    if c1 == c2: # Cùng hàng dọc
        step = 1 if r2 > r1 else -1
        for r in range(r1 + step, r2, step):
            if GLOBAL_GRID[r, c1] == 1: return False
        return True
    return False

def bfs_path(start, target):
    """Tìm đường ngắn nhất từ start đến target bằng BFS"""
    if start == target: return []
    queue = [(start, [])]
    visited = {start}
    
    while queue:
        (curr, path) = queue.pop(0)
        if curr == target: return path
        
        for nr, nc in get_valid_neighbors(curr):
            if (nr, nc) not in visited:
                visited.add((nr, nc))
                new_path = list(path)
                dr, dc = nr - curr[0], nc - curr[1]
                move_enum = None
                for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                    if m.value == (dr, dc):
                        move_enum = m
                        break
                if move_enum:
                    new_path.append(move_enum)
                    queue.append(((nr, nc), new_path))
    return []

# --- 3. GHOST AGENT CLASS ---

class GhostAgent(BaseGhostAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Parkour_Ghost" # Tên mới cho phong cách chạy luồn lách
        
        # Mobility Map
        self.mobility_map = np.zeros((HEIGHT, WIDTH))
        self.compute_mobility_map()
        
        # State & Memory
        self.history = deque(maxlen=4)
        self.last_known_pacman = (15, 10) 
        self.turns_since_seen = 0
        
        # CHIẾN THUẬT KHAI CUỘC (Ambush)
        self.opening_target = (5, 12) 
        self.opening_moves = [] 
        
        # TRẠNG THÁI
        self.in_opening_phase = True
        self.in_camping_phase = False 

        # Sectors
        self.sectors = {
            'TR': (5, 15), 'TL': (5, 5),
            'BL': (15, 5), 'BR': (15, 15)
        }

    def compute_mobility_map(self):
        for r in range(HEIGHT):
            for c in range(WIDTH):
                if GLOBAL_GRID[r, c] == 1: continue
                neighbors = get_valid_neighbors((r, c))
                score = len(neighbors) * 1.0
                for nr, nc in neighbors:
                    score += len(get_valid_neighbors((nr, nc))) * 0.5
                self.mobility_map[r, c] = score

    def get_pacman_sector(self, pac_pos):
        r, c = pac_pos
        if r < 10 and c >= 10: return 'TR'
        if r < 10 and c < 10: return 'TL'
        if r >= 10 and c < 10: return 'BL'
        return 'BR'

    def step(self, map_state, my_position, enemy_position, step_number):
        # --- 0. BÁO ĐỘNG ĐỎ ---
        if enemy_position:
            if self.in_opening_phase or self.in_camping_phase:
                print(f"⚠️ Pacman detected! PARKOUR MODE ACTIVATED!")
                self.in_opening_phase = False
                self.in_camping_phase = False
            
            self.last_known_pacman = enemy_position
            self.turns_since_seen = 0
        else:
            self.turns_since_seen += 1

        # --- PHASE 1: OPENING ---
        if self.in_opening_phase:
            if my_position == self.opening_target:
                self.in_opening_phase = False
                self.in_camping_phase = True
                print(f"🏕️ Reached ambush spot. Camping...")
                return Move.STAY
            else:
                if not self.opening_moves:
                    self.opening_moves = bfs_path(my_position, self.opening_target)
                
                if self.opening_moves:
                    next_move = self.opening_moves.pop(0)
                    if next_move:
                        dr, dc = next_move.value
                        self.history.append((my_position[0]+dr, my_position[1]+dc))
                        return next_move
                self.in_opening_phase = False 

        # --- PHASE 2: CAMPING ---
        if self.in_camping_phase:
            return Move.STAY

        # --- PHASE 3: ACTIVE EVASION (PARKOUR STYLE) ---
        pac_sector = self.get_pacman_sector(self.last_known_pacman)
        
        target_sector_name = 'TL' 
        if pac_sector == 'TR': target_sector_name = 'BL'
        elif pac_sector == 'TL': target_sector_name = 'BR'
        elif pac_sector == 'BL': target_sector_name = 'TR'
        elif pac_sector == 'BR': target_sector_name = 'TL'
        
        target_pos = self.sectors[target_sector_name]

        best_move = self.evaluate_best_move(my_position, self.last_known_pacman, target_pos)
        
        if best_move is None: best_move = Move.STAY

        # Anti-stuck
        if best_move == Move.STAY and self.turns_since_seen < 5:
             valid = get_valid_neighbors(my_position)
             candidates = [n for n in valid if n not in self.history]
             if candidates:
                 next_pos = random.choice(candidates)
                 dr, dc = next_pos[0]-my_position[0], next_pos[1]-my_position[1]
                 for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                     if m.value == (dr, dc): return m

        dr, dc = best_move.value
        self.history.append((my_position[0]+dr, my_position[1]+dc))
        return best_move

    def evaluate_best_move(self, my_pos, enemy_pos, target_pos):
        valid_neighbors = get_valid_neighbors(my_pos)
        moves_score = []

        for nr, nc in valid_neighbors:
            # --- TÍNH ĐIỂM CHIẾN THUẬT ---
            score = 0
            
            # 1. SAFETY (Vẫn quan trọng nhất)
            dist_to_enemy = abs(nr - enemy_pos[0]) + abs(nc - enemy_pos[1])
            if dist_to_enemy < 4: 
                score -= 1000 # Tử địa
                score += dist_to_enemy * 50 # Vớt vát: Cố gắng xa thêm chút nào hay chút đó
            
            # 2. LOS BREAKING (Tàng hình)
            if not has_line_of_sight((nr, nc), enemy_pos):
                score += 300 # Ưu tiên số 1: Khuất tầm nhìn

            # 3. ANTI-CORRIDOR & JUNCTION (Khắc chế đường thẳng)
            next_valid_moves = get_valid_neighbors((nr, nc))
            num_exits = len(next_valid_moves)
            
            is_corridor = False
            if num_exits == 2:
                # Kiểm tra 2 lối ra có thẳng hàng không
                r1, c1 = next_valid_moves[0]
                r2, c2 = next_valid_moves[1]
                if r1 == r2 or c1 == c2: # Thẳng hàng dọc hoặc ngang
                    is_corridor = True
            
            if num_exits >= 3:
                score += 100 # Ngã 3/Ngã 4: Rất Tốt! Dễ đánh võng.
            elif num_exits == 1:
                score -= 500 # Ngõ cụt: Chết chắc!
            elif is_corridor:
                score -= 100 # Hành lang thẳng: Nguy hiểm với Speed 2!
            else:
                score += 50 # Góc cua (Corner): Tốt, cắt đuôi tốt.

            # 4. WALL HUGGING (Bám tường để dễ núp)
            # Kiểm tra 4 ô xung quanh xem có bao nhiêu tường
            adjacent_walls = 0
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                 # Check biên map để không lỗi index
                 check_r, check_c = nr+dr, nc+dc
                 if 0 <= check_r < HEIGHT and 0 <= check_c < WIDTH:
                     if GLOBAL_GRID[check_r, check_c] == 1:
                         adjacent_walls += 1
            
            if adjacent_walls > 0:
                score += 20 * adjacent_walls # Càng nhiều tường che chắn càng tốt

            # 5. TARGET DIRECTION (Hướng chạy về khu an toàn)
            dist_to_target = abs(nr - target_pos[0]) + abs(nc - target_pos[1])
            score -= dist_to_target * 5 

            # 6. HISTORY (Tránh lặp)
            if (nr, nc) in self.history:
                score -= 200

            # Safe Convert Move
            dr, dc = nr - my_pos[0], nc - my_pos[1]
            move_enum = Move.STAY
            for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                if m.value == (dr, dc): 
                    move_enum = m
                    break
            
            moves_score.append((score, move_enum))

        if not moves_score: return Move.STAY
        
        # Randomize nhẹ để tránh AI bị đoán trước nếu điểm bằng nhau
        moves_score.sort(key=lambda x: x[0], reverse=True)
        
        # Chọn nước đi tốt nhất (có thể thêm random trong top 2 nếu muốn)
        return moves_score[0][1]