"""
Example student submission showing the required interface.

Students should implement their own PacmanAgent and/or GhostAgent
following this template.
"""

import heapq
import sys
from pathlib import Path
from collections import deque
from heapq import heappop, heappush
from sb3_contrib import RecurrentPPO

# Add src to path to import the interface
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np
import random
import time


class PacmanAgent(BasePacmanAgent):
    """
    Example Pacman agent using a simple greedy strategy.
    Students should implement their own search algorithms here.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the Pacman agent.
        Students can set up any data structures they need here.
        """
        super().__init__(**kwargs)
        self.name = "Example Greedy Pacman"
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        # Memory for limited observation mode
        self.last_known_enemy_pos = None
        self.map_size = 21
        self.known_map = np.full((self.map_size, self.map_size), -1)  
    
    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int):
        """
        Simple greedy strategy: move towards the ghost.
        
        When enemy_position is None (limited observation mode),
        uses last known position or explores randomly.
        
        Students should implement better search algorithms like:
        - BFS (Breadth-First Search)
        - DFS (Depth-First Search)
        - A* Search
        - Greedy Best-First Search
        - etc.
        """
        visible_map = (map_state != -1)
        self.known_map[visible_map] = map_state[visible_map]

        # Update memory if the enemy is visible
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
        elif self.last_known_enemy_pos == my_position:
            self.last_known_enemy_pos = None
        
        # Use current sighting, fall back to last known position, or explore
        target = None
        if enemy_position:
            target = enemy_position
        elif self.last_known_enemy_pos:
            target = self.last_known_enemy_pos
        else:
            target = self._find_nearest_unknown(my_position)
        
        if target is None:
            return self._explore(my_position, map_state)
        
        best_move = self.a_star_search(my_position, target)        
        steps = self._max_valid_steps(my_position, best_move, self.known_map, self.pacman_speed)

        if steps > 1:
            # Position if moving 1 step
            r1 = my_position[0] + best_move.value[0]
            c1 = my_position[1] + best_move.value[1]
            pos1 = (r1, c1)
            
            # Position if moving 2 steps
            r2 = my_position[0] + best_move.value[0] * 2
            c2 = my_position[1] + best_move.value[1] * 2
            pos2 = (r2, c2)
            
            # If moving 2 steps does not get closer to the target than 1 step, reduce speed to 1
            if self.heuristic(pos2, target) >= self.heuristic(pos1, target):
                steps = 1
        
        if steps > 0:
            return (best_move, steps)
        
        return (Move.STAY, 1)

    def _explore(self, my_position: tuple, map_state: np.ndarray):
        """Random exploration when the enemy position is unknown."""
        all_moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        random.shuffle(all_moves)
        
        for move in all_moves:
            steps = self._max_valid_steps(my_position, move, map_state, self.pacman_speed)
            if steps > 0:
                return (move, steps)
        
        return (Move.STAY, 1)
    
    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        """Check if a position is valid (not a wall and within bounds)."""
        row, col = pos
        height, width = map_state.shape
        
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        
        return map_state[row, col] == 0

    def _max_valid_steps(self, pos: tuple, move: Move, map_state: np.ndarray, desired_steps: int) -> int:
        steps = 0
        max_steps = min(self.pacman_speed, max(1, desired_steps))
        current = pos
        for _ in range(max_steps):
            delta_row, delta_col = move.value
            next_pos = (current[0] + delta_row, current[1] + delta_col)
            if not self._is_valid_position(next_pos, map_state):
                break
            steps += 1
            current = next_pos
        return steps

    def _desired_steps(self, move: Move, row_diff: int, col_diff: int) -> int:
        if move in (Move.UP, Move.DOWN):
            return abs(row_diff)
        if move in (Move.LEFT, Move.RIGHT):
            return abs(col_diff)
        return 1

    def _find_nearest_unknown(self, start_pos):
        """BFS to find the nearest unknown cell (-1)."""
        queue = [(start_pos, 0)]
        visited = {start_pos}
        directions = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]

        while queue:
            (r, c), dist = queue.pop(0)
            if self.known_map[r, c] == -1:
                return (r, c)
            for mv in directions:
                dr, dc = mv.value
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.map_size and 0 <= nc < self.map_size:
                    if (nr, nc) not in visited and self.known_map[nr, nc] != 1:
                        visited.add((nr, nc))
                        queue.append(((nr, nc), dist + 1))

        return None
    
    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def a_star_search(self, start, goal):
        """A* implementation to find the best move toward the goal."""
        if start == goal: 
            return Move.STAY
        
        frontier = []
        heapq.heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}
        found = False

        dir_map = {(-1, 0): Move.UP, (1, 0): Move.DOWN, (0, -1): Move.LEFT, (0, 1): Move.RIGHT}

        while frontier:
            _, current = heapq.heappop(frontier)

            if current == goal:
                found = True
                break

            for (dr, dc), move_enum in dir_map.items():
                next_pos = (current[0] + dr, current[1] + dc)
                if not (0 <= next_pos[0] < self.map_size and 0 <= next_pos[1] < self.map_size):
                    continue

                if self.known_map[next_pos[0], next_pos[1]] == 1:
                    continue

                new_cost = cost_so_far[current] + 1
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + self.heuristic(goal, next_pos)
                    heapq.heappush(frontier, (priority, next_pos))
                    came_from[next_pos] = current
        
        if found:
            curr = goal
            while came_from[curr] != start:
                curr = came_from[curr]
                if curr is None:
                    return Move.STAY
            
            diff = (curr[0] - start[0], curr[1] - start[1])
            return dir_map.get(diff, Move.STAY)
        
        return Move.STAY

class GhostAgent(BaseGhostAgent):
    """
    Two-mode algorithmic Ghost under partial observability:

    Mode 1 (Scout): triggered when Pacman hasn't been seen for >= 30 steps (or never seen).
      - Aggressively turn: prefer perpendicular moves to last_move.
      - Quick fix: only force turning if the perpendicular move leads into a "fresh turn" cell
        (junction/corner that is unvisited or frontier). If that turn is already cached (visited and not frontier),
        do NOT force turning; fall back to normal scoring.
      - Cache the map progressively from FOV.
      - Prefer entering junctions (observed_degree >= 3) and frontier cells (free with adjacent unknown).
      - Avoid negative-marked rays (short clear alleys) and deadends.

    Mode 2 (Avoid): when Pacman is visible or was seen within the last 30 steps.
      - Use the cached map to move to the nearest "turn" cell:
        * Turn = junction (deg >= 3) OR corner (deg == 2 with perpendicular neighbors).
      - BFS on known free cells to the closest such cell (ignore negative-marked cells if possible).
      - Deadend heuristic from FOV: rays < 5 (not blocked by wall) mark first 4 visible free cells as "bad" (-50).
      - If no path to a turn exists, fallback to a local step that increases distance from Pacman while avoiding bad cells.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Ghost Algo (Aggressive-Turn + Turn-Seeking)"

        # Map and mode params
        self.map_size = 21
        self.obs_radius = int(kwargs.get("ghost_obs_radius", 5))
        self.mode2_persist_steps = 30  # remain in Mode 2 if Pacman seen within last 30 steps
        self.deadend_junc_dist_thresh = 5  # near-deadend if dist to nearest junction < 5

        # Aggressive turning weights (Mode 1)
        self.w_turn_perp = 2.0
        self.w_frontier = 0.8
        self.w_degree = 1.2
        self.w_corridor = 0.3
        self.w_deadend = 2.5
        self.w_recent = 0.5
        self.w_junc_bonus = 0.6

        # Episode-local caches (in-memory only)
        self.known_map = np.full((self.map_size, self.map_size), -1, dtype=np.int8)  # -1 unknown, 0 free, 1 wall
        self.observed_degree = np.zeros((self.map_size, self.map_size), dtype=np.int8)
        self.neg_mask = np.zeros((self.map_size, self.map_size), dtype=bool)          # "bad" cells (deadend ray)
        self.visited = set()
        self.visited_recent = deque(maxlen=12)

        # Pacman tracking
        self.last_seen_pac = None
        self.last_seen_age = 999

        # Motion memory
        self.last_move = Move.STAY

    # ---------------- Main decision ----------------
    def step(self, map_state: np.ndarray,
             my_position: tuple,
             enemy_position: tuple,
             step_number: int) -> Move:
        # Reset per-episode state if first step (no file cache)
        if step_number == 0:
            self._reset_episode()

        # Update map cache and visit memory
        self._update_known_map(map_state)
        self.visited.add(tuple(my_position))
        self.visited_recent.append(tuple(my_position))

        # Update Pacman memory and mode timing
        if enemy_position is not None:
            self.last_seen_pac = enemy_position
            self.last_seen_age = 0
        else:
            self.last_seen_age += 1

        # Update derived structures
        self._update_observed_degree()
        self._update_negative_from_fov(map_state, my_position)

        # Emergency: if we are at a near-deadend, try to exit it quickly
        if self._is_deadend_near(my_position):
            mv = self._escape_deadend(my_position)
            self.last_move = mv if mv != Move.STAY else self.last_move
            return mv

        # Mode selection
        in_mode2 = (self.last_seen_pac is not None) and (self.last_seen_age < self.mode2_persist_steps)
        if in_mode2:
            mv = self._mode2_seek_nearest_turn(my_position)
        else:
            mv = self._mode1_aggressive_turn_quickfix(my_position)

        self.last_move = mv if mv != Move.STAY else self.last_move
        return mv

    # ---------------- Mode 1: Aggressive turning with quick fix ----------------
    def _mode1_aggressive_turn_quickfix(self, my_pos):
        """
        Quick fix:
        - Always turn (perpendicular to last_move) WHEN POSSIBLE,
          BUT only if the perpendicular move leads into a "fresh turn" target:
            * next cell is a turn (junction/corner), AND
            * next cell is frontier (adjacent unknown) OR unvisited.
        - If all perpendicular turns lead into already-cached turns (visited & not frontier),
          do NOT force turning; fall back to normal scoring across all valid moves.
        """
        candidates = self._valid_moves(my_pos)
        if not candidates:
            return Move.STAY

        # Collect perpendicular candidates
        perp_candidates = []
        for mv in candidates:
            if not self._is_perpendicular(self.last_move, mv):
                continue
            nxt = self._apply_move(my_pos, mv)
            if not self._is_valid(nxt):
                continue
            is_turn_next = self._is_turn_cell(nxt)
            is_frontier_next = self._is_frontier(nxt)
            is_unvisited_next = (self.known_map[nxt] == 0 and tuple(nxt) not in self.visited and not self.neg_mask[nxt])
            is_fresh_turn = is_turn_next and (is_frontier_next or is_unvisited_next)
            perp_candidates.append((mv, nxt, is_turn_next, is_fresh_turn))

        # If any perpendicular move enters a fresh turn ? force turning and choose best among them
        fresh_perp = [(mv, nxt) for (mv, nxt, is_turn, is_fresh) in perp_candidates if is_fresh]
        if fresh_perp:
            best_mv, best_score = Move.STAY, -1e9
            for mv, nxt in fresh_perp:
                deg_next = self.observed_degree[nxt]
                frontier = self._is_frontier(nxt)
                recent_pen = self.w_recent if tuple(nxt) in self.visited_recent else 0.0
                corr_len = self._corridor_len(nxt, mv)
                is_dead = (deg_next == 1)

                score = 0.0
                score += self.w_turn_perp  # forced turn bonus
                score += self.w_frontier * (1.0 if frontier else 0.0)
                score += self.w_degree * deg_next
                if deg_next >= 3:
                    score += self.w_junc_bonus
                score -= self.w_corridor * corr_len
                score -= self.w_deadend * (1.0 if is_dead else 0.0)
                score -= recent_pen
                if self.neg_mask[nxt]:
                    score -= 1000.0

                if score > best_score:
                    best_mv, best_score = mv, score
            return best_mv

        # Else: do NOT force turning; apply normal local scoring across all candidates
        best_mv, best_score = Move.STAY, -1e9
        for mv in candidates:
            nxt = self._apply_move(my_pos, mv)
            if not self._is_valid(nxt):
                continue

            deg_next = self.observed_degree[nxt]
            frontier = self._is_frontier(nxt)
            is_perp = self._is_perpendicular(self.last_move, mv)
            recent_pen = self.w_recent if tuple(nxt) in self.visited_recent else 0.0
            corr_len = self._corridor_len(nxt, mv)
            is_dead = (deg_next == 1)

            score = 0.0
            # still prefer turning, but not forced
            score += self.w_turn_perp * (1.0 if is_perp else 0.0)
            score += self.w_frontier * (1.0 if frontier else 0.0)
            score += self.w_degree * deg_next
            if deg_next >= 3:
                score += self.w_junc_bonus
            score -= self.w_corridor * corr_len
            score -= self.w_deadend * (1.0 if is_dead else 0.0)
            score -= recent_pen
            if self.neg_mask[nxt]:
                score -= 1000.0

            # Additional penalty: avoid turning into an already-cached turn
            if is_perp and self._is_turn_cell(nxt) and (not frontier) and (tuple(nxt) in self.visited):
                score -= 2.0  # small penalty to not prefer this "known turn"

            if score > best_score:
                best_mv, best_score = mv, score

        return best_mv

    # ---------------- Mode 2: Nearest turn via cached map ----------------
    def _mode2_seek_nearest_turn(self, my_pos):
        """
        Move toward the nearest "turn" cell using cached map information.

        Definition of a "turn" cell:
          - junction: observed_degree >= 3
          - corner: observed_degree == 2 AND the two neighbors are perpendicular (not opposite)
        BFS on known free cells (avoiding negative-marked cells if possible) to find the closest such cell.
        If no path exists, fallback to a local step that increases distance from last_seen_pac.
        """
        # If Pacman is visible, first try an immediate perpendicular "ankle-break" to disrupt pursuit
        if self.last_seen_pac is not None:
            approach_dir = self._approach_dir(self.last_seen_pac, my_pos)
            for mv in self._perpendiculars(approach_dir):
                nxt = self._apply_move(my_pos, mv)
                if self._is_valid(nxt) and self.observed_degree[nxt] >= 2:
                    return mv

        # BFS to nearest turn cell on known free graph
        target = self._nearest_turn_cell(my_pos)
        if target is not None:
            mv = self._a_star_first_step(my_pos, target)
            if mv is not None:
                return mv

        # Fallback: take a local step that increases distance from Pacman (while avoiding bad cells)
        if self.last_seen_pac is not None:
            return self._local_away_from_pac(my_pos, self.last_seen_pac)

        # If no Pacman information, fallback to Mode 1 behavior
        return self._mode1_aggressive_turn_quickfix(my_pos)

    # ---------------- Deadend ray marking (heuristic -50) ----------------
    def _update_negative_from_fov(self, view, my_pos):
        """
        For each FOV ray (up to 5 cells), if the ray length < 5 and is NOT blocked by a wall (1),
        mark up to the first 4 visible free cells on that ray as "negative" (equivalent to a -50 penalty).
        We then avoid those cells whenever possible.
        """
        for mv in [Move.UP, Move.DOWN, Move.RIGHT, Move.LEFT]:
            ray_cells, blocked_by_wall = self._vision_ray(view, my_pos, mv)
            if len(ray_cells) == 0:
                continue
            if len(ray_cells) < 5 and not blocked_by_wall:
                for (r, c) in ray_cells[:min(4, len(ray_cells))]:
                    if 0 <= r < self.map_size and 0 <= c < self.map_size:
                        if self.known_map[r, c] == 0:
                            self.neg_mask[r, c] = True

    def _vision_ray(self, view, origin, mv):
        """
        Walk along mv up to 5 cells in the current FOV:
        - Stop if out of bounds or a wall (1) is encountered (mark as blocked_by_wall=True).
        - Collect visible free cells (0). If view == -1 (outside FOV), stop (not blocked by wall).
        Returns (list_of_visible_free_cells_on_ray, blocked_by_wall_flag).
        """
        dr, dc = mv.value
        r0, c0 = origin
        ray = []
        blocked_by_wall = False
        for step in range(1, 6):
            r, c = r0 + dr * step, c0 + dc * step
            if not (0 <= r < self.map_size and 0 <= c < self.map_size):
                break
            cell = view[r, c]
            if cell == -1:
                break
            if cell == 1:
                blocked_by_wall = True
                break
            # cell == 0 ? visible free cell on this ray
            ray.append((r, c))
        return ray, blocked_by_wall

    # ---------------- Turn detection and BFS routing ----------------
    def _nearest_turn_cell(self, start):
        """
        BFS over known free cells (avoiding negative-marked cells if possible) to the closest "turn" cell.
        Returns the target cell or None if none found/reachable.
        """
        q = deque([start])
        seen = {start}
        while q:
            cur = q.popleft()
            if self._is_turn_cell(cur):
                return cur
            for mv in [Move.UP, Move.DOWN, Move.RIGHT, Move.LEFT]:
                nxt = self._apply_move(cur, mv)
                if nxt in seen:
                    continue
                if not self._is_valid(nxt):
                    continue
                seen.add(nxt)
                q.append(nxt)
        return None

    def _is_turn_cell(self, pos):
        """
        A cell is a "turn" if:
          - observed_degree >= 3 (junction), or
          - observed_degree == 2 and the two neighbors are perpendicular (a corner).
        Cells with degree 2 that are opposite (straight corridor) are NOT turns.
        """
        if not self._is_valid(pos):
            return False
        deg = self.observed_degree[pos]
        if deg >= 3:
            return True
        if deg == 2:
            dirs = self._neighbor_dirs(pos)
            return len(dirs) == 2 and not self._are_opposite(dirs[0], dirs[1])
        return False

    def _local_away_from_pac(self, my_pos, pac_pos):
        """
        Local greedy fallback: pick a move that increases Manhattan distance from Pacman,
        breaking ties by preferring higher degree and non-negative cells.
        """
        candidates = self._valid_moves(my_pos)
        if not candidates:
            return Move.STAY
        best, best_score = Move.STAY, -1e9
        for mv in candidates:
            nxt = self._apply_move(my_pos, mv)
            if not self._is_valid(nxt):
                continue
            dist = self._manhattan(nxt, pac_pos)
            deg = self.observed_degree[nxt]
            score = dist + 0.3 * deg
            if self.neg_mask[nxt]:
                score -= 1000.0
            if score > best_score:
                best, best_score = mv, score
        return best

    # ---------------- Deadend escape and degree updates ----------------
    def _is_deadend_near(self, pos):
        """
        Consider a cell near-deadend if it's valid, degree == 1, and the BFS distance
        to the nearest junction (deg >= 3) is < 5.
        """
        if not self._is_valid(pos):
            return False
        if self.observed_degree[pos] != 1:
            return False
        return self._dist_to_nearest_junction(pos) < self.deadend_junc_dist_thresh

    def _escape_deadend(self, pos):
        """
        Move toward the neighbor with highest observed degree to quickly leave deadend/corridor.
        """
        best, best_deg = Move.STAY, -1
        for mv in [Move.UP, Move.DOWN, Move.RIGHT, Move.LEFT]:
            nxt = self._apply_move(pos, mv)
            if self._is_valid(nxt):
                deg = self.observed_degree[nxt]
                if deg > best_deg:
                    best, best_deg = mv, deg
        return best

    def _update_observed_degree(self):
        """
        observed_degree counts how many known free neighbors (ignoring negative-marked cells).
        """
        h, w = self.map_size, self.map_size
        for r in range(h):
            for c in range(w):
                if self.known_map[r, c] != 0 or self.neg_mask[r, c]:
                    self.observed_degree[r, c] = 0
                    continue
                cnt = 0
                for mv in [Move.UP, Move.DOWN, Move.RIGHT, Move.LEFT]:
                    nr, nc = self._apply_move((r, c), mv)
                    if 0 <= nr < h and 0 <= nc < w and self.known_map[nr, nc] == 0 and not self.neg_mask[nr, nc]:
                        cnt += 1
                self.observed_degree[r, c] = cnt

    def _dist_to_nearest_junction(self, pos):
        """
        BFS distance from pos to the nearest junction (observed_degree >= 3).
        """
        if not self._is_valid(pos):
            return 999
        q = deque([pos])
        seen = {pos}
        dist = {pos: 0}
        while q:
            cur = q.popleft()
            if self.observed_degree[cur] >= 3:
                return dist[cur]
            for mv in [Move.UP, Move.DOWN, Move.RIGHT, Move.LEFT]:
                nxt = self._apply_move(cur, mv)
                if self._is_valid(nxt) and nxt not in seen:
                    seen.add(nxt)
                    dist[nxt] = dist[cur] + 1
                    q.append(nxt)
        return 999

    # ---------------- Map, validity, and utilities ----------------
    def _reset_episode(self):
        self.known_map.fill(-1)
        self.observed_degree.fill(0)
        self.neg_mask.fill(False)
        self.visited.clear()
        self.visited_recent.clear()
        self.last_seen_pac = None
        self.last_seen_age = 999
        self.last_move = Move.STAY

    def _update_known_map(self, view):
        """
        Walls (1) are always visible; free (0) only within FOV; unknown elsewhere (-1).
        """
        mask = (view != -1)
        self.known_map[mask] = view[mask]

    def _valid_moves(self, pos):
        moves = []
        for mv in [Move.UP, Move.DOWN, Move.RIGHT, Move.LEFT]:
            nxt = self._apply_move(pos, mv)
            if self._is_valid(nxt):
                moves.append(mv)
        if not moves:
            moves = [Move.STAY]
        return moves

    def _apply_move(self, pos, mv):
        dr, dc = mv.value
        return (pos[0] + dr, pos[1] + dc)

    def _is_valid(self, pos):
        r, c = pos
        return (0 <= r < self.map_size and 0 <= c < self.map_size and
                self.known_map[r, c] == 0 and not self.neg_mask[r, c])

    def _is_perpendicular(self, mv_prev, mv_next):
        if mv_prev in [Move.UP, Move.DOWN] and mv_next in [Move.LEFT, Move.RIGHT]:
            return True
        if mv_prev in [Move.LEFT, Move.RIGHT] and mv_next in [Move.UP, Move.DOWN]:
            return True
        return False

    def _neighbor_dirs(self, pos):
        """
        Return list of directions to valid neighbors from pos.
        """
        dirs = []
        for mv in [Move.UP, Move.DOWN, Move.RIGHT, Move.LEFT]:
            nxt = self._apply_move(pos, mv)
            if self._is_valid(nxt):
                dirs.append(mv)
        return dirs

    def _is_frontier(self, pos):
        """Return True if `pos` is a free cell adjacent to any unknown cell."""
        r, c = pos
        if not (0 <= r < self.map_size and 0 <= c < self.map_size):
            return False
        if self.known_map[r, c] != 0:
            return False
        for mv in [Move.UP, Move.DOWN, Move.RIGHT, Move.LEFT]:
            nr, nc = self._apply_move(pos, mv)
            if 0 <= nr < self.map_size and 0 <= nc < self.map_size:
                if self.known_map[nr, nc] == -1:
                    return True
        return False

    def _are_opposite(self, d1, d2):
        return ((d1 == Move.UP and d2 == Move.DOWN) or
                (d1 == Move.DOWN and d2 == Move.UP) or
                (d1 == Move.LEFT and d2 == Move.RIGHT) or
                (d1 == Move.RIGHT and d2 == Move.LEFT))

    def _corridor_len(self, pos, dir_move):
        """
        Look ahead up to 10 cells in dir_move; stop if blocked or when a turn option appears (deg >= 2).
        """
        length = 0
        cur = pos
        for _ in range(10):
            nxt = self._apply_move(cur, dir_move)
            if not self._is_valid(nxt):
                break
            length += 1
            if self.observed_degree[nxt] >= 2:
                break
            cur = nxt
        return length

    def _a_star_first_step(self, start, goal):
        """
        A* on known free cells to goal. Return the first move or None if unreachable.
        """
        if start == goal or goal is None or not self._is_valid(goal):
            return None
        frontier = []
        heapq.heappush(frontier, (self._manhattan(start, goal), 0, start, None))
        g = {start: 0}
        while frontier:
            f, cost, cur, first_mv = heapq.heappop(frontier)
            if cur == goal:
                return first_mv if first_mv is not None else Move.STAY
            for mv in [Move.UP, Move.DOWN, Move.RIGHT, Move.LEFT]:
                nxt = self._apply_move(cur, mv)
                if not self._is_valid(nxt):
                    continue
                new_g = cost + 1
                if nxt not in g or new_g < g[nxt]:
                    g[nxt] = new_g
                    h = self._manhattan(nxt, goal)
                    heapq.heappush(frontier, (new_g + h, new_g, nxt, mv if cur == start else first_mv))
        return None

    def _manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _approach_dir(self, pac_pos, ghost_pos):
        """
        Dominant-axis direction from Pacman to Ghost (approximate approach vector).
        """
        dr = ghost_pos[0] - pac_pos[0]
        dc = ghost_pos[1] - pac_pos[1]
        if abs(dr) >= abs(dc):
            if dr > 0:  return Move.DOWN
            if dr < 0:  return Move.UP
            return Move.RIGHT if dc > 0 else Move.LEFT
        else:
            if dc > 0:  return Move.RIGHT
            if dc < 0:  return Move.LEFT
            return Move.DOWN if dr > 0 else Move.UP

    def _perpendiculars(self, dir_move):
        if dir_move in [Move.UP, Move.DOWN]:
            return [Move.LEFT, Move.RIGHT]
        if dir_move in [Move.LEFT, Move.RIGHT]:
            return [Move.UP, Move.DOWN]
        return [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]


