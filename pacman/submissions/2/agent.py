"""
Bayesian Network Agents for Pacman Game.
Centroid Targeting with Entropy Switching (CTES) version.
This module implements Pacman and Ghost agents using Bayesian filtering
to track and predict enemy positions under limited visibility.
"""

import sys
import numpy as np
from collections import deque
from pathlib import Path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))
from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move

class BayesianFilter:
    """
    A Bayesian filter for tracking enemy positions on a grid.
    
    Uses a predict-update cycle:
    - Predict: Diffuses probability based on possible enemy movements
    - Update: Incorporates new observations to refine beliefs
    """
    
    def __init__(self, map_shape, walls):
        """
        Initialize the Bayesian filter with uniform prior distribution.
        
        Args:
            map_shape: Tuple (rows, cols) representing grid dimensions
            walls: Binary mask where 1 indicates wall, 0 indicates free space
        """
        self.rows, self.cols = map_shape
        self.walls = walls
        
        # Start with uniform distribution over non-wall cells
        self.belief = np.ones(map_shape)
        self.belief[self.walls == 1] = 0
        self.normalize()

    def normalize(self):
        """Normalize belief distribution to sum to 1.0"""
        total = np.sum(self.belief)
        if total > 0:
            self.belief /= total
        else:
            # Reset to uniform if belief collapses
            self.belief = np.ones((self.rows, self.cols))
            self.belief[self.walls == 1] = 0
            self.belief /= np.sum(self.belief)

    def predict(self):
        """
        Prediction step: Diffuse probability to adjacent cells.
        
        Models enemy movement uncertainty by spreading belief to neighboring
        cells (up, down, left, right, stay) with equal probability.
        """
        # Apply diffusion kernel: enemy can move to any adjacent cell or stay
        up = np.roll(self.belief, -1, axis=0)
        down = np.roll(self.belief, 1, axis=0)
        left = np.roll(self.belief, -1, axis=1)
        right = np.roll(self.belief, 1, axis=1)
        stay = self.belief
        
        new_belief = (up + down + left + right + stay) * 0.2
        new_belief[self.walls == 1] = 0
        
        self.belief = new_belief
        self.normalize()

    def update(self, visible_mask, enemy_pos=None):
        """
        Update step: Incorporate observation to refine belief.
        
        Args:
            visible_mask: Boolean array indicating visible cells
            enemy_pos: Tuple (r, c) if enemy is seen, None otherwise
        """
        if enemy_pos is not None:
            # Direct observation: collapse belief to observed position
            self.belief.fill(0)
            self.belief[enemy_pos] = 1.0
        else:
            # Negative observation: eliminate visible cells from belief
            self.belief[visible_mask] = 0
            self.normalize()
            
    def get_most_likely_position(self):
        """Return the grid position with highest probability."""
        flat_idx = np.argmax(self.belief)
        return np.unravel_index(flat_idx, self.belief.shape)


class PacmanAgent(BasePacmanAgent):
    """
    Improved Pacman Agent with Centroid-to-Peak Hunting and Mode Locking.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.filter = None
        self.map_walls = None
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.name = "Bayesian Smart"
        
        # State tracking
        self.unseen_cells = None
        self.mode = "hunt"  # 'hunt' or 'explore'
        self.mode_step_counter = 0  # Lock the mode for N steps
        self.last_target = None

    def get_centroid(self):
        """Calculate the center of mass of the probability distribution."""
        rows, cols = np.indices(self.filter.belief.shape)
        total_prob = np.sum(self.filter.belief)
        if total_prob == 0: return self.filter.get_most_likely_position()
        
        avg_r = np.sum(rows * self.filter.belief) / total_prob
        avg_c = np.sum(cols * self.filter.belief) / total_prob
        return int(round(avg_r)), int(round(avg_c))

    def step(self, map_state, my_pos, enemy_pos, step_num):
        # 1. Initialization
        if self.filter is None:
            self.map_walls = (map_state == 1).astype(int)
            self.filter = BayesianFilter(map_state.shape, self.map_walls)
            self.unseen_cells = np.ones(map_state.shape, dtype=int)
            self.unseen_cells[self.map_walls == 1] = 0

        # 2. Update Belief
        self.filter.predict()
        visible_mask = (map_state != -1) & (map_state != 1)
        self.filter.update(visible_mask, enemy_pos)
        self.unseen_cells[visible_mask] = 0  # Clear seen cells

        # 3. Decision Logic
        target = my_pos
        
        # CASE A: Ghost is visible - CHASE
        if enemy_pos:
            target = enemy_pos
            self.mode_step_counter = 0 # Reset lock
            # Reset exploration memory since we found him
            self.unseen_cells = np.ones(map_state.shape, dtype=int)
            self.unseen_cells[self.map_walls == 1] = 0
            
        # CASE B: Ghost is hidden - STRATEGIZE
        else:
            # Manage Mode Switching (Hysteresis)
            # Only switch modes if the lock has expired (counter <= 0)
            if self.mode_step_counter <= 0:
                max_belief = np.max(self.filter.belief)
                
                # If we are confused (low probability peak), Explore.
                # If we are confident (high probability peak), Hunt.
                if max_belief < 0.05:
                    self.mode = "explore"
                    self.mode_step_counter = 20  # Lock explore for 20 steps
                else:
                    self.mode = "hunt"
                    self.mode_step_counter = 10  # Lock hunt for 10 steps
            else:
                self.mode_step_counter -= 1

            # Execute Mode
            if self.mode == "explore":
                target = self._get_nearest_unseen_cell(my_pos, map_state)
            else: # mode == "hunt"
                centroid = self.get_centroid()
                dist_to_centroid = abs(my_pos[0] - centroid[0]) + abs(my_pos[1] - centroid[1])
                
                # STRATEGY: Approach the zone (Centroid), then kill the target (Peak)
                # If we are far from the average location, go there.
                if dist_to_centroid > 5:
                    target = centroid
                else:
                    # We are in the zone, go to the specific highest probability cell
                    target = self.filter.get_most_likely_position()

        # 4. Anti-Stuck Mechanism
        # If target is wall or self, pick a random valid neighbor to force movement
        if target == my_pos or map_state[target] == 1:
             valid_neighbors = []
             for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                 nr, nc = my_pos[0]+dr, my_pos[1]+dc
                 if map_state[nr, nc] != 1:
                     valid_neighbors.append((nr, nc))
             if valid_neighbors:
                 # Pick neighbor closest to original target if possible, else random
                 target = valid_neighbors[np.random.choice(len(valid_neighbors))]

        return self._bfs_move(map_state, my_pos, target)

    def _bfs_move(self, map_state, start, goal):
        if start == goal: return Move.STAY, 1
        queue = [(start, [])]
        visited = {start}
        while queue:
            (curr, path) = queue.pop(0)
            if curr == goal:
                return path[0], self.pacman_speed
            
            for m, (dr, dc) in [(Move.UP, (-1,0)), (Move.DOWN, (1,0)), 
                                (Move.LEFT, (0,-1)), (Move.RIGHT, (0,1))]:
                nr, nc = curr[0]+dr, curr[1]+dc
                if (0 <= nr < map_state.shape[0] and 0 <= nc < map_state.shape[1] 
                    and map_state[nr, nc] != 1 and (nr, nc) not in visited):
                    visited.add((nr, nc))
                    queue.append(((nr, nc), path + [m]))
        return Move.STAY, 1

    def _get_nearest_unseen_cell(self, my_pos, map_state):
        rows, cols = map_state.shape
        unseen = np.argwhere(self.unseen_cells == 1)
        
        # If map is fully explored, reset exploration mask
        if len(unseen) == 0:
            self.unseen_cells = np.ones(map_state.shape, dtype=int)
            self.unseen_cells[self.map_walls == 1] = 0
            unseen = np.argwhere(self.unseen_cells == 1)

        # BFS to find nearest unseen that is NOT my_pos
        queue = [my_pos]
        visited = {my_pos}
        
        # Optimization: Create a set for O(1) lookup
        unseen_set = set(map(tuple, unseen))
        # Remove current position from targets to prevent STUCK bug
        if my_pos in unseen_set:
            unseen_set.remove(my_pos)
            # Also clear it in the mask so we don't try again immediately
            self.unseen_cells[my_pos] = 0 

        while queue:
            curr = queue.pop(0)
            if curr in unseen_set:
                return curr
            
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = curr[0]+dr, curr[1]+dc
                if (0 <= nr < rows and 0 <= nc < cols 
                    and map_state[nr, nc] != 1 and (nr, nc) not in visited):
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        
        # Fallback if unreachable
        return self.filter.get_most_likely_position()


# GhostAgent
class GhostAgent(BaseGhostAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Ghost BFS"

        self.last_known_enemy_pos = None

        self._last_move = Move.STAY
        self._recent_positions = []

        self.strict_unknown = bool(kwargs.get("strict_unknown", True))

    # Helpers
    def _in_bounds(self, pos, map_state) -> bool:
        r, c = pos
        h, w = map_state.shape
        return 0 <= r < h and 0 <= c < w

    def _is_valid_position(self, pos, map_state) -> bool:
        if not self._in_bounds(pos, map_state):
            return False
        r, c = pos
        if self.strict_unknown:
            return map_state[r, c] == 0
        return map_state[r, c] != 1

    def _apply_move(self, pos, move: Move):
        dr, dc = move.value
        return (pos[0] + dr, pos[1] + dc)

    def _get_neighbors_4(self, pos, map_state):
        out = []
        for mv in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            nxt = self._apply_move(pos, mv)
            if self._is_valid_position(nxt, map_state):
                out.append((nxt, mv))
        return out

    def _bfs_distance_map(self, map_state: np.ndarray, start_pos: tuple):
        """
        BFS distance from start_pos to all reachable cells.
        returns dict pos->dist
        """
        if start_pos is None:
            return {}

        if not self._is_valid_position(start_pos, map_state):
            return {}

        q = deque([start_pos])
        dist = {start_pos: 0}

        while q:
            cur = q.popleft()
            for nxt, _ in self._get_neighbors_4(cur, map_state):
                if nxt not in dist:
                    dist[nxt] = dist[cur] + 1
                    q.append(nxt)
        return dist

    def _inverse(self, mv: Move) -> Move:
        table = {
            Move.UP: Move.DOWN,
            Move.DOWN: Move.UP,
            Move.LEFT: Move.RIGHT,
            Move.RIGHT: Move.LEFT,
            Move.STAY: Move.STAY,
        }
        return table[mv]

    # Main step
    def step(self, map_state: np.ndarray, my_position: tuple, enemy_position: tuple, step_number: int) -> Move:
        # update memory
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
        threat = enemy_position or self.last_known_enemy_pos

        # build distance map from threat
        dist_from_threat = self._bfs_distance_map(map_state, threat)

        # collect possible moves (NO STAY unless stuck)
        candidates = []
        for mv in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            nxt = self._apply_move(my_position, mv)
            if self._is_valid_position(nxt, map_state):
                candidates.append((mv, nxt))

        if not candidates:
            self._last_move = Move.STAY
            return Move.STAY

        # anti-oscillation memory
        self._recent_positions.append(my_position)
        if len(self._recent_positions) > 10:
            self._recent_positions.pop(0)

        # If we don't know threat (can't see + no memory), just move to avoid looping
        if threat is None or not dist_from_threat:
            np.random.shuffle(candidates)
            inv = self._inverse(self._last_move)
            for mv, nxt in candidates:
                if mv == inv:
                    continue
                if nxt in self._recent_positions[-3:]:
                    continue
                self._last_move = mv
                return mv
            self._last_move = candidates[0][0]
            return candidates[0][0]

        # choose move that maximizes distance from threat (tie-break: avoid reverse + avoid recent)
        best_mv = None
        best_key = None

        inv = self._inverse(self._last_move)

        for mv, nxt in candidates:
            d = dist_from_threat.get(nxt, -1)  # if unreachable in our belief => -1

            rev_pen = 1 if mv == inv else 0
            recent_pen = 1 if nxt in self._recent_positions[-3:] else 0
            deg = len(self._get_neighbors_4(nxt, map_state))

            # maximize: distance, degree; minimize penalties
            key = (d, deg, -rev_pen, -recent_pen)

            if best_key is None or key > best_key:
                best_key = key
                best_mv = mv

        self._last_move = best_mv if best_mv is not None else candidates[0][0]
        return self._last_move