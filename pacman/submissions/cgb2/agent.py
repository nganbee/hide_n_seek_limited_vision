"""
Advanced Agent Implementation with Bayesian Estimation and Smart Tactics

Features:
- Ghost: Bayesian threat tracking, safe zone identification, strategic evasion
- Pacman: Pattern recognition, cut-off strategy, intelligent pursuit
"""

import sys
from pathlib import Path
import numpy as np
from collections import deque
import time

src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move

UNKNOWN = -1
EMPTY = 0
WALL = 1


from collections import deque, defaultdict
import numpy as np
from environment import Move
from agent_interface import GhostAgent as BaseGhostAgent



import numpy as np
import random
import time
from enum import Enum
from collections import deque



import numpy as np
import random
import time
import math
from enum import Enum
from collections import deque



class GhostAgent:
    def __init__(self, **kwargs):
        self.name = "Euclidean-Sequence Ghost"
        self.map_layout = np.full((21, 21), -1)
        self.last_pacman_pos = None
        self.last_seen_step = -999
        self.visit_history = {} 
        
        # Exact Move Sequence
        self.OPENING_SEQUENCE = [
            Move.RIGHT, Move.RIGHT, Move.RIGHT, Move.RIGHT,
            Move.UP, Move.UP,
            Move.LEFT, Move.LEFT,
            Move.UP, Move.UP,
            Move.RIGHT
        ]
        
    def _is_valid(self, pos):
        r, c = pos
        return 0 <= r < 21 and 0 <= c < 21 and self.map_layout[r, c] != 1

    def _euclidean_dist(self, pos1, pos2):
        """Calculates straight-line distance."""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def _evaluate_position(self, g_pos, p_pos, step_number):
        # Base: Maximize Euclidean distance
        dist = self._euclidean_dist(g_pos, p_pos)
        
        # Detention penalty to prevent jittering
        if g_pos in self.visit_history:
            recency = step_number - self.visit_history[g_pos]
            if recency < 12: dist -= (12 - recency) * 1.5
            
        return dist

    def _minimax(self, g_pos, p_pos, depth, is_ghost, alpha, beta, start_time, step_number):
        if time.time() - start_time > 0.8 or depth == 0 or g_pos == p_pos:
            return self._evaluate_position(g_pos, p_pos, step_number), Move.STAY

        moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]
        if is_ghost:
            val, best_m = float('-inf'), Move.STAY
            for m in moves:
                nxt = (g_pos[0]+m.value[0], g_pos[1]+m.value[1])
                if self._is_valid(nxt):
                    res, _ = self._minimax(nxt, p_pos, depth-1, False, alpha, beta, start_time, step_number)
                    if res > val: val, best_m = res, m
                    alpha = max(alpha, val)
                    if beta <= alpha: break
            return val, best_m
        else:
            # Pacman Speed 2 Simulation
            val = float('inf')
            for m in moves:
                nxt = (p_pos[0]+m.value[0], p_pos[1]+m.value[1])
                if self._is_valid(nxt):
                    res, _ = self._minimax(g_pos, nxt, depth-1, True, alpha, beta, start_time, step_number)
                    val = min(val, res)
                    beta = min(beta, val)
                    if beta <= alpha: break
            return val, Move.STAY

    def step(self, map_state, my_position, enemy_position, step_number):
        start_time = time.time()
        
        # 1. Update Map Memory
        for r in range(21):
            for c in range(21):
                if map_state[r, c] != -1: self.map_layout[r, c] = map_state[r, c]
        
        self.visit_history[my_position] = step_number

        # 2. Euclidean Tripwire Check
        is_critically_close = False
        if enemy_position:
            # We use Euclidean distance here to decide whether to break the lurk
            e_dist = self._euclidean_dist(my_position, enemy_position)
            if e_dist < 5.0:
                is_critically_close = True

        # PHASE 1: Execution of Sequence
        if not is_critically_close and step_number < len(self.OPENING_SEQUENCE):
            planned_move = self.OPENING_SEQUENCE[step_number]
            target_pos = (my_position[0] + planned_move.value[0], my_position[1] + planned_move.value[1])
            if self._is_valid(target_pos):
                return planned_move

        # PHASE 2: Parked State
        if not is_critically_close:
            return Move.STAY

        # PHASE 3: Escape (Triggered if Euclidean Distance < 5)
        _, move = self._minimax(my_position, enemy_position, 4, True, float('-inf'), float('inf'), start_time, step_number)
        
        if move == Move.STAY:
            valid_moves = []
            for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                nxt = (my_position[0]+m.value[0], my_position[1]+m.value[1])
                if self._is_valid(nxt):
                    d = self._euclidean_dist(nxt, enemy_position)
                    valid_moves.append((m, d))
            if valid_moves:
                valid_moves.sort(key=lambda x: x[1], reverse=True)
                return valid_moves[0][0]

        return move# =====================================================
# PACMAN AGENT (SEEK) — All Issues Fixed
# =====================================================

from collections import deque
import numpy as np

class PacmanAgent(BasePacmanAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))

        # Persistent global memory
        self.global_map = np.full((21, 21), UNKNOWN, dtype=int)

        # Ghost tracking
        self.last_known_ghost = None
        self.ghost_history = []
        self.ghost_last_seen_step = -999

        # Path commitment
        self.committed_path = []
        self.target_position = None
        self.steps_on_current_path = 0
        self.max_path_age = 20

        # Direction memory
        self.last_direction = None
        self.direction_persistence = 0

        # Search state
        self.search_area_center = None
        self.steps_in_search_area = 0
        self.max_search_area_steps = 15
        self.visited_in_search = set()

        # Anti-oscillation memory
        self.recent_positions = deque(maxlen=12)

        # Cost shaping
        self.REVISIT_PENALTY = 6
        self.UNKNOWN_PENALTY = 3

    # =====================================================
    # MAIN STEP
    # =====================================================

    def step(self, map_state, my_position, enemy_position, step_number):

        self.recent_positions.append(my_position)
        self._update_global_map(map_state)

        ghost_visible = enemy_position is not None
        ghost_moved_significantly = False

        if ghost_visible:
            # Check if ghost moved significantly
            if self.last_known_ghost is not None:
                dist = abs(enemy_position[0] - self.last_known_ghost[0]) + \
                       abs(enemy_position[1] - self.last_known_ghost[1])
                ghost_moved_significantly = dist > 5
            
            self.last_known_ghost = enemy_position
            self.ghost_last_seen_step = step_number
            self.ghost_history.append(enemy_position)
            self.ghost_history = self.ghost_history[-10:]

            # Clear search state when ghost found
            self.search_area_center = None
            self.steps_in_search_area = 0
            self.visited_in_search.clear()

        else:
            # Track search progress
            if self.search_area_center:
                dist = abs(my_position[0] - self.search_area_center[0]) + \
                       abs(my_position[1] - self.search_area_center[1])
                if dist <= 10:
                    self.steps_in_search_area += 1
                    self.visited_in_search.add(my_position)
                else:
                    self.search_area_center = None
                    self.steps_in_search_area = 0
                    self.visited_in_search.clear()

        ghost_staleness = step_number - self.ghost_last_seen_step
        stuck_searching = self.steps_in_search_area > self.max_search_area_steps
        
        # Check if ghost is dangerously close
        ghost_is_close = False
        if ghost_visible:
            dist_to_ghost = abs(enemy_position[0] - my_position[0]) + \
                           abs(enemy_position[1] - my_position[1])
            ghost_is_close = dist_to_ghost <= 8  # Within 8 tiles

        should_replan = (
            not self.committed_path or
            self.steps_on_current_path >= self.max_path_age or
            stuck_searching or
            ghost_is_close or  # Always replan when ghost is nearby!
            (ghost_visible and ghost_moved_significantly)
        )

        if not should_replan and self.committed_path:
            self.steps_on_current_path += 1
            return self._execute_committed_path(my_position)

        self.steps_on_current_path = 0

        if ghost_visible:
            return self._create_pursuit_plan(my_position, enemy_position)

        if self.last_known_ghost and ghost_staleness < 30 and not stuck_searching:
            if not self.search_area_center:
                self.search_area_center = self.last_known_ghost
            return self._create_search_plan(my_position)

        return self._create_exploration_plan(my_position)

    # =====================================================
    # MAP UPDATE
    # =====================================================

    def _update_global_map(self, local_map):
        for i in range(21):
            for j in range(21):
                if local_map[i, j] != UNKNOWN:
                    self.global_map[i, j] = local_map[i, j]

    # =====================================================
    # PLANNING
    # =====================================================

    def _create_pursuit_plan(self, my_pos, ghost_pos):
        path = self._astar(my_pos, ghost_pos)
        if path and len(path) > 1:
            self.committed_path = path[1:]
            return self._execute_committed_path(my_pos)
        return self._greedy_move(my_pos, ghost_pos)

    def _create_search_plan(self, my_pos):
        target = self._predict_ghost_position(self.last_known_ghost)
        if target:
            path = self._astar(my_pos, target)
            if path and len(path) > 1:
                self.committed_path = path[1:]
                return self._execute_committed_path(my_pos)

        target = self._find_unvisited_search_target(my_pos)
        if target:
            path = self._astar(my_pos, target)
            if path and len(path) > 1:
                self.committed_path = path[1:]
                return self._execute_committed_path(my_pos)

        # Give up on this search area
        self.steps_in_search_area = self.max_search_area_steps + 1
        return self._create_exploration_plan(my_pos)

    def _create_exploration_plan(self, my_pos):
        # First priority: frontier tiles (adjacent to unknown)
        target = self._find_frontier_tile(my_pos)
        
        if not target:
            # No frontiers - map fully explored, search distant areas
            target = self._find_farthest_empty_tile(my_pos)
        
        if target:
            path = self._astar(my_pos, target)
            if path and len(path) > 1:
                self.committed_path = path[1:]
                return self._execute_committed_path(my_pos)

        # Hard anti-jitter fallback - avoid recent positions
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if self._is_valid_move(my_pos, move):
                dx, dy = move.value
                nxt = (my_pos[0] + dx, my_pos[1] + dy)
                if nxt not in self.recent_positions:
                    return (move, 1)

        # Last resort - just move anywhere valid
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if self._is_valid_move(my_pos, move):
                return (move, 1)

        return (Move.STAY, 1)

    # =====================================================
    # FRONTIER EXPLORATION
    # =====================================================

    def _find_frontier_tile(self, my_pos):
        """Find closest empty tile adjacent to unknown areas"""
        best_score = float('inf')
        best = None

        for x in range(21):
            for y in range(21):
                if self.global_map[x, y] != EMPTY:
                    continue

                # Count unknown neighbors
                unknown_neighbors = 0
                for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < 21 and 0 <= ny < 21:
                        if self.global_map[nx, ny] == UNKNOWN:
                            unknown_neighbors += 1

                if unknown_neighbors == 0:
                    continue

                # Calculate distance
                dist = abs(x - my_pos[0]) + abs(y - my_pos[1])
                
                # FIXED: Distance is primary, unknown neighbors is tiebreaker
                # Lower distance = lower score = better
                # More unknown neighbors = lower score = better (as tiebreaker)
                score = dist * 100 - unknown_neighbors * 10
                
                # Penalty for recently visited tiles
                if (x, y) in self.recent_positions:
                    score += 50

                if score < best_score:
                    best_score = score
                    best = (x, y)

        return best

    def _find_farthest_empty_tile(self, my_pos):
        """Find empty tile farthest from current position"""
        max_dist = 0
        best = None
        
        for x in range(21):
            for y in range(21):
                if self.global_map[x, y] == EMPTY:
                    dist = abs(x - my_pos[0]) + abs(y - my_pos[1])
                    
                    # Bonus for not being recently visited
                    if (x, y) not in self.recent_positions:
                        dist += 2
                    
                    if dist > max_dist:
                        max_dist = dist
                        best = (x, y)
        
        return best

    # =====================================================
    # A* PATHFINDING (ANTI-LOOP)
    # =====================================================

    def _astar(self, start, goal):
        from heapq import heappush, heappop

        frontier = []
        heappush(frontier, (0, start, [start]))
        visited = {start: 0}

        max_iterations = 500
        iterations = 0

        while frontier and iterations < max_iterations:
            iterations += 1
            _, current, path = heappop(frontier)
            
            if current == goal:
                return path

            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dx, dy = move.value
                neighbor = (current[0] + dx, current[1] + dy)

                if not self._is_valid_position(neighbor):
                    continue

                g = len(path)

                # Penalize revisiting recent positions
                if neighbor in self.recent_positions:
                    g += self.REVISIT_PENALTY

                # Slight penalty for unknown tiles (prefer known safe paths)
                if self.global_map[neighbor[0], neighbor[1]] == UNKNOWN:
                    g += self.UNKNOWN_PENALTY

                if neighbor not in visited or g < visited[neighbor]:
                    visited[neighbor] = g
                    h = abs(neighbor[0] - goal[0]) + abs(neighbor[1] - goal[1])
                    heappush(frontier, (g + h, neighbor, path + [neighbor]))

        return []

    # =====================================================
    # EXECUTION
    # =====================================================

    def _execute_committed_path(self, my_pos):
        """Execute path with correct multi-step handling"""
        
        # Remove positions already reached
        while self.committed_path and self.committed_path[0] == my_pos:
            self.committed_path.pop(0)

        if not self.committed_path:
            return (Move.STAY, 1)

        # Determine move direction
        next_pos = self.committed_path[0]
        dx = next_pos[0] - my_pos[0]
        dy = next_pos[1] - my_pos[1]

        move = None
        if dx == 1 and dy == 0:
            move = Move.DOWN
        elif dx == -1 and dy == 0:
            move = Move.UP
        elif dy == 1 and dx == 0:
            move = Move.RIGHT
        elif dy == -1 and dx == 0:
            move = Move.LEFT

        if move is None:
            # Path is invalid
            self.committed_path = []
            return (Move.STAY, 1)

        # Count consecutive steps in same direction
        move_dx, move_dy = move.value
        steps = 0
        current_pos = my_pos

        for i in range(min(len(self.committed_path), self.pacman_speed)):
            expected_next = (current_pos[0] + move_dx, current_pos[1] + move_dy)
            
            # Verify path continues in same direction and is valid
            if i < len(self.committed_path) and self.committed_path[i] == expected_next:
                if self._is_valid_position(expected_next):
                    steps += 1
                    current_pos = expected_next
                else:
                    break
            else:
                break

        steps = max(1, steps)
        self.committed_path = self.committed_path[steps:]
        
        return (move, steps)

    # =====================================================
    # HELPERS
    # =====================================================

    def _predict_ghost_position(self, last):
        """Predict ghost position based on recent movement"""
        if len(self.ghost_history) < 2:
            return None
        
        # Use last two positions to calculate velocity
        dx = self.ghost_history[-1][0] - self.ghost_history[-2][0]
        dy = self.ghost_history[-1][1] - self.ghost_history[-2][1]
        
        # Predict 4 steps ahead
        pred = (last[0] + dx * 4, last[1] + dy * 4)
        pred = (max(0, min(20, pred[0])), max(0, min(20, pred[1])))
        
        if self.global_map[pred[0], pred[1]] != WALL:
            return pred
        return None

    def _find_unvisited_search_target(self, my_pos):
        """Find unvisited position in search area"""
        best = None
        best_score = -float('inf')
        
        for dx in range(-8, 9):
            for dy in range(-8, 9):
                x = self.search_area_center[0] + dx
                y = self.search_area_center[1] + dy
                
                if not (0 <= x < 21 and 0 <= y < 21):
                    continue
                if self.global_map[x, y] == WALL:
                    continue
                if (x, y) in self.visited_in_search:
                    continue
                
                # Prefer closer positions
                dist = abs(x - my_pos[0]) + abs(y - my_pos[1])
                score = -dist
                
                if score > best_score:
                    best_score = score
                    best = (x, y)
        
        return best

    def _is_valid_move(self, pos, move):
        dx, dy = move.value
        return self._is_valid_position((pos[0] + dx, pos[1] + dy))

    def _is_valid_position(self, pos):
        x, y = pos
        return 0 <= x < 21 and 0 <= y < 21 and self.global_map[x, y] != WALL

    def _max_valid_steps(self, pos, move, max_steps):
        dx, dy = move.value
        steps = 0
        cur = pos
        for _ in range(max_steps):
            nxt = (cur[0] + dx, cur[1] + dy)
            if not self._is_valid_position(nxt):
                break
            cur = nxt
            steps += 1
        return steps

    def _greedy_move(self, start, goal):
        """Greedy move toward goal"""
        dx = goal[0] - start[0]
        dy = goal[1] - start[1]
        
        prefs = []
        if abs(dx) > abs(dy):
            prefs.append(Move.DOWN if dx > 0 else Move.UP)
            prefs.append(Move.RIGHT if dy > 0 else Move.LEFT)
        else:
            prefs.append(Move.RIGHT if dy > 0 else Move.LEFT)
            prefs.append(Move.DOWN if dx > 0 else Move.UP)

        for m in prefs:
            if self._is_valid_move(start, m):
                return (m, 1)

        for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if self._is_valid_move(start, m):
                return (m, 1)

        return (Move.STAY, 1)