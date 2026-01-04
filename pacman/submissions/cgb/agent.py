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



class GhostAgent(BaseGhostAgent):
    """
    Hiding Agent – Turn-first with tabu memory

    Fixes:
    - Corridor re-entry
    - Turn oscillation
    - Directional backtracking
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.global_map = np.full((21, 21), UNKNOWN, dtype=int)

        self.recent_positions = deque(maxlen=6)
        self.tabu_positions = deque(maxlen=5)   # <<< NEW

        self.bad_directions = defaultdict(int)
        self.last_move = None

        self.last_seen_pacman = None
        self.steps_since_seen = 999

    # =====================================================
    # MAIN STEP
    # =====================================================
    def step(self, map_state, my_pos, enemy_pos, step_number):

        self._update_map(map_state)
        self._decay_bad_directions()

        self.recent_positions.append(my_pos)
        self.tabu_positions.append(my_pos)   # <<< track tabu

        if enemy_pos is not None:
            self.last_seen_pacman = enemy_pos
            self.steps_since_seen = 0
        else:
            self.steps_since_seen += 1

        if enemy_pos:
            move = self._choose_move(my_pos, enemy_pos)
        elif self.last_seen_pacman and self.steps_since_seen <= 6:
            move = self._choose_move(my_pos, self.last_seen_pacman)
        else:
            move = self._choose_move(my_pos, None)

        self.last_move = move
        return move

    # =====================================================
    # MOVE SELECTION (TABU-AWARE)
    # =====================================================
    def _choose_move(self, my_pos, pacman_pos):

        legal = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            nxt = self._apply(my_pos, move)
            if self._walkable(nxt):
                legal.append((move, nxt))

        if not legal:
            return Move.STAY
        
        # === HARD TURN COMMITMENT ===
        if self.last_move is not None:
            turning_moves = [
                (m, p) for m, p in legal
                if self._is_turn_move(m) and p not in self.tabu_positions
            ]

            if turning_moves:
                # ONLY consider turning moves
                legal = turning_moves

        # --- HARD TABU FILTER ---
        non_tabu = [(m, p) for m, p in legal if p not in self.tabu_positions]
        candidates = non_tabu if non_tabu else legal

        # --- Tiering ---
        turn_moves = []
        intersection_moves = []
        safe_moves = []
        fallback_moves = []

        for move, nxt in candidates:
            if self._is_turn_tile(nxt):
                turn_moves.append((move, nxt))
            elif self._exits(nxt) >= 3:
                intersection_moves.append((move, nxt))
            elif not self._leads_to_long_corridor(nxt, move):
                safe_moves.append((move, nxt))
            else:
                fallback_moves.append((move, nxt))

        for tier in [turn_moves, intersection_moves, safe_moves, fallback_moves]:
            if tier:
                return self._score_and_choose(my_pos, pacman_pos, tier)

        return Move.STAY

    # =====================================================
    # SCORING
    # =====================================================
    def _score_and_choose(self, my_pos, pacman_pos, tier):

        best_move = Move.STAY
        best_score = -1e9

        for move, nxt in tier:
            score = 0

            if (my_pos, move) in self.bad_directions:
                score -= 80

            if pacman_pos:
                d0 = self._manhattan(my_pos, pacman_pos)
                d1 = self._manhattan(nxt, pacman_pos)
                score += (d1 - d0) * 30

                if nxt[0] != pacman_pos[0] and nxt[1] != pacman_pos[1]:
                    score += 25

            score += self._exits(nxt) * 15

            if nxt in self.recent_positions:
                score -= 20

            if self._leads_to_long_corridor(nxt, move):
                score -= 60
                self.bad_directions[(my_pos, move)] = 6

            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    # =====================================================
    # MAP
    # =====================================================
    def _update_map(self, obs):
        for i in range(21):
            for j in range(21):
                if obs[i, j] != UNKNOWN:
                    self.global_map[i, j] = obs[i, j]

    def _decay_bad_directions(self):
        for k in list(self.bad_directions.keys()):
            self.bad_directions[k] -= 1
            if self.bad_directions[k] <= 0:
                del self.bad_directions[k]

    # =====================================================
    # STRUCTURE DETECTION
    # =====================================================
    def _is_turn_tile(self, pos):
        if self._exits(pos) != 2:
            return False
        moves = [m for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
                 if self._walkable(self._apply(pos, m))]
        dx1, dy1 = moves[0].value
        dx2, dy2 = moves[1].value
        return not (dx1 == -dx2 and dy1 == -dy2)

    def _leads_to_long_corridor(self, start, move):
        dx, dy = move.value
        cur = start
        for _ in range(10):
            if not self._walkable(cur):
                return False
            if self._exits(cur) >= 3:
                return False
            cur = (cur[0] + dx, cur[1] + dy)
        return True

    # =====================================================
    # HELPERS
    # =====================================================
    def _apply(self, pos, move):
        dx, dy = move.value
        return (pos[0] + dx, pos[1] + dy)

    def _walkable(self, pos):
        x, y = pos
        if not (0 <= x < 21 and 0 <= y < 21):
            return False
        return self.global_map[x, y] == EMPTY

    def _exits(self, pos):
        return sum(
            self._walkable(self._apply(pos, m))
            for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        )

    def _manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _is_turn_move(self, move):
        if self.last_move is None:
            return False
        dx1, dy1 = self.last_move.value
        dx2, dy2 = move.value
        return not (dx1 == dx2 and dy1 == dy2)

# =====================================================
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