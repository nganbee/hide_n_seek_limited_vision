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
# PACMAN AGENT (SEEK) - Fixed Oscillation Issues
# =====================================================
class PacmanAgent(BasePacmanAgent):

    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        
        # Memory
        self.global_map = np.full((21, 21), UNKNOWN, dtype=int)
        self.last_known_ghost = None
        self.ghost_history = []
        
        # Committed path and target
        self.committed_path = []
        self.target_position = None
        self.steps_on_current_path = 0
        self.min_commitment_steps = 3  # Minimum steps before reconsidering
        
    def step(self, map_state, my_position, enemy_position, step_number):
        """Main decision logic with single decision point"""
        
        # Update map
        self._update_global_map(map_state)
        
        # Update Ghost tracking
        ghost_moved_significantly = False
        if enemy_position is not None:
            if self.last_known_ghost is not None:
                dist = abs(enemy_position[0] - self.last_known_ghost[0]) + \
                       abs(enemy_position[1] - self.last_known_ghost[1])
                ghost_moved_significantly = dist > 3
            
            self.last_known_ghost = enemy_position
            self.ghost_history.append(enemy_position)
            if len(self.ghost_history) > 10:
                self.ghost_history.pop(0)
        
        # Check if we should replan
        should_replan = (
            not self.committed_path or  # No current path
            ghost_moved_significantly or  # Ghost moved significantly
            self.steps_on_current_path >= 10 or  # Path is stale
            (self.target_position and enemy_position and 
             self.target_position != enemy_position and
             self.steps_on_current_path >= self.min_commitment_steps)  # Target changed
        )
        
        # Continue on committed path if we shouldn't replan
        if not should_replan and self.committed_path:
            self.steps_on_current_path += 1
            return self._execute_committed_path(my_position)
        
        # Need to create new plan
        self.steps_on_current_path = 0
        
        if enemy_position is not None:
            # Direct pursuit
            self.target_position = enemy_position
            return self._create_pursuit_plan(my_position, enemy_position)
        
        if self.last_known_ghost is not None:
            # Search last known area
            self.target_position = self.last_known_ghost
            return self._create_search_plan(my_position, self.last_known_ghost)
        
        # Explore systematically
        self.target_position = None
        return self._create_exploration_plan(my_position)
    
    # ===== MAP UPDATE =====
    
    def _update_global_map(self, local_map):
        """Merge observation into global memory"""
        for i in range(21):
            for j in range(21):
                if local_map[i, j] != UNKNOWN:
                    self.global_map[i, j] = local_map[i, j]
    
    # ===== PLAN CREATION =====
    
    def _create_pursuit_plan(self, my_pos, ghost_pos):
        """Create committed pursuit plan"""
        path = self._astar(my_pos, ghost_pos)
        
        if path and len(path) > 1:
            self.committed_path = path[1:]  # Exclude current position
            return self._execute_committed_path(my_pos)
        
        # Fallback: greedy move
        return self._greedy_move(my_pos, ghost_pos)
    
    def _create_search_plan(self, my_pos, last_known):
        """Create search plan around last known position"""
        # Find best search target in area
        search_target = self._find_search_target(my_pos, last_known)
        
        if search_target:
            path = self._astar(my_pos, search_target)
            if path and len(path) > 1:
                self.committed_path = path[1:]
                return self._execute_committed_path(my_pos)
        
        return self._greedy_move(my_pos, last_known)
    
    def _create_exploration_plan(self, my_pos):
        """Create systematic exploration plan"""
        target = self._find_closest_unknown_cluster(my_pos)
        
        if target:
            path = self._astar(my_pos, target)
            if path and len(path) > 1:
                self.committed_path = path[1:]
                return self._execute_committed_path(my_pos)
        
        # Random valid move as fallback
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if self._is_valid_move(my_pos, move):
                return (move, 1)
        
        return (Move.STAY, 1)
    
    def _find_search_target(self, my_pos, last_known):
        """Find best position to search near last known ghost location"""
        search_radius = 6
        best_score = -1
        best_target = None
        
        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                x, y = last_known[0] + dx, last_known[1] + dy
                
                if not (0 <= x < 21 and 0 <= y < 21):
                    continue
                
                if self.global_map[x, y] != EMPTY:
                    continue
                
                # Score based on distance from last known and from Pacman
                dist_from_last = abs(dx) + abs(dy)
                dist_from_pacman = abs(x - my_pos[0]) + abs(y - my_pos[1])
                
                if dist_from_last > search_radius or dist_from_pacman == 0:
                    continue
                
                # Prefer positions slightly away from last known
                score = 10 - dist_from_last - dist_from_pacman * 0.1
                
                if score > best_score:
                    best_score = score
                    best_target = (x, y)
        
        return best_target
    
    def _find_closest_unknown_cluster(self, my_pos):
        """Find closest unknown area"""
        min_dist = float('inf')
        target = None
        
        for x in range(21):
            for y in range(21):
                if self.global_map[x, y] == UNKNOWN:
                    # Count unknown neighbors
                    unknown_neighbors = sum(
                        1 for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]
                        if 0 <= x+dx < 21 and 0 <= y+dy < 21
                        and self.global_map[x+dx, y+dy] == UNKNOWN
                    )
                    
                    if unknown_neighbors >= 2:
                        dist = abs(x - my_pos[0]) + abs(y - my_pos[1])
                        if dist < min_dist:
                            min_dist = dist
                            target = (x, y)
        
        return target
    
    # ===== PATH EXECUTION =====
    
    def _execute_committed_path(self, my_pos):
        """Execute committed path with proper multi-step handling"""
        
        if not self.committed_path:
            return (Move.STAY, 1)
        
        # Remove any positions we've already reached
        while self.committed_path and self.committed_path[0] == my_pos:
            self.committed_path.pop(0)
        
        if not self.committed_path:
            return (Move.STAY, 1)
        
        # Determine move direction from current position to next position
        next_pos = self.committed_path[0]
        dx = next_pos[0] - my_pos[0]
        dy = next_pos[1] - my_pos[1]
        
        # Find the move direction
        move_direction = None
        if dx > 0 and dy == 0:
            move_direction = Move.DOWN
        elif dx < 0 and dy == 0:
            move_direction = Move.UP
        elif dy > 0 and dx == 0:
            move_direction = Move.RIGHT
        elif dy < 0 and dx == 0:
            move_direction = Move.LEFT
        
        if move_direction is None:
            # Path is invalid, clear it
            self.committed_path = []
            return (Move.STAY, 1)
        
        # Count consecutive steps in same direction along path
        move_dx, move_dy = move_direction.value
        steps = 0
        current_pos = my_pos
        
        for i in range(min(len(self.committed_path), self.pacman_speed)):
            expected_next = (current_pos[0] + move_dx, current_pos[1] + move_dy)
            
            # Check if path continues in same direction
            if i < len(self.committed_path) and self.committed_path[i] == expected_next:
                if self._is_valid_position(expected_next):
                    steps += 1
                    current_pos = expected_next
                else:
                    break
            else:
                break
        
        steps = max(1, steps)
        
        # Remove consumed positions from path
        self.committed_path = self.committed_path[steps:]
        
        return (move_direction, steps)
    
    # ===== PATHFINDING =====
    
    def _astar(self, start, goal):
        """A* pathfinding"""
        from heapq import heappush, heappop
        
        frontier = []
        heappush(frontier, (0, start, [start]))
        visited = {start: 0}
        
        max_iterations = 500  # Prevent infinite loops
        iterations = 0
        
        while frontier and iterations < max_iterations:
            iterations += 1
            f_cost, current, path = heappop(frontier)
            
            if current == goal:
                return path
            
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dx, dy = move.value
                neighbor = (current[0] + dx, current[1] + dy)
                
                if not self._is_valid_position(neighbor):
                    continue
                
                g_cost = len(path)
                
                if neighbor not in visited or g_cost < visited[neighbor]:
                    visited[neighbor] = g_cost
                    h_cost = abs(neighbor[0] - goal[0]) + abs(neighbor[1] - goal[1])
                    f = g_cost + h_cost
                    heappush(frontier, (f, neighbor, path + [neighbor]))
        
        return []
    
    def _greedy_move(self, start, goal):
        """Greedy movement towards goal"""
        dx = goal[0] - start[0]
        dy = goal[1] - start[1]
        
        # Prefer the axis with larger distance
        moves_to_try = []
        if abs(dx) > abs(dy):
            moves_to_try = [
                Move.DOWN if dx > 0 else Move.UP,
                Move.RIGHT if dy > 0 else Move.LEFT
            ]
        else:
            moves_to_try = [
                Move.RIGHT if dy > 0 else Move.LEFT,
                Move.DOWN if dx > 0 else Move.UP
            ]
        
        # Try preferred moves first
        for move in moves_to_try:
            steps = self._max_valid_steps(start, move, self.pacman_speed)
            if steps > 0:
                return (move, steps)
        
        # Try all other moves
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if move not in moves_to_try:
                steps = self._max_valid_steps(start, move, self.pacman_speed)
                if steps > 0:
                    return (move, steps)
        
        return (Move.STAY, 1)
    
    # ===== HELPERS =====
    
    def _is_valid_move(self, pos, move):
        """Check if move is valid"""
        dx, dy = move.value
        new_pos = (pos[0] + dx, pos[1] + dy)
        return self._is_valid_position(new_pos)
    
    def _is_valid_position(self, pos):
        """Check if position is valid"""
        x, y = pos
        if not (0 <= x < 21 and 0 <= y < 21):
            return False
        return self.global_map[x, y] == EMPTY
    
    def _max_valid_steps(self, pos, move, max_steps):
        """Calculate maximum valid steps in direction"""
        steps = 0
        current = pos
        dx, dy = move.value
        
        for _ in range(max_steps):
            next_pos = (current[0] + dx, current[1] + dy)
            if not self._is_valid_position(next_pos):
                break
            steps += 1
            current = next_pos
        
        return steps