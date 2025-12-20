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


# =====================================================
# GHOST AGENT (HIDE) - Advanced Evasion
# =====================================================
class GhostAgent(BaseGhostAgent):
    """
    Ghost (Hider) - Advanced evasion with Bayesian threat estimation
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Memory
        self.global_map = np.full((21, 21), UNKNOWN, dtype=int)
        
        # Bayesian threat tracking
        self.danger_map = np.zeros((21, 21), dtype=float)
        self.last_seen_pacman = None
        self.steps_since_seen = 0
        self.pacman_history = []
        
        # Safe zones
        self.safe_zones_cache = []
        self.last_safe_update = -10
        
    def step(self, map_state, my_position, enemy_position, step_number):
        """Main decision logic"""
        start_time = time.time()
        
        # 1. Update map memory
        self._update_global_map(map_state)
        
        # 2. Get visible cells
        visible_cells = self._get_visible_cells(map_state, my_position)
        
        # 3. Update danger estimation
        self._update_danger_map(my_position, enemy_position, visible_cells)
        
        # 4. Track Pacman pattern
        if enemy_position is not None:
            self.pacman_history.append(enemy_position)
            if len(self.pacman_history) > 10:
                self.pacman_history.pop(0)
        
        # 5. Decision tree based on threat level
        if enemy_position is not None:
            # EMERGENCY: Visible threat
            return self._emergency_escape(my_position, enemy_position)
        
        if self._is_high_danger(my_position):
            # HIGH DANGER: Defensive move
            return self._defensive_move(my_position)
        
        # SAFE: Strategic positioning
        return self._strategic_move(my_position, step_number)
    
    # ===== MAP & OBSERVATION =====
    
    def _update_global_map(self, local_map):
        """Merge observation into global memory"""
        for i in range(21):
            for j in range(21):
                if local_map[i, j] != UNKNOWN:
                    self.global_map[i, j] = local_map[i, j]
    
    def _get_visible_cells(self, obs, pos):
        """Extract visible cells from cross-shaped observation"""
        visible = [pos]
        x, y = pos
        
        # Four directions, max 5 steps each
        for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
            for step in range(1, 6):
                nx, ny = x + dx*step, y + dy*step
                if not (0 <= nx < 21 and 0 <= ny < 21):
                    break
                
                visible.append((nx, ny))
                
                # Stop at walls
                if obs[nx, ny] == WALL:
                    break
        
        return visible
    
    # ===== BAYESIAN THREAT ESTIMATION =====
    
    def _update_danger_map(self, my_pos, pacman_pos, visible_cells):
        """Update danger belief with Bayesian prediction"""
        
        # Temporal decay
        self.danger_map *= 0.7
        
        # Update tracking
        if pacman_pos is not None:
            self.last_seen_pacman = pacman_pos
            self.steps_since_seen = 0
        else:
            self.steps_since_seen += 1
        
        if self.last_seen_pacman is None:
            return
        
        px, py = self.last_seen_pacman
        
        # Predict Pacman movement TOWARDS Ghost
        max_reachable = self.steps_since_seen + 5
        
        for x in range(max(0, px-max_reachable), min(21, px+max_reachable+1)):
            for y in range(max(0, py-max_reachable), min(21, py+max_reachable+1)):
                if self.global_map[x, y] == WALL:
                    continue
                
                # Distance from last seen
                d_from_last = abs(x - px) + abs(y - py)
                
                if d_from_last > max_reachable:
                    continue
                
                # Distance to Ghost (Pacman chases Ghost)
                d_to_ghost = abs(x - my_pos[0]) + abs(y - my_pos[1])
                
                # Closer to Ghost = Higher threat
                threat = 1.0 / (d_to_ghost + 1)
                
                # Reachability factor
                reachability = 1.0 - (d_from_last / max_reachable)
                
                self.danger_map[x, y] += threat * reachability
        
        # Zero out visible cells (Pacman NOT there)
        for x, y in visible_cells:
            self.danger_map[x, y] = 0
    
    def _is_high_danger(self, pos):
        """Check if current position is dangerous"""
        return self.danger_map[pos] > 0.5
    
    # ===== MOVE STRATEGIES =====
    
    def _emergency_escape(self, my_pos, pacman_pos):
        """Immediate escape from visible Pacman"""
        best_move = Move.STAY
        best_score = -1e9
        
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            new_pos = self._apply_move(my_pos, move)
            if not self._is_walkable(new_pos):
                continue
            
            # Distance from Pacman
            dist = abs(new_pos[0] - pacman_pos[0]) + abs(new_pos[1] - pacman_pos[1])
            
            # Number of exits (avoid dead-ends)
            exits = self._count_exits(new_pos)
            
            # Score
            score = dist * 20 + exits * 10
            
            # Heavy penalty for dead-ends
            if exits <= 1:
                score -= 100
            
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move
    
    def _defensive_move(self, my_pos):
        """Move away from danger zones"""
        best_move = Move.STAY
        best_score = -1e9
        
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            new_pos = self._apply_move(my_pos, move)
            if not self._is_walkable(new_pos):
                continue
            
            score = self._evaluate_position(new_pos)
            
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move
    
    def _strategic_move(self, my_pos, step_num):
        """Long-term strategic positioning"""
        
        # Update safe zones periodically
        if step_num - self.last_safe_update > 10:
            self.safe_zones_cache = self._find_safe_zones(my_pos)
            self.last_safe_update = step_num
        
        # Already in safe zone? Explore or stay
        if self.safe_zones_cache and my_pos in self.safe_zones_cache[:3]:
            return self._explore_unknown(my_pos)
        
        # Move towards safe zone
        if self.safe_zones_cache:
            target = self.safe_zones_cache[0]
            return self._move_towards(my_pos, target)
        
        # Fallback: move to safest neighbor
        return self._defensive_move(my_pos)
    
    # ===== POSITION EVALUATION =====
    
    def _evaluate_position(self, pos):
        """Comprehensive position scoring"""
        x, y = pos
        
        danger = self.danger_map[x, y]
        exits = self._count_exits(pos)
        
        score = 0
        
        # Core factors
        score -= danger * 50
        score += exits * 10
        
        # Distance to highest danger
        if self.danger_map.max() > 0:
            max_danger_pos = np.unravel_index(
                self.danger_map.argmax(), 
                self.danger_map.shape
            )
            dist_danger = abs(x - max_danger_pos[0]) + abs(y - max_danger_pos[1])
            score += dist_danger * 3
        
        # Depth in map
        depth = min(x, 20-x, y, 20-y)
        score += depth * 2
        
        # Unknown cells nearby (exploration value)
        unknown = 0
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                nx, ny = x + dx, y + dy
                if 0 <= nx < 21 and 0 <= ny < 21:
                    if self.global_map[nx, ny] == UNKNOWN:
                        unknown += 1
        score += unknown * 1.5
        
        # Penalties
        if exits <= 1:
            score -= 100
        
        # Walls nearby (cover)
        walls = sum(
            1 for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]
            if 0 <= x+dx < 21 and 0 <= y+dy < 21 
            and self.global_map[x+dx, y+dy] == WALL
        )
        score += walls * 3
        
        return score
    
    def _find_safe_zones(self, my_pos):
        """Find top safe positions"""
        candidates = []
        
        for x in range(21):
            for y in range(21):
                if self.global_map[x, y] == EMPTY:
                    score = self._evaluate_position((x, y))
                    candidates.append(((x, y), score))
        
        candidates.sort(key=lambda item: item[1], reverse=True)
        return [pos for pos, _ in candidates[:5]]
    
    # ===== HELPERS =====
    
    def _apply_move(self, pos, move):
        """Apply move to position"""
        dx, dy = move.value
        return (pos[0] + dx, pos[1] + dy)
    
    def _is_walkable(self, pos):
        """Check if position is walkable"""
        x, y = pos
        if not (0 <= x < 21 and 0 <= y < 21):
            return False
        return self.global_map[x, y] == EMPTY
    
    def _count_exits(self, pos):
        """Count available exits from position"""
        count = 0
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            new_pos = self._apply_move(pos, move)
            if self._is_walkable(new_pos):
                count += 1
        return count
    
    def _explore_unknown(self, my_pos):
        """Move towards unknown areas"""
        best_move = Move.STAY
        max_unknown = -1
        
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            new_pos = self._apply_move(my_pos, move)
            if not self._is_walkable(new_pos):
                continue
            
            # Count unknown in direction
            unknown = 0
            dx, dy = move.value
            for step in range(1, 6):
                check_pos = (new_pos[0] + dx*step, new_pos[1] + dy*step)
                if 0 <= check_pos[0] < 21 and 0 <= check_pos[1] < 21:
                    if self.global_map[check_pos] == UNKNOWN:
                        unknown += 1
            
            if unknown > max_unknown:
                max_unknown = unknown
                best_move = move
        
        return best_move
    
    def _move_towards(self, start, goal):
        """Simple pathfinding towards goal"""
        path = self._bfs(start, goal)
        if path and len(path) > 1:
            next_pos = path[1]
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                if self._apply_move(start, move) == next_pos:
                    return move
        
        # Fallback: greedy
        dx = goal[0] - start[0]
        dy = goal[1] - start[1]
        
        if abs(dx) > abs(dy):
            move = Move.DOWN if dx > 0 else Move.UP
        else:
            move = Move.RIGHT if dy > 0 else Move.LEFT
        
        if self._is_walkable(self._apply_move(start, move)):
            return move
        
        return Move.STAY
    
    def _bfs(self, start, goal):
        """BFS pathfinding"""
        queue = deque([(start, [start])])
        visited = {start}
        
        while queue:
            pos, path = queue.popleft()
            
            if pos == goal:
                return path
            
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                new_pos = self._apply_move(pos, move)
                if new_pos not in visited and self._is_walkable(new_pos):
                    visited.add(new_pos)
                    queue.append((new_pos, path + [new_pos]))
        
        return []


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