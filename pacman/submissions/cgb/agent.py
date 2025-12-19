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
# PACMAN AGENT (SEEK) - Intelligent Pursuit
# =====================================================
class PacmanAgent(BasePacmanAgent):
    """
    Pacman (Seeker) - Intelligent pursuit with prediction
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        
        # Memory
        self.global_map = np.full((21, 21), UNKNOWN, dtype=int)
        self.last_known_ghost = None
        self.ghost_history = []
        
        # Probability map for Ghost location
        self.ghost_prob_map = np.zeros((21, 21), dtype=float)
        
    def step(self, map_state, my_position, enemy_position, step_number):
        """Main decision logic"""
        
        # Update map
        self._update_global_map(map_state)
        
        # Update Ghost tracking
        if enemy_position is not None:
            self.last_known_ghost = enemy_position
            self.ghost_history.append(enemy_position)
            if len(self.ghost_history) > 10:
                self.ghost_history.pop(0)
        
        # Update probability map
        self._update_ghost_probability(my_position, enemy_position)
        
        # Decision based on visibility
        if enemy_position is not None:
            # Direct pursuit with prediction
            return self._pursue_with_prediction(my_position, enemy_position)
        
        if self.last_known_ghost is not None:
            # Search last known area
            return self._search_area(my_position, self.last_known_ghost)
        
        # Explore
        return self._explore(my_position)
    
    # ===== MAP UPDATE =====
    
    def _update_global_map(self, local_map):
        """Merge observation into global memory"""
        for i in range(21):
            for j in range(21):
                if local_map[i, j] != UNKNOWN:
                    self.global_map[i, j] = local_map[i, j]
    
    # ===== GHOST PROBABILITY ESTIMATION =====
    
    def _update_ghost_probability(self, my_pos, ghost_pos):
        """Estimate Ghost location probability"""
        
        # Decay old beliefs
        self.ghost_prob_map *= 0.6
        
        if ghost_pos is not None:
            # Direct observation
            self.ghost_prob_map.fill(0)
            self.ghost_prob_map[ghost_pos] = 1.0
            return
        
        if self.last_known_ghost is None:
            return
        
        # Predict Ghost movement (away from Pacman)
        gx, gy = self.last_known_ghost
        
        for x in range(21):
            for y in range(21):
                if self.global_map[x, y] != EMPTY:
                    continue
                
                # Distance from last known
                d_from_last = abs(x - gx) + abs(y - gy)
                
                if d_from_last > 8:
                    continue
                
                # Distance from Pacman (Ghost runs away)
                d_from_pacman = abs(x - my_pos[0]) + abs(y - my_pos[1])
                
                # Further from Pacman = more likely
                prob = d_from_pacman / (d_from_last + 1)
                self.ghost_prob_map[x, y] += prob * 0.1
    
    # ===== PURSUIT STRATEGIES =====
    
    def _pursue_with_prediction(self, my_pos, ghost_pos):
        """Chase with cut-off prediction"""
        
        # Predict Ghost's escape direction
        predicted_pos = self._predict_ghost_position(ghost_pos)
        
        if predicted_pos:
            # Try to cut off
            cutoff_point = self._find_cutoff_point(my_pos, ghost_pos, predicted_pos)
            if cutoff_point:
                return self._move_towards_multi(my_pos, cutoff_point)
        
        # Direct chase
        return self._move_towards_multi(my_pos, ghost_pos)
    
    def _predict_ghost_position(self, ghost_pos):
        """Predict where Ghost will move"""
        if len(self.ghost_history) < 2:
            return None
        
        # Calculate movement vector
        prev_pos = self.ghost_history[-2]
        dx = ghost_pos[0] - prev_pos[0]
        dy = ghost_pos[1] - prev_pos[1]
        
        # Predict 2 steps ahead
        pred_x = ghost_pos[0] + dx * 2
        pred_y = ghost_pos[1] + dy * 2
        
        if 0 <= pred_x < 21 and 0 <= pred_y < 21:
            if self.global_map[pred_x, pred_y] == EMPTY:
                return (pred_x, pred_y)
        
        return None
    
    def _find_cutoff_point(self, my_pos, ghost_pos, predicted_pos):
        """Find optimal interception point"""
        # Path from Ghost to predicted
        path = self._bfs(ghost_pos, predicted_pos)
        
        if not path:
            return None
        
        # Find point where Pacman arrives first
        for i, point in enumerate(path):
            ghost_dist = i
            pacman_dist = abs(point[0] - my_pos[0]) + abs(point[1] - my_pos[1])
            
            if pacman_dist <= ghost_dist:
                return point
        
        return None
    
    def _search_area(self, my_pos, last_known):
        """Search around last known position"""
        
        # Find highest probability area
        max_prob_pos = np.unravel_index(
            self.ghost_prob_map.argmax(),
            self.ghost_prob_map.shape
        )
        
        if self.ghost_prob_map[max_prob_pos] > 0.1:
            return self._move_towards_multi(my_pos, max_prob_pos)
        
        # Move to last known
        return self._move_towards_multi(my_pos, last_known)
    
    def _explore(self, my_pos):
        """Explore unknown areas"""
        # Find closest unknown area
        min_dist = float('inf')
        target = None
        
        for x in range(21):
            for y in range(21):
                if self.global_map[x, y] == UNKNOWN:
                    dist = abs(x - my_pos[0]) + abs(y - my_pos[1])
                    if dist < min_dist:
                        min_dist = dist
                        target = (x, y)
        
        if target:
            return self._move_towards_multi(my_pos, target)
        
        # Random valid move
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if self._is_valid_move(my_pos, move):
                return (move, 1)
        
        return (Move.STAY, 1)
    
    # ===== PATHFINDING =====
    
    def _move_towards_multi(self, start, goal):
        """Multi-step pathfinding"""
        path = self._bfs(start, goal)
        
        if not path or len(path) <= 1:
            return self._greedy_move(start, goal)
        
        # Calculate how many steps we can take
        steps = min(len(path) - 1, self.pacman_speed)
        
        # Determine direction
        next_pos = path[1]
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dx, dy = move.value
            if (start[0] + dx, start[1] + dy) == next_pos:
                return (move, steps)
        
        return (Move.STAY, 1)
    
    def _greedy_move(self, start, goal):
        """Greedy movement towards goal"""
        dx = goal[0] - start[0]
        dy = goal[1] - start[1]
        
        if abs(dx) > abs(dy):
            move = Move.DOWN if dx > 0 else Move.UP
            desired_steps = abs(dx)
        else:
            move = Move.RIGHT if dy > 0 else Move.LEFT
            desired_steps = abs(dy)
        
        steps = self._max_valid_steps(start, move, min(desired_steps, self.pacman_speed))
        
        if steps > 0:
            return (move, steps)
        
        # Try other directions
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            steps = self._max_valid_steps(start, move, self.pacman_speed)
            if steps > 0:
                return (move, steps)
        
        return (Move.STAY, 1)
    
    def _bfs(self, start, goal):
        """BFS pathfinding"""
        queue = deque([(start, [start])])
        visited = {start}
        
        while queue:
            pos, path = queue.popleft()
            
            if pos == goal:
                return path
            
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dx, dy = move.value
                new_pos = (pos[0] + dx, pos[1] + dy)
                
                if new_pos not in visited and self._is_valid_position(new_pos):
                    visited.add(new_pos)
                    queue.append((new_pos, path + [new_pos]))
        
        return []
    
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