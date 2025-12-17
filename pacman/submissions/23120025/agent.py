"""
Compact agent implementation - optimized for brevity and performance
"""

import sys
from pathlib import Path

# Add src to path to import the interface
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np
import heapq
from collections import deque
import random


class PacmanAgent(BasePacmanAgent):
    """Pacman v8.0 Compact - Speed Demon Optimized"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Pacman v8.0 (Speed Demon Compact)"
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 2)))
        
        self.map_size = (21, 21)
        self.global_map = np.full(self.map_size, -1)
        self.last_known_enemy_pos = None
        self.enemy_history = deque(maxlen=5)
        self.my_history = deque(maxlen=15) 
        self.current_target = None

    def step(self, map_state: np.ndarray, my_position: tuple, enemy_position: tuple, step_number: int):
        # Update map and history
        visible_mask = map_state != -1
        self.global_map[visible_mask] = map_state[visible_mask]
        self.my_history.append(my_position)
        
        # Anti-loop check
        if self.my_history.count(my_position) >= 3:
            return self._escape_loop(my_position)
        
        # Set target - prioritize direct chase when enemy is visible
        if enemy_position:
            self.last_known_enemy_pos = enemy_position
            self.enemy_history.append(enemy_position)
            dist = self._manhattan_distance(my_position, enemy_position)
            
            # Direct axis-aligned chase strategy
            if self._on_same_axis(my_position, enemy_position):
                # Same row or column - direct pursuit
                straight_move, steps = self._get_straight_advantage(my_position, enemy_position)
                if straight_move and steps > 0:
                    return (straight_move, steps)
            
            if dist <= 1:
                self.current_target = enemy_position
            elif dist <= 3:
                self.current_target = self._corner_cut(my_position, enemy_position) or enemy_position
            else:
                self.current_target = self._predict_target(my_position, enemy_position)
        else:
            self.current_target = self.last_known_enemy_pos

        # Move with speed optimization
        if self.current_target:
            # Try straight line with speed=2
            straight_move, steps = self._get_straight_advantage(my_position, self.current_target)
            if straight_move and steps > 0:
                return (straight_move, steps)
            
            # A* pathfinding
            next_move = self._a_star(my_position, self.current_target)
            if next_move:
                steps = self._max_steps(my_position, next_move)
                if steps > 0:
                    return (next_move, steps)

        # Explore
        frontier_move = self._find_frontier(my_position)
        if frontier_move:
            steps = self._max_steps(my_position, frontier_move)
            if steps > 0:
                return (frontier_move, steps)

        # Fallback: always return a valid random move
        return self._random_move(my_position)

    def _escape_loop(self, pos):
        self.my_history.clear()
        moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        random.shuffle(moves)
        for m in moves:
            if self._is_passable(self._get_next_pos(pos, m)):
                return (m, 1)
        return (Move.STAY, 1)

    def _corner_cut(self, my_pos, ghost_pos):
        escape_positions = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            next_pos = self._get_next_pos(ghost_pos, move)
            if self._is_passable(next_pos):
                escape_positions.append(next_pos)
                extended = self._get_next_pos(next_pos, move)
                if self._is_passable(extended):
                    escape_positions.append(extended)
        
        if len(escape_positions) <= 2:
            best_target = None
            min_time = float('inf')
            for pos in escape_positions:
                pacman_time = (self._manhattan_distance(my_pos, pos) + 1) // self.pacman_speed
                ghost_time = self._manhattan_distance(ghost_pos, pos)
                if pacman_time <= ghost_time and pacman_time < min_time:
                    min_time = pacman_time
                    best_target = pos
            return best_target
        return None

    def _predict_target(self, my_pos, ghost_pos):
        if len(self.enemy_history) < 2:
            return ghost_pos
        
        # Analyze recent moves
        recent_moves = []
        for i in range(1, min(3, len(self.enemy_history))):
            curr = self.enemy_history[-i]
            prev = self.enemy_history[-i-1]
            recent_moves.append((curr[0] - prev[0], curr[1] - prev[1]))
        
        # Predict positions
        candidates = [ghost_pos]
        for dr, dc in recent_moves:
            for steps in range(1, 4):
                pred_pos = (ghost_pos[0] + dr*steps, ghost_pos[1] + dc*steps)
                if self._is_passable(pred_pos):
                    candidates.append(pred_pos)
        
        return min(candidates, key=lambda p: self._manhattan_distance(my_pos, p))

    def _get_straight_advantage(self, start_pos, target_pos):
        dr = target_pos[0] - start_pos[0]
        dc = target_pos[1] - start_pos[1]
        
        # Calculate maximum possible steps in each direction
        if dr == 0 or dc == 0:
            # Same row or column
            if dc > 0:
                move = Move.RIGHT
                max_dist = dc
            elif dc < 0:
                move = Move.LEFT
                max_dist = -dc
            elif dr > 0:
                move = Move.DOWN
                max_dist = dr
            else:  # dr < 0
                move = Move.UP
                max_dist = -dr
            
            max_steps = self._max_steps(start_pos, move)
            # Return minimum of: max possible steps, max distance, or pacman speed
            return move, min(max_steps, max_dist, self.pacman_speed)
        return None, 0

    def _a_star(self, start, goal):
        pq = [(0, start, None)]
        visited = {start: 0}
        
        while pq:
            f, current, first_move = heapq.heappop(pq)
            if current == goal:
                return first_move
            if visited[current] > 20:
                continue
            
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                max_steps = self._max_steps(current, move)
                if max_steps > 0:
                    for steps in range(min(max_steps, self.pacman_speed), 0, -1):
                        neighbor = current
                        for _ in range(steps):
                            neighbor = self._get_next_pos(neighbor, move)
                        
                        if self._is_passable(neighbor):
                            new_g = visited[current] + 1
                            if neighbor not in visited or new_g < visited[neighbor]:
                                visited[neighbor] = new_g
                                h = self._manhattan_distance(neighbor, goal)
                                new_first = first_move or move
                                heapq.heappush(pq, (new_g + h, neighbor, new_first))
                        break
        return None

    def _find_frontier(self, start):
        queue = deque([(start, None)])
        visited = {start}
        while queue:
            curr, move = queue.popleft()
            if self._is_frontier(curr):
                return move
            for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                nxt = self._get_next_pos(curr, m)
                if self._is_passable(nxt) and nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, move or m))
        return None

    def _random_move(self, pos):
        moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        random.shuffle(moves)
        for m in moves:
            if self._is_passable(self._get_next_pos(pos, m)):
                steps = self._max_steps(pos, m)
                if steps > 0:
                    return (m, min(steps, self.pacman_speed))
        # Absolute fallback
        return (Move.STAY, 1)

    # Helper methods
    def _on_same_axis(self, pos1, pos2):
        """Check if two positions are on the same row or column"""
        return pos1[0] == pos2[0] or pos1[1] == pos2[1]

    def _manhattan_distance(self, p1, p2):
        return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])

    def _is_frontier(self, pos):
        if self.global_map[pos] != 0:
            return False
        r, c = pos
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = r+dr, c+dc
            if self._is_in_bounds((nr, nc)) and self.global_map[nr, nc] == -1:
                return True
        return False

    def _is_passable(self, pos):
        return self._is_in_bounds(pos) and self.global_map[pos] != 1

    def _is_in_bounds(self, pos):
        return 0 <= pos[0] < 21 and 0 <= pos[1] < 21

    def _get_next_pos(self, pos, move):
        dr, dc = move.value
        return (pos[0] + dr, pos[1] + dc)

    def _max_steps(self, pos, move):
        steps = 0
        curr = pos
        for _ in range(self.pacman_speed):
            next_p = self._get_next_pos(curr, move)
            if not self._is_passable(next_p):
                break
            steps += 1
            curr = next_p
        return steps


class GhostAgent(BaseGhostAgent):
    """Ghost v5.0 Compact - Phantom Fortress Optimized"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Ghost v5.0 (Phantom Fortress Compact)"
        
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 2)))
        self.capture_threshold = 3
        
        self.map_size = (21, 21)
        self.global_map = np.full(self.map_size, -1)
        
        self.last_known_enemy_pos = None
        self.turns_since_seen = 0
        self.enemy_history = deque(maxlen=6)
        self.danger_zones = set()
        
        # Thresholds
        self.ULTRA_DANGER = 3
        self.EXTREME_DANGER = 5
        self.MINIMAX_TRIGGER = 8
        self.MINIMAX_DEPTH = 4
        self.ESCAPE_THRESHOLD = 3

    def step(self, map_state: np.ndarray, my_position: tuple, enemy_position: tuple, step_number: int):
        # Update map and enemy tracking
        visible_mask = map_state != -1
        self.global_map[visible_mask] = map_state[visible_mask]
        
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
            self.enemy_history.append(enemy_position)
            self.turns_since_seen = 0
            self._update_danger_zones(enemy_position)
        else:
            self.turns_since_seen += 1
            if self.turns_since_seen > 8:
                self.last_known_enemy_pos = None
                self.danger_zones.clear()

        target_enemy = enemy_position or self.last_known_enemy_pos

        # Tactical decision
        if target_enemy is None:
            return self._paranoid_roam(my_position)
            
        dist = self._manhattan_distance(my_position, target_enemy)
        
        if dist <= self.ULTRA_DANGER and enemy_position is not None:
            return self._panic_escape(my_position, enemy_position)
        elif dist <= self.EXTREME_DANGER and enemy_position is not None:
            return self._extreme_escape(my_position, enemy_position)
        elif dist <= self.MINIMAX_TRIGGER and enemy_position is not None:
            return self._minimax_escape(my_position, enemy_position)
        else:
            return self._strategic_escape(my_position, target_enemy)

    def _update_danger_zones(self, pacman_pos):
        self.danger_zones.clear()
        for turn in range(1, 4):
            straight_reach = self.pacman_speed * turn
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                for dist in range(1, straight_reach + 1):
                    danger_pos = (pacman_pos[0] + dr*dist, pacman_pos[1] + dc*dist)
                    if self._is_in_bounds(danger_pos) and self.global_map[danger_pos] != 1:
                        self.danger_zones.add(danger_pos)

    def _panic_escape(self, my_pos, pacman_pos):
        if self._is_straight_threat(my_pos, pacman_pos):
            return self._perpendicular_escape(my_pos, pacman_pos)
        
        best_move = Move.STAY
        max_safety = -1
        
        for next_pos, move in self._get_neighbors(my_pos):
            safety = self._effective_distance(next_pos, pacman_pos) * 10
            safety += self._count_escapes(next_pos, 2) * 5
            if next_pos in self.danger_zones:
                safety -= 200
            
            if safety > max_safety:
                max_safety = safety
                best_move = move
        
        return best_move

    def _extreme_escape(self, my_pos, pacman_pos):
        best_move = Move.STAY
        best_score = -float('inf')
        
        for next_pos, move in self._get_neighbors(my_pos):
            score = self._effective_distance(next_pos, pacman_pos) * 15
            score += self._count_escapes(next_pos, 3) * 10
            
            if next_pos in self.danger_zones:
                score -= 300
            if self._is_corner(next_pos):
                score += 30
                
            if score > best_score:
                best_score = score
                best_move = move
                
        return best_move

    def _minimax_escape(self, my_pos, pacman_pos):
        _, best_move = self._minimax(pacman_pos, my_pos, self.MINIMAX_DEPTH, True)
        return best_move or Move.STAY

    def _strategic_escape(self, my_pos, enemy_pos):
        best_move = Move.STAY
        best_score = -float('inf')
        
        for next_pos, move in self._get_neighbors(my_pos):
            score = self._effective_distance(next_pos, enemy_pos) * 20
            
            if next_pos in self.danger_zones:
                score -= 400
            
            escapes = self._count_escapes(next_pos, 4)
            if escapes < self.ESCAPE_THRESHOLD:
                score -= 2000
            else:
                score += escapes * 15
            
            if not self._on_same_axis(next_pos, enemy_pos):
                score += 100
                
            if score > best_score:
                best_score = score
                best_move = move
                
        return best_move

    def _paranoid_roam(self, my_pos):
        best_move = Move.STAY
        best_score = -float('inf')
        
        for next_pos, move in self._get_neighbors(my_pos):
            score = 0
            
            if self._is_corner(next_pos):
                score += 200
            
            center_dist = abs(next_pos[0] - 10) + abs(next_pos[1] - 10)
            if center_dist < 8:
                score -= center_dist * 15
            
            escapes = self._count_escapes(next_pos, 3)
            if escapes < 4:
                score -= 300
            else:
                score += escapes * 10
                
            if score > best_score:
                best_score = score
                best_move = move
        return best_move

    def _minimax(self, pacman_pos, ghost_pos, depth, is_ghost_turn):
        dist = self._effective_distance(pacman_pos, ghost_pos)
        if dist < self.capture_threshold:
            return -1000, None
        if depth == 0:
            return dist * 10, None
        
        if is_ghost_turn:
            max_eval = -float('inf')
            best_move = Move.STAY
            
            for next_pos, move in self._get_neighbors(ghost_pos):
                safety = self._effective_distance(pacman_pos, next_pos) * 5
                if next_pos in self.danger_zones:
                    safety -= 100
                
                eval_score, _ = self._minimax(pacman_pos, next_pos, depth-1, False)
                eval_score += safety
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
            return max_eval, best_move
        else:
            min_eval = float('inf')
            # Simplified Pacman moves
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)]:
                next_pos = (pacman_pos[0] + dr*self.pacman_speed, pacman_pos[1] + dc*self.pacman_speed)
                if self._is_in_bounds(next_pos) and self.global_map[next_pos] != 1:
                    eval_score, _ = self._minimax(next_pos, ghost_pos, depth-1, True)
                    if eval_score < min_eval:
                        min_eval = eval_score
            return min_eval, None

    # Helper methods
    def _is_straight_threat(self, my_pos, pacman_pos):
        dr = my_pos[0] - pacman_pos[0]
        dc = my_pos[1] - pacman_pos[1]
        if dr == 0 or dc == 0:
            dist = abs(dr) + abs(dc)
            return dist <= self.pacman_speed * 2
        return False

    def _perpendicular_escape(self, my_pos, pacman_pos):
        dr = my_pos[0] - pacman_pos[0]
        dc = my_pos[1] - pacman_pos[1]
        
        moves = []
        if dr == 0:
            moves = [Move.UP, Move.DOWN]
        elif dc == 0:
            moves = [Move.LEFT, Move.RIGHT]
        
        best_move = Move.STAY
        best_safety = -1
        for move in moves:
            next_pos = self._get_next_pos(my_pos, move)
            if self._is_passable(next_pos):
                safety = self._count_escapes(next_pos, 2)
                if safety > best_safety:
                    best_safety = safety
                    best_move = move
        return best_move

    def _effective_distance(self, pos1, pos2):
        manhattan = self._manhattan_distance(pos1, pos2)
        if self._on_same_axis(pos1, pos2):
            return max(1, manhattan / self.pacman_speed)
        return manhattan

    def _is_corner(self, pos):
        r, c = pos
        corner_dist = min(r + c, r + (20 - c), (20 - r) + c, (20 - r) + (20 - c))
        return corner_dist <= 5

    def _on_same_axis(self, pos1, pos2):
        return pos1[0] == pos2[0] or pos1[1] == pos2[1]

    def _count_escapes(self, pos, depth):
        visited = {pos}
        queue = deque([(pos, 0)])
        count = 0
        while queue and len(visited) < 50:  # Limit for performance
            curr, d = queue.popleft()
            if d >= depth:
                continue
            for next_p, _ in self._get_neighbors(curr):
                if next_p not in visited:
                    visited.add(next_p)
                    queue.append((next_p, d+1))
                    count += 1
        return count

    def _get_neighbors(self, pos):
        neighbors = []
        r, c = pos
        for nr, nc, move in [(r-1, c, Move.UP), (r+1, c, Move.DOWN), 
                            (r, c-1, Move.LEFT), (r, c+1, Move.RIGHT)]:
            if self._is_passable((nr, nc)):
                neighbors.append(((nr, nc), move))
        return neighbors

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _is_passable(self, pos):
        return self._is_in_bounds(pos) and self.global_map[pos] != 1

    def _is_in_bounds(self, pos):
        return 0 <= pos[0] < 21 and 0 <= pos[1] < 21

    def _get_next_pos(self, pos, move):
        dr, dc = move.value
        return (pos[0] + dr, pos[1] + dc)