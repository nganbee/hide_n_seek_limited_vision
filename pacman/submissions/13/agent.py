"""
Template for student agent implementation.

INSTRUCTIONS:
1. Copy this file to submissions/<your_student_id>/agent.py
2. Implement the PacmanAgent and/or GhostAgent classes
3. Replace the simple logic with your search algorithm
4. Test your agent using: python arena.py --seek <your_id> --hide example_student

IMPORTANT:
- Do NOT change the class names (PacmanAgent, GhostAgent)
- Do NOT change the method signatures (step, __init__)
- Pacman step must return either a Move or a (Move, steps) tuple where
    1 <= steps <= pacman_speed (provided via kwargs)
- Ghost step must return a Move enum value
- You CAN add your own helper methods
- You CAN import additional Python standard libraries
- Agents are STATEFUL - you can store memory across steps
- enemy_position may be None when limited observation is enabled
- map_state cells: 1=wall, 0=empty, -1=unseen (fog)
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
import random
UNKNOWN = -1
EMPTY = 0
WALL = 1



class PacmanAgent(BasePacmanAgent):
    """
    Pacman (Seeker) Agent - Goal: Catch the Ghost
    
    Implement your search algorithm to find and catch the ghost.
    Suggested algorithms: BFS, DFS, A*, Greedy Best-First
    """
    
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
    
    # Helpers
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

class GhostAgent(BaseGhostAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Survivalist - Minimax"
        self.last_pacman_pos = None
        self.map_size = (21, 21)
        self.global_map = None

    def step(self, map_state, my_position, enemy_position, step_number):
        if self.global_map is None: self.global_map = map_state
        else: self.global_map[map_state != -1] = map_state[map_state != -1]

        if enemy_position: self.last_pacman_pos = enemy_position

        # Nếu không biết Pacman ở đâu, đi random an toàn
        if not self.last_pacman_pos:
            return self._patrol(my_position)

        # Kích hoạt Minimax
        best_move = self._minimax_move(my_position, self.last_pacman_pos)
        return best_move

    def _minimax_move(self, ghost_pos, pacman_pos):
        valid_moves = self._get_valid_moves(ghost_pos)
        if not valid_moves: return Move.STAY

        best_score = -float('inf')
        best_moves = []

        # 1. GHOST MOVE (MAX LAYER)
        for move in valid_moves:
            next_g_pos = self._get_next_pos(ghost_pos, move)
            
            # Nếu đi vào chỗ chết ngay lập tức -> Bỏ qua
            if self._manhattan_distance(next_g_pos, pacman_pos) <= 2:
                score = -9999
            else:
                # 2. PACMAN MOVE (MIN LAYER - PREDICTION)
                # Dự đoán Pacman sẽ đi 2 bước tối ưu nhất để bắt Ghost
                pred_p_pos = self._predict_pacman_best_move(pacman_pos, next_g_pos, steps=2)
                
                # 3. EVALUATE STATE
                score = self._evaluate_state(next_g_pos, pred_p_pos)

            if score > best_score:
                best_score = score
                best_moves = [move]
            elif score == best_score:
                best_moves.append(move)
        
        return random.choice(best_moves) if best_moves else Move.STAY

    def _predict_pacman_best_move(self, p_pos, target_g, steps=2):
        """Giả lập Pacman lao tới Ghost nhanh nhất có thể"""
        curr = p_pos
        for _ in range(steps):
            # Tìm nước đi giảm khoảng cách Manhattan nhiều nhất
            moves = self._get_valid_moves(curr)
            best_next = curr
            min_dist = float('inf')
            
            for m in moves:
                nxt = self._get_next_pos(curr, m)
                # Pacman cũng phải né tường
                dist = self._manhattan_distance(nxt, target_g)
                if dist < min_dist:
                    min_dist = dist
                    best_next = nxt
            curr = best_next
            if curr == target_g: break
        return curr

    def _evaluate_state(self, g_pos, p_pos):
        dist = self._manhattan_distance(g_pos, p_pos)
        
        if dist <= 2:
            return -1000 + dist # Ưu tiên chết xa hơn chút
        
        score = dist * 10
        
        # Thưởng nếu phá trục
        if g_pos[0] != p_pos[0] and g_pos[1] != p_pos[1]:
            score += 50
            
        # Thưởng nếu gần ngã rẽ
        if not self._is_dead_end(g_pos):
            score += 20
            
        return score

    def _patrol(self, pos):
        moves = self._get_valid_moves(pos)
        safe = [m for m in moves if not self._is_dead_end(self._get_next_pos(pos, m))]
        return random.choice(safe) if safe else (random.choice(moves) if moves else Move.STAY)

    def _get_valid_moves(self, pos):
        moves = []
        for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if self._is_passable(self._get_next_pos(pos, m)): moves.append(m)
        return moves

    def _get_next_pos(self, pos, move):
        return (pos[0]+move.value[0], pos[1]+move.value[1])

    def _is_passable(self, pos):
        r, c = pos
        if 0 <= r < 21 and 0 <= c < 21: return self.global_map[r, c] != 1
        return False

    def _manhattan_distance(self, p1, p2):
        return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])

    def _is_dead_end(self, pos):
        exits = 0
        for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if self._is_passable(self._get_next_pos(pos, m)): exits += 1
        return exits <= 1