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


class PacmanAgent(BasePacmanAgent):
    """
    Pacman (Seeker) Agent - Goal: Catch the Ghost
    
    Implement your search algorithm to find and catch the ghost.
    Suggested algorithms: BFS, DFS, A*, Greedy Best-First
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        # TODO: Initialize any data structures you need
        # Examples:
        # - self.path = []  # Store planned path
        # - self.visited = set()  # Track visited positions
        self.name = "Template Pacman"
        # Memory for limited observation mode
        self.memory_map = np.full((21, 21), -1)
        self.last_known_enemy_pos = None
    
    def _bfs(self, start, target_condition):
        """Thuật toán BFS tìm đường ngắn nhất đến mục tiêu"""
        queue = collections.deque([(start, [])])
        visited = {start}
        while queue:
            (r, c), path = queue.popleft()
            if target_condition(r, c):
                return path[0] if path else Move.STAY
            
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move.value
                nr, nc = r + dr, c + dc
                if 0 <= nr < 21 and 0 <= nc < 21 and self.memory_map[nr, nc] != 1:
                    if (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append(((nr, nc), path + [move]))
        return None

    def step(self, map_state: np.ndarray, my_position: tuple, enemy_position: tuple, step_number: int):
        # Update memory if enemy is visible
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
        
        # Cập nhật bản đồ quan sát một phần vào bộ nhớ
        mask = (map_state != -1)
        self.memory_map[mask] = map_state[mask]

        # TODO: Implement your search algorithm here
        target = None
        if self.last_known_enemy_pos:
            if my_position == self.last_known_enemy_pos:
                self.last_known_enemy_pos = None
            else:
                target = lambda r, c: (r, c) == self.last_known_enemy_pos
        
        # Nếu không thấy địch, BFS tìm ô chưa biết (-1) gần nhất để khám phá
        if not target:
            target = lambda r, c: self.memory_map[r, c] == -1

        action = self._bfs(my_position, target)
        
        if action:
            # Tính toán số bước tối đa (steps) có thể đi dựa trên pacman_speed
            desired_steps = self.pacman_speed
            actual_steps = self._max_valid_steps(my_position, action, map_state, desired_steps)
            return action, actual_steps
        
        return Move.STAY, 1

    def _max_valid_steps(self, pos, move, map_state, max_steps):
        steps = 0
        curr_r, curr_c = pos
        dr, dc = move.value
        for _ in range(max_steps):
            nr, nc = curr_r + dr, curr_c + dc
            if self._is_valid_position((nr, nc), map_state):
                steps += 1
                curr_r, curr_c = nr, nc
            else:
                break
        return steps

    def _is_valid_position(self, pos, map_state):
        r, c = pos
        return 0 <= r < 21 and 0 <= c < 21 and map_state[r, c] != 1



class GhostAgent(BaseGhostAgent):
    """
    Ghost (Hider) Agent - Goal: Avoid being caught
    
    Implement your search algorithm to evade Pacman as long as possible.
    Suggested algorithms: BFS (find furthest point), Minimax, Monte Carlo
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # TODO: Initialize any data structures you need
        self.memory_map = np.full((21, 21), -1)
        self.last_known_enemy_pos = None

    def step(self, map_state: np.ndarray, my_position: tuple, enemy_position: tuple, step_number: int):
        # Update memory if enemy is visible
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
        
        mask = (map_state != -1)
        self.memory_map[mask] = map_state[mask]

        # TODO: Implement your search algorithm here
        if enemy_position:
            # Chiến thuật trốn: Chọn hướng làm tăng khoảng cách Manhattan lớn nhất
            best_move = Move.STAY
            max_dist = -1
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]:
                dr, dc = move.value if move != Move.STAY else (0, 0)
                nr, nc = my_position[0] + dr, my_position[1] + dc
                if self._is_valid_position((nr, nc), map_state):
                    dist = abs(nr - enemy_position[0]) + abs(nc - enemy_position[1])
                    if dist > max_dist:
                        max_dist = dist
                        best_move = move
            return best_move

        # Di chuyển ngẫu nhiên vào các ô trống nếu không thấy địch
        valid_moves = [m for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT] 
                       if self._is_valid_position((my_position[0]+m.value[0], my_position[1]+m.value[1]), map_state)]
        return random.choice(valid_moves) if valid_moves else Move.STAY

    def _is_valid_position(self, pos, map_state):
        r, c = pos
        return 0 <= r < 21 and 0 <= c < 21 and map_state[r, c] == 0