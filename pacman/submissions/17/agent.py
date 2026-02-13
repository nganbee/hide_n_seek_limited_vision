"""
Template for student agent implementation.

INSTRUCTIONS:
1. Copy this file to submissions/<your_student_id>/agent.py
2. Implement the PacmanAgent and/or GhostAgent classes
3. Replace the simple logic with your search algorithm
4. Test your agent using: python arena.py --seek <your_id> --hide example_student --pacman-speed 2 --capture-distance 2

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
import pickle
import random
from collections import deque
from heapq import heappush, heappop
from itertools import count

# Add src to path to import the interface
src_path = Path(__file__).parent.parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np

# Pacman
class PacmanAgent(BasePacmanAgent):
    """
    Pacman (Seeker)
    Strategy:
    - Maintain known_map (memory)
    - Chase enemy if visible or last known
    - Otherwise explore nearest unknown (-1)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Pacman A* Limited Vision"
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))

        self.known_map = None
        self.last_known_enemy_pos = None
        # Ghi nhớ hướng đi ở bước trước để quyết định số bước (1 hay 2)
        self.last_move = None

    # ---------------- MAIN STEP ----------------
    def step(self, map_state, my_pos, enemy_pos, step_number):
        self._update_known_map(map_state)

        if enemy_pos is not None:
            self.last_known_enemy_pos = enemy_pos

        target = enemy_pos or self.last_known_enemy_pos

        # Mặc định đứng yên
        chosen_move = Move.STAY

        # 1. Chase enemy if possible
        if target is not None:
            path = self._astar(my_pos, target)
            if path:
                chosen_move = path[0]

        # 2. Explore unknown area nếu chưa có hướng đuổi
        if chosen_move == Move.STAY:
            explore_target = self._nearest_unknown(my_pos)
            if explore_target is not None:
                path = self._astar(my_pos, explore_target)
                if path:
                    chosen_move = path[0]

        # Quy tắc bước đi:
        # - Nếu đi thẳng (cùng hướng với bước trước) => 1 bước
        # - Nếu quẹo (đổi hướng so với bước trước)  => 2 bước (tối đa pacman_speed)
        steps = 1
        if chosen_move != Move.STAY:
            if self.last_move is not None and chosen_move != self.last_move:
                steps = min(2, self.pacman_speed)

        self.last_move = chosen_move
        return (chosen_move, steps)

    # ---------------- MEMORY ----------------
    def _update_known_map(self, map_state):
        if self.known_map is None:
            self.known_map = map_state.copy()
        else:
            mask = (map_state != -1)
            self.known_map[mask] = map_state[mask]

    # ---------------- A* SEARCH ----------------
    def _astar(self, start, goal):
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        tie_breaker = count()
        frontier = []
        heappush(frontier, (0, next(tie_breaker), start, []))
        visited = set()

        while frontier:
            _, _, current, path = heappop(frontier)

            if current == goal:
                return path

            if current in visited:
                continue
            visited.add(current)

            for next_pos, move in self._neighbors(current):
                if next_pos not in visited:
                    new_path = path + [move]
                    cost = len(new_path) + heuristic(next_pos, goal)
                    heappush(frontier, (cost, next(tie_breaker), next_pos, new_path))

        return []

    # ---------------- EXPLORATION ----------------
    def _nearest_unknown(self, start):
        queue = deque([start])
        visited = {start}

        while queue:
            current = queue.popleft()
            if self.known_map[current] == -1: # pyright: ignore[reportOptionalSubscript]
                return current

            for next_pos, _ in self._neighbors(current):
                if next_pos not in visited:
                    visited.add(next_pos)
                    queue.append(next_pos)
        return None

    # ---------------- HELPERS ----------------
    def _neighbors(self, pos):
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            delta = move.value
            next_pos = (pos[0] + delta[0], pos[1] + delta[1])
            if self._is_valid(next_pos):
                yield next_pos, move

    def _is_valid(self, pos):
        r, c = pos
        h, w = self.known_map.shape # pyright: ignore[reportOptionalMemberAccess]
        if r < 0 or r >= h or c < 0 or c >= w:
            return False
        return self.known_map[r, c] == 0 # pyright: ignore[reportOptionalSubscript]

# Ghost
class GhostAgent(BaseGhostAgent):
    """
    Class đại diện cho Ghost.
    Nhiệm vụ: Sống sót càng lâu càng tốt.
    Chiến thuật: Tối đa hóa khoảng cách thực tế (BFS distance) tới Pacman.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_known_enemy_pos = None 
        self.known_map = None
    
    def step(self, map_state: np.ndarray, my_position: tuple, enemy_position: tuple, step_number: int) -> Move:
        # cập nhật trí bản đồ
        if self.known_map is None:
            self.known_map = map_state.copy()
        else:
            # Chỉ cập nhật những ô nhìn thấy (khác -1) vào trí nhớ
            visible_mask = (map_state != -1)
            self.known_map[visible_mask] = map_state[visible_mask]

        # Cập nhật vị trí địch
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
        
        threat_pos = enemy_position or self.last_known_enemy_pos
        
        if threat_pos is None:
            return self._patrol(my_position, self.known_map)
        else:
            return self._smart_run_away(my_position, threat_pos, self.known_map)

    def _smart_run_away(self, my_pos, threat_pos, map_state):
        """
        Đánh giá từng nước đi có thể.
        Với mỗi nước đi, tính xem khoảng cách tới Pacman là bao nhiêu (dùng BFS).
        Chọn nước đi có khoảng cách xa nhất.
        """
        best_move = Move.STAY
        max_safety_score = -1 # Điểm an toàn cao nhất (càng xa càng tốt)
        
        # Lấy danh sách các ô có thể đi tiếp theo
        valid_moves = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            next_pos = self._apply_move(my_pos, move)
            if self._is_valid_move(next_pos, map_state):
                valid_moves.append((next_pos, move))
        
        # Nếu bị kẹt không đi được đâu -> Đứng yên
        if not valid_moves:
            return Move.STAY

        # Duyệt qua từng nước đi khả thi
        for next_pos, move in valid_moves:
            # Tính khoảng cách từ vị trí dự kiến (next_pos) tới Pacman (threat_pos)
            # Dùng BFS chính xác hơn Manhattan vì nó tính đường đi vòng qua tường
            dist = self._bfs_distance(next_pos, threat_pos, map_state)
            
            # Nếu khoảng cách này tốt hơn kỷ lục cũ -> Cập nhật
            if dist > max_safety_score:
                max_safety_score = dist
                best_move = move
            elif dist == max_safety_score:
                # Nếu khoảng cách bằng nhau -> Chọn ngẫu nhiên 50/50 để khó đoán
                if random.random() > 0.5:
                    best_move = move
                    
        return best_move

    def _bfs_distance(self, start, target, map_state):
        """Hàm đo khoảng cách ngắn nhất giữa 2 điểm (xuyên qua mê cung)."""
        queue = deque([(start, 0)]) # Lưu (vị trí, khoảng cách)
        visited = {start}
        
        while queue:
            current, dist = queue.popleft()
            
            if current == target:
                return dist # Đã tìm thấy Pacman, trả về khoảng cách
            
            # Nếu khoảng cách > 30 bước thì coi như rất an toàn.
            if dist > 30: 
                return dist
            
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                next_pos = self._apply_move(current, move)
                # Ghost đi được vào ô trống (0) và sương mù (-1), tránh tường (1)
                if self._is_valid_move(next_pos, map_state) and next_pos not in visited:
                    visited.add(next_pos)
                    queue.append((next_pos, dist + 1))
        
        # Nếu không có đường tới Pacman (bị chặn hoàn toàn) -> Rất an toàn -> Trả về số lớn
        return 100

    def _patrol(self, pos, map_state):
        """Đi tuần tra ngẫu nhiên, ưu tiên di chuyển hơn là đứng yên."""
        valid_moves = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            # Kiểm tra thử nước đi
            if self._is_valid_move(self._apply_move(pos, move), map_state):
                valid_moves.append(move)
        
        if valid_moves:
            return random.choice(valid_moves) # Chọn bừa 1 hướng đi được
        return Move.STAY

    def _apply_move(self, pos, move):
        return (pos[0] + move.value[0], pos[1] + move.value[1])

    def _is_valid_move(self, pos, map_state):
        """Kiểm tra ô đó có hợp lệ cho Ghost không (trong bản đồ và không phải tường)."""
        rows, cols = map_state.shape
        if not (0 <= pos[0] < rows and 0 <= pos[1] < cols):
            return False
        return map_state[pos] != 1 # Khác 1 là đi được (bao gồm 0 và -1)