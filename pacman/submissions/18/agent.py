"""
Uses A* for Pacman and evasion strategy for Ghost
"""

import sys
from pathlib import Path
from collections import deque
import heapq
import random

src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np

class PacmanAgent(BasePacmanAgent):
    """
    Pacman (Seeker) - Uses A* pathfinding with fog of war handling
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.name = "Smart Pacman"
        
        # Memory systems
        self.last_known_enemy_pos = None
        self.explored_map = None  # Track what we've seen
        self.path = []  # Current planned path
        self.exploration_targets = []  # Unexplored areas to check
        
    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int):
        """
        Step with A* pathfinding and exploration strategy
        """
        # Initialize explored map on first step
        if self.explored_map is None:
            self.explored_map = np.copy(map_state)
        else:
            # Update explored map with new observations
            self.explored_map = np.where(
                map_state != -1,  # Where we can see
                map_state,  # Update with current view
                self.explored_map  # Keep old knowledge
            )
        
        # Update enemy tracking
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
            self.path = []  # Replan when enemy is spotted
        
        # Strategy 1: Chase if we know where enemy is
        if enemy_position is not None:
            target = enemy_position
            path = self._a_star(my_position, target, self.explored_map)
            if path and len(path) > 1:
                return self._follow_path(my_position, path)
        
        # Strategy 2: Go to last known position
        if self.last_known_enemy_pos is not None:
            if self.last_known_enemy_pos != my_position:
                path = self._a_star(my_position, self.last_known_enemy_pos, self.explored_map)
                if path and len(path) > 1:
                    return self._follow_path(my_position, path)
            else:
                # Reached last known position, start exploring
                self.last_known_enemy_pos = None
        
        # Strategy 3: Explore unknown areas
        unexplored = self._find_nearest_unexplored(my_position, self.explored_map)
        if unexplored:
            path = self._a_star(my_position, unexplored, self.explored_map)
            if path and len(path) > 1:
                return self._follow_path(my_position, path)
        
        # Strategy 4: Random valid move
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if self._is_valid_move(my_position, move, self.explored_map):
                return (move, 1)
        
        return (Move.STAY, 1)
    
    def _a_star(self, start: tuple, goal: tuple, map_state: np.ndarray):
        """A* pathfinding algorithm"""
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path
            
            # Check all neighbors
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move.value
                neighbor = (current[0] + dr, current[1] + dc)
                
                if not self._is_valid_position(neighbor, map_state):
                    continue
                
                tentative_g = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return None  # No path found
    
    def _find_nearest_unexplored(self, start: tuple, map_state: np.ndarray):
        """Find nearest unexplored area using BFS"""
        queue = deque([start])
        visited = {start}
        
        while queue:
            pos = queue.popleft()
            
            # Check if this position borders unexplored area
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move.value
                neighbor = (pos[0] + dr, pos[1] + dc)
                
                if not self._in_bounds(neighbor, map_state):
                    continue
                
                # Found unexplored cell adjacent to explored area
                if map_state[neighbor] == -1:
                    return pos  # Return the explored cell next to unknown
                
                if neighbor not in visited and self._is_valid_position(neighbor, map_state):
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return None
    
    def _follow_path(self, current_pos: tuple, path: list):
        """Convert path to move action with multiple steps"""
        if len(path) < 2:
            return (Move.STAY, 1)
        
        # Determine direction
        next_pos = path[1]
        dr = next_pos[0] - current_pos[0]
        dc = next_pos[1] - current_pos[1]
        
        if dr > 0:
            move = Move.DOWN
        elif dr < 0:
            move = Move.UP
        elif dc > 0:
            move = Move.RIGHT
        elif dc < 0:
            move = Move.LEFT
        else:
            return (Move.STAY, 1)
        
        # Calculate how many steps we can take in this direction
        steps = self._max_valid_steps(current_pos, move, self.explored_map, self.pacman_speed)
        return (move, max(1, steps))
    
    def _max_valid_steps(self, pos: tuple, move: Move, map_state: np.ndarray, max_steps: int) -> int:
        steps = 0
        current = pos
        for _ in range(max_steps):
            dr, dc = move.value
            next_pos = (current[0] + dr, current[1] + dc)
            if not self._is_valid_position(next_pos, map_state):
                break
            steps += 1
            current = next_pos
        return steps
    
    def _is_valid_move(self, pos: tuple, move: Move, map_state: np.ndarray) -> bool:
        return self._max_valid_steps(pos, move, map_state, 1) == 1
    
    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        if not self._in_bounds(pos, map_state):
            return False
        # Consider -1 (unknown) as potentially valid for exploration
        return map_state[pos] == 0
    
    def _in_bounds(self, pos: tuple, map_state: np.ndarray) -> bool:
        row, col = pos
        height, width = map_state.shape
        return 0 <= row < height and 0 <= col < width



class GhostAgent(BaseGhostAgent):
    """
    Ghost (Hider) - Chiến thuật: Né tránh tầm nhìn & Ưu tiên trốn vào góc xa Pacman nhất.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Corner Hider Ghost"
        self.last_known_enemy_pos = None
        self.explored_map = None
        self.target_corner = None  # Mục tiêu góc hiện tại
        self.last_moves = []
        
    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int) -> Move:
        
        # 1. Cập nhật bản đồ và vị trí địch
        if self.explored_map is None:
            self.explored_map = np.copy(map_state)
        else:
            self.explored_map = np.where(map_state != -1, map_state, self.explored_map)
        
        if enemy_position:
            self.last_known_enemy_pos = enemy_position
            
        threat = enemy_position or self.last_known_enemy_pos
        
        # --- CHIẾN THUẬT 1: Né tầm nhìn chữ thập ---
        # Nếu Pacman đang nhìn thấy hoặc ở quá gần (<= 4 ô), bỏ qua việc chạy về góc mà lo né trước
        if threat:
            dist_to_threat = self._manhattan_distance(my_position, threat)
            in_vision = self._in_cross_vision(my_position, threat)
            
            if in_vision or dist_to_threat <= 4:
                # Reset mục tiêu góc vì đang bị săn đuổi
                self.target_corner = None 
                
                # Ưu tiên né theo đường chéo (Diagonal Escape) như code cũ
                move = self._diagonal_escape(my_position, threat, map_state)
                if move:
                    self._record_move(move)
                    return move
                
                # Nếu không né chéo được, chạy ra xa nhất có thể
                return self._immediate_evade(my_position, threat, map_state)

        # --- CHIẾN THUẬT 2: CORNER HIDING ---
        # Tìm góc tốt nhất để trốn (Góc xa Pacman nhất)
        best_corner = self._find_best_hiding_corner(my_position, threat, map_state)
        
        if best_corner:
            # Nếu đã đến góc an toàn, cố gắng ở lại đó hoặc đảo vị trí nhỏ để không bị timeout
            if my_position == best_corner:
                 # Nếu Pacman chưa đến gần, có thể đứng yên hoặc đi vòng quanh nhỏ
                 return Move.STAY 
            
            # Tìm đường đi ngắn nhất đến góc đó
            path = self._bfs_path(my_position, best_corner, map_state)
            if path and len(path) > 1:
                move = self._get_move_direction(my_position, path[1])
                self._record_move(move)
                return move

        # --- CHIẾN THUẬT 3: NGẪU NHIÊN ---
        # Nếu không có mục tiêu góc hoặc không có mối đe dọa, đi ngẫu nhiên thông minh
        valid_moves = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if self._is_valid_move(my_position, move, map_state) and not self._is_recent_move(move):
                valid_moves.append(move)
        
        if valid_moves:
            move = random.choice(valid_moves)
            self._record_move(move)
            return move
            
        return Move.STAY

    def _find_best_hiding_corner(self, my_pos: tuple, threat_pos: tuple, map_state: np.ndarray):
        """
        Tìm góc an toàn nhất.
        Tiêu chí: Phải đi được (valid) và XA Pacman nhất (Maximize distance to threat).
        """
        height, width = map_state.shape
        
        # Xác định 4 khu vực góc của bản đồ (Top-Left, Top-Right, Bot-Left, Bot-Right)
        # Tìm ô đi được (value=0) gần các góc hình học nhất
        candidates = []
        
        # Các điểm mốc góc lý thuyết
        corners_geometric = [
            (1, 1),                 # Top-Left
            (1, width - 2),         # Top-Right
            (height - 2, 1),        # Bottom-Left
            (height - 2, width - 2) # Bottom-Right
        ]
        
        # Tìm ô hợp lệ gần nhất cho mỗi góc hình học
        valid_corners = []
        for r, c in corners_geometric:
            nearest = self._find_nearest_valid_cell((r, c), map_state)
            if nearest:
                valid_corners.append(nearest)
        
        if not valid_corners:
            return None
            
        # Nếu không biết vị trí Pacman, chọn góc xa mình nhất (để khám phá)
        if not threat_pos:
            valid_corners.sort(key=lambda p: self._manhattan_distance(my_pos, p), reverse=True)
            return valid_corners[0]
            
        # Nếu biết vị trí Pacman, chọn góc XA Pacman nhất
        # Score = Khoảng cách từ Góc tới Pacman
        best_corner = None
        max_safety_score = -1
        
        for corner in valid_corners:
            dist_to_threat = self._manhattan_distance(corner, threat_pos)
            dist_from_me = self._manhattan_distance(my_pos, corner)
            
            # Chỉ chọn góc nếu Ghost có thể đến đó nhanh hơn Pacman (hoặc ít nhất là an toàn)
            # Hệ số an toàn = Khoảng cách địch tới góc
            score = dist_to_threat
            
            # Nếu góc đó quá gần Ghost hiện tại (Ghost đang đứng ở đó), ưu tiên giữ vị trí
            if dist_from_me == 0:
                score += 5 
                
            if score > max_safety_score:
                max_safety_score = score
                best_corner = corner
                
        return best_corner

    def _find_nearest_valid_cell(self, start_pos: tuple, map_state: np.ndarray):
        """BFS tìm ô hợp lệ gần nhất từ một toạ độ (dùng để tìm điểm đi được ở góc)"""
        if self._is_valid_position(start_pos, map_state):
            return start_pos
            
        queue = deque([start_pos])
        visited = {start_pos}
        
        # Giới hạn tìm kiếm cục bộ để không chạy hết map
        limit = 0
        while queue and limit < 50:
            pos = queue.popleft()
            limit += 1
            
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move.value
                neighbor = (pos[0] + dr, pos[1] + dc)
                
                
                h, w = map_state.shape
                if 0 <= neighbor[0] < h and 0 <= neighbor[1] < w:
                    if map_state[neighbor] == 0: # Tìm thấy đường đi
                        return neighbor
                    
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
        return None

    
    
    def _in_cross_vision(self, my_pos: tuple, enemy_pos: tuple) -> bool:
        my_row, my_col = my_pos
        enemy_row, enemy_col = enemy_pos
        if my_row == enemy_row or my_col == enemy_col:
            return self._manhattan_distance(my_pos, enemy_pos) <= 5
        return False
    
    def _diagonal_escape(self, my_pos: tuple, threat_pos: tuple, map_state: np.ndarray):
        
        my_row, my_col = my_pos
        threat_row, threat_col = threat_pos
        
        best_move = None
        max_priority = -1
        
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if not self._is_valid_move(my_pos, move, map_state):
                continue
                
            dr, dc = move.value
            new_pos = (my_pos[0] + dr, my_pos[1] + dc)
            
            # Tính điểm cho nước đi
            priority = 0
            
            # 1. Thoát khỏi tầm nhìn chữ thập
            curr_in_cross = self._in_cross_vision(my_pos, threat_pos)
            new_in_cross = self._in_cross_vision(new_pos, threat_pos)
            if curr_in_cross and not new_in_cross:
                priority += 10
            
            # 2. Tăng khoảng cách Manhattan
            old_dist = self._manhattan_distance(my_pos, threat_pos)
            new_dist = self._manhattan_distance(new_pos, threat_pos)
            if new_dist > old_dist:
                priority += 2
                
            # 3. Ưu tiên di chuyển chéo (thay đổi hàng nếu đang cùng hàng, cột nếu cùng cột)
            if my_row == threat_row: # Đang cùng hàng -> Ưu tiên đi dọc (UP/DOWN)
                if move in [Move.UP, Move.DOWN]: priority += 5
            elif my_col == threat_col: # Đang cùng cột -> Ưu tiên đi ngang (LEFT/RIGHT)
                if move in [Move.LEFT, Move.RIGHT]: priority += 5
                
            if priority > max_priority:
                max_priority = priority
                best_move = move
                
        return best_move

    def _immediate_evade(self, my_pos: tuple, threat_pos: tuple, map_state: np.ndarray):
        """Chạy ra xa nhất có thể"""
        best_move = None
        max_dist = -1
        
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if self._is_valid_move(my_pos, move, map_state):
                dr, dc = move.value
                new_pos = (my_pos[0] + dr, my_pos[1] + dc)
                dist = self._manhattan_distance(new_pos, threat_pos)
                if dist > max_dist:
                    max_dist = dist
                    best_move = move
        return best_move

    def _bfs_path(self, start: tuple, goal: tuple, map_state: np.ndarray):
        queue = deque([(start, [start])])
        visited = {start}
        while queue:
            pos, path = queue.popleft()
            if pos == goal:
                return path
            
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move.value
                neighbor = (pos[0] + dr, pos[1] + dc)
                if neighbor not in visited and self._is_valid_position(neighbor, map_state):
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None

    def _record_move(self, move: Move):
        self.last_moves.append(move)
        if len(self.last_moves) > 4:
            self.last_moves.pop(0)
    
    def _is_recent_move(self, move: Move) -> bool:
        # Giữ logic chống lặp lại đơn giản
        if len(self.last_moves) >= 2:
            # Nếu vừa đi UP, không đi DOWN ngay lập tức trừ khi đường cùng
            opposite = {Move.UP: Move.DOWN, Move.DOWN: Move.UP, Move.LEFT: Move.RIGHT, Move.RIGHT: Move.LEFT}
            if move == opposite.get(self.last_moves[-1]):
                return True
        return False

    def _get_move_direction(self, from_pos: tuple, to_pos: tuple) -> Move:
        dr = to_pos[0] - from_pos[0]
        dc = to_pos[1] - from_pos[1]
        if dr > 0: return Move.DOWN
        elif dr < 0: return Move.UP
        elif dc > 0: return Move.RIGHT
        elif dc < 0: return Move.LEFT
        return Move.STAY
        
    def _manhattan_distance(self, pos1: tuple, pos2: tuple) -> int:
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _is_valid_move(self, pos: tuple, move: Move, map_state: np.ndarray) -> bool:
        dr, dc = move.value
        new_pos = (pos[0] + dr, pos[1] + dc)
        return self._is_valid_position(new_pos, map_state)
    
    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        row, col = pos
        h, w = map_state.shape
        if not (0 <= row < h and 0 <= col < w): return False
        return map_state[pos] == 0