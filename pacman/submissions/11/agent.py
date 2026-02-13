"""
Advanced Agent for Hide and Seek Arena - Limited Vision
- PacmanAgent: A* với tối ưu speed trên đường thẳng
- GhostAgent: Thuật toán heuristic để chạy thoát
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
    Pacman (Seeker) - A* với tối ưu speed
    Chiến thuật: Đuổi theo đường thẳng khi có thể để tận dụng speed=2
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Advanced Pacman"
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 2)))
        
        self.known_map = None
        self.last_known_enemy_pos = None
        self.last_enemy_seen_step = 0
        self.current_path = []
        self.prediction_history = []
        
        # Chiến thuật Check & Switch đầu game
        self.initial_check_mode = 0  # 0: None, 1: Check Left, 2: Force Right
        self.check_steps = 0
    
    def step(self, map_state: np.ndarray, my_position: tuple, 
             enemy_position: tuple, step_number: int):
        
        # Cập nhật bản đồ đã biết
        # RESET game state khi bắt đầu game mới
        if step_number == 1:
            self.known_map = None
            self.last_known_enemy_pos = None
            self.last_enemy_seen_step = 0
            self.prediction_history = []
            self.current_path = []
            self.initial_check_mode = 0
            self.check_steps = 0
        
        if self.known_map is None:
            self.known_map = np.copy(map_state)
        else:
            mask = map_state != -1
            self.known_map[mask] = map_state[mask]
        
        # Cập nhật vị trí địch
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
            self.last_enemy_seen_step = step_number
            self.prediction_history.append(enemy_position)
            if len(self.prediction_history) > 10:
                self.prediction_history.pop(0)
        
        # Chiến thuật
        if enemy_position is not None:
            # Địch visible: đuổi với tối ưu speed
            return self._pursue_enemy(my_position, enemy_position)
        elif self.last_known_enemy_pos is not None:
            # Có thông tin cũ -> Xử lý như bình thường
            staleness = step_number - self.last_enemy_seen_step
            if staleness < 8:
                # Dự đoán vị trí địch và đuổi
                predicted = self._predict_enemy()
                target = predicted if predicted else self.last_known_enemy_pos
                self.initial_check_mode = 0 # Reset check mode khi đã có target
                return self._pursue_enemy(my_position, target)
            else:
                return self._explore(my_position)
        else:
            # Chưa thấy địch bao giờ hoặc mất dấu lâu
            # LOGIC MỚI: Check & Switch tại hàng 9-11
            r, c = my_position
            if 8 <= r <= 11 and step_number < 40:
                if self.initial_check_mode == 0:
                     # Bắt đầu check: Mặc định check TRÁI trước
                     self.initial_check_mode = 1 
                     self.check_steps = 0
                
                if self.initial_check_mode == 1: # Check Left
                    if self.check_steps < 3: # Đi trái 3 nhịp (cover tầm 6-10 ô)
                        self.check_steps += 1
                        # Kiểm tra xem đi được không
                        steps = self._max_steps(my_position, Move.LEFT, self.known_map)
                        if steps > 0:
                            return (Move.LEFT, min(steps, self.pacman_speed))
                        else:
                            # Tắc đường trái -> Switch sang phải luôn
                            self.initial_check_mode = 2
                    else:
                        # Timeout -> Switch sang phải
                        self.initial_check_mode = 2
                
                if self.initial_check_mode == 2: # Force Right
                    steps = self._max_steps(my_position, Move.RIGHT, self.known_map)
                    if steps > 0:
                        return (Move.RIGHT, min(steps, self.pacman_speed))
                    # Nếu tắc phải thì explore bình thường
            
            return self._explore(my_position)
    
    def _predict_ghost_target(self, ghost_pos: tuple, my_pos: tuple, map_state: np.ndarray) -> tuple:
        """
        Dự đoán vị trí Ghost sẽ đến để chặn đầu (Interception).
        Look-ahead dựa trên khoảng cách.
        """
        dist = abs(ghost_pos[0] - my_pos[0]) + abs(ghost_pos[1] - my_pos[1])
        
        # Determine lookahead based on distance
        if dist <= 6:
            return ghost_pos # Chase directly if close enough (speed advantage)
        else:
            lookahead = 4
            
        # Try to estimate ghost direction from history
        dr, dc = 0, 0
        if len(self.prediction_history) >= 2:
            last = self.prediction_history[-1]
            prev = self.prediction_history[-2]
            dr, dc = last[0] - prev[0], last[1] - prev[1]
        
        # If no history/movement, verify valid neighbors to guess direction or just target current
        if dr == 0 and dc == 0:
             return ghost_pos

        # Simulate ghost movement
        curr_r, curr_c = ghost_pos
        for _ in range(lookahead):
            next_r, next_c = curr_r + dr, curr_c + dc
            if self._is_valid((next_r, next_c), map_state):
                curr_r, curr_c = next_r, next_c
            else:
                # Ghost hit a wall, simplistic assumption: it stops or turns. 
                # For interception, aiming at the corner/wall is often good enough to trap.
                break
                
        return (curr_r, curr_c)

    def _predict_enemy(self) -> tuple:
        """Dự đoán vị trí địch khi mất dấu (cho logic explore/chase blind)"""
        if len(self.prediction_history) < 2:
            return None
        last = self.prediction_history[-1]
        prev = self.prediction_history[-2]
        dr, dc = last[0] - prev[0], last[1] - prev[1]
        
        # Simple linear projection for blind chase
        predicted = (last[0] + dr * 2, last[1] + dc * 2)
        if self._is_valid(predicted, self.known_map):
            return predicted
        return last
    
    def _pursue_enemy(self, my_pos: tuple, enemy_pos: tuple):
        """Đuổi địch với A* và tối ưu speed + INTERCEPTION"""
        # Determine interception target
        target_pos = self._predict_ghost_target(enemy_pos, my_pos, self.known_map)
        
        # Use A* to get to the interception point
        path = self._astar(my_pos, target_pos)
        
        # Fallback: if cannot path to intercept (e.g. wall/blocked), path to ghost directly
        if not path:
             path = self._astar(my_pos, enemy_pos)
        
        if path and len(path) > 0:
            first_move = path[0]
            # Đếm số bước liên tiếp cùng hướng
            consecutive = self._count_consecutive(path, first_move)
            # Số bước có thể đi
            available = self._max_steps(my_pos, first_move, self.known_map)
            
            # QUAN TRỌNG: Nếu trên đường thẳng với địch, đi tối đa!
            dr, dc = first_move.value
            if dr != 0 and my_pos[1] == enemy_pos[1]:
                # Cùng cột - đường thẳng dọc
                optimal = min(available, self.pacman_speed)
            elif dc != 0 and my_pos[0] == enemy_pos[0]:
                # Cùng hàng - đường thẳng ngang
                optimal = min(available, self.pacman_speed)
            else:
                # Không thẳng hàng - cần rẽ
                optimal = min(consecutive, available, self.pacman_speed)
            
            return (first_move, max(1, optimal))
        
        return self._greedy_toward(my_pos, enemy_pos)
    
    def _explore(self, my_pos: tuple):
        """Khám phá khu vực chưa biết"""
        target = self._find_unexplored(my_pos)
        if target:
            path = self._astar(my_pos, target)
            if path and len(path) > 0:
                first_move = path[0]
                consecutive = self._count_consecutive(path, first_move)
                available = self._max_steps(my_pos, first_move, self.known_map)
                optimal = min(consecutive, available, self.pacman_speed)
                return (first_move, max(1, optimal))
        
        # Random exploration
        moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        random.shuffle(moves)
        for move in moves:
            steps = self._max_steps(my_pos, move, self.known_map)
            if steps > 0:
                return (move, min(steps, self.pacman_speed))
        return (Move.STAY, 1)
    
    def _count_consecutive(self, path: list, target_move: Move) -> int:
        """Đếm số move liên tiếp cùng hướng"""
        count = 0
        for move in path:
            if move == target_move:
                count += 1
            else:
                break
        return count
    
    def _find_unexplored(self, my_pos: tuple) -> tuple:
        """Tìm ô gần nhất cạnh khu vực chưa khám phá"""
        h, w = self.known_map.shape
        min_dist = float('inf')
        best = None
        
        for r in range(h):
            for c in range(w):
                if self.known_map[r, c] == 0:
                    # Kiểm tra có ô -1 cạnh không
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and self.known_map[nr, nc] == -1:
                            dist = abs(my_pos[0] - r) + abs(my_pos[1] - c)
                            if dist < min_dist:
                                min_dist = dist
                                best = (r, c)
                            break
        return best
    
    def _astar(self, start: tuple, goal: tuple) -> list:
        """A* trả về list các Move"""
        if start == goal:
            return []
        
        frontier = [(0, 0, start, [])]
        visited = set()
        counter = 0
        
        while frontier:
            _, _, cur, path = heapq.heappop(frontier)
            
            if cur == goal:
                return path
            
            if cur in visited:
                continue
            visited.add(cur)
            
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move.value
                nxt = (cur[0] + dr, cur[1] + dc)
                if nxt not in visited and self._is_valid(nxt, self.known_map):
                    new_path = path + [move]
                    g = len(new_path)
                    h = abs(nxt[0] - goal[0]) + abs(nxt[1] - goal[1])
                    counter += 1
                    heapq.heappush(frontier, (g + h, counter, nxt, new_path))
        
        return []
    
    def _greedy_toward(self, my_pos: tuple, target: tuple):
        """Di chuyển greedy về phía target"""
        rd, cd = target[0] - my_pos[0], target[1] - my_pos[1]
        
        moves = []
        if abs(rd) >= abs(cd):
            moves = [Move.DOWN if rd > 0 else Move.UP, Move.RIGHT if cd > 0 else Move.LEFT]
        else:
            moves = [Move.RIGHT if cd > 0 else Move.LEFT, Move.DOWN if rd > 0 else Move.UP]
        moves.extend([Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT])
        
        for move in moves:
            steps = self._max_steps(my_pos, move, self.known_map)
            if steps > 0:
                return (move, min(steps, self.pacman_speed))
        return (Move.STAY, 1)
    
    def _max_steps(self, pos: tuple, move: Move, map_state: np.ndarray) -> int:
        """Số bước tối đa có thể đi theo hướng move"""
        steps = 0
        cur = pos
        dr, dc = move.value
        for _ in range(self.pacman_speed):
            nxt = (cur[0] + dr, cur[1] + dc)
            if not self._is_valid(nxt, map_state):
                break
            steps += 1
            cur = nxt
        return steps
    
    def _is_valid(self, pos: tuple, map_state: np.ndarray) -> bool:
        r, c = pos
        h, w = map_state.shape
        if r < 0 or r >= h or c < 0 or c >= w:
            return False
        return map_state[r, c] == 0


class GhostAgent(BaseGhostAgent):
    """
    Ghost (Hider) - TRÁNH ĐƯỜNG THẲNG vì Pacman có speed=2
    Chiến thuật: 
    1. Tránh đường thẳng với Pacman (tránh speed=2)
    2. Tìm khu vực phức tạp nhiều ngã rẽ
    3. Không bao giờ đi vào dead-end
    4. Ưu tiên di chuyển vuông góc với hướng Pacman đến
    """
    
    # Vị trí spawn mặc định (theo map chuẩn)
    DEFAULT_PACMAN_SPAWN = (15, 10)
    DEFAULT_GHOST_SPAWN = (9, 10)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Advanced Ghost"
        
        self.known_map = None
        self.last_known_enemy_pos = None
        self.last_enemy_seen_step = 0
        self.visited_recently = []
        self.max_recent = 25
        self.safe_zones = []  # Các khu vực an toàn đã tìm được
        
        # QUAN TRỌNG: Giả định Pacman spawn ở vị trí mặc định
        self.assumed_pacman_pos = self.DEFAULT_PACMAN_SPAWN
    
    def step(self, map_state: np.ndarray, my_position: tuple, 
             enemy_position: tuple, step_number: int) -> Move:
        
        # Cập nhật bản đồ
        # RESET game state khi bắt đầu game mới
        if step_number == 1:
            self.known_map = None
            self.last_known_enemy_pos = None
            self.last_enemy_seen_step = 0
            self.visited_recently = []
            self.safe_zones = []
            self.last_escape_direction = None
        
        if self.known_map is None:
            self.known_map = np.copy(map_state)
        else:
            mask = map_state != -1
            self.known_map[mask] = map_state[mask]
        
        # Track visited
        self.visited_recently.append(my_position)
        if len(self.visited_recently) > self.max_recent:
            self.visited_recently.pop(0)
        
        # Cập nhật vị trí địch
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
            self.last_enemy_seen_step = step_number
        
        # Tìm safe zone nếu chưa có
        if step_number == 1:
            self.safe_zones = self._find_safe_zones()
            self.last_escape_direction = None
        
        # ====== CHIẾN THUẬT CHẠY THOÁT NÂNG CAO ======
        
        # Xác định vị trí nguy hiểm (Pacman hoặc giả định)
        if enemy_position is not None:
            danger_pos = enemy_position
            distance = self._manhattan(my_position, enemy_position)
            is_visible = True
        elif self.last_known_enemy_pos is not None:
            danger_pos = self.last_known_enemy_pos
            staleness = step_number - self.last_enemy_seen_step
            # Dự đoán Pacman đang đuổi theo
            distance = self._manhattan(my_position, danger_pos) - staleness * 2
            distance = max(1, distance)
            is_visible = False
        else:
            danger_pos = self.assumed_pacman_pos
            distance = self._manhattan(my_position, danger_pos)
            is_visible = False
        
        # LUÔN dùng BFS để tìm hướng thoát tốt nhất
        return self._bfs_best_escape(my_position, danger_pos, distance, is_visible, step_number)
    
    def _bfs_best_escape(self, my_pos: tuple, danger_pos: tuple, 
                         distance: int, is_visible: bool, step_number: int = 0) -> Move:
        """
        BFS TÌM HƯỚNG THOÁT TỐI ƯU
        Với mỗi hướng có thể đi, dùng BFS để đánh giá:
        1. Khoảng cách xa nhất có thể đạt được từ Pacman
        2. Số lối thoát trên đường đi
        3. Tránh đường thẳng với Pacman ở mọi bước
        """
        moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        best_move = Move.STAY
        best_score = -float('inf')
        
        # Shuffle để không bị dự đoán (trừ bước 1 ưu tiên RIGHT)
        if step_number != 1:
            random.shuffle(moves)
        
        for move in moves:
            if not self._is_valid_move(my_pos, move):
                continue
            
            nxt = self._apply_move(my_pos, move)
            score = 0
            
            # Ưu tiên đi PHẢI ở bước đầu tiên
            if step_number == 1 and move == Move.RIGHT:
                score += 500
            
            # ===== 1. TRÁNH ĐƯỜNG THẲNG - ƯU TIÊN CAO NHẤT =====
            on_same_row = nxt[0] == danger_pos[0]
            on_same_col = nxt[1] == danger_pos[1]
            
            if on_same_row or on_same_col:
                # NGUY HIỂM! Pacman có thể dùng speed=2
                line_dist = abs(nxt[1] - danger_pos[1]) if on_same_row else abs(nxt[0] - danger_pos[0])
                if line_dist <= 3:
                    score -= 500  # CỰC KỲ NGUY HIỂM
                elif line_dist <= 5:
                    score -= 300
                elif line_dist <= 7:
                    score -= 150
                else:
                    score -= 50
            else:
                score += 100  # BONUS LỚN cho vị trí an toàn
            
            # ===== 2. KHOẢNG CÁCH VÀ LOOK-AHEAD =====
            current_dist = self._manhattan(nxt, danger_pos)
            score += current_dist * 15
            
            # BFS look-ahead: tìm khoảng cách xa nhất có thể đạt được
            max_escape_dist, safe_path_score = self._bfs_evaluate_escape(nxt, danger_pos, max_depth=6)
            score += max_escape_dist * 10
            score += safe_path_score * 5
            
            # ===== 3. DEAD-END CHECK - CỰC KỲ QUAN TRỌNG =====
            neighbors = self._count_neighbors(nxt)
            if neighbors == 0:
                score -= 1000  # Không có lối ra = chết
            elif neighbors == 1:
                # Dead-end, chỉ chấp nhận nếu tăng khoảng cách đáng kể
                if current_dist > distance + 3:
                    score -= 100
                else:
                    score -= 400  # Dead-end gần = rất nguy hiểm
            elif neighbors == 2:
                score += 20  # Hành lang, OK
            elif neighbors >= 3:
                score += 50  # Ngã ba/tư = nhiều lựa chọn
            
            # ===== 4. HƯỚNG VUÔNG GÓC VỚI PACMAN =====
            dr = danger_pos[0] - my_pos[0]
            dc = danger_pos[1] - my_pos[1]
            move_dr, move_dc = move.value
            
            # Nếu Pacman đến từ dọc (dr lớn), ta rẽ ngang (move_dc != 0)
            if abs(dr) > abs(dc):
                if move_dc != 0:
                    score += 40  # Rẽ vuông góc
            else:
                if move_dr != 0:
                    score += 40
            
            # ===== 5. TRÁNH ĐI LẠI CHỖ CŨ =====
            if nxt in self.visited_recently:
                count = self.visited_recently.count(nxt)
                score -= count * 20
            
            # ===== 6. ZIGZAG BONUS =====
            if hasattr(self, 'last_escape_direction') and self.last_escape_direction:
                last = self.last_escape_direction
                # Bonus cho đổi hướng (zigzag)
                if last in [Move.UP, Move.DOWN] and move in [Move.LEFT, Move.RIGHT]:
                    score += 30
                elif last in [Move.LEFT, Move.RIGHT] and move in [Move.UP, Move.DOWN]:
                    score += 30
            
            if score > best_score:
                best_score = score
                best_move = move
        
        # Lưu hướng cho zigzag
        if best_move != Move.STAY:
            self.last_escape_direction = best_move
        
        return best_move
    
    def _bfs_evaluate_escape(self, start: tuple, danger_pos: tuple, 
                             max_depth: int) -> tuple:
        """
        BFS đánh giá khả năng thoát từ vị trí start.
        Returns: (max_distance, safe_path_score)
        """
        queue = deque([(start, 0)])
        visited = {start}
        max_dist = self._manhattan(start, danger_pos)
        safe_score = 0
        
        while queue:
            cur, depth = queue.popleft()
            if depth >= max_depth:
                continue
            
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nxt = (cur[0] + dr, cur[1] + dc)
                if nxt in visited or not self._is_valid(nxt, self.known_map):
                    continue
                
                visited.add(nxt)
                dist = self._manhattan(nxt, danger_pos)
                max_dist = max(max_dist, dist)
                
                # Bonus cho vị trí không trên đường thẳng
                if nxt[0] != danger_pos[0] and nxt[1] != danger_pos[1]:
                    safe_score += 1
                
                # Bonus cho junction
                neighbors = self._count_neighbors(nxt)
                if neighbors >= 3:
                    safe_score += 2
                
                queue.append((nxt, depth + 1))
        
        return max_dist, safe_score
    
    def _safe_escape(self, my_pos: tuple, danger_pos: tuple) -> Move:
        """
        Chạy thoát AN TOÀN - KHÔNG BAO GIỜ đi vào cùng hàng/cột với danger_pos
        """
        best_move = Move.STAY
        best_score = -float('inf')
        
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if not self._is_valid_move(my_pos, move):
                continue
            
            nxt = self._apply_move(my_pos, move)
            score = 0
            
            # 1. CRITICAL: Không bao giờ đi vào cùng hàng/cột với danger
            if nxt[0] == danger_pos[0] or nxt[1] == danger_pos[1]:
                score -= 200  # TUYỆT ĐỐI TRÁNH
            else:
                score += 50
            
            # 2. Tăng khoảng cách
            dist = self._manhattan(nxt, danger_pos)
            score += dist * 10
            
            # 3. Số lối thoát
            neighbors = self._count_neighbors(nxt)
            if neighbors >= 3:
                score += 30
            elif neighbors == 2:
                score += 15
            elif neighbors <= 1:
                score -= 80  # Dead end
            
            # 4. Tránh đi lại
            if nxt in self.visited_recently:
                score -= 10 * self.visited_recently.count(nxt)
            
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move

    def _emergency_turn(self, my_pos: tuple, enemy_pos: tuple) -> Move:
        """
        EMERGENCY: Đang trên đường thẳng với Pacman!
        Phải rẽ vuông góc NGAY LẬP TỨC để thoát speed=2 của Pacman.
        """
        on_same_row = my_pos[0] == enemy_pos[0]
        on_same_col = my_pos[1] == enemy_pos[1]
        
        # Xác định hướng vuông góc cần rẽ
        if on_same_row:
            # Cùng hàng -> Rẽ DỌC (UP hoặc DOWN)
            perpendicular_moves = [Move.UP, Move.DOWN]
        else:
            # Cùng cột -> Rẽ NGANG (LEFT hoặc RIGHT)
            perpendicular_moves = [Move.LEFT, Move.RIGHT]
        
        best_move = None
        best_score = -float('inf')
        
        # Ưu tiên các move vuông góc
        for move in perpendicular_moves:
            if not self._is_valid_move(my_pos, move):
                continue
            
            nxt = self._apply_move(my_pos, move)
            score = 0
            
            # 1. Khoảng cách sau khi rẽ
            dist = self._manhattan(nxt, enemy_pos)
            score += dist * 30
            
            # 2. Số lối thoát (tránh dead-end)
            neighbors = self._count_neighbors(nxt)
            if neighbors >= 3:
                score += 50
            elif neighbors == 2:
                score += 20
            elif neighbors == 1:
                score -= 100  # Dead end = chết
            
            # 3. Sau khi rẽ, có còn thẳng hàng không?
            if nxt[0] != enemy_pos[0] and nxt[1] != enemy_pos[1]:
                score += 40  # Không còn thẳng hàng = PERFECT
            
            # 4. Tránh đi lại chỗ cũ
            if nxt in self.visited_recently:
                score -= 15 * self.visited_recently.count(nxt)
            
            if score > best_score:
                best_score = score
                best_move = move
        
        # Nếu không có perpendicular move tốt, thử các move khác
        if best_move is None:
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                if not self._is_valid_move(my_pos, move):
                    continue
                nxt = self._apply_move(my_pos, move)
                
                # Ưu tiên move làm tăng distance
                dist = self._manhattan(nxt, enemy_pos)
                neighbors = self._count_neighbors(nxt)
                
                score = dist * 10 + neighbors * 5
                if nxt[0] != enemy_pos[0] and nxt[1] != enemy_pos[1]:
                    score += 30
                
                if score > best_score:
                    best_score = score
                    best_move = move
        
        return best_move if best_move else Move.STAY
    
    def _zigzag_escape(self, my_pos: tuple, enemy_pos: tuple, step: int) -> Move:
        """
        Chiến thuật Zigzag: Liên tục đổi hướng để Pacman không thể dùng speed=2.
        Ý tưởng: Nếu Pacman chỉ nhanh trên đường thẳng, ta buộc Pacman phải rẽ liên tục.
        """
        # Tìm các move hợp lệ và không phải dead-end
        valid_moves = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if not self._is_valid_move(my_pos, move):
                continue
            nxt = self._apply_move(my_pos, move)
            neighbors = self._count_neighbors(nxt)
            if neighbors >= 2:  # Không đi vào dead-end
                valid_moves.append((move, nxt, neighbors))
        
        if not valid_moves:
            # Không có lựa chọn tốt, dùng critical_evade
            return self._critical_evade(my_pos, enemy_pos)
        
        # Phân loại moves theo hướng (vertical vs horizontal)
        vertical_moves = [(m, p, n) for m, p, n in valid_moves if m in [Move.UP, Move.DOWN]]
        horizontal_moves = [(m, p, n) for m, p, n in valid_moves if m in [Move.LEFT, Move.RIGHT]]
        
        # Hướng Pacman đang đến
        dr = enemy_pos[0] - my_pos[0]
        dc = enemy_pos[1] - my_pos[1]
        
        best_move = None
        best_score = -float('inf')
        
        for move, nxt, neighbors in valid_moves:
            score = 0
            
            # 1. Khoảng cách xa Pacman
            dist = self._manhattan(nxt, enemy_pos)
            score += dist * 20
            
            # 2. PHẢI tránh đường thẳng
            if nxt[0] == enemy_pos[0] or nxt[1] == enemy_pos[1]:
                score -= 150  # Rất nguy hiểm
            else:
                score += 60  # Bonus lớn
            
            # 3. Bonus cho junction
            score += neighbors * 15
            
            # 4. Đổi hướng so với lần trước (zigzag)
            if hasattr(self, 'last_escape_direction') and self.last_escape_direction:
                last_mv = self.last_escape_direction
                # Nếu lần trước đi dọc, lần này ưu tiên ngang và ngược lại
                if last_mv in [Move.UP, Move.DOWN] and move in [Move.LEFT, Move.RIGHT]:
                    score += 40  # Bonus đổi hướng
                elif last_mv in [Move.LEFT, Move.RIGHT] and move in [Move.UP, Move.DOWN]:
                    score += 40
            
            # 5. Bonus nếu di chuyển vuông góc với hướng Pacman đến
            move_dr, move_dc = move.value
            if abs(dr) > abs(dc):  # Pacman đến từ dọc
                if move_dc != 0:  # Ta rẽ ngang
                    score += 35
            else:  # Pacman đến từ ngang
                if move_dr != 0:  # Ta rẽ dọc
                    score += 35
            
            # 6. Tránh đi vào visited
            if nxt in self.visited_recently:
                count = self.visited_recently.count(nxt)
                score -= count * 15
            
            if score > best_score:
                best_score = score
                best_move = move
        
        if best_move:
            self.last_escape_direction = best_move
            return best_move
        
        return self._critical_evade(my_pos, enemy_pos)

    def _critical_evade(self, my_pos: tuple, enemy_pos: tuple) -> Move:
        """
        CRITICAL: Tránh đường thẳng với Pacman!
        Pacman có speed=2 trên đường thẳng nên PHẢI rẽ góc.
        Sử dụng Minimax đơn giản để dự đoán Pacman.
        """
        # Tìm safe zone xa nhất từ Pacman
        best_safe_zone = self._find_nearest_safe_zone_away(my_pos, enemy_pos)
        
        # Đánh giá tất cả các move với look-ahead
        best_move = Move.STAY
        best_score = -float('inf')
        
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if not self._is_valid_move(my_pos, move):
                continue
            
            nxt = self._apply_move(my_pos, move)
            
            # Simulate Pacman chasing (2 bước nếu đường thẳng)
            pacman_next = self._simulate_pacman_chase(enemy_pos, my_pos, speed=2)
            
            score = 0
            
            # 1. Khoảng cách mới từ PACMAN SAU KHI DI CHUYỂN
            new_dist = self._manhattan(nxt, pacman_next)
            score += new_dist * 25
            
            # 2. PHẠT CỰC NẶNG nếu trên đường thẳng với địch
            if nxt[0] == enemy_pos[0]:
                # Cùng hàng - Pacman có thể rush ngang
                col_dist = abs(nxt[1] - enemy_pos[1])
                if col_dist <= 4:
                    score -= 200  # RẤT NGUY HIỂM
                elif col_dist <= 6:
                    score -= 100
                else:
                    score -= 30
            elif nxt[1] == enemy_pos[1]:
                # Cùng cột - Pacman có thể rush dọc  
                row_dist = abs(nxt[0] - enemy_pos[0])
                if row_dist <= 4:
                    score -= 200  # RẤT NGUY HIỂM
                elif row_dist <= 6:
                    score -= 100
                else:
                    score -= 30
            else:
                score += 50  # Bonus lớn khi không thẳng hàng
            
            # 3. BONUS nếu di chuyển hướng về safe zone
            if best_safe_zone:
                dist_to_safe = self._manhattan(nxt, best_safe_zone)
                old_dist_to_safe = self._manhattan(my_pos, best_safe_zone)
                if dist_to_safe < old_dist_to_safe:
                    score += 30  # Đang tiến về safe zone
            # 3. Số lối thoát (critical)
            neighbors = self._count_neighbors(nxt)
            if neighbors >= 3:
                score += 35  # Ngã 3/4 rất tốt
            elif neighbors == 2:
                score += 15
            elif neighbors == 1:
                score -= 100  # Dead end = chết chắc!
            elif neighbors == 0:
                score -= 200  # Không có lối ra
            
            # 4. Tránh đi lại chỗ cũ (tránh bị dồn)
            if nxt in self.visited_recently:
                score -= 20 * self.visited_recently.count(nxt)
            
            # 5. BFS tìm khoảng cách thoát tối đa (look ahead)
            escape_dist = self._bfs_max_escape(nxt, pacman_next, max_depth=8)
            score += escape_dist * 5
            
            # 6. BONUS: Di chuyển vuông góc với hướng Pacman đến
            pacman_dir_row = my_pos[0] - enemy_pos[0]
            pacman_dir_col = my_pos[1] - enemy_pos[1]
            move_dr, move_dc = move.value
            # Nếu Pacman đến từ dọc (pacman_dir_row != 0), ta rẽ ngang (move_dc != 0)
            if pacman_dir_row != 0 and move_dc != 0:
                score += 25
            if pacman_dir_col != 0 and move_dr != 0:
                score += 25
            
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move
    
    def _simulate_pacman_chase(self, pacman_pos: tuple, ghost_pos: tuple, speed: int) -> tuple:
        """Dự đoán vị trí Pacman sau khi đuổi (với speed)"""
        dr = 0 if ghost_pos[0] == pacman_pos[0] else (1 if ghost_pos[0] > pacman_pos[0] else -1)
        dc = 0 if ghost_pos[1] == pacman_pos[1] else (1 if ghost_pos[1] > pacman_pos[1] else -1)
        
        # Nếu thẳng hàng, Pacman di chuyển speed bước
        if pacman_pos[0] == ghost_pos[0] or pacman_pos[1] == ghost_pos[1]:
            steps = speed
        else:
            steps = 1
        
        # Di chuyển theo hướng ưu tiên
        new_pos = pacman_pos
        if abs(ghost_pos[0] - pacman_pos[0]) >= abs(ghost_pos[1] - pacman_pos[1]):
            # Ưu tiên dọc
            for _ in range(steps):
                test = (new_pos[0] + dr, new_pos[1])
                if self._is_valid(test, self.known_map):
                    new_pos = test
        else:
            # Ưu tiên ngang
            for _ in range(steps):
                test = (new_pos[0], new_pos[1] + dc)
                if self._is_valid(test, self.known_map):
                    new_pos = test
        
        return new_pos
    
    def _bfs_max_escape(self, start: tuple, enemy_pos: tuple, max_depth: int) -> int:
        """BFS tìm khoảng cách xa nhất có thể đạt được từ địch"""
        queue = deque([(start, 0)])
        visited = {start}
        max_dist = self._manhattan(start, enemy_pos)
        
        while queue:
            cur, depth = queue.popleft()
            if depth >= max_depth:
                continue
            
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nxt = (cur[0] + dr, cur[1] + dc)
                if nxt not in visited and self._is_valid(nxt, self.known_map):
                    visited.add(nxt)
                    dist = self._manhattan(nxt, enemy_pos)
                    # Bonus nếu không trên đường thẳng
                    if nxt[0] != enemy_pos[0] and nxt[1] != enemy_pos[1]:
                        dist += 2  # Bonus cho vị trí không thẳng hàng
                    max_dist = max(max_dist, dist)
                    queue.append((nxt, depth + 1))
        
        return max_dist
    
    def _move_to_complex_area(self, my_pos: tuple, enemy_pos: tuple) -> Move:
        """Di chuyển vào khu vực có nhiều góc cua, tường - tránh đường thẳng"""
        best_move = Move.STAY
        best_score = -float('inf')
        
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if not self._is_valid_move(my_pos, move):
                continue
            
            nxt = self._apply_move(my_pos, move)
            
            # Score = complexity + distance - straight_line_penalty
            complexity = self._area_complexity(nxt)
            distance = self._manhattan(nxt, enemy_pos)
            
            # PHẠT CỰC NẶNG nếu nằm trên đường thẳng với địch
            straight_penalty = 0
            if nxt[0] == enemy_pos[0]:
                col_dist = abs(nxt[1] - enemy_pos[1])
                straight_penalty = -80 if col_dist <= 6 else -30
            elif nxt[1] == enemy_pos[1]:
                row_dist = abs(nxt[0] - enemy_pos[0])
                straight_penalty = -80 if row_dist <= 6 else -30
            else:
                straight_penalty = 30  # Bonus không thẳng hàng
            
            # Phạt dead end
            neighbors = self._count_neighbors(nxt)
            neighbor_score = 0
            if neighbors >= 3:
                neighbor_score = 25
            elif neighbors == 2:
                neighbor_score = 10
            elif neighbors <= 1:
                neighbor_score = -50  # Dead end nguy hiểm
            
            # Phạt đi lại chỗ cũ
            repeat_penalty = 0
            if nxt in self.visited_recently:
                repeat_penalty = -10 * self.visited_recently.count(nxt)
            
            score = complexity * 2 + distance * 2 + neighbor_score + straight_penalty + repeat_penalty
            
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move
    
    def _cautious_evade(self, my_pos: tuple) -> Move:
        """Né tránh cẩn thận khi không thấy địch nhưng mới gặp"""
        if self.last_known_enemy_pos is None:
            return self._strategic_position(my_pos)
        
        return self._critical_evade(my_pos, self.last_known_enemy_pos)
    
    def _strategic_position(self, my_pos: tuple) -> Move:
        """Di chuyển chiến thuật - ưu tiên ngã 3, ngã 4"""
        best_move = Move.STAY
        best_neighbors = 0
        
        moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        random.shuffle(moves)  # Thêm tính ngẫu nhiên
        
        for move in moves:
            if not self._is_valid_move(my_pos, move):
                continue
            
            nxt = self._apply_move(my_pos, move)
            neighbors = self._count_neighbors(nxt)
            
            # Ưu tiên 3-4 lối thoát (ngã 3, ngã 4)
            score = neighbors * 10
            if nxt in self.visited_recently:
                score -= 5
            
            # Bonus nếu đến safe zone
            if nxt in self.safe_zones:
                score += 15
            
            # Thêm random nhỏ để không dễ đoán
            score += random.uniform(0, 3)
            
            if score > best_neighbors:
                best_neighbors = score
                best_move = move
        
        return best_move
    
    def _run_away(self, my_pos: tuple, enemy_pos: tuple) -> Move:
        """
        Chạy xa khỏi Pacman - ưu tiên tăng khoảng cách VÀ tránh đường thẳng
        """
        best_move = Move.STAY
        best_score = -float('inf')
        
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if not self._is_valid_move(my_pos, move):
                continue
            
            nxt = self._apply_move(my_pos, move)
            score = 0
            
            # 1. Khoảng cách từ Pacman
            dist = self._manhattan(nxt, enemy_pos)
            old_dist = self._manhattan(my_pos, enemy_pos)
            score += (dist - old_dist) * 50  # Reward tăng khoảng cách
            score += dist * 10
            
            # 2. TRÁNH ĐƯỜNG THẲNG
            if nxt[0] == enemy_pos[0] or nxt[1] == enemy_pos[1]:
                score -= 100  # Nguy hiểm!
            else:
                score += 30  # An toàn
            
            # 3. Số lối thoát
            neighbors = self._count_neighbors(nxt)
            if neighbors >= 3:
                score += 25
            elif neighbors == 2:
                score += 10
            elif neighbors <= 1:
                score -= 50  # Dead end
            
            # 4. Tránh đi lại
            if nxt in self.visited_recently:
                score -= 10 * self.visited_recently.count(nxt)
            
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move
    
    def _escape_to_corner(self, my_pos: tuple) -> Move:
        """
        Khi không biết Pacman ở đâu - chạy về góc xa nhất từ center
        Vì Pacman thường spawn ở góc đối diện
        """
        h, w = self.known_map.shape
        center = (h // 2, w // 2)
        
        # Xác định góc xa nhất từ center mà có thể đi được
        corners = [(1, 1), (1, w-2), (h-2, 1), (h-2, w-2)]
        
        # Góc xa nhất từ vị trí hiện tại của mình
        best_corner = max(corners, key=lambda c: self._manhattan(my_pos, c))
        
        # Di chuyển về phía góc đó
        dr = 0 if best_corner[0] == my_pos[0] else (1 if best_corner[0] > my_pos[0] else -1)
        dc = 0 if best_corner[1] == my_pos[1] else (1 if best_corner[1] > my_pos[1] else -1)
        
        # Thử các move ưu tiên
        if abs(best_corner[0] - my_pos[0]) >= abs(best_corner[1] - my_pos[1]):
            primary = Move.DOWN if dr > 0 else Move.UP
            secondary = Move.RIGHT if dc > 0 else Move.LEFT
        else:
            primary = Move.RIGHT if dc > 0 else Move.LEFT
            secondary = Move.DOWN if dr > 0 else Move.UP
        
        for move in [primary, secondary]:
            if self._is_valid_move(my_pos, move):
                nxt = self._apply_move(my_pos, move)
                # Tránh dead end
                if self._count_neighbors(nxt) >= 2:
                    return move
        
        # Fallback
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if self._is_valid_move(my_pos, move):
                return move
        
        return Move.STAY
    
    def _find_safe_zones(self) -> list:
        """Tìm các vị trí có >= 3 lối thoát (ngã 3, ngã 4)"""
        safe = []
        h, w = self.known_map.shape
        for r in range(h):
            for c in range(w):
                if self.known_map[r, c] == 0:
                    if self._count_neighbors((r, c)) >= 3:
                        safe.append((r, c))
        return safe
    

    
    def _find_nearest_safe_zone_away(self, my_pos: tuple, enemy_pos: tuple) -> tuple:
        """Tìm safe zone xa nhất từ enemy mà có thể đến được"""
        if not self.safe_zones:
            return None
        
        best_zone = None
        best_score = -float('inf')
        
        for zone in self.safe_zones:
            dist_from_enemy = self._manhattan(zone, enemy_pos)
            dist_from_me = self._manhattan(zone, my_pos)
            
            # Ưu tiên zone xa enemy nhưng không quá xa từ ta
            score = dist_from_enemy * 2 - dist_from_me * 0.5
            
            # Bonus nếu zone không trên đường thẳng với enemy
            if zone[0] != enemy_pos[0] and zone[1] != enemy_pos[1]:
                score += 10
            
            if score > best_score:
                best_score = score
                best_zone = zone
        
        return best_zone
    
    def _area_complexity(self, pos: tuple) -> int:
        """Đếm độ phức tạp (số tường) xung quanh"""
        complexity = 0
        h, w = self.known_map.shape
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                r, c = pos[0] + dr, pos[1] + dc
                if 0 <= r < h and 0 <= c < w:
                    if self.known_map[r, c] == 1:
                        complexity += 1
        return complexity
    
    def _count_neighbors(self, pos: tuple) -> int:
        """Đếm số ô trống xung quanh (số lối thoát)"""
        count = 0
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nxt = (pos[0] + dr, pos[1] + dc)
            if self._is_valid(nxt, self.known_map):
                count += 1
        return count
    
    def _apply_move(self, pos: tuple, move: Move) -> tuple:
        dr, dc = move.value
        return (pos[0] + dr, pos[1] + dc)
    
    def _is_valid_move(self, pos: tuple, move: Move) -> bool:
        nxt = self._apply_move(pos, move)
        return self._is_valid(nxt, self.known_map)
    
    def _is_valid(self, pos: tuple, map_state: np.ndarray) -> bool:
        r, c = pos
        h, w = map_state.shape
        if r < 0 or r >= h or c < 0 or c >= w:
            return False
        return map_state[r, c] == 0
    
    def _manhattan(self, p1: tuple, p2: tuple) -> int:
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
