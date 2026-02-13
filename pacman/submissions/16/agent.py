import numpy as np
import collections
import random
from environment import Move
from agent_interface import GhostAgent as BaseGhostAgent
import sys
from pathlib import Path
from collections import deque
import heapq
import itertools

src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np


class PacmanAgent(BasePacmanAgent):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        
        # Biến nhớ vị trí cuối cùng nhìn thấy kẻ thù
        self.last_known_enemy_pos = None
        
    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int):
        
        # Nếu nhìn thấy địch, cập nhật vị trí vào bộ nhớ
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position

        # Nếu địch đã chạy mất -> Xóa ký ức
        if my_position == self.last_known_enemy_pos and enemy_position is None:
            self.last_known_enemy_pos = None

        target = None

        # Nếu nhìn thấy địch trực tiếp, lao vào bắt ngay.
        if enemy_position is not None:
            target = enemy_position
        
        # Không thấy địch nhưng nhớ vị trí cũ -> Đến đó kiểm tra
        elif self.last_known_enemy_pos is not None:
            target = self.last_known_enemy_pos
        
        # Nếu không có thông tin -> Tìm ô sương mù gần nhất để mở rộng tầm nhìn.
        else:
            target = self.find_nearest_unseen(my_position, map_state)

        if target:
            # Dùng thuật toán A* tìm đường đi ngắn nhất đến mục tiêu
            path = self.astar(my_position, target, map_state)
            
            if path:
                first_move = path[0]
                steps_to_take = 1
                
                while steps_to_take < self.pacman_speed and steps_to_take < len(path):
                    if path[steps_to_take] == first_move:
                        steps_to_take += 1
                    else:
                        break
                return (first_move, steps_to_take)
        
        # Nếu không tìm được đường hoặc bị kẹt, đi random 
        return self.random_valid_move_fast(my_position, map_state)


    def astar(self, start, goal, map_state):
        """
        Thuật toán tìm đường A* (A-Star).
        Sử dụng Tie-breaker (bộ đếm) để tránh lỗi so sánh Move trong Priority Queue.
        """
        # Tạo bộ đếm vô tận (0, 1, 2...). Dùng làm yếu tố so sánh phụ trong heap
        counter = itertools.count() 
        
        # Hàng đợi ưu tiên chứa: (F-cost, G-cost, Count, Position, Path)
        frontier = [(0, 0, next(counter), start, [])]
        visited = set()
        
        # Lưu chi phí G thấp nhất đến từng điểm để tối ưu hóa
        g_scores = {start: 0}
        
        while frontier:
            # Lấy phần tử có F-cost nhỏ nhất ra
            _, g_cost, _, current, path = heapq.heappop(frontier)
            
            # Nếu đến đích -> Trả về đường đi
            if current == goal:
                return path
            
            # Nếu đường hiện tại đến 'current' tốn kém hơn đường đã tìm thấy trước đó -> Bỏ qua
            if current in visited and g_cost > g_scores.get(current, float('inf')):
                continue
            visited.add(current)
            
            # Duyệt 4 hướng di chuyển
            for move in [Move.UP, Move.LEFT, Move.RIGHT, Move.DOWN]:
                next_pos = self._apply_move(current, move)
                
                # Kiểm tra ô tiếp theo có đi được không
                if self._is_valid_walkable(next_pos, map_state):
                    new_g = g_cost + 1
                    
                    # Nếu tìm thấy con đường mới rẻ hơn để đến next_pos
                    if new_g < g_scores.get(next_pos, float('inf')):
                        g_scores[next_pos] = new_g
                        # Heuristic: Khoảng cách Manhattan đến đích
                        new_h = abs(next_pos[0] - goal[0]) + abs(next_pos[1] - goal[1])
                        new_path = path + [move]
                        # Đẩy vào hàng đợi
                        heapq.heappush(frontier, (new_g + new_h, new_g, next(counter), next_pos, new_path))
        return []

    def find_nearest_unseen(self, start, map_state):
        """
        Thuật toán BFS (Loang) để tìm ô sương mù (-1) gần nhất.
        Giúp Pacman chủ động đi mở rộng bản đồ thay vì đi random.
        """
        queue = deque([(start)])
        visited = {start}
        while queue:
            current = queue.popleft()
            # Nếu gặp ô -1, đây là mục tiêu gần nhất!
            if map_state[current[0], current[1]] == -1:
                return current
            
            # Loang ra 4 hướng
            for move in [Move.UP, Move.LEFT, Move.RIGHT, Move.DOWN]:
                next_pos = self._apply_move(current, move)
                if self._is_valid_walkable(next_pos, map_state) and next_pos not in visited:
                    visited.add(next_pos)
                    queue.append(next_pos)
        return None # Không tìm thấy 

    def random_valid_move_fast(self, pos, map_state):
        valid_moves = []
        for move in [Move.UP, Move.LEFT, Move.RIGHT, Move.DOWN]:
            # Kiểm tra xem hướng này đi thẳng được tối đa bao nhiêu bước
            steps = 0
            curr_check = pos
            for _ in range(self.pacman_speed):
                next_p = self._apply_move(curr_check, move)
                if self._is_valid_walkable(next_p, map_state):
                    steps += 1
                    curr_check = next_p
                else:
                    break # Gặp tường thì dừng đếm
            
            if steps > 0:
                valid_moves.append((move, steps))
        
        if valid_moves:
            idx = np.random.choice(len(valid_moves))
            return valid_moves[idx]
            
        return (Move.STAY, 1)

    def _apply_move(self, pos, move):
        """Tính tọa độ mới dựa trên nước đi"""
        return (pos[0] + move.value[0], pos[1] + move.value[1])

    def _is_valid_walkable(self, pos, map_state):
        """Kiểm tra ô có đi vào được không (Không phải tường)"""
        h, w = map_state.shape
        if not (0 <= pos[0] < h and 0 <= pos[1] < w):
            return False
        return map_state[pos[0], pos[1]] != 1 



class GhostAgent(BaseGhostAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Sniper Ghost"
        self.grid_size = 21
        self.last_move = None 
        
        # Ma trận niềm tin (Belief Matrix) cho Particle Filter.
        # Ban đầu giả định Pacman có thể ở bất cứ đâu.
        self.belief = np.ones((self.grid_size, self.grid_size))
        
        # Khoảng cách bắt
        self.CAPTURE_DIST = 2 

    def _valid(self, r, c, map_state):
        """Kiểm tra tọa độ hợp lệ và không phải tường"""
        return (
            0 <= r < self.grid_size and
            0 <= c < self.grid_size and
            map_state[r, c] != 1
        )

    def _neighbors(self, pos):
        """Trả về các ô hàng xóm hợp lệ"""
        r, c = pos
        moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        random.shuffle(moves) # Xáo trộn để tạo tính ngẫu nhiên khi các hướng ngang nhau
        for m in moves:
            nr, nc = r + m.value[0], c + m.value[1]
            if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                yield m, (nr, nc)

    def _bfs_distance_map(self, target, map_state):
        """
        Dùng BFS để tính bảng khoảng cách thực tế từ 'target' đến TẤT CẢ các ô khác.
        Giúp Ghost biết đi hướng nào thì gần/xa Pacman nhất (tránh đi vào ngõ cụt).
        """
        dist = np.full((self.grid_size, self.grid_size), 999) # Khởi tạo vô cực
        q = collections.deque([target])
        dist[target] = 0
        while q:
            r, c = q.popleft()
            for m, (nr, nc) in self._neighbors((r, c)):
                if self._valid(nr, nc, map_state) and dist[nr, nc] == 999:
                    dist[nr, nc] = dist[r, c] + 1
                    q.append((nr, nc))
        return dist

    def _manhattan_dist(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def update_belief(self, map_state, my_pos, enemy_pos):
        """Cập nhật ma trận xác suất vị trí của Pacman"""
        
        # 1. Cập nhật dựa trên quan sát (Observation)
        if enemy_pos is not None:
            # Nếu nhìn thấy Pacman -> Xác suất tại đó là 100%, chỗ khác là 0%
            self.belief.fill(0.0)
            self.belief[enemy_pos] = 1.0
            return

        # Nếu không thấy: Chắc chắn không có Pacman
        # map_state != -1 nghĩa là các ô sáng
        visible_mask = (map_state != -1)
        self.belief[visible_mask] = 0.0

        # 2. Lan truyền xác suất (Diffusion)
        # Mô phỏng việc Pacman di chuyển. Vì không biết nó đi đâu, ta tản xác suất ra xung quanh.
        # Lặp 2 lần vì Pacman có thể có tốc độ 2 (đi được 2 ô).
        for _ in range(2): 
            padded = np.pad(self.belief, 1, mode='constant', constant_values=0)
            diffusion = (
                padded[0:-2, 1:-1] +
                padded[2:, 1:-1] +
                padded[1:-1, 0:-2] +
                padded[1:-1, 2:]
            ) 
            self.belief = diffusion
            # Đảm bảo xác suất không lan vào tường
            walls = (map_state == 1)
            self.belief[walls] = 0.0

        # Chuẩn hóa lại để tổng xác suất = 1
        total = np.sum(self.belief)
        if total > 0:
            self.belief /= total
        else:
            # Trường hợp lỗi (tất cả về 0), reset lại đều
            self.belief = np.ones_like(self.belief)
            walls = (map_state == 1)
            self.belief[walls] = 0.0


    def step(self, map_state, my_position, enemy_position, step_number):
        def make_move(move):
            self.last_move = move
            return move

        # 1. Update Belief
        self.update_belief(map_state, my_position, enemy_position)

        # 2. Xác định mối đe dọa (Threat)
        threat_pos = None
        if enemy_position:
            threat_pos = enemy_position
        else:
            # Nếu không thấy, mối đe dọa là nơi có xác suất Pacman cao nhất
            threat_pos = np.unravel_index(np.argmax(self.belief), self.belief.shape)

        # 3. Tính khoảng cách đến mối đe dọa
        dist_map = self._bfs_distance_map(threat_pos, map_state)
        
        candidates = []
        opposites = {Move.UP:Move.DOWN, Move.DOWN:Move.UP, Move.LEFT:Move.RIGHT, Move.RIGHT:Move.LEFT}
        
        for m, (nr, nc) in self._neighbors(my_position):
            if self._valid(nr, nc, map_state):
                candidates.append((m, (nr, nc)))
        
        if not candidates: return Move.STAY

        scores = {}
        for move, nxt in candidates:
            d = dist_map[nxt] # Khoảng cách đến Pacman
            
            # 1. Ưu tiên khoảng cách xa (Run Away)
            # Điểm càng cao nếu càng xa mối đe dọa
            score = d * 10  
            
            # 2. Né những nơi Pacman có thể đang ẩn nấp (Dựa trên Belief)
            # Trừ điểm nặng nếu đi vào vùng có xác suất Pacman cao
            score -= self.belief[nxt] * 500 

            # 3. Phạt nặng nếu đi vào ngõ cụt (Dead End)
            # Kiểm tra xem ô tiếp theo có bao nhiêu lối thoát
            valid_neighbors = sum(1 for _, (r, c) in self._neighbors(nxt) if self._valid(r, c, map_state))
            if valid_neighbors <= 1: 
                score -= 1000

            # 4. Ưu tiên bẻ cua (Juke/Kite) để cắt tầm nhìn
            if self.last_move and move != self.last_move:
                score += 15 

            if self.last_move and move == opposites[self.last_move]:
                score -= 50

            scores[move] = score

        # Chọn nước đi tốt nhất
        best_move = max(scores, key=scores.get)
        return make_move(best_move)