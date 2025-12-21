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
from collections import deque
import heapq


import numpy as np
import random
import heapq

class PacmanAgent(BasePacmanAgent):
    """
    Pure A* Agent: 
    - Luôn chọn đường ngắn nhất.
    - Ưu tiên mục tiêu GẦN NHẤT.
    - Tận dụng tối đa luật đi 2 bước.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Pure A* Pacman (Fixed)"
        # Lấy tốc độ từ config, mặc định là 2
        speed_conf = kwargs.get("pacman_speed", kwargs.get("pacman-speed", 2))
        self.pacman_speed = max(1, int(speed_conf))
        
        self.vision_range = 5
        self.last_known_enemy_pos = None 
        self.internal_map = None 
        self.height = 0
        self.width = 0
        
        
    def step(self, map_state, my_position, enemy_position, step_number):
        # 1. Init
        if self.internal_map is None:
            self.height, self.width = map_state.shape
            self.internal_map = np.full((self.height, self.width), -1, dtype=int)
            
        # 2. Update Vision
        self._update_vision(my_position, map_state)
        
        # 3. LOGIC CHÍNH
        
        # ƯU TIÊN 1: GHOST (Nếu nhìn thấy hoặc nhớ vị trí)
        target_ghost = None
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
            target_ghost = enemy_position
        elif self.last_known_enemy_pos is not None:
            if my_position == self.last_known_enemy_pos:
                self.last_known_enemy_pos = None
            else:
                target_ghost = self.last_known_enemy_pos
        
        # Nếu có mục tiêu Ghost, thử A* tới đó
        if target_ghost:
            move = self._try_move_to_target(my_position, target_ghost, map_state)
            if move: return move
        
        # ƯU TIÊN 2: KHÁM PHÁ (Frontier Exploration)
        # Lấy danh sách tất cả frontier, sắp xếp từ gần đến xa
        sorted_frontiers = self._get_sorted_frontiers(my_position)
        
        # Thử lần lượt từng frontier. 
        # Nếu frontier gần nhất không đi được (A* fail), thử cái tiếp theo.
        for target in sorted_frontiers:
            move = self._try_move_to_target(my_position, target, map_state)
            if move: # Nếu tìm thấy đường đi hợp lệ
                return move
            
        # Nếu không đi đâu được cả -> Random để thoát kẹt
        return self._explore_randomly(my_position, map_state)

    def _get_sorted_frontiers(self, my_pos):
        """
        Trả về danh sách các Frontier được sắp xếp theo khoảng cách Manhattan tăng dần.
        """
        frontiers = []
        for r in range(self.height):
            for c in range(self.width):
                # Chỉ xét ô đường (0) và KHÔNG phải vị trí đang đứng
                if self.internal_map[r, c] == 0 and (r, c) != my_pos:
                    if self._is_frontier_cell(r, c):
                        dist = abs(r - my_pos[0]) + abs(c - my_pos[1])
                        frontiers.append(((r, c), dist))
        
        # Sort theo distance (phần tử thứ 2 trong tuple)
        frontiers.sort(key=lambda x: x[1])
        
        # Trả về list tọa độ đã sort
        return [f[0] for f in frontiers]

    def _try_move_to_target(self, start, goal, map_state):
        """
        Thử tìm đường A* đến mục tiêu. 
        Trả về (move, steps) nếu thành công, None nếu thất bại.
        """
        path = self._a_star_search(start, goal)
        
        if path:
            first_move = path[0]
            desired_steps = 1
            if len(path) >= 2 and path[0] == path[1]:
                desired_steps = 2
            
            steps = self._max_valid_steps(start, first_move, map_state, desired_steps)
            return (first_move, steps)
            
        return None # A* không tìm thấy đường

    def _find_nearest_frontier(self, my_pos):
        """
        Tìm Frontier gần nhất theo khoảng cách Manhattan.
        Đây là cách thuần túy nhất để mở rộng bản đồ.
        """
        best_target = None
        min_dist = float('inf')
        
        for r in range(self.height):
            for c in range(self.width):
                # Chỉ xét ô đường (0) và KHÔNG phải vị trí đang đứng
                if self.internal_map[r, c] == 0 and (r, c) != my_pos:
                    # Kiểm tra xem có phải Frontier không
                    if self._is_frontier_cell(r, c):
                        # Tính khoảng cách Manhattan thuần
                        dist = abs(r - my_pos[0]) + abs(c - my_pos[1])
                        if dist < min_dist:
                            min_dist = dist
                            best_target = (r, c)
                            
        return best_target

    def _move_with_a_star(self, start, goal, map_state):
        """
        Dùng A* để đi đến đích nhanh nhất.
        Kết hợp logic đi 2 bước nếu đường thẳng.
        """
        path = self._a_star_search(start, goal)
        
        if path:
            first_move = path[0]
            desired_steps = 1
            
            # Logic tối ưu bước đi (Luật game)
            if len(path) >= 2 and path[0] == path[1]:
                desired_steps = 2
            
            # Kiểm tra thực tế trên map
            steps = self._max_valid_steps(start, first_move, map_state, desired_steps)
            return (first_move, steps)
            
        return (Move.STAY, 1)

    # --- CÁC HÀM CƠ BẢN (KHÔNG ĐỔI) ---
    
    def _update_vision(self, my_pos, true_map_state):
        r, c = my_pos
        self.internal_map[r, c] = true_map_state[r, c]
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            for dist in range(1, self.vision_range + 1):
                nr, nc = r + (dr * dist), c + (dc * dist)
                if 0 <= nr < self.height and 0 <= nc < self.width:
                    val = true_map_state[nr, nc]
                    self.internal_map[nr, nc] = val
                    if val == 1: break
                else: break

    def _is_frontier_cell(self, r, c):
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.height and 0 <= nc < self.width:
                if self.internal_map[nr, nc] == -1: return True
        return False

    def _a_star_search(self, start, goal):
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: abs(start[0]-goal[0]) + abs(start[1]-goal[1])}
        open_set_hash = {start}

        while open_set:
            current_f, current = heapq.heappop(open_set)
            open_set_hash.discard(current)
            if current == goal: return self._reconstruct_path(came_from, current)

            r, c = current
            moves_check = [(Move.UP, (-1, 0)), (Move.LEFT, (0, -1)), (Move.DOWN, (1, 0)), (Move.RIGHT, (0, 1))]
            for move_enum, (dr, dc) in moves_check:
                neighbor = (r + dr, c + dc)
                nr, nc = neighbor
                if 0 <= nr < self.height and 0 <= nc < self.width:
                    # Chỉ đi qua đường đã biết (0) hoặc đích
                    if (self.internal_map[nr, nc] == 0) or (neighbor == goal):
                        tentative_g = g_score[current] + 1
                        if tentative_g < g_score.get(neighbor, float('inf')):
                            came_from[neighbor] = (current, move_enum)
                            g_score[neighbor] = tentative_g
                            f_score[neighbor] = tentative_g + abs(nr-goal[0]) + abs(nc-goal[1])
                            if neighbor not in open_set_hash:
                                heapq.heappush(open_set, (f_score[neighbor], neighbor))
                                open_set_hash.add(neighbor)
        return None

    def _reconstruct_path(self, came_from, current):
        total_path = []
        while current in came_from:
            parent, move = came_from[current]
            total_path.append(move)
            current = parent
        total_path.reverse()
        return total_path

    def _explore_randomly(self, my_position, map_state):
        all_moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        random.shuffle(all_moves)
        for move in all_moves:
            steps = self._max_valid_steps(my_position, move, map_state, 1)
            if steps > 0: return (move, steps)
        return (Move.STAY, 1)

    def _is_valid_position(self, pos, map_state):
        row, col = pos
        h, w = map_state.shape
        if row < 0 or row >= h or col < 0 or col >= w: return False
        return map_state[row, col] == 0

    def _max_valid_steps(self, pos, move, map_state, desired_steps):
        steps = 0
        max_steps = min(self.pacman_speed, max(1, desired_steps))
        current = pos
        for _ in range(max_steps):
            dr, dc = move.value
            next_pos = (current[0] + dr, current[1] + dc)
            if not self._is_valid_position(next_pos, map_state): break
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