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


import numpy as np
import random
import os
import json
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move

class GhostAgent(BaseGhostAgent):
    """
    Ghost Agent - Heuristic Pro + Lookahead Safety Check
    - Vẫn giữ nền tảng Heuristic thông minh (nhanh, mượt).
    - Thêm lớp bảo vệ: Nhìn trước 1 lượt (Ghost đi 1, Pacman lao tới 2) để loại bỏ nước đi tử thần.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # --- CẤU HÌNH THAM SỐ (HARD-CODED) ---
        self.EARLY_GAME_LIMIT = 4  # 4 bước đầu ưu tiên nấp
        self.SAFE_DISTANCE    = 6   # Khoảng cách an toàn
        self.SURVIVAL_HORIZON = 12  # Nhìn trước 12 bước (Ghost 12 - Pacman 24)
        self.VISION_RADIUS    = 5   # Tầm nhìn hình chữ thập: 10 ô (radius 5: ±5 ô theo 4 hướng)

        # --- CẤU HÌNH ---
        self.params = {
            "EARLY_GAME_LIMIT": 4,
            "SAFE_DISTANCE": 6,
            "W_DISTANCE": 37,
            "W_AXIS_PENALTY": 1991,
            "W_CORNER_PENALTY": 1031,
            "W_DEAD_END_BAD": 7569,
            "W_VISIT_PENALTY": 500,
            "W_INERTIA": 50
        }
        
        # --- DATA ---
        self.map_size = None
        self.ghost_map = None  # Bản đồ cá nhân của Ghost (-1 = chưa nhìn thấy, 1 = tường, 0 = trống)
        self.walls = None
        self.dead_ends = None
        self.corners = None
        self.visit_map = None
        self.last_known_enemy_pos = None
        self.last_move = None
        
        self.direction_bias = {Move.UP: 5, Move.DOWN: 0, Move.LEFT: 5, Move.RIGHT: 10}
        
        # --- LOGGING ---
        self.log_file = open("ghost_log.json", "w", encoding="utf-8")
        self.map_file = open("ghost_map.json", "w", encoding="utf-8")  # Chế độ write (ghi đè) để chỉ giữ lại map cuối cùng

    def step(self, map_state, my_pos, enemy_pos, step_num):
        # 1. PRE-COMPUTE
        if self.walls is None:
            self.map_size = map_state.shape
            self.ghost_map = np.full(self.map_size, -1, dtype=np.int8)  # -1 = chưa nhìn thấy
            self.walls = (map_state == 1)
            self.visit_map = np.zeros(self.map_size)
            self._precompute_map_features()
        
        # 2. CẬP NHẬT BẢN ĐỒ RIÊNG (Chỉ nhìn thấy hình chữ thập 5 ô)
        self._update_ghost_map(my_pos, map_state)
        
        self.visit_map[my_pos] += 1
        if step_num <= 10:
            return Move.RIGHT  # Bước đầu tiên đi vào kẹt góc phải cho chắc chắn
        # 3. UPDATE PACMAN
        pacman_visible = enemy_pos is not None
        if pacman_visible:
            self.last_known_enemy_pos = enemy_pos
        else:
            # Pacman không thấy - có thể vừa escape khỏi tầm nhìn hay bị capture
            pass
        
        target_pacman = self.last_known_enemy_pos
        
        # LOG
        log_entry = {
            "step": step_num,
            "ghost_pos": my_pos,
            "pacman_visible": pacman_visible,
            "pacman_pos": target_pacman
        }
        self.log_file.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        self.log_file.flush()
        
        # LOG GHOST MAP (chỉ lưu map cuối cùng, ghi đè các lần trước)
        map_entry = {
            "step": step_num,
            "ghost_pos": my_pos,
            "ghost_map": [self.ghost_map[i].tolist() for i in range(self.map_size[0])]
        }
        # Mở file ở chế độ write (ghi đè) để chỉ giữ lại dòng cuối cùng
        with open("ghost_map.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(map_entry, ensure_ascii=False) + "\n")
            
        # 4. EARLY GAME
        is_safe = True
        if target_pacman and self._manhattan_distance(my_pos, target_pacman) <= 5: 
            is_safe = False
        
        # Trong early game, ưu tiên nấp nếu còn an toàn và pacman visible
        if step_num <= self.EARLY_GAME_LIMIT:
            if self._is_good_hiding_spot(my_pos): 
                return Move.RIGHT
            move = self._find_nearest_cover(my_pos)
            if move != Move.STAY:
                return move
        
        # 5. MAIN LOGIC
        if target_pacman:
            # Chạy có tính toán sâu (Deep Check)
            final_move = self._momentum_aware_escape(my_pos, target_pacman)
        else:
            # Khi không thấy pacman: exploration + tránh cẫm
            final_move = self._safe_exploration(my_pos)
            
        self.last_move = final_move
        return final_move

    def _update_ghost_map(self, my_pos, map_state):
        """
        Cập nhật bản đồ riêng của Ghost dựa trên tầm nhìn hình chữ thập (10 ô).
        Tầm nhìn bán kính 5: nhìn thấy ±5 ô theo mỗi hướng chính (UP, DOWN, LEFT, RIGHT).
        """
        r, c = my_pos
        h, w = self.map_size
        
        # Cập nhật vị trí hiện tại
        self.ghost_map[r, c] = map_state[r, c]
        
        # Cập nhật 4 hướng chính (chữ thập), mỗi hướng radius 5
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            for dist in range(1, 6):  # 1 đến 5 bước theo hướng
                nr, nc = r + dr*dist, c + dc*dist
                if 0 <= nr < h and 0 <= nc < w:
                    self.ghost_map[nr, nc] = map_state[nr, nc]

    # =========================================================================
    # LOGIC NÉ ĐÒN VỚI TẦM NHÌN SÂU (BFS SURVIVAL)
    # =========================================================================

    def _momentum_aware_escape(self, my_pos, pacman_pos):
        valid_moves = self._get_valid_moves(my_pos)
        if not valid_moves: return Move.STAY
        
        candidates = []
        on_highway = self._is_on_same_axis(my_pos, pacman_pos)
        
        # BƯỚC 1: Lọc bằng Heuristic (Nhanh)
        for move in valid_moves:
            next_pos = self._get_next_pos(my_pos, move)
            score = 0
            
            d = self._manhattan_distance(next_pos, pacman_pos)
            score += d * self.params["W_DISTANCE"]
            
            # Phạt nặng địa hình xấu
            if next_pos in self.dead_ends: score -= self.params["W_DEAD_END_BAD"]
            if next_pos in self.corners:   score -= self.params["W_CORNER_PENALTY"]
            if self._is_on_same_axis(next_pos, pacman_pos): score -= self.params["W_AXIS_PENALTY"]
            
            if self.last_move and move == self.last_move and not on_highway:
                score += self.params["W_INERTIA"]
            
            score += self.direction_bias.get(move, 0)
            candidates.append((score, move, next_pos))
            
            # LOG HEURISTIC
            move_log = {"move": move.name, "next_pos": next_pos, "distance": d, "score": score}
            self.log_file.write(json.dumps(move_log, ensure_ascii=False) + "\n")
            self.log_file.flush()

        # Sắp xếp điểm cao nhất lên đầu
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        # BƯỚC 2: Kiểm tra sinh tồn chiều sâu (Chậm hơn nên chỉ check Top 3)
        # Chỉ cần tìm được 1 nước đi trong Top 3 mà sống sót được qua 12 bước là CHỐT luôn.
        
        for score, move, next_pos in candidates[:3]:
            # Gọi hàm BFS check sâu
            if self._is_safe_deep_check(next_pos, pacman_pos):
                chosen_log = {"chosen": move.name, "next_pos": next_pos, "score": score, "safe": True}
                self.log_file.write(json.dumps(chosen_log, ensure_ascii=False) + "\n")
                self.log_file.flush()
                return move # Tìm thấy đường sống -> Đi ngay
            
        # Nếu cả 3 nước tốt nhất đều dẫn đến cái chết sau N bước -> Rất nguy hiểm
        # Fallback: Vẫn chọn nước có điểm Heuristic cao nhất (hy vọng Pacman sai lầm)
        if candidates:
            score, move, next_pos = candidates[0]
            chosen_log = {"chosen": move.name, "next_pos": next_pos, "score": score, "safe": False}
            self.log_file.write(json.dumps(chosen_log, ensure_ascii=False) + "\n")
            self.log_file.flush()
            return move
            
        return Move.STAY

    def _is_safe_deep_check(self, start_node, pacman_pos):
        """
        Dùng BFS để kiểm tra: Liệu từ start_node, Ghost có thể sống sót
        trong self.SURVIVAL_HORIZON bước tiếp theo không?
        """
        # Hàng đợi BFS: (Vị trí Ghost, Thời gian t)
        queue = deque([(start_node, 1)])
        visited = set([(start_node, 1)]) # Visited theo (pos, time) để cho phép quay đầu nếu cần
        
        initial_pacman_dist = self._manhattan_distance(start_node, pacman_pos)
        
        # Nếu ngay bước đầu đã bị bắt -> False
        if initial_pacman_dist <= 2: return False 

        while queue:
            curr_pos, time = queue.popleft()
            
            # Nếu đã sống sót đủ lâu (vượt qua Horizon) -> Nước đi này AN TOÀN
            if time >= self.SURVIVAL_HORIZON:
                return True
            
            # Mở rộng các nước đi tiếp theo của Ghost
            next_moves = self._get_valid_moves_coords(curr_pos)
            
            for next_pos in next_moves:
                # KIỂM TRA TỬ THẦN:
                # Tại thời điểm (time + 1), Ghost ở next_pos.
                # Pacman đã di chuyển tổng cộng (time + 1) * 2 bước.
                # Khoảng cách an toàn tối thiểu cần thiết = (time + 1) * 2
                
                # Tuy nhiên, tính chính xác Pacman đi đường nào rất khó (vì tường).
                # Ta dùng "Vùng Nguy Hiểm Ước Lượng" (Conservative Estimate):
                # Giả sử Pacman đi xuyên tường (hoặc đi tối ưu nhất) để bắt mình.
                # Nếu khoảng cách Manhattan hiện tại > Tốc độ đuổi của Pacman -> Sống.
                
                # Pacman Speed 2 thu hẹp khoảng cách tối đa 1 ô mỗi lượt (Nó đi 2, mình đi 1 -> Net = 1)
                # Vậy sau t lượt, Pacman gần thêm t ô.
                # Điều kiện sống: Dist_Ban_Dau - t > 1 (để không bị bắt)
                
                # Cách kiểm tra chặt chẽ hơn:
                # Tính khoảng cách thực tế tới Pacman tại vị trí gốc
                dist_to_pacman_origin = self._manhattan_distance(next_pos, pacman_pos)
                
                # Pacman có thể đi tối đa (time + 1) * 2 bước.
                # Nếu khoảng cách < Tầm với của Pacman -> Coi như chết (cho an toàn)
                # Lưu ý: Đây là check "Worst Case" (Pacman đi xuyên tường). 
                # Nếu qua được bài test này thì 100% sống.
                if dist_to_pacman_origin <= (time + 1) * 2:
                    continue # Nhánh này chết, bỏ qua
                
                # Nếu sống, thêm vào hàng đợi để check tiếp bước sau
                state = (next_pos, time + 1)
                if state not in visited:
                    visited.add(state)
                    queue.append(state)
        
        # Nếu đi hết tất cả các nhánh mà không nhánh nào chạm mốc Horizon -> Chết chắc
        return False

    def _check_future_safety(self, ghost_future_pos, pacman_current_pos):
        """
        Giả lập Pacman Speed 2 đuổi theo Ghost.
        Trả về True nếu Ghost còn sống sau lượt này, False nếu chết.
        """
        # 1. Pacman bước 1 (Greedy về phía Ghost)
        p_step_1 = self._predict_pacman_greedy(pacman_current_pos, ghost_future_pos)
        if p_step_1 == ghost_future_pos: return False # Bị bắt ngay bước 1
        
        # 2. Pacman bước 2
        p_step_2 = self._predict_pacman_greedy(p_step_1, ghost_future_pos)
        if p_step_2 == ghost_future_pos: return False # Bị bắt bước 2
        
        # 3. Kiểm tra xem sau khi chạy xong, Ghost có bị dồn vào đường cụt không?
        # Nếu vị trí tương lai là ngõ cụt và Pacman đang bịt cửa -> Chết chắc
        if ghost_future_pos in self.dead_ends:
             # Nếu Pacman đang ở rất gần (<=3 ô) mà mình chui vào ngõ cụt -> Coi như chết
             if self._manhattan_distance(ghost_future_pos, p_step_2) <= 3:
                 return False

        return True

    def _predict_pacman_greedy(self, p_pos, g_pos):
        """Dự đoán Pacman đi đâu (giả định nó đi hướng ngắn nhất tới mình)"""
        best_p = p_pos
        min_dist = float('inf')
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            np_pos = (p_pos[0]+dr, p_pos[1]+dc)
            # Pacman không đi xuyên tường
            if self._is_valid_coord(np_pos):
                d = self._manhattan_distance(np_pos, g_pos)
                if d < min_dist:
                    min_dist = d
                    best_p = np_pos
        return best_p

    # =========================================================================
    # CÁC HÀM CŨ (GIỮ NGUYÊN)
    # =========================================================================
    def _safe_exploration(self, my_pos):
        """Khám phá an toàn: ưu tiên ô chưa đi qua, tránh cẫm"""
        valid_moves = self._get_valid_moves(my_pos)
        if not valid_moves: 
            return Move.STAY
        
        # Phân loại nước đi
        good_moves = []  # Ô chưa đi qua và không phải cẫm
        okay_moves = []  # Ô chưa đi qua nhưng là cẫm
        bad_moves = []   # Ô đã đi qua
        
        for move in valid_moves:
            next_pos = self._get_next_pos(my_pos, move)
            visits = self.visit_map[next_pos]
            is_dead_end = next_pos in self.dead_ends
            
            if visits == 0:
                if not is_dead_end:
                    good_moves.append((move, next_pos))
                else:
                    okay_moves.append((move, next_pos))
            else:
                bad_moves.append((move, next_pos))
        
        # Ưu tiên: ô chưa đi + không cẫm > ô chưa đi + cẫm > ô đã đi
        candidates = good_moves or okay_moves or bad_moves
        
        if not candidates:
            return Move.STAY
        
        # Chọn từ best candidates
        best_move = candidates[0][0]
        best_score = -float('inf')
        
        for move, next_pos in candidates:
            score = 0
            score += self.direction_bias.get(move, 0)
            if self.last_move and move == self.last_move:
                score += self.params["W_INERTIA"]
            if score > best_score:
                best_score = score
                best_move = move
            
            # LOG EXPLORATION
            move_log = {"move": move.name, "next_pos": next_pos, "distance": 0, "score": score, "mode": "exploration"}
            self.log_file.write(json.dumps(move_log, ensure_ascii=False) + "\n")
            self.log_file.flush()
        
        chosen_log = {"chosen": best_move.name, "mode": "exploration", "safe": True}
        self.log_file.write(json.dumps(chosen_log, ensure_ascii=False) + "\n")
        self.log_file.flush()
        
        return best_move

    def _smart_exploration(self, my_pos):
        valid_moves = self._get_valid_moves(my_pos)
        if not valid_moves: return Move.STAY
        
        moves_list = list(valid_moves)
        random.shuffle(moves_list)
        best_move = Move.STAY
        best_score = -float('inf')

        for move in moves_list:
            next_pos = self._get_next_pos(my_pos, move)
            score = 0
            visits = self.visit_map[next_pos]
            if visits == 0: score += 500
            else: score -= visits * self.params["W_VISIT_PENALTY"]
            if next_pos in self.dead_ends: score -= 2000
            if self.last_move and move == self.last_move: score += self.params["W_INERTIA"]
            score += self.direction_bias.get(move, 0)
            if score > best_score:
                best_score = score
                best_move = move
        return best_move

    def _precompute_map_features(self):
        """
        Tính toán các tính năng bản đồ (dead-end, corner).
        Dùng self.walls (từ map_state gốc) để xác định chính xác các bức tường.
        """
        self.dead_ends = set()
        self.corners = set()
        h, w = self.map_size
        for r in range(h):
            for c in range(w):
                if self.walls[r, c]: 
                    continue
                # Đếm bao nhiêu hướng bị chặn bởi tường hoặc biên
                walls = 0
                for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                    nr, nc = r+dr, c+dc
                    if not (0 <= nr < h and 0 <= nc < w) or self.walls[nr, nc]:
                        walls += 1
                if walls >= 3: 
                    self.dead_ends.add((r, c))
                elif walls >= 2: 
                    self.corners.add((r, c))

    def _get_valid_moves_coords(self, pos):
        """Trả về list toạ độ (row, col) đi được theo bản đồ riêng của Ghost"""
        valid = []
        r, c = pos
        h, w = self.map_size
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < h and 0 <= nc < w:
                cell_value = self.ghost_map[nr, nc]
                # Nếu là -1 (chưa nhìn thấy) hoặc 0 (trống), có thể đi
                if cell_value != 1:
                    valid.append((nr, nc))
        return valid

    def _is_good_hiding_spot(self, pos):
        return (pos in self.corners) or (pos in self.dead_ends)

    def _find_nearest_cover(self, my_pos):
        valid_moves = self._get_valid_moves(my_pos)
        if not valid_moves: return Move.STAY
        for move in valid_moves:
            next_pos = self._get_next_pos(my_pos, move)
            if (next_pos in self.dead_ends) or (next_pos in self.corners):
                return move
        return valid_moves[0]

    def _get_valid_moves(self, pos):
        valid = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if self._is_valid_move(pos, move): valid.append(move)
        return valid

    def _is_valid_move(self, pos, move):
        delta_row, delta_col = move.value
        nr, nc = pos[0] + delta_row, pos[1] + delta_col
        return self._is_valid_coord((nr, nc))

    def _is_valid_coord(self, pos):
        r, c = pos
        h, w = self.map_size
        if r < 0 or r >= h or c < 0 or c >= w: 
            return False
        # Kiểm tra xem ô này là tường theo bản đồ riêng
        cell_value = self.ghost_map[r, c]
        # Nếu là -1 (chưa nhìn thấy), cứ coi là có thể đi được
        # Nếu là 1 (tường), không đi được
        # Nếu là 0 (trống), đi được
        return cell_value != 1

    def _get_next_pos(self, pos, move):
        delta_row, delta_col = move.value
        return (pos[0] + delta_row, pos[1] + delta_col)

    def _manhattan_distance(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def _is_on_same_axis(self, pos1, pos2):
        return pos1[0] == pos2[0] or pos1[1] == pos2[1]