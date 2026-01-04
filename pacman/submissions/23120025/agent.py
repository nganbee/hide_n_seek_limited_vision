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
    
    # ========== DEFAULT TUNING CÁC THAM SỐ TẠI ĐÂY ==========
    # Nếu có file weights.json sẽ override những giá trị này
    EARLY_GAME_LIMIT = 16       # Số bước đầu ưu tiên nấp (0-20)
    SAFE_DISTANCE = 5          # Khoảng cách an toàn (0-8)
    W_DISTANCE = 30            # Trọng số khoảng cách (20-60)
    W_AXIS_PENALTY = 1474      # Phạt cùng trục (1000-3000)
    W_CORNER_PENALTY = 810    # Phạt góc (500-1500)
    W_DEAD_END_BAD = 12072      # Phạt ngõ cụt (5000-15000)
    W_VISIT_PENALTY = 719      # Phạt ô cũ (100-800)
    W_INERTIA = 93             # Quán tính (20-100)
    BASE_TRAP_DURATION = 10    # Base duration trong kẹt (3-20)
    # ===============================================
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Load parameters từ weights.json nếu có
        self._load_parameters_from_weights()
        
        # --- CẤU HÌNH THAM SỐ (HARD-CODED hoặc từ JSON) ---
        self.SURVIVAL_HORIZON = 12  # Nhìn trước 12 bước (Ghost 12 - Pacman 24)

        # --- CẤU HÌNH ---
        self.params = {
            "EARLY_GAME_LIMIT": self.EARLY_GAME_LIMIT,
            "SAFE_DISTANCE": self.SAFE_DISTANCE,
            "W_DISTANCE": self.W_DISTANCE,
            "W_AXIS_PENALTY": self.W_AXIS_PENALTY,
            "W_CORNER_PENALTY": self.W_CORNER_PENALTY,
            "W_DEAD_END_BAD": self.W_DEAD_END_BAD,
            "W_VISIT_PENALTY": self.W_VISIT_PENALTY,
            "W_INERTIA": self.W_INERTIA,
            "BASE_TRAP_DURATION": self.BASE_TRAP_DURATION   # Base duration trong kẹt
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
        
        self.direction_bias = {Move.UP: 5, Move.DOWN: 0, Move.LEFT: 5, Move.RIGHT: 6}
        
        # --- TRAPPING LOGIC ---
        self.trapped_pos = None  # Vị trí kẹt
        self.trapped_step = 0     # Số bước đã ở trong kẹt
        
        
        # --- PATTERN RECOGNITION ---
        self.pacman_history = deque(maxlen=10)  # Track Pacman positions
        self.pacman_last_seen_step = 0
        
        # --- EXPLORATION ---
        self.exploration_targets = []  # Frontier cells to explore
        self.last_frontier_update = 0
        
        # --- MEMORY DECAY & RISK ASSESSMENT ---
        self.cell_last_seen = {}  # Track when each cell was last seen
        self.risk_map = None  # Will be initialized when map_size is known
        self.pacman_activity_zones = []  # Vùng Pacman hoạt động thường xuyên
        
        # --- PERFORMANCE TRACKING ---
        self.survival_time = 0
        self.escape_count = 0
        self.trap_success_count = 0
        self.death_trap_count = 0  # Số lần gặp death trap
        self.current_step = 0  # Track current step number
        
        # --- LOGGING ---
        self.log_file = open("../submissions/23120025/ghost_log.json", "w", encoding="utf-8")  # Append mode để ghi tiếp
        self.map_file = open("../submissions/23120025/ghost_map.json", "w", encoding="utf-8")  # Write mode để chỉ giữ map cuối

    def _load_parameters_from_weights(self):
        """Load parameters từ weights.json nếu tồn tại"""
        import json
        from pathlib import Path
        
        weights_path = Path(__file__).parent / "weights.json"
        if weights_path.exists():
            try:
                with open(weights_path, 'r') as f:
                    weights = json.load(f)
                
                # Override class attributes với values từ file
                for param in ["GHOST_OBS_RADIUS", "PACMAN_OBS_RADIUS", "EARLY_GAME_LIMIT", 
                             "SAFE_DISTANCE", "W_DISTANCE", "W_AXIS_PENALTY", 
                             "W_CORNER_PENALTY", "W_DEAD_END_BAD", "W_VISIT_PENALTY", "W_INERTIA"]:
                    if param in weights:
                        setattr(self, param, weights[param])
            except Exception as e:
                pass  # Nếu lỗi, dùng default values

    def step(self, map_state, my_pos, enemy_pos, step_num):
        # Store current step for use in other methods
        self.current_step = step_num
        
        # 1. PRE-COMPUTE
        if self.walls is None:
            self.map_size = map_state.shape
            self.ghost_map = np.full(self.map_size, -1, dtype=np.int8)  # -1 = chưa nhìn thấy
            self.walls = (map_state == 1)
            self.visit_map = np.zeros(self.map_size)
            self.risk_map = np.zeros(self.map_size)  # Initialize risk map here
            self._precompute_map_features()
        
        # 2. CẬP NHẬT BẢN ĐỒ RIÊNG (Chỉ nhìn thấy hình chữ thập 5 ô)
        self._update_ghost_map(my_pos, map_state)
        # Cập nhật lại map features sau khi có thông tin mới
        self._precompute_map_features()
        # Update memory decay và risk assessment
        self._update_memory_and_risk(my_pos, step_num)
        
        self.visit_map[my_pos] += 1
        self.survival_time = step_num
        
        # 3. UPDATE PACMAN
        pacman_visible = enemy_pos is not None
        if pacman_visible:
            self.last_known_enemy_pos = enemy_pos
            self.pacman_history.append((enemy_pos, step_num))
            self.pacman_last_seen_step = step_num
            self._update_pacman_activity_zones(enemy_pos)
            
            # Performance tracking
            if self.trapped_pos:  # Escape từ trap thành công
                self.escape_count += 1
        else:
            # Pacman không thấy - có thể vừa escape khỏi tầm nhìn hay bị capture
            pass
        
        target_pacman = self.last_known_enemy_pos
        
        # 4. EARLY GAME - Đứng yên hoặc chạy tới vùng an toàn
        if step_num <= self.EARLY_GAME_LIMIT:
            if not target_pacman:
                # Pacman không thấy -> Đứng yên an toàn
                return Move.RIGHT
            
            distance = self._manhattan_distance(my_pos, target_pacman)
            if distance > self.SAFE_DISTANCE:
                # Đã ở vùng an toàn -> Đứng yên
                return Move.STAY
            else:
                # Pacman gần -> Chạy tới vùng an toàn (khoảng cách > SAFE_DISTANCE)
                valid_moves = self._get_valid_moves(my_pos)
                best_move = Move.STAY
                best_dist = distance
                
                for move in valid_moves:
                    next_pos = self._get_next_pos(my_pos, move)
                    next_dist = self._manhattan_distance(next_pos, target_pacman)
                    if next_dist > best_dist:
                        best_dist = next_dist
                        best_move = move
                
                return best_move
        
        # 5. MAIN LOGIC
        if target_pacman:
            # Thấy pacman -> Thoát khỏi vùng kẹt
            self.trapped_pos = None
            self.trapped_step = 0
            
            distance = self._manhattan_distance(my_pos, target_pacman)
            
            # CRITICAL DISTANCE: Pacman cực gần -> panic escape  
            if distance <= 2:  # Pacman chỉ cách 1-2 ô
                valid_moves = self._get_valid_moves(my_pos)
                final_move = self._panic_escape(my_pos, target_pacman, valid_moves)
                if final_move is None:
                    final_move = self._momentum_aware_escape(my_pos, target_pacman)
                self._log_decision(step_num, my_pos, pacman_visible, target_pacman, final_move, "panic_escape")
            else:
                # Chạy có tính toán sâu (Deep Check)
                final_move = self._momentum_aware_escape(my_pos, target_pacman)
                self._log_decision(step_num, my_pos, pacman_visible, target_pacman, final_move, "momentum_escape")
        else:
            # Không thấy pacman - check logic kẹt với cải thiện
            is_in_trap = my_pos in self.dead_ends
            
            # Emergency escape: Nếu đang trap mà Pacman gần đây xuất hiện
            should_emergency_escape = False
            if self.trapped_pos and (step_num - self.pacman_last_seen_step) <= 3:
                should_emergency_escape = True
            
            # DEATH TRAP DETECTION: Pacman gần + ở dead-end = chết chắc
            is_death_trap = False
            if target_pacman and my_pos in self.dead_ends:
                distance = self._manhattan_distance(my_pos, target_pacman)
                # Pacman speed 2, Ghost speed 1 -> nếu Pacman <= 4 ô trong dead-end = chết
                if distance <= 4:
                    is_death_trap = True
                    should_emergency_escape = True
                    self.death_trap_count += 1
            
            if is_in_trap and not should_emergency_escape:
                # Vào hoặc ở trong trap
                if self.trapped_pos != my_pos:
                    self.trapped_pos = my_pos
                    self.trapped_step = 0
                
                self.trapped_step += 1
                
                # Dynamic trap duration dựa trên khoảng cách tới last known Pacman
                trap_duration = self._calculate_dynamic_trap_duration(target_pacman)
                
                if self.trapped_step < trap_duration:
                    final_move = Move.STAY
                    if self.trapped_step == trap_duration - 1:  # Sắp thoát
                        self.trap_success_count += 1
                    self._log_decision(step_num, my_pos, pacman_visible, target_pacman, final_move, "trapping")
                else:
                    # Thoát khỏi kẹt
                    self.trapped_pos = None
                    self.trapped_step = 0
                    final_move = self._smart_exploration(my_pos)
                    self._log_decision(step_num, my_pos, pacman_visible, target_pacman, final_move, "exit_trap")
            else:
                # Emergency escape hoặc death trap -> RUN!
                if should_emergency_escape or is_death_trap:
                    self.escaped_from_trap = True
                    self.trapped_pos = None
                    self.trap_duration = 0
                    self.escape_count += 1
                    
                    if is_death_trap:
                        # PANIC MODE: Chạy theo hướng tốt nhất ngay lập tức
                        valid_moves = self._get_valid_moves(my_pos)
                        panic_action = self._panic_escape(my_pos, target_pacman, valid_moves)
                        if panic_action:
                            final_move = panic_action
                        else:
                            final_move = self._momentum_aware_escape(my_pos, target_pacman)
                        self._log_decision(step_num, my_pos, pacman_visible, target_pacman, final_move, "death_trap_escape")
                    else:
                        # Normal escape với deep safety check
                        final_move = self._momentum_aware_escape(my_pos, target_pacman)
                        self._log_decision(step_num, my_pos, pacman_visible, target_pacman, final_move, "emergency_escape")
                else:
                    # Không trong emergency - normal exploration
                    final_move = self._smart_exploration(my_pos)
                    self._log_decision(step_num, my_pos, pacman_visible, target_pacman, final_move, "exploration")
        
        self.last_move = final_move
        return final_move

    def _log_decision(self, step_num, my_pos, pacman_visible, target_pacman, final_move, context=""):
        """Helper method để log decision"""
        valid_moves = self._get_valid_moves(my_pos)
        decision_log = {
            "step": step_num,
            "ghost_pos": my_pos,
            "pacman_visible": pacman_visible,
            "pacman_pos": target_pacman if pacman_visible else None,
            "valid_moves": [m.name for m in valid_moves],
            "chosen_move": final_move.name,
            "context": context
        }
        self.log_file.write(json.dumps(decision_log, ensure_ascii=False) + "\n")
        self.log_file.flush()
        
        # LOG GHOST MAP (chỉ lưu map cuối cùng, ghi đè các lần trước)
        map_entry = {
            "step": step_num,
            "ghost_pos": my_pos,
            "ghost_map": [self.ghost_map[i].tolist() for i in range(self.map_size[0])]
        }
        # Ghi đè file để chỉ giữ lại dòng cuối cùng
        with open("../submissions/23120025/ghost_map.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(map_entry, ensure_ascii=False) + "\n")
        
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
        """Khám phá thông minh: ưu tiên frontier và vùng chưa thấy"""
        valid_moves = self._get_valid_moves(my_pos)
        if not valid_moves: return Move.STAY
        
        # Update exploration targets mỗi 5 bước
        if (self.pacman_last_seen_step - self.last_frontier_update) > 5:
            self._update_exploration_targets()
            self.last_frontier_update = self.pacman_last_seen_step
        
        best_move = Move.STAY
        best_score = -float('inf')

        for move in valid_moves:
            next_pos = self._get_next_pos(my_pos, move)
            score = 0
            visits = self.visit_map[next_pos]
            
            # Prioritize unvisited cells
            if visits == 0: score += 800
            else: score -= visits * self.params["W_VISIT_PENALTY"]
            
            # Heavy penalty for dead ends
            if next_pos in self.dead_ends: 
                score -= 3000
                # EXTRA PENALTY nếu Pacman gần last known position
                if (self.last_known_enemy_pos and 
                    self._manhattan_distance(next_pos, self.last_known_enemy_pos) <= 6):
                    score -= 2000  # Death trap avoidance
            
            # Bonus for moving toward exploration targets (frontier)
            if self.exploration_targets:
                min_dist_to_frontier = min(self._manhattan_distance(next_pos, target) 
                                         for target in self.exploration_targets)
                score += max(0, 200 - min_dist_to_frontier * 20)
            
            # Inertia and direction bias
            if self.last_move and move == self.last_move: score += self.params["W_INERTIA"]
            score += self.direction_bias.get(move, 0)
            
            # Pacman prediction avoidance với risk assessment
            if len(self.pacman_history) >= 2:
                predicted_pacman_pos = self._predict_pacman_position()
                if predicted_pacman_pos:
                    dist_to_predicted = self._manhattan_distance(next_pos, predicted_pacman_pos)
                    if dist_to_predicted <= 3:
                        score -= 1000  # Avoid predicted Pacman area
            
            # Risk-based scoring
            if self._is_in_bounds(next_pos) and self.risk_map is not None:
                risk_score = self.risk_map[next_pos]
                score -= risk_score * 50  # Penalty for high-risk areas
            
            # Memory decay bonus - ưu tiên vùng lâu chưa thăm
            if str(next_pos) in self.cell_last_seen:
                time_since_seen = self.current_step - self.cell_last_seen[str(next_pos)]
                if time_since_seen > 20:  # Vùng lâu chưa thăm
                    score += min(200, time_since_seen * 5)
            
            if score > best_score:
                best_score = score
                best_move = move
        return best_move

    def _panic_escape(self, my_pos, pacman_pos, legal_actions):
        """Panic mode: Escape ngay lập tức khỏi death trap"""
        if not pacman_pos:
            return None
            
        best_action = None
        max_distance = -1
        
        for action in legal_actions:
            next_pos = self._get_next_pos(my_pos, action)
            if self._is_valid_coord(next_pos):
                # Tìm vị trí xa Pacman nhất và không phải dead-end
                distance = self._manhattan_distance(next_pos, pacman_pos)
                
                # Bonus cho việc thoát khỏi dead-end
                if next_pos not in self.dead_ends and my_pos in self.dead_ends:
                    distance += 10  # Big bonus for escaping dead-end
                
                # Penalty cho việc vào dead-end khác
                if next_pos in self.dead_ends:
                    distance -= 5
                
                # Penalty cho việc đi về phía Pacman
                if distance < self._manhattan_distance(my_pos, pacman_pos):
                    distance -= 3
                
                if distance > max_distance:
                    max_distance = distance
                    best_action = action
        
        return best_action

    def _precompute_map_features(self):
        """
        Tính toán các tính năng bản đồ (dead-end, corner).
        Dùng self.ghost_map (bản đồ riêng với tầm nhìn hạn chế) thay vì walls.
        """
        self.dead_ends = set()
        self.corners = set()
        h, w = self.map_size
        for r in range(h):
            for c in range(w):
                # Chỉ xét những ô Ghost đã nhìn thấy (không phải -1)
                if self.ghost_map[r, c] == -1:  # Chưa nhìn thấy
                    continue
                if self.ghost_map[r, c] == 1:   # Là tường
                    continue
                
                # Đếm bao nhiêu hướng bị chặn bởi tường hoặc biên hoặc chưa nhìn thấy
                walls = 0
                for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                    nr, nc = r+dr, c+dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        walls += 1
                    elif self.ghost_map[nr, nc] == 1:  # Là tường
                        walls += 1
                    elif self.ghost_map[nr, nc] == -1:  # Chưa nhìn thấy - coi như tường để an toàn
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

    def _get_visible_area(self, my_pos):
        """Lấy tầm nhìn hiện tại (chữ thập radius 5)"""
        r, c = my_pos
        h, w = self.map_size
        visible = {}
        
        # Center
        visible[str(my_pos)] = int(self.ghost_map[r, c])
        
        # 4 hướng chính, mỗi hướng radius 5
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            for dist in range(1, 6):
                nr, nc = r + dr*dist, c + dc*dist
                if 0 <= nr < h and 0 <= nc < w:
                    visible[str((nr, nc))] = int(self.ghost_map[nr, nc])
        
        return visible
    
    def _calculate_dynamic_trap_duration(self, pacman_pos):
        """Tính trap duration dựa trên khoảng cách, risk và performance"""
        base_duration = self.BASE_TRAP_DURATION
        
        if pacman_pos:
            distance = self._manhattan_distance(self.trapped_pos or (0,0), pacman_pos)
            # Càng xa Pacman, ở lại càng lâu
            distance_multiplier = min(2.0, distance / 8.0)
            duration = int(base_duration * distance_multiplier)
            
            # Adjust based on success rate
            if self.escape_count + self.trap_success_count > 0:
                success_rate = self.trap_success_count / (self.escape_count + self.trap_success_count)
                if success_rate > 0.7:  # High success rate
                    duration = int(duration * 1.2)  # Stay longer
                elif success_rate < 0.3:  # Low success rate
                    duration = int(duration * 0.8)  # Stay shorter
        else:
            # Không biết Pacman đâu -> ở lại lâu hơn nhưng có giới hạn
            duration = int(base_duration * 1.3)
        
        return max(5, min(35, duration))  # Optimized range 5-35 bước
    
    def _update_exploration_targets(self):
        """Cập nhật danh sách frontier cells để khám phá"""
        self.exploration_targets = []
        h, w = self.map_size
        
        for r in range(h):
            for c in range(w):
                if self.ghost_map[r, c] == 0:  # Ô trống đã thấy
                    # Check if adjacent to unexplored area
                    for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                        nr, nc = r+dr, c+dc
                        if (0 <= nr < h and 0 <= nc < w and 
                            self.ghost_map[nr, nc] == -1):  # Chưa khám phá
                            if (r, c) not in self.exploration_targets:
                                self.exploration_targets.append((r, c))
                            break
    
    def _predict_pacman_position(self):
        """Dự đoán vị trí Pacman dựa trên movement pattern"""
        if len(self.pacman_history) < 3:
            return None
        
        # Analyze recent movement vectors
        recent_moves = []
        for i in range(1, min(4, len(self.pacman_history))):
            curr_pos, curr_time = self.pacman_history[-i]
            prev_pos, prev_time = self.pacman_history[-i-1]
            
            if curr_time > prev_time:  # Valid time sequence
                dx = curr_pos[0] - prev_pos[0]
                dy = curr_pos[1] - prev_pos[1]
                recent_moves.append((dx, dy))
        
        if not recent_moves:
            return None
        
        # Calculate average movement vector
        avg_dx = sum(dx for dx, dy in recent_moves) / len(recent_moves)
        avg_dy = sum(dy for dx, dy in recent_moves) / len(recent_moves)
        
        # Predict future position
        last_pos, last_time = self.pacman_history[-1]
        time_diff = self.pacman_last_seen_step - last_time
        
        predicted_x = int(last_pos[0] + avg_dx * time_diff)
        predicted_y = int(last_pos[1] + avg_dy * time_diff)
        
        # Clamp to map bounds
        predicted_x = max(0, min(self.map_size[0]-1, predicted_x))
        predicted_y = max(0, min(self.map_size[1]-1, predicted_y))
        
        return (predicted_x, predicted_y)
    
    def _update_memory_and_risk(self, my_pos, step_num):
        """Cập nhật memory decay và risk assessment"""
        if self.risk_map is None:
            return  # Skip if risk map not initialized yet
            
        # Update cell last seen times
        r, c = my_pos
        h, w = self.map_size
        
        # Center
        self.cell_last_seen[str(my_pos)] = step_num
        
        # Visible area (chữ thập)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            for dist in range(1, 6):
                nr, nc = r + dr*dist, c + dc*dist
                if 0 <= nr < h and 0 <= nc < w:
                    self.cell_last_seen[str((nr, nc))] = step_num
        
        # Decay risk map over time
        self.risk_map = self.risk_map * 0.95  # 5% decay mỗi bước
        
        # Update risk based on Pacman activity
        for pos, time in self.pacman_history:
            if step_num - time <= 10:  # Recent activity
                self._increase_risk_around(pos, step_num - time)
    
    def _increase_risk_around(self, center, age):
        """Tăng risk level xung quanh vị trí Pacman hoạt động"""
        r, c = center
        h, w = self.map_size
        intensity = max(0.1, 1.0 - age / 10.0)  # Intensity giảm theo thời gian
        
        for dr in range(-3, 4):
            for dc in range(-3, 4):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    distance = abs(dr) + abs(dc)
                    if distance <= 3:
                        risk_increase = intensity * (4 - distance) / 4
                        self.risk_map[nr, nc] = min(10.0, self.risk_map[nr, nc] + risk_increase)
    
    def _update_pacman_activity_zones(self, pacman_pos):
        """Track các vùng Pacman hoạt động thường xuyên"""
        # Simple clustering - merge nearby positions
        for i, (zone_center, count) in enumerate(self.pacman_activity_zones):
            if self._manhattan_distance(pacman_pos, zone_center) <= 4:
                # Update existing zone
                new_center = ((zone_center[0] + pacman_pos[0]) // 2, 
                             (zone_center[1] + pacman_pos[1]) // 2)
                self.pacman_activity_zones[i] = (new_center, count + 1)
                return
        
        # Add new zone
        if len(self.pacman_activity_zones) < 5:  # Limit to 5 zones
            self.pacman_activity_zones.append((pacman_pos, 1))
        else:
            # Replace least active zone
            min_idx = min(range(len(self.pacman_activity_zones)), 
                         key=lambda i: self.pacman_activity_zones[i][1])
            self.pacman_activity_zones[min_idx] = (pacman_pos, 1)
    
    def _is_in_bounds(self, pos):
        """Check if position is within map bounds"""
        r, c = pos
        return 0 <= r < self.map_size[0] and 0 <= c < self.map_size[1]