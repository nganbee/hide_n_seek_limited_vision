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
    """Pacman v8.0 Compact - Speed Demon Optimized"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Pacman"
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        
        self.map_size = (21, 21)
        self.global_map = np.full(self.map_size, -1)
        self.last_known_enemy_pos = None
        self.enemy_history = deque(maxlen=5)
        self.my_history = deque(maxlen=15) 
        self.current_target = None
        self.ghost_probability = np.zeros((21, 21))

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
            
            self.ghost_probability.fill(0)
            self.ghost_probability[enemy_position] = 1.0    
            
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
            
            new_prob = np.zeros_like(self.ghost_probability)
            for r in range(21):
                for c in range(21):
                    if self.ghost_probability[r, c] > 0:
                        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                            nr, nc = self._get_next_pos((r, c), move)
                            if self._is_passable((nr, nc)):
                                new_prob[nr, nc] += self.ghost_probability[r, c] * 0.25
            self.ghost_probability = new_prob
            
        if not enemy_position:
            idx = np.unravel_index(self.ghost_probability.argmax(), self.ghost_probability.shape)
            if self.ghost_probability[idx] > 0:
                self.current_target = idx

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
    
import os, pickle

# class GhostAgent(BaseGhostAgent):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
        
#         # 1. Cấu hình Hyperparameters
#         self.last_state = None
#         self.last_action = None
#         self.last_my_pos = None
#         self.epsilon = 0.2  # Đặt 0.0 để thi đấu (Inference), đặt 0.2 nếu đang Train
#         # Learning hyperparameters
#         self.alpha = 0.01
#         self.discount = 0.9
#         self.weights_file = 'ghost_weights.pkl'
        
#         # 2. Khởi tạo Trọng số (Weights)
#         # Nếu không có file, dùng bộ weights "Hard-coded" đã được tối ưu sơ bộ
#         self.weights = self.load_weights()
        
#         # 3. Memory cho Partial Observability
#         self.last_known_enemy_pos = None
#         self.turns_since_seen = 0
#         self.safe_fallback_pos = None # Vị trí an toàn để rút lui
        
#         self.visited_map = {} # Dùng dict cho nhẹ: key=(row, col), value=count
#         self.decay_factor = 0.9
        
#         self.save_frequency = 10
#         self.steps_count = 0

#     def load_weights(self):
#         """Load weights từ file hoặc dùng default"""
#         if os.path.exists(self.weights_file):
#             try:
#                 with open(self.weights_file, 'rb') as f:
#                     return pickle.load(f)
#             except:
#                 pass
        
#         return {
#             'bias': 10.0,             # Điểm cơ bản để khuyến khích di chuyển
#             'dist_danger': -50.0,     # Rất sợ bị Pacman lại gần (Manhattan <= 4)
#             'dist_critical': -200.0,  # Cực sợ chết (Manhattan <= 2)
#             'is_dead_end': -20.0,     # Ghét đường cụt
#             'is_junction': 5.0,       # Thích đứng ở ngã ba/ngã tư
#             'action_stay' : -2.0,     # Phạt đứng yên
#             'visited_count': -2.0,    # Càng đi nhiều càng bị trừ điểm -> Né ra
#             'vertical_bias': 1.0,
#             'reverse_move': -10.0,    # Mạnh mẽ cấm quay đầu liên tục
#             'away_from_enemy': 30.0,   # Khuyến khích di chuyển xa mục tiêu (tránh bị bắt)
#             'hidden_from_pacman': 10.0 # Thích nấp sau tường
#         }

#     def step(self, map_state: np.ndarray, 
#              my_position: tuple, 
#              enemy_position: tuple,
#              step_number: int) -> Move:
        
#         self.steps_count += 1
        
#         # --- BƯỚC 1: CẬP NHẬT MEMORY ---
#         if enemy_position is not None:
#             self.last_known_enemy_pos = enemy_position
#             self.turns_since_seen = 0
#         else:
#             self.turns_since_seen += 1
#             # Nếu quá lâu không thấy (20 bước), quên vị trí cũ để tránh ám ảnh
#             if self.turns_since_seen > 20:
#                 self.last_known_enemy_pos = None
                
#         if my_position not in self.visited_map:
#             self.visited_map[my_position] = 0
#         self.visited_map[my_position] += 1
        
#         if step_number % 100 == 0: 
#              self.visited_map = {}
          
#         # Training        
#         if self.last_state is not None and self.last_action is not None:
#             # 1. Tính Reward: Hành động trước đó có tốt không?
#             reward = self.calculate_reward(map_state, my_position, enemy_position)
            
#             # 2. Tính Q_max hiện tại (Best score có thể đạt được ở trạng thái mới này)
#             # Ta cần xem xét các nước đi hợp lệ TẠI THỜI ĐIỂM NÀY
#             current_legal_moves = []
#             for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]:
#                 if self._is_valid_move(my_position, m, map_state):
#                     current_legal_moves.append(m)
            
#             if current_legal_moves:
#                 current_q_values = [self.get_q_value(map_state, my_position, m) for m in current_legal_moves]
#                 max_next_q = max(current_q_values)
#             else:
#                 max_next_q = 0 # Game over hoặc bị kẹt
            
#             # 3. Tính Q_value cũ (của trạng thái quá khứ)
#             # Lưu ý: Phải dùng last_my_pos chứ không phải my_position hiện tại
#             # (Bạn cần lưu thêm self.last_my_pos trong __init__)
#             old_next_pos = self._get_next_pos(self.last_my_pos, self.last_action)
#             old_features = self.extract_features(self.last_state, self.last_my_pos, old_next_pos, self.last_action)
#             predicted_q = sum(self.weights.get(k,0) * old_features[k] for k in old_features)
            
#             # 4. Update Weights (Gradient Descent)
#             # Difference = (Reward thực tế + Viễn cảnh tương lai) - Dự đoán cũ
#             difference = (reward + self.discount * max_next_q) - predicted_q
            
#             for feature, value in old_features.items():
#                 self.weights[feature] = self.weights.get(feature, 0) + self.alpha * difference * value

#         # --- BƯỚC 2: LẤY DANH SÁCH NƯỚC ĐI HỢP LỆ ---
#         legal_moves = []
#         all_moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]
        
#         for move in all_moves:
#             if self._is_valid_move(my_position, move, map_state):
#                 legal_moves.append(move)
                
#         # Avoid choosing STAY by default: only keep it if it's the sole legal move
#         if Move.STAY in legal_moves and len(legal_moves) > 1:
#             legal_moves.remove(Move.STAY)
        
#         if self.last_known_enemy_pos is not None and self.last_action is not None:
#             reverse_move = self._get_opposite_move(self.last_action)
            
#             if reverse_move in legal_moves and len(legal_moves) > 1:
#                 legal_moves.remove(reverse_move)
                
#             if Move.STAY in legal_moves and len(legal_moves) > 1:
#                 legal_moves.remove(Move.STAY)   
                
#         if not legal_moves:
#             # No valid move (shouldn't normally happen) -> stay as a fallback
#             return Move.STAY

#         # --- BƯỚC 3: CHIẾN THUẬT (EPSILON-GREEDY) ---
        
#         # Nếu đang train: Thử nghiệm ngẫu nhiên với xác suất epsilon
#         if random.random() < self.epsilon:
#             best_move = random.choice(legal_moves)
#         else:
#             best_score = -float('inf')
#             best_moves = []
#             for move in legal_moves:
#                 score = self.get_q_value(map_state, my_position, move)
#                 if score > best_score:
#                     best_score = score
#                     best_moves = [move]
#                 elif score == best_score:
#                     best_moves.append(move)
#             # Try to avoid immediate reversal if possible
#             if len(best_moves) > 1 and self.last_action is not None:
#                 opposite = self._get_opposite_move(self.last_action)
#                 non_reverse = [m for m in best_moves if m != opposite]
#                 if non_reverse:
#                     best_moves = non_reverse
#             best_move = random.choice(best_moves)

#         # Chọn ngẫu nhiên trong số các nước đi tốt nhất (để tránh lặp lại pattern)
        
#         self.last_state = map_state.copy() # Copy mảng để tránh bị ghi đè tham chiếu
#         self.last_action = best_move
#         self.last_my_pos = my_position
        
#         if self.steps_count % self.save_frequency == 0:
#             self.save_weights()
        
#         return best_move

#     # --- CÁC HÀM FEATURE ENGINEERING (PHẦN QUAN TRỌNG NHẤT) ---
    
#     def calculate_reward(self, current_map, my_pos, enemy_pos):
#         """
#         Hàm tính điểm thưởng/phạt dựa trên kết quả hành động vừa rồi.
#         """
#         reward = 0
        
#         # 1. Phần thưởng sinh tồn (Survival Reward)
#         # Mỗi lượt sống sót là một thành công nhỏ
#         reward += 1.0 
        
#         # 2. Phạt nếu Pacman quá gần (bị dồn vào chân tường)
#         if enemy_pos: # Nếu nhìn thấy địch
#             dist = abs(my_pos[0] - enemy_pos[0]) + abs(my_pos[1] - enemy_pos[1])
#             if dist <= 1:
#                 reward -= 100.0 # Bị bắt (hoặc sắp bị bắt) -> Phạt cực nặng
#             elif dist <= 3:
#                 reward -= 10.0  # Vùng nguy hiểm
        
#         # 3. Phạt nếu đứng im (tùy chiến thuật)
#         # Nếu muốn agent linh hoạt, phạt nhẹ khi đứng im
#         if self.last_action == Move.STAY:
#             reward -= 0.5

#         return reward
    
#     def save_weights(self):
#         try:
#             with open(self.weights_file, 'wb') as f:
#                 pickle.dump(self.weights, f)
                
#             with open("log.txt", 'a') as f:
#                 f.write("Write weight\n")
#         except Exception as e:
#             print(f"Save error: {e}")

#     def get_q_value(self, map_state, my_pos, move):
#         """Tính điểm cho một nước đi dựa trên features và weights"""
#         next_pos = self._get_next_pos(my_pos, move)
#         features = self.extract_features(map_state, my_pos, next_pos, move)
        
#         # Công thức: Q = w1*f1 + w2*f2 + ...
#         score = 0
#         for key in features:
#             score += self.weights.get(key, 0.0) * features[key]
#         return score

#     def extract_features(self, map_state, current_pos, next_pos, move):
#         """
#         Trích xuất đặc trưng từ trạng thái dự kiến (next_pos).
#         Đây là 'bộ não' của Ghost.
#         """
#         features = {'bias': 1.0}
        
#         # Xác định vị trí mục tiêu (Pacman)
#         target_pos = self.last_known_enemy_pos
        
#         # 1. Feature: SAFETY (An toàn khoảng cách)
#         if target_pos:
#             dist = abs(next_pos[0] - target_pos[0]) + abs(next_pos[1] - target_pos[1])
            
#             # Pacman đi 2 bước + đánh đồng thời -> Vùng chết là <= 2, Vùng nguy hiểm <= 4
#             if dist <= 2:
#                 features['dist_critical'] = 1.0 # Rất gần cái chết
#             elif dist <= 5:
#                 features['dist_danger'] = 1.0   # Nguy hiểm
            
#             # Logic cắt tầm nhìn (Raycast check) - Rất quan trọng
#             if self.check_wall_between(map_state, next_pos, target_pos):
#                  features['hidden_from_pacman'] = 1.0
        
#         # 2. Feature: DEAD END (Đường cụt)
#         # Đếm số lối đi từ next_pos (không tính tường)
#         num_exits = self.count_exits(map_state, next_pos)
#         if num_exits <= 1:
#             features['is_dead_end'] = 1.0
        
#         # 3. Feature: JUNCTION (Ngã rẽ)
#         # Pacman chạy nhanh, Ghost cần đứng ở ngã rẽ để dễ "lạng lách"
#         if num_exits >= 3:
#             features['is_junction'] = 1.0

#         # 4. Feature: STAY PENALTY
#         if move == Move.STAY:
#              features['action_stay'] = -1.0
             
#         # 5. Penalize immediate reverse (to avoid left-right oscillation)
#         if self.last_action is not None:
#             # If the considered move is the opposite of last_action -> penalize
#             if move == self._get_opposite_move(self.last_action):
#                 features['reverse_move'] = 1.0

#         # 6. Reward moving away from the known enemy if any
#         if target_pos:
#             curr_dist = abs(current_pos[0] - target_pos[0]) + abs(current_pos[1] - target_pos[1])
#             next_dist = abs(next_pos[0] - target_pos[0]) + abs(next_pos[1] - target_pos[1])
#             if next_dist > curr_dist:
#                 features['away_from_enemy'] = 1.0

#         visit_count = self.visited_map.get(next_pos, 0)
        
#         # Nếu ô đó đã đi 5 lần -> count=5. Nếu chưa đi -> count=0.
#         features['visited_count'] = float(visit_count)

#         return features

#     def _get_next_pos(self, pos, move):
#         """Tính tọa độ tiếp theo"""
#         dr, dc = move.value
#         return (pos[0] + dr, pos[1] + dc)

#     def count_exits(self, map_state, pos):
#         """Đếm số ô trống xung quanh một vị trí"""
#         exits = 0
#         h, w = map_state.shape
#         r, c = pos
#         # Kiểm tra 4 hướng
#         for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
#             nr, nc = r + dr, c + dc
#             if 0 <= nr < h and 0 <= nc < w:
#                 # 0 là đường, 1 là tường. -1 là sương mù (coi như tường cho an toàn)
#                 if map_state[nr, nc] == 0: 
#                     exits += 1
#         return exits

#     def check_wall_between(self, map_state, pos1, pos2):
#         """
#         Kiểm tra đơn giản xem có tường chắn giữa 2 điểm không.
#         Dùng để tính feature 'hidden_from_pacman'.
#         Chỉ check trường hợp thẳng hàng (row hoặc col) vì Pacman nhìn theo tia.
#         """
#         r1, c1 = pos1
#         r2, c2 = pos2
        
#         # Nếu không thẳng hàng, tạm coi là đã khuất tầm nhìn (hoặc an toàn hơn)
#         if r1 != r2 and c1 != c2:
#             return True 
            
#         # Check dọc
#         if c1 == c2:
#             start, end = min(r1, r2), max(r1, r2)
#             for r in range(start + 1, end):
#                 if map_state[r, c1] == 1: return True
                
#         # Check ngang
#         if r1 == r2:
#             start, end = min(c1, c2), max(c1, c2)
#             for c in range(start + 1, end):
#                 if map_state[r1, c] == 1: return True
                
#         return False

#     def _is_valid_move(self, pos: tuple, move: Move, map_state: np.ndarray) -> bool:
#         """Logic check move gốc của bạn"""
#         delta_row, delta_col = move.value
#         new_pos = (pos[0] + delta_row, pos[1] + delta_col)
#         return self._is_valid_position(new_pos, map_state)
    
#     def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
#         row, col = pos
#         height, width = map_state.shape
#         if row < 0 or row >= height or col < 0 or col >= width:
#             return False
#         # Lưu ý: -1 là unseen, tùy rule mà có thể đi vào hay không. 
#         # Ở đây giả định chỉ đi vào ô 0 (empty). Nếu game cho phép đi vào -1 thì sửa thành != 1
#         return map_state[row, col] == 0

#     def _get_opposite_move(self, move: Move) -> Move:
#         """Return Move that is opposite of the given move"""
#         mapping = {
#             Move.UP: Move.DOWN,
#             Move.DOWN: Move.UP,
#             Move.LEFT: Move.RIGHT,
#             Move.RIGHT: Move.LEFT,
#             Move.STAY: Move.STAY
#         }
#         return mapping.get(move, Move.STAY)
    
#     def __del__(self):
#         self.save_weights()
        
        
import numpy as np
import random
import pickle
import os

class GhostAgent(BaseGhostAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # --- CONFIG ---
        self.weights_file = '../submissions/23120060/ghost_weights.pkl'
        self.weights = self.load_weights()
        self.visited_map = {}
        self.step_count = 0
        
        self.global_map = None
        self.last_known_enemy_pos = None
        self.turns_since_seen = 0
        
        self.last_action = None
        
        # --- OPENING BOOK CONFIG (QUAN TRỌNG) ---
        # Mục tiêu: Chạy đến góc trên cùng bên trái (1, 1) hoặc phải (1, 19)
        # Nơi đó xa Pacman nhất (vì Pacman spawn ở dưới)
        self.safe_haven = (1, 1) 
        self.opening_path = [] # Sẽ chứa danh sách nước đi đã tính toán sẵn
        self.is_opening_phase = True
        
    def load_weights(self):
        # Load lại file weight bạn đã train xịn xò từ bài trước
        if os.path.exists(self.weights_file):
            try:
                with open(self.weights_file, 'rb') as f:
                    return pickle.load(f)
            except: pass
        # Fallback weights
        return {'bias': 10.0, 
                'visited_count': -5.0, 
                'is_junction': 10.0,
                'is_dead_end': -20.0, 
                'action_stay': -50.0,
                'dist_to_pacman': 50.0,
                'is_turning': 15.0,
                'straight_when_visible': -60.0}
        
    def save_weights(self):
        return
        try:
            with open(self.weights_file, 'wb') as f:
                pickle.dump(self.weights, f)
                
            with open("log.txt", 'a') as f:
                f.write(f"{self.weights}\n")
        except Exception as e:
            print(f"Save error: {e}")

    def step(self, map_state, my_position, enemy_position, step_number):
        self.step_count += 1
        
        # 1. INIT MAP
        if self.global_map is None:
            self.global_map = map_state.copy()
            self.opening_path = self._bfs_find_path(my_position, self.safe_haven)
        else:
            visible_mask = (map_state != -1)
            self.global_map[visible_mask] = map_state[visible_mask]

        # 2. UPDATE INFO
        if enemy_position:
            self.last_known_enemy_pos = enemy_position
            self.turns_since_seen = 0
        else:
            self.turns_since_seen += 1
            if self.turns_since_seen > 50: self.last_known_enemy_pos = None

        if my_position not in self.visited_map: self.visited_map[my_position] = 0
        self.visited_map[my_position] += 1
        
        # --- QUYẾT ĐỊNH NƯỚC ĐI (Lưu vào biến final_move chứ chưa return vội) ---
        final_move = Move.STAY
        
        old_pos = my_position

        dist_to_enemy = float('inf')
        if self.last_known_enemy_pos:
            dist_to_enemy = self._manhattan_distance(my_position, self.last_known_enemy_pos)
            
        # CASE 1: NGUY HIỂM -> MINIMAX
        if self.last_known_enemy_pos and dist_to_enemy <= 8:
            self.is_opening_phase = False 
            final_move = self._minimax_move(my_position, self.last_known_enemy_pos)

        # CASE 2: KHAI CUỘC (SCRIPTED OPENING)
        elif self.is_opening_phase and self.opening_path:
            next_target = self.opening_path[0]
            move_to_target = self._get_move_from_points(my_position, next_target)
            
            if self._is_passable(next_target):
                self.opening_path.pop(0)
                final_move = move_to_target
            else:
                self.is_opening_phase = False
                # Nếu fail opening thì chuyển sang patrol ngay lập tức
                final_move = self._strategic_patrol(my_position)
        
        # CASE 3: ĐI TUẦN TRA (PATROL)
        else:
            self.is_opening_phase = False # Đảm bảo tắt opening nếu path rỗng
            final_move = self._strategic_patrol(my_position)
            
        reward = self._calculate_reward(old_pos, final_move, enemy_position)

        # 4. CẬP NHẬT TRỌNG SỐ (Q-LEARNING UPDATE)
        new_pos = self._get_next_pos(old_pos, final_move)
        self.update_learning(old_pos, final_move, new_pos, reward)

        self.last_action = final_move
        
        if self.step_count % 20 == 0:
            self.save_weights()
        
        return final_move
    
    def _calculate_reward(self, old_pos, action, enemy_pos):
        reward = -1 # Phạt nhẹ mỗi bước đi để tránh đi lòng vòng
        
        new_pos = self._get_next_pos(old_pos, action)
        
        if enemy_pos is not None:
            if action == self.last_action:
                reward -= 30
        
        if self.last_action and action != self.last_action and action != self._get_opposite_move(self.last_action):
            reward += 5
        
        if enemy_pos:
            old_dist = self._manhattan_distance(old_pos, enemy_pos)
            new_dist = self._manhattan_distance(new_pos, enemy_pos)
            
            if new_dist > old_dist:
                reward += 1.5  # Thưởng vì đã chạy xa ra
            elif new_dist < old_dist:
                reward -= 10
                
            if new_dist == 0:
                reward -= 500
                
        if self._is_dead_end(new_pos):
            reward -= 20 # Phạt vì chui vào ngõ cụt
            
        return reward
    
    def _get_features(self, pos, action, enemy_pos):
        """
        Trả về các đặc trưng dựa trên bộ weights bạn đã định nghĩa:
        'bias', 'visited_count', 'is_junction', 'is_dead_end', 'action_stay'
        """
        next_pos = self._get_next_pos(pos, action)
        features = {}

        # 1. Bias: Luôn là 1
        features['bias'] = 1.0

        # 2. Visited Count: Số lần đã đi qua ô tiếp theo
        features['visited_count'] = float(self.visited_map.get(next_pos, 0))

        # 3. Is Junction: Có phải ngã 3, ngã 4 không?
        exits = len(self._get_valid_moves(next_pos))
        features['is_junction'] = 1.0 if exits > 2 else 0.0

        # 4. Is Dead End: Có phải ngõ cụt không?
        features['is_dead_end'] = 1.0 if self._is_dead_end(next_pos) else 0.0

        # 5. Action Stay: Phạt nếu đứng yên
        features['action_stay'] = 1.0 if action == Move.STAY else 0.0
        
        # 6. Turning
        is_turning = 0.0
        if self.last_action is not None and action != Move.STAY:
            # Nếu hành động hiện tại khác với hành động trước đó và không phải đi lùi
            if action != self.last_action and action != self._get_opposite_move(self.last_action):
                is_turning = 1.0
                
        features['is_turning'] = is_turning

        # 6. Distance to Pacman: normalized so closer -> larger (0..1)
        if self.last_known_enemy_pos:
            # Use global map bounds if available, else fallback to 21x21
            h, w = self.global_map.shape if self.global_map is not None else (21, 21)
            max_dist = h + w
            dist = self._manhattan_distance(next_pos, self.last_known_enemy_pos)
            features['dist_to_pacman'] = dist / max_dist
        else:
            features['dist_to_pacman'] = 0.5
            
        is_straight = 1.0 if action == self.last_action and action != Move.STAY else 0.0
        is_visible = 1.0 if enemy_pos is not None else 0.0
        
        features['straight_when_visible'] = is_straight * is_visible

        return features
    
    def update_learning(self, state, action, next_state, reward):
        """
        Cập nhật self.weights dựa trên trải nghiệm vừa có
        """
        alpha = 0.05   # Tốc độ học (Learning Rate)
        gamma = 0.9   # Hệ số giảm (Discount Factor)

        # 1. Tính Q(s, a) hiện tại: Tổng (weight * feature)
        features = self._get_features(state, action, self.last_known_enemy_pos)
        current_q = sum(self.weights.get(f, 0) * val for f, val in features.items())

        # 2. Tìm Max Q(s') của trạng thái tiếp theo
        next_moves = self._get_valid_moves(next_state)
        if not next_moves:
            next_max_q = 0.0
        else:
            # Tính Q cho tất cả các hành động có thể ở ô tiếp theo
            all_next_qs = []
            for m in next_moves:
                f_next = self._get_features(next_state, m, self.last_known_enemy_pos)
                q_val = sum(self.weights.get(f, 0) * val for f, val in f_next.items())
                all_next_qs.append(q_val)
            next_max_q = max(all_next_qs)

        # 3. Tính sai số (Difference/Temporal Difference Error)
        # Công thức: Difference = [Reward + γ * Max Q(s')] - Q(s, a)
        difference = (reward + gamma * next_max_q) - current_q

        # 4. Cập nhật từng trọng số trong self.weights
        for feature_name, feature_value in features.items():
            if feature_name not in self.weights:
                self.weights[feature_name] = 0.0
            
            # Cập nhật: w = w + alpha * difference * feature_value
            self.weights[feature_name] += alpha * difference * feature_value

    def _bfs_find_path(self, start, goal):
        """Tìm đường đi ngắn nhất từ Start đến Goal (trả về list các tọa độ)"""
        queue = deque([(start, [])])
        visited = set([start])
        
        while queue:
            (curr, path) = queue.popleft()
            
            if curr == goal:
                return path # Trả về danh sách các bước đi
            
            # Thử 4 hướng
            neighbors = []
            r, c = curr
            # Thứ tự ưu tiên: Lên, Trái, Phải, Xuống (Để ưu tiên chạy lên trên)
            candidates = [(r-1, c), (r, c-1), (r, c+1), (r+1, c)]
            
            for nr, nc in candidates:
                if 0 <= nr < 21 and 0 <= nc < 21:
                    # Check tường (giả sử tường là 1)
                    if self.global_map[nr, nc] != 1 and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append(((nr, nc), path + [(nr, nc)]))
        
        return [] # Không tìm thấy đường

    def _get_move_from_points(self, current, next_p):
        """Chuyển đổi tọa độ sang Enum Move"""
        dr = next_p[0] - current[0]
        dc = next_p[1] - current[1]
        if dr == -1: return Move.UP
        if dr == 1: return Move.DOWN
        if dc == -1: return Move.LEFT
        if dc == 1: return Move.RIGHT
        return Move.STAY
    
    def _minimax_move(self, ghost_pos, pacman_pos):
        valid_moves = self._get_valid_moves(ghost_pos)
        if not valid_moves: return Move.STAY
        best_score = -float('inf')
        best_moves = []
        for move in valid_moves:
            next_g_pos = self._get_next_pos(ghost_pos, move)
            dist_now = self._manhattan_distance(next_g_pos, pacman_pos)
            if dist_now <= 2: score = -99999
            else:
                pred_p_pos = self._predict_pacman_best_move(pacman_pos, next_g_pos, steps=2)
                score = self._evaluate_minimax_state(next_g_pos, pred_p_pos)
            if score > best_score:
                best_score = score
                best_moves = [move]
            elif score == best_score: best_moves.append(move)
        return random.choice(best_moves) if best_moves else Move.STAY

    def _predict_pacman_best_move(self, p_pos, target_g, steps=2):
        curr = p_pos
        for _ in range(steps):
            moves = self._get_valid_moves(curr)
            best_next = curr
            min_dist = float('inf')
            for m in moves:
                nxt = self._get_next_pos(curr, m)
                dist = self._manhattan_distance(nxt, target_g)
                if dist < min_dist:
                    min_dist = dist
                    best_next = nxt
            curr = best_next
            if curr == target_g: break
        return curr
        
    def _evaluate_minimax_state(self, g_pos, p_pos):
        features = self._get_features(g_pos,Move.STAY, p_pos)
            
        # Tính score dựa trên weights đã học
        score = sum(self.weights.get(f, 0) * val for f, val in features.items())
        return score
    
    # def _get_features_for_eval(self, g_pos, p_pos):
    #     dist = self._manhattan_distance(g_pos, p_pos)
    #     return {
    #         'bias': 1.0,
    #         'dist_to_pacman': dist, # Càng xa càng bị điểm âm (để ghost muốn lại gần)
    #         'is_dead_end': 1.0 if self._is_dead_end(g_pos) else 0.0,
    #         'visited_count': self.visited_map.get(g_pos, 0)
    #     }

    def _strategic_patrol(self, my_pos):
        # (Copy logic Patrol từ bài trước - nhớ Logic Cấm Quay Đầu)
        valid_moves = self._get_valid_moves(my_pos)
        if not valid_moves: return Move.STAY
        if Move.STAY in valid_moves and len(valid_moves) > 1: valid_moves.remove(Move.STAY)
        # Cấm quay đầu
        if self.last_action:
            rev = self._get_opposite_move(self.last_action)
            if rev in valid_moves and len(valid_moves) > 1: valid_moves.remove(rev)
            
        best_moves = []
        best_score = -float('inf')
        for move in valid_moves:
            next_pos = self._get_next_pos(my_pos, move)
            
            v_weight = self.weights.get('visited_count', -5.0)
            score = self.visited_map.get(next_pos, 0) * v_weight # Phạt nặng lối mòn
            
            if self._is_dead_end(next_pos): 
                score += self.weights.get('is_dead_end', -20.0)
                
            if score > best_score:
                best_score = score
                best_moves = [move]
            elif score == best_score: best_moves.append(move)
        return random.choice(best_moves) if best_moves else Move.STAY
    
    def _get_valid_moves(self, pos):
        moves = []
        for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            nxt = self._get_next_pos(pos, m)
            if self._is_passable(nxt): moves.append(m)
        return moves

    def _is_passable(self, pos):
        r, c = pos
        h, w = self.global_map.shape
        if 0 <= r < h and 0 <= c < w: return self.global_map[r, c] != 1 
        return False

    def _get_next_pos(self, pos, move):
        return (pos[0]+move.value[0], pos[1]+move.value[1])

    def _manhattan_distance(self, p1, p2):
        return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])

    def _is_dead_end(self, pos):
        exits = 0
        for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if self._is_passable(self._get_next_pos(pos, m)): exits += 1
        return exits <= 1
    
    def _get_opposite_move(self, move):
        mapping = {Move.UP: Move.DOWN, 
                   Move.DOWN: Move.UP, 
                   Move.LEFT: Move.RIGHT, 
                   Move.RIGHT: Move.LEFT, 
                   Move.STAY: Move.STAY}
        return mapping.get(move, Move.STAY)
    
    def __del__(self):
        self.save_weights()