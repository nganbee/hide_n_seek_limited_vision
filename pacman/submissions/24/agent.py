"""
Example student submission showing the required interface.

Students should implement their own PacmanAgent and/or GhostAgent
following this template.
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

# Import
from collections import deque, Counter
import heapq
import itertools


# ======================
# COMMON UTILITIES
# ======================

DIRECTIONS = {
    Move.UP: (-1, 0),
    Move.DOWN: (1, 0),
    Move.LEFT: (0, -1),
    Move.RIGHT: (0, 1)
}


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# ====================================================
# PACMAN AGENT
# ====================================================
class PacmanAgent(BasePacmanAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Master Pacman"
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        
        # 1. BỘ NHỚ & BẢN ĐỒ
        # self.internal_map: -1 = Chưa biết, 0 = Đường, 1 = Tường
        self.h, self.w = None, None
        self.internal_map = None 
        
        # self.visit_count: Heatmap đếm số lần đi qua mỗi ô
        self.visit_count = None 
        
        # 2. THEO DÕI ĐỊCH (TRACKING)
        self.last_known_ghost = None
        self.prev_ghost_pos = None # Dùng để tính hướng di chuyển của Ghost
        
        # 3. TRẠNG THÁI (STATE)
        self.stuck_counter = 0
        self.last_pos = None
        self.last_move = Move.STAY # để áp dụng quy tắc rẽ/đi lùi
        
        # 4. TIE-BREAKER 
        self.counter = itertools.count()

    def step(self, map_state: np.ndarray, my_position: tuple, enemy_position: tuple, step_number: int):
        # --- BƯỚC 1: KHỞI TẠO & CẬP NHẬT DỮ LIỆU ---
        if self.internal_map is None:
            self.h, self.w = map_state.shape
            self.internal_map = np.full((self.h, self.w), -1)
            self.visit_count = np.zeros((self.h, self.w), dtype=float)

        # Cập nhật những gì mắt thấy vào bộ nhớ vĩnh viễn
        self._update_map(map_state)
        
        # Đánh dấu heatmap tại vị trí đang đứng (cộng điểm để lần sau hạn chế đi lại)
        self.visit_count[my_position] += 1.0 
        
        # Kiểm tra xem có bị kẹt không (đứng yên tại chỗ)
        if self.last_pos == my_position: self.stuck_counter += 1
        else: self.stuck_counter = 0
        self.last_pos = my_position

        # --- BƯỚC 2: XÁC ĐỊNH MỤC TIÊU (TARGET SELECTION) ---
        target = None
        
        if enemy_position:
            # === ƯU TIÊN 1: SĂN ĐUỔI (HUNT) ===
            # Chiến thuật: Đón đầu (Intercept)
            target = enemy_position
            
            # Nếu biết vị trí trước đó, tính vector hướng đi của Ghost
            if self.prev_ghost_pos:
                dr = enemy_position[0] - self.prev_ghost_pos[0]
                dc = enemy_position[1] - self.prev_ghost_pos[1]
                # Dự đoán Ghost sẽ ở đâu sau 2 bước nữa
                pr, pc = enemy_position[0] + dr*2, enemy_position[1] + dc*2
                
                # Nếu điểm đón đầu an toàn, nhắm vào đó. Nếu không, nhắm thẳng vào Ghost.
                if self._is_safe(pr, pc):
                    target = (pr, pc)
            
            self.last_known_ghost = enemy_position
            self.prev_ghost_pos = enemy_position
            
        elif self.last_known_ghost:
            # === ƯU TIÊN 2: TRUY VẾT (CHASE) ===
            # Ghost vừa biến mất -> Chạy đến đúng chỗ biến mất
            target = self.last_known_ghost
            self.prev_ghost_pos = None # Mất dấu vector
            
            # Nếu đã đến nơi mà không thấy -> Xóa dấu vết
            if my_position == target:
                self.last_known_ghost = None
                target = None

        if not target:
            # === ƯU TIÊN 3: KHÁM PHÁ (EXPLORE) ===
            # Tìm các ô -1 (Unknown) để mở rộng bản đồ
            target = self._find_best_exploration_target(my_position)
            
            # === ƯU TIÊN 4: TUẦN TRA (PATROL) ===
            # Nếu hàm trên trả về None (tức là map đã sáng hết 100%)
            # Ta chuyển sang tìm các ô đã lâu không ghé qua
            if not target:
                target = self._find_patrol_target(my_position)

        # Fallback: Nếu vẫn không có target hoặc bị kẹt quá lâu -> Random
        if not target or self.stuck_counter > 3:
            return (self._get_random_valid_move(my_position), 1)

        # --- BƯỚC 3: TÌM ĐƯỜNG (MOMENTUM A*) ---
        # Tìm đường đi tối ưu đến target, ưu tiên đường thẳng
        path = self._momentum_a_star(my_position, target)

        if not path:
            return (self._get_random_valid_move(my_position), 1)

        # --- BƯỚC 4: QUYẾT ĐỊNH SỐ BƯỚC (SPEED LOGIC) ---
        first_move = path[0]
        steps = 1

        # --- HANDLE RETURN: APPLY TURNING RULE ---
        if first_move:
            is_turning = (
                first_move != self.last_move
                and self.last_move != Move.STAY
                and not (
                    (first_move == Move.UP and self.last_move == Move.DOWN) or
                    (first_move == Move.DOWN and self.last_move == Move.UP) or
                    (first_move == Move.LEFT and self.last_move == Move.RIGHT) or
                    (first_move == Move.RIGHT and self.last_move == Move.LEFT)
                )
            )
            if is_turning:
                steps = 1  # Quẹo → 1 bước
            else:
                # Kiểm tra đường thẳng ít nhất 2 bước
                if len(path) >= 2 and first_move == path[1]:
                    # Kiểm tra có thể đi 2 bước an toàn không
                    if self._can_move_steps(my_position, first_move, 2):
                        steps = min(self.pacman_speed, 2)
            # Lưu hướng lần trước
            self.last_move = first_move
        return (first_move, steps)

    # ======================================================
    # ALGORITHM 1: MOMENTUM A* (Tối ưu Speed & Tránh Rẽ)
    # ======================================================
    def _momentum_a_star(self, start, goal):
        # Nếu đích là tường, tìm ô trống cạnh đó
        if self.internal_map[goal[0], goal[1]] == 1:
            neighbors = [n for n in self._get_neighbors(goal) if self._is_safe(n[0], n[1])]
            if neighbors: goal = neighbors[0]
            else: return []

        pq = []
        # Cấu trúc Heap: (F-Score, G-Score, TIE_BREAKER, Current_Pos, Path)
        # TIE_BREAKER giúp tránh lỗi so sánh Move object
        count = next(self.counter)
        heapq.heappush(pq, (0, 0, count, start, []))
        
        visited = {start: 0}

        while pq:
            _, g, _, curr, path = heapq.heappop(pq)
            
            if curr == goal: return path
            if len(path) > 40: continue # Giới hạn độ sâu để tránh timeout

            last_move = path[-1] if path else None

            for move_enum, (dr, dc) in [(Move.UP, (-1,0)), (Move.DOWN, (1,0)), (Move.LEFT, (0,-1)), (Move.RIGHT, (0,1))]:
                nr, nc = curr[0] + dr, curr[1] + dc
                
                if self._is_safe(nr, nc):
                    # --- TÍNH CHI PHÍ DI CHUYỂN ---
                    move_cost = 1.0
                    
                    # 1. PHẠT RẼ (Turning Penalty) 
                    # Phạt 2.5 điểm nếu đổi hướng -> Ép Agent đi đường thẳng để dùng Speed x2
                    if last_move and move_enum != last_move:
                        move_cost += 2.5
                    
                    # 2. PHẠT ĐƯỜNG CŨ (Heatmap Penalty)
                    # Cộng thêm 10% số lần đã đi qua vào chi phí -> Tránh đi lại lối mòn
                    move_cost += self.visit_count[nr, nc] * 0.1

                    new_g = g + move_cost

                    if (nr, nc) not in visited or new_g < visited[(nr, nc)]:
                        visited[(nr, nc)] = new_g
                        # Heuristic: Manhattan Distance
                        h = abs(nr - goal[0]) + abs(nc - goal[1])
                        
                        new_path = list(path)
                        new_path.append(move_enum)
                        
                        count = next(self.counter)
                        heapq.heappush(pq, (new_g + h, new_g, count, (nr, nc), new_path))
        return []

    # ======================================================
    # ALGORITHM 2: TÌM MỤC TIÊU KHÁM PHÁ (Mapping)
    # ======================================================
    def _find_best_exploration_target(self, start):
        """Tìm ô -1 (Unknown) tốt nhất để đi tới"""
        candidates = []
        queue = deque([(start, 0)])
        visited = {start}
        
        found_unknown = False
        
        while queue:
            curr, dist = queue.popleft()
            if dist > 30: break 
            
            val = self.internal_map[curr]
            
            # Nếu tìm thấy ô chưa biết (-1)
            if val == -1:
                # Điểm số = 1000 - khoảng cách (Ưu tiên cái gần nhất)
                return curr 
            
            # Mở rộng tìm kiếm
            neighbors = self._get_neighbors(curr)
            random.shuffle(neighbors) # Shuffle để khám phá tự nhiên hơn
            
            for n in neighbors:
                if n not in visited and self._is_safe(n[0], n[1]):
                    visited.add(n)
                    queue.append((n, dist + 1))
        
        return None

    # ======================================================
    # ALGORITHM 3: TÌM MỤC TIÊU TUẦN TRA (Patrol)
    # ======================================================
    def _find_patrol_target(self, start):
        """
        Chạy khi Map đã hoàn thiện.
        Tìm ô có visit_count thấp nhất (lâu chưa đến).
        """
        candidates = []
        queue = deque([(start, 0)])
        visited = {start}
        count = 0
        
        while queue:
            curr, dist = queue.popleft()
            count += 1
            if count > 150: break # Chỉ tìm trong khu vực lân cận
            
            if self.internal_map[curr] == 1: continue
            
            # Score thấp là tốt: (Số lần thăm * 10) + Khoảng cách
            # Ưu tiên: Thăm ít > Gần
            score = self.visit_count[curr] * 10 + dist
            candidates.append((score, curr))
            
            for n in self._get_neighbors(curr):
                if n not in visited and self._is_safe(n[0], n[1]):
                    visited.add(n)
                    queue.append((n, dist + 1))
        
        if candidates:
            # Sort lấy score nhỏ nhất
            candidates.sort(key=lambda x: x[0])
            return candidates[0][1]
        
        return None

    # ======================================================
    # CÁC HÀM HỖ TRỢ (HELPERS)
    # ======================================================
    def _update_map(self, obs_map):
        """Hợp nhất tầm nhìn hiện tại vào bản đồ nhớ"""
        rows, cols = obs_map.shape
        for r in range(rows):
            for c in range(cols):
                if obs_map[r, c] != -1:
                    self.internal_map[r, c] = obs_map[r, c]

    def _get_neighbors(self, pos):
        """Lấy 4 ô xung quanh"""
        return [(pos[0]+dr, pos[1]+dc) for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]]

    def _is_safe(self, r, c):
        """Kiểm tra xem ô có đi được không (Trong biên và không phải Tường)"""
        # Lưu ý: -1 (Unknown) vẫn coi là safe để đi vào khám phá
        return 0 <= r < self.h and 0 <= c < self.w and self.internal_map[r, c] != 1

    def _can_move_steps(self, pos, move, steps):
        """Kiểm tra xem có thể đi thẳng n bước không"""
        r, c = pos
        dr, dc = move.value
        for _ in range(steps):
            r += dr
            c += dc
            if not self._is_safe(r, c): return False
        return True

    def _get_random_valid_move(self, pos):
        """Random một nước đi hợp lệ"""
        valid = [m for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT] if self._can_move_steps(pos, m, 1)]
        return random.choice(valid) if valid else Move.STAY





# ====================================================
# GHOST AGENT
# ====================================================
class GhostAgent(BaseGhostAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Master Ghost"
        
        # --- TUNING WEIGHTS ---
        self.W = {
            'dist_panic': 300.0,      # Ưu tiên 1: Khoảng cách sinh tồn
            'dist_safe': 15.0,        # Giữ khoảng cách khi an toàn
            'strong_safe': 120.0,     # Thưởng lớn cho Safe Zone tốt
            'weak_safe': 30.0,        # Safe Zone lởm (dẫn vào choke/dead)
            'escape_bias': 50.0,      # Thưởng cho việc tiến về Escape Target
            'chokepoint': -2500.0,    # Tử địa hành lang (khi panic)
            'danger_open': -600.0,    # Vùng quá thoáng (dễ bị bao vây)
            'dead_end': -999999.0,    # Cấm tuyệt đối
            'loop': -60.0
        }

        # --- MAP STRUCTURES ---
        self.rows = 0
        self.cols = 0
        self.map_analyzed = False
        
        # Sets lưu tọa độ
        self.dead_zones = set()
        self.chokepoints = set()
        self.strong_safe_zones = set() 
        self.weak_safe_zones = set()   
        self.danger_open_zones = set() 
        
        # --- STATE ---
        self.history = deque(maxlen=4)
        self.belief_state = None 
        self.escape_target = None      

    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int) -> Move:
        
        # 1. INIT & ANALYZE (1 lần đầu game)
        if not self.map_analyzed:
            self._analyze_grandmaster_map(map_state)
            # Init Belief uniform
            self.belief_state = np.zeros_like(map_state, dtype=float)
            zeros = np.where(map_state == 0)
            self.belief_state[zeros] = 1.0 / len(zeros[0])

        # 2. XÁC ĐỊNH TRẠNG THÁI 
        # Cần ước lượng Panic trước để dùng cho Belief Update
        # Lấy vị trí địch (thực hoặc dự đoán cũ)
        est_threat = enemy_position or self._get_most_likely_enemy_pos()
        dist_map_est = self._compute_distance_map(est_threat, map_state)
        current_dist = dist_map_est[my_position]
        
        # Panic nếu Pacman ở gần HOẶC đầu game 
        is_panic = (current_dist <= 8) or (step_number < 12)

        # 3. BELIEF UPDATE 
        if enemy_position:
            self.belief_state.fill(0)
            self.belief_state[enemy_position] = 1.0
            target_threat = enemy_position
        else:
            self._update_belief_adaptive(map_state, my_position, is_panic)
            target_threat = self._get_most_likely_enemy_pos()
            # Tính toán distance map với target mới 
            dist_map_est = self._compute_distance_map(target_threat, map_state)

        # 4. XÁC ĐỊNH ESCAPE TARGET
        # Nếu đang Panic, chọn 1 Strong Safe Zone gần nhất để chạy về
        self.escape_target = None
        if is_panic:
            self.escape_target = self._find_nearest_strong_safe_zone(my_position, dist_map_est)

        # 5. MOVE EVALUATION
        valid_moves = self._get_valid_moves(my_position, map_state)
        if not valid_moves: return Move.STAY

        best_move = None
        best_score = -float('inf')

        for move in valid_moves:
            next_pos = self._get_next_pos(my_position, move)
            
            score = self._evaluate_move(
                move, next_pos, dist_map_est, 
                map_state, my_position, is_panic
            )
            
            if score > best_score:
                best_score = score
                best_move = move
            elif score == best_score:
                if random.random() > 0.5: best_move = move

        if best_move:
            self.history.append(my_position)
            return best_move
        return Move.STAY

    # =========================================================================
    # CORE LOGIC: EVALUATION
    # =========================================================================
    def _evaluate_move(self, move, next_pos, dist_map, map_state, current_pos, is_panic):
        dist_to_pacman = dist_map[next_pos]
        
        # 1. DEAD ZONE 
        # Nếu next_pos là Dead Zone
        if next_pos in self.dead_zones:
            # Nếu đang ở ngoài thì cấm vào (trừ khi Pacman ở rất xa > 12)
            if current_pos not in self.dead_zones and dist_to_pacman < 12:
                return self.W['dead_end']
            # Nếu đang kẹt trong đó rồi thì ưu tiên lối ra (đã xử lý bởi BFS distance)

        score = 0
        
        # 2. DISTANCE (BFS)
        if is_panic:
            score += dist_to_pacman * self.W['dist_panic']
        else:
            score += dist_to_pacman * self.W['dist_safe']

        # 3. SAFE ZONE TIERS
        if next_pos in self.strong_safe_zones:
            score += self.W['strong_safe']
        elif next_pos in self.weak_safe_zones:
            score += self.W['weak_safe']

        # 4. CHOKEPOINT (Anti-Corridor)
        if is_panic and next_pos in self.chokepoints:
            # Nếu hành lang này dẫn lại gần Pacman -> Tự sát
            if dist_to_pacman <= dist_map[current_pos]:
                score += self.W['chokepoint']
            else:
                score -= 100 # Vẫn phạt nhẹ vì rủi ro bị chặn đầu bên kia

        # 5. DANGER OPEN ZONE (Center Box)
        # Chỉ phạt khi panic (lúc bình thường có thể đi qua để đổi cánh)
        if is_panic and next_pos in self.danger_open_zones:
             score += self.W['danger_open']

        # 6. ESCAPE TARGET BIAS
        # Nếu có mục tiêu rút lui, thưởng cho bước đi tiến gần mục tiêu đó
        if self.escape_target:
            # Dùng Manhattan cho nhanh (từ next_pos tới escape_target)
            curr_dist_esc = abs(current_pos[0] - self.escape_target[0]) + abs(current_pos[1] - self.escape_target[1])
            next_dist_esc = abs(next_pos[0] - self.escape_target[0]) + abs(next_pos[1] - self.escape_target[1])
            
            # Nếu bước này giảm khoảng cách tới Safe Zone -> Thưởng
            score += (curr_dist_esc - next_dist_esc) * self.W['escape_bias']

        # 7. Loop Penalty
        if next_pos in self.history:
            score += self.W['loop']

        return score

    # =========================================================================
    # LOGIC: MAP ANALYSIS 
    # =========================================================================
    def _analyze_grandmaster_map(self, map_state):
        self.rows, self.cols = map_state.shape
        
        # 1. Base Detection (Exits, Dead Zones, Chokepoints)
        base_safe_zones = set()
        
        # --- A. Wall Distance Transform ---
        # Tính khoảng cách từ mỗi ô tới bức tường gần nhất
        wall_dist_map = self._compute_wall_distance(map_state)
        
        for r in range(self.rows):
            for c in range(self.cols):
                if map_state[r, c] == 0:
                    exits = self._count_exits((r, c), map_state)
                    
                    if exits >= 3: base_safe_zones.add((r, c))
                    if exits <= 1: self.dead_zones.add((r, c))
                    
                    # Định nghĩa Danger Zone bằng khoảng cách tới tường
                    # Nếu cách tường >= 2 ô (tức là ở giữa khoảng trống 5x5 trở lên)
                    if wall_dist_map[r, c] >= 3: 
                        self.danger_open_zones.add((r, c))

        # --- Chokepoint Detection ---
        # Cần xác định chokepoint trước để phân loại Safe Zone
        self._detect_chokepoints(map_state, base_safe_zones)
        
        # --- C. Peeling Dead Zones ---
        self._peel_dead_zones(map_state)

        # --- D. CLASSIFY SAFE ZONES (Strong vs Weak) ---
        for sz in base_safe_zones:
            bad_exits = 0
            neighbors = self._get_neighbors(sz, map_state)
            
            for n in neighbors:
                # Lối đi xấu nếu dẫn vào Chokepoint hoặc Dead Zone
                if n in self.chokepoints or n in self.dead_zones:
                    bad_exits += 1
            
            # Nếu có tối đa 1 lối xấu (vẫn còn >=2 lối thoát tốt) -> Strong
            if bad_exits <= 1:
                self.strong_safe_zones.add(sz)
            else:
                self.weak_safe_zones.add(sz)
        
        self.map_analyzed = True

    def _compute_wall_distance(self, map_state):
        """BFS Multi-source từ tất cả các tường ra ngoài"""
        dist = np.full((self.rows, self.cols), 999)
        queue = deque()
        
        # Thêm tất cả tường vào queue
        walls = np.where(map_state == 1)
        for r, c in zip(walls[0], walls[1]):
            dist[r, c] = 0
            queue.append((r, c))
            
        while queue:
            r, c = queue.popleft()
            d = dist[r, c]
            
            for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    if dist[nr, nc] == 999:
                        dist[nr, nc] = d + 1
                        queue.append((nr, nc))
        return dist

    # =========================================================================
    # LOGIC: ADAPTIVE BELIEF 
    # =========================================================================
    def _update_belief_adaptive(self, map_state, my_pos, is_panic):
        new_belief = np.zeros_like(self.belief_state)
        active_indices = np.argwhere(self.belief_state > 0.001)
        
        # Check xem Ghost có đang an toàn không
        ghost_in_safe = my_pos in self.strong_safe_zones
        
        for r, c in active_indices:
            prob = self.belief_state[r, c]
            neighbors = []
            for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                nr, nc = r+dr, c+dc
                if self._is_valid((nr, nc), map_state):
                    neighbors.append((nr, nc))
            
            if neighbors:
                split_prob = prob / len(neighbors)
                for nr, nc in neighbors:
                    multiplier = 1.0

                    # --- BELIEF MODE SWITCH ---
                    is_choke = (nr, nc) in self.chokepoints
                    
                    if ghost_in_safe:
                        # Ghost đang thủ: Pacman ít dám chui vào corridor thẳng mặt,
                        # Pacman sẽ tìm cách flank qua đường open khác
                        if is_choke: 
                            multiplier *= 0.7
                        else: 
                            multiplier *= 1.2
                    elif is_panic:
                        # Ghost đang chạy loạn: Pacman sẽ lao vào corridor để cắt đường (intercept)
                        if is_choke: 
                            multiplier *= 1.6
                        else: 
                            multiplier *= 0.8
                    else:
                        # Normal mode
                        if is_choke: 
                            multiplier *= 1.3
                    
                    new_belief[nr, nc] += split_prob * multiplier
            else:
                 new_belief[r, c] += prob

        # Vision Update
        visible = self._get_cross_vision(my_pos, map_state)
        for vr, vc in visible: new_belief[vr, vc] = 0.0
        
        # Normalize
        total = np.sum(new_belief)
        if total > 0: self.belief_state = new_belief / total
        else: 
            zeros = np.where(map_state == 0)
            self.belief_state[zeros] = 1.0 / len(zeros[0])

    # =========================================================================
    # UTILS
    # =========================================================================
    def _find_nearest_strong_safe_zone(self, current_pos, dist_map):
        """Tìm Strong Safe Zone gần nhất dựa trên dist_map"""
        if not self.strong_safe_zones: return None
        
        # Lọc ra safe zone nào có khoảng cách nhỏ nhất trên dist_map
        best_target = None
        min_dist = float('inf')
        
        # Duyệt qua tất cả Strong Safe Zone
        for sz in self.strong_safe_zones:
            d = abs(current_pos[0] - sz[0]) + abs(current_pos[1] - sz[1])
            
            # Heuristic: Chọn Safe Zone gần mình NHƯNG phải xa Pacman
            dist_to_enemy = dist_map[sz]
            if dist_to_enemy < 6:
                continue # Safe zone này đang bị Pacman camp -> Bỏ
            
            if d < min_dist:
                min_dist = d
                best_target = sz
        
        return best_target

    def _detect_chokepoints(self, map_state, safe_zones_set):
        visited = set()
        for r in range(self.rows):
            for c in range(self.cols):
                if map_state[r,c] == 0 and (r,c) not in safe_zones_set and (r,c) not in visited:
                     if self._count_exits((r,c), map_state) == 2:
                         segment = self._trace_segment((r,c), map_state, safe_zones_set)
                         if len(segment) >= 4:
                             for cell in segment: self.chokepoints.add(cell)
                         visited.update(segment)

    def _trace_segment(self, start, map_state, safe_zones):
        # Simple BFS/DFS to find connected corridor cells
        q = deque([start])
        segment = {start}
        while q:
            curr = q.popleft()
            for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                nr, nc = curr[0]+dr, curr[1]+dc
                if self._is_valid((nr, nc), map_state) and (nr,nc) not in safe_zones:
                    if self._count_exits((nr,nc), map_state) == 2 and (nr,nc) not in segment:
                        # Check linearity (thẳng hàng) - optional nhưng tốt
                        segment.add((nr,nc))
                        q.append((nr,nc))
        return list(segment)

    def _peel_dead_zones(self, map_state):
        temp_map = map_state.copy()
        found = True
        while found:
            found = False
            for r in range(self.rows):
                for c in range(self.cols):
                    if temp_map[r,c] == 0:
                        exits = 0
                        for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                            if 0<=r+dr<self.rows and 0<=c+dc<self.cols and temp_map[r+dr, c+dc]==0:
                                exits+=1
                        if exits <= 1:
                            self.dead_zones.add((r,c))
                            temp_map[r,c] = 1
                            found = True

    def _compute_distance_map(self, start, map_state):
        dist_map = np.full((self.rows, self.cols), 999)
        dist_map[start] = 0
        queue = deque([start])
        while queue:
            r, c = queue.popleft()
            d = dist_map[r, c]
            if d > 25: continue
            for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                nr, nc = r+dr, c+dc
                if self._is_valid((nr, nc), map_state) and dist_map[nr, nc] == 999:
                    dist_map[nr, nc] = d + 1
                    queue.append((nr, nc))
        return dist_map

    def _get_neighbors(self, pos, map_state):
        neighbors = []
        for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
            nr, nc = pos[0]+dr, pos[1]+dc
            if self._is_valid((nr, nc), map_state):
                neighbors.append((nr, nc))
        return neighbors

    def _get_most_likely_enemy_pos(self):
        return np.unravel_index(np.argmax(self.belief_state), self.belief_state.shape)

    def _get_cross_vision(self, pos, map_state):
        cells = [pos]
        r, c = pos
        for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
            for i in range(1, 6):
                nr, nc = r + dr*i, c + dc*i
                if not (0<=nr<self.rows and 0<=nc<self.cols): break
                if map_state[nr, nc] == 1: break
                cells.append((nr, nc))
        return cells

    def _count_exits(self, pos, map_state):
        c = 0
        for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
            if self._is_valid((pos[0]+dr, pos[1]+dc), map_state): c += 1
        return c

    def _is_valid(self, pos, map_state):
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols and map_state[r, c] == 0
        
    def _get_valid_moves(self, pos, map_state):
        moves = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if self._is_valid(self._get_next_pos(pos, move), map_state):
                moves.append(move)
        return moves

    def _get_next_pos(self, pos, move):
        return (pos[0] + move.value[0], pos[1] + move.value[1])
    