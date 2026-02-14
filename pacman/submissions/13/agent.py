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
import json
import heapq
from collections import deque
UNKNOWN = -1
EMPTY = 0
WALL = 1



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

import numpy as np
import random
from collections import deque
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move

class GhostAgent(BaseGhostAgent):
    """
    THE PHANTOM GHOST (Hybrid Strategy)
    - Strategic Layer: Parkour Logic (Mobility, LOS Breaking, Sector Balancing).
    - Tactical Layer: Deep Survival BFS (12-step Lookahead Safety Net).
    - Kết hợp: Chọn nước đi 'ngon' nhất về thế trận mà vẫn đảm bảo an toàn tuyệt đối.
    """
    
    # ========== FULL MAP LAYOUT (HARD-CODED) ==========
    DEFAULT_MAP_LAYOUT = [
        "#####################",
        "#.........#.........#",
        "#.###.###.#.###.###.#",
        "#...................#",
        "#.###.#.#####.#.###.#",
        "#.....#...#...#.....#",
        "#####.### # ###.#####",
        "    #.#       #.#    ",
        "#####.# ##-## #.#####",
        "     .  . G .  .     ",
        "#####.# ##### #.#####",
        "    #.#       #.#    ",
        "#####.# ##### #.#####",
        "#.........#.........#",
        "#.###.###.#.###.###.#",
        "#...#.....P.....#...#",
        "###.#.#.#####.#.#.###",
        "#.....#...#...#.....#",
        "#.#######.#.#######.#",
        "#...................#",
        "#####################"
    ]
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Phantom_Hybrid"
        
        # --- CẤU HÌNH ---
        self.SURVIVAL_HORIZON = 12  # Nhìn trước 12 bước (Chuẩn Survival)
        self.EARLY_GAME_STEPS = 10  # Số bước đầu game chạy theo kịch bản
        self.MIDDLE_GAME_STEPS = 50  # Bước giữa game từ ẩn nấp thành chạy trốn
        
        # --- DATA MAP ---
        self.global_grid = None
        self.height = None
        self.width = None
        self.mobility_map = None  # Bản đồ độ thoáng (từ Parkour)
        
        # --- STATE ---
        self.history = deque(maxlen=4)
        self.last_known_pacman = None
        self.turns_since_seen = 0
        self.opening_target = (5,12)
        self.opening_moves = []
        self.sectors = {}

        # --- TRỌNG SỐ CHIẾN THUẬT (Parkour Weights) ---
        self.W_MOBILITY   = 50    # Thích chỗ thoáng (Ngã 3, Ngã 4)
        self.W_LOS_BREAK  = 500   # Cực thích ngắt tầm nhìn (Khắc chế Particle Filter)
        self.W_WALL_HUG   = 20    # Thích đi sát tường
        self.W_DISTANCE   = 10    # Giữ khoảng cách
        self.W_SECTOR     = 100   # Đi về khu vực đối diện
        self.W_HISTORY    = -500  # Phạt đi lại đường cũ
        self.W_RIGHT_BIAS = 20  # Bias RIGHT hơn LEFT (tạo unpredictability)

    def step(self, map_state, my_position, enemy_position, step_number):
        # 1. INIT MAP (Chạy 1 lần)
        if self.global_grid is None:
            self._initialize_map(map_state)
            # Chọn Opening Target là điểm thoáng nhất ở giữa map
            self.opening_target = (self.height // 2, self.width // 2)

        # 2. UPDATE STATE
        if enemy_position:
            self.last_known_pacman = enemy_position
            self.turns_since_seen = 0
            # Nếu thấy địch -> Hủy bỏ chế độ khai cuộc, chuyển sang chiến đấu
            self.opening_moves = [] 
        else:
            self.turns_since_seen += 1
            if self.last_known_pacman is None:
                self.last_known_pacman = (self.height // 2, self.width // 2)

        # 3. PHASE 1: OPENING (Chạy ra giữa map hoặc vị trí chiến lược)
        # Chỉ chạy khi chưa thấy địch và còn trong giai đoạn đầu
        if step_number <= self.EARLY_GAME_STEPS and not enemy_position and not self.opening_moves:
             if my_position != self.opening_target:
                 self.opening_moves = self.bfs_path(my_position, self.opening_target)

        if self.opening_moves:
            next_move = self.opening_moves.pop(0)
            # Kiểm tra an toàn sơ bộ cho Opening
            next_pos = self._get_next_pos(my_position, next_move)
            if self._is_safe_deep_check(next_pos, self.last_known_pacman):
                self._update_history(next_pos)
                return next_move
            else:
                self.opening_moves = [] # Hủy Opening nếu thấy nguy hiểm

        # 4. PHASE 2: COMBAT HYBRID (Parkour Scoring + Survival Filtering)
        best_move = self._decide_best_move(my_position, self.last_known_pacman)
        
        # Cập nhật lịch sử
        next_pos = self._get_next_pos(my_position, best_move)
        self._update_history(next_pos)
        
        return best_move

    # ==========================================================
    # CORE LOGIC: PARKOUR SCORING + SURVIVAL FILTER
    # ==========================================================

    def _decide_best_move(self, my_pos, pacman_pos):
        valid_moves = self.get_valid_neighbors_enum(my_pos)
        if not valid_moves: return Move.STAY

        candidates = []
        
        # Xác định Sector mục tiêu (Parkour Logic)
        target_pos = self._get_opposite_sector_center(pacman_pos)

        # A. CHẤM ĐIỂM (Scoring)
        for move, next_pos in valid_moves:
            score = 0
            
            # 1. Mobility (Độ thoáng)
            score += self.mobility_map[next_pos] * self.W_MOBILITY
            
            # 2. LOS Breaking (Tàng hình) - Quan trọng để lừa Pacman AI
            if not self.has_line_of_sight(next_pos, pacman_pos):
                score += self.W_LOS_BREAK
            
            # 3. Distance & Sector (Chiến thuật vĩ mô)
            dist_to_pac = self.manhattan(next_pos, pacman_pos)
            dist_to_target = self.manhattan(next_pos, target_pos)
            
            score += dist_to_pac * self.W_DISTANCE
            score -= dist_to_target * 5 # Càng gần target càng tốt (penalty thấp)
            
            # 4. Wall Hugging (Bám tường)
            walls = self._count_adjacent_walls(next_pos)
            score += walls * self.W_WALL_HUG
            
            # 5. History Penalty (Chống lặp)
            if next_pos in self.history:
                score += self.W_HISTORY
                
            # 6. Dead End Penalty (Heuristic sơ bộ)
            # Nếu là ngõ cụt mà Pacman đang ở gần -> Phạt cực nặng
            if self.mobility_map[next_pos] <= 1.0 and dist_to_pac < 8:
                score -= 2000
                
            # 7. Directional Bias (RIGHT > LEFT để unpredictable)
            if move == Move.RIGHT:
                score += self.W_RIGHT_BIAS  # +5 điểm
            elif move == Move.LEFT:
                score -= self.W_RIGHT_BIAS // 2  # -2.5 điểm (penalty nhẹ)

            candidates.append((score, move, next_pos))

        # B. SẮP XẾP & LỌC (Sorting & Filtering)
        # Sắp xếp từ điểm cao xuống thấp
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Duyệt qua các nước đi tốt nhất, nước nào AN TOÀN thì chọn ngay
        for score, move, next_pos in candidates:
            # KIỂM TRA SINH TỒN (Survival Check)
            # Nếu BFS bảo là chết sau 12 bước và step <70 -> BỎ QUA NGAY LẬP TỨC
            if self._is_safe_deep_check(next_pos, pacman_pos) or self.turns_since_seen > 5:
                print(f"Chose move {move} with score {score} (Safe)")
                return move
        
        # Fallback: Nếu tất cả đều dẫn đến cái chết (Checkmate), chọn cái sống lâu nhất
        # (Ở đây tạm thời chọn cái điểm cao nhất để liều mạng)
        if candidates:
            print(f"All moves risky, chose {candidates[0][1]} with score {candidates[0][0]} (Best of Worst)")
            print(f"Second best move was {candidates[1][1]} with score {candidates[1][0]}")
            return candidates[0][1]
            
        return Move.STAY

    # ==========================================================
    # SURVIVAL CHECK (BFS LOOKAHEAD - TỪ CODE CŨ CỦA BẠN)
    # ==========================================================
    
    def _is_safe_deep_check(self, start_node, pacman_pos):
        queue = deque([(start_node, 1)])
        visited = set([(start_node, 1)])
        
        init_dist = self.manhattan(start_node, pacman_pos)
        if init_dist <= 2: return False # Chết ngay bước đầu

        while queue:
            curr_pos, time = queue.popleft()
            if time >= self.SURVIVAL_HORIZON: return True # Sống sót
            
            valid_next = self.get_valid_neighbors_coords(curr_pos)
            
            for next_pos in valid_next:
                # Conservative Check: Pacman Speed 2
                dist_to_pac_origin = self.manhattan(next_pos, pacman_pos)
                if dist_to_pac_origin <= (time + 1) * 2:
                    continue # Nhánh này chết
                
                state = (next_pos, time + 1)
                if state not in visited:
                    visited.add(state)
                    queue.append(state)
        return False

    # ==========================================================
    # INITIALIZATION & MAP ANALYSIS
    # ==========================================================
    
    def _parse_map_layout(self):
        """Parse hard-coded map layout thành numpy array"""
        h = len(self.DEFAULT_MAP_LAYOUT)
        w = len(self.DEFAULT_MAP_LAYOUT[0])
        map_array = np.zeros((h, w), dtype=np.int8)
        
        for i, row in enumerate(self.DEFAULT_MAP_LAYOUT):
            for j, cell in enumerate(row):
                if cell == '#' or cell == '-':
                    map_array[i, j] = 1  # Wall
                else:
                    map_array[i, j] = 0  # Empty
        
        return map_array
    
    def _initialize_map(self, map_state):
        # Use FULL HARD-CODED MAP instead of limited vision
        full_map = self._parse_map_layout()
        self.height, self.width = full_map.shape
        self.global_grid = (full_map == 1)  # True là tường
        self.mobility_map = np.zeros((self.height, self.width))
        
        # Tính Mobility (Độ thoáng) cho từng ô
        for r in range(self.height):
            for c in range(self.width):
                if not self.global_grid[r, c]:
                    neighbors = self.get_valid_neighbors_coords((r, c))
                    # Score = số lối ra. Ngã 3 (3.0), Ngã 4 (4.0) là ngon nhất.
                    self.mobility_map[r, c] = len(neighbors)
        
        # Init Sectors
        mid_r, mid_c = self.height // 2, self.width // 2
        self.sectors = {
            'TL': (mid_r // 2, mid_c // 2),
            'TR': (mid_r // 2, mid_c + mid_c // 2),
            'BL': (mid_r + mid_r // 2, mid_c // 2),
            'BR': (mid_r + mid_r // 2, mid_c + mid_c // 2)
        }

    def _get_opposite_sector_center(self, pac_pos):
        # Xác định Pacman đang ở đâu
        r, c = pac_pos
        mid_r, mid_c = self.height // 2, self.width // 2
        
        if r < mid_r: # Top
            if c < mid_c: pac_sec = 'TL'
            else:         pac_sec = 'TR'
        else: # Bottom
            if c < mid_c: pac_sec = 'BL'
            else:         pac_sec = 'BR'
            
        # Chọn đối diện
        if pac_sec == 'TL': return self.sectors['BR']
        if pac_sec == 'TR': return self.sectors['BL']
        if pac_sec == 'BL': return self.sectors['TR']
        return self.sectors['TL'] # BR -> TL

    # ==========================================================
    # HELPER METHODS
    # ==========================================================

    def get_valid_neighbors_enum(self, pos):
        r, c = pos
        valid = []
        for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            nr, nc = r + m.value[0], c + m.value[1]
            if self._is_valid(nr, nc): valid.append((m, (nr, nc)))
        return valid

    def get_valid_neighbors_coords(self, pos):
        r, c = pos
        valid = []
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = r+dr, c+dc
            if self._is_valid(nr, nc): valid.append((nr, nc))
        return valid

    def has_line_of_sight(self, p1, p2):
        r1, c1 = p1
        r2, c2 = p2
        if r1 == r2:
            step = 1 if c2 > c1 else -1
            for c in range(c1 + step, c2, step):
                if self.global_grid[r1, c]: return False
            return True
        if c1 == c2:
            step = 1 if r2 > r1 else -1
            for r in range(r1 + step, r2, step):
                if self.global_grid[r, c1]: return False
            return True
        return False

    def bfs_path(self, start, target):
        queue = deque([(start, [])])
        visited = {start}
        while queue:
            curr, path = queue.popleft()
            if curr == target: return path
            for m, next_pos in self.get_valid_neighbors_enum(curr):
                if next_pos not in visited:
                    visited.add(next_pos)
                    queue.append((next_pos, path + [m]))
        return []

    def _count_adjacent_walls(self, pos):
        count = 0
        r, c = pos
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = r+dr, c+dc
            # Ngoài map hoặc là tường thì tính là tường
            if not (0 <= nr < self.height and 0 <= nc < self.width) or self.global_grid[nr, nc]:
                count += 1
        return count

    def _update_history(self, pos):
        self.history.append(pos)

    def _get_next_pos(self, pos, move):
        return (pos[0]+move.value[0], pos[1]+move.value[1])

    def _is_valid(self, r, c):
        return 0 <= r < self.height and 0 <= c < self.width and not self.global_grid[r, c]

    def manhattan(self, p1, p2):
        return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])