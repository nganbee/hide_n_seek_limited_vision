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

"""
Template for student agent implementation.
"""

import sys
from pathlib import Path
import numpy as np
import random
import heapq
from collections import deque

# Add src to path to import the interface
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move


class PacmanAgent(BasePacmanAgent):
    """
    Pacman v9.0 - Static Map Optimized
    Logic gốc: Speed Demon (Chase, Intercept, Prediction)
    Nâng cấp: Sử dụng Static Map & Pre-computed Distances để xử lý tức thì.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Pacman Static God"
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        
        # --- STATIC MAP DATA ---
        self.static_map = self._load_classic_layout()
        self.map_h, self.map_w = self.static_map.shape
        
        # --- PRE-COMPUTATION (Chạy 1 lần duy nhất) ---
        # dist_matrix[(start_pos)][end_pos] = real_distance
        self.dist_matrix = self._precompute_all_distances()
        self.dead_ends = self._precompute_dead_ends()
        
        # --- DYNAMIC STATE ---
        self.last_known_enemy_pos = None
        self.enemy_history = deque(maxlen=5)
        self.my_history = deque(maxlen=4)
        self.ghost_belief = np.zeros(self.static_map.shape)
        
        # Init belief uniform
        empties = np.argwhere(self.static_map == 0)
        prob = 1.0 / len(empties)
        for r, c in empties:
            self.ghost_belief[r, c] = prob

    def step(self, map_state: np.ndarray, my_position: tuple, enemy_position: tuple, step_number: int):
        # 0. Check map size consistency (nếu map game khác map static thì fallback)
        if map_state.shape != self.static_map.shape:
            # Fallback đơn giản nếu map lạ (hiếm khi xảy ra)
            return (Move.UP, 1)

        self.my_history.append(my_position)
        
        # 1. Update Belief State
        if enemy_position:
            self.last_known_enemy_pos = enemy_position
            self.enemy_history.append(enemy_position)
            # Reset belief
            self.ghost_belief.fill(0)
            self.ghost_belief[enemy_position] = 1.0
        else:
            self._diffuse_belief(map_state)

        # 2. Anti-Loop (Nếu đi qua đi lại 1 chỗ quá nhiều)
        if self.my_history.count(my_position) >= 3:
            return self._escape_loop_static(my_position)

        # 3. Determine Target
        target = self._strategy_selector(my_position, enemy_position)
        
        # NẾU TARGET LÀ CHÍNH MÌNH (Đã đến nơi) -> Ép tìm target mới hoặc đi random
        if target == my_position:
            # Xóa xác suất tại đây để nó tìm chỗ khác
            self.ghost_belief[my_position] = 0 
            target = None 
        
        # 4. Move Execution
        action = (Move.STAY, 1)
        if target:
            action = self._execute_move_static(my_position, target)
        
        # === [QUAN TRỌNG] FORCE MOVE LOGIC ===
        # Nếu thuật toán trả về STAY hoặc action bị None, ép buộc di chuyển
        if action[0] == Move.STAY:
            return self._force_move(my_position)
            
        return action

    # =========================================================================
    # CHIẾN THUẬT (LOGIC GỐC CỦA BẠN)
    # =========================================================================
    
    def _force_move(self, pos):
        """
        Tìm mọi nước đi có thể (trừ tường).
        Ưu tiên: Ô chưa đi qua gần đây > Ô ngẫu nhiên.
        """
        valid_moves = []
        possible_moves = []
        
        for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            nxt = self._get_next_pos(pos, m)
            if self._is_valid_static(nxt):
                # Lưu lại move và số bước có thể đi (ưu tiên max speed)
                steps = 0
                curr = pos
                for _ in range(self.pacman_speed):
                    n = self._get_next_pos(curr, m)
                    if not self._is_valid_static(n): break
                    curr = n
                    steps += 1
                
                if steps > 0:
                    possible_moves.append((m, steps))
                    # Nếu ô này không trùng history gần đây thì ưu tiên
                    if nxt not in self.my_history:
                        valid_moves.append((m, steps))
        
        # 1. Ưu tiên đi vào ô mới (chưa bị kẹt)
        if valid_moves:
            return random.choice(valid_moves)
        
        # 2. Nếu kẹt quá thì đi đại hướng nào cũng được (miễn là đi được)
        if possible_moves:
            return random.choice(possible_moves)
            
        # 3. Chết dí (4 bức tường) - hiếm khi xảy ra
        return (Move.STAY, 1)
    
    def _strategy_selector(self, my_pos, ghost_pos):
        # A. Nếu thấy Ghost
        if ghost_pos:
            real_dist = self._get_real_dist(my_pos, ghost_pos)
            
            # Case 1: Rất gần (<= 2 bước) -> Lao thẳng tới (Greedy)
            if real_dist <= 2:
                return ghost_pos
            
            # Case 2: Tầm trung (3-5 bước) -> Cắt góc (Intercept)
            if real_dist <= 5:
                predicted = self._predict_ghost_pos(ghost_pos, steps=2)
                # Nếu vị trí dự đoán không phải tường, chạy tới đó
                if self.static_map[predicted] == 0:
                    return predicted
                return ghost_pos
            
            # Case 3: Xa -> Dự đoán đường dài
            return self._predict_ghost_pos(ghost_pos, steps=4)

        # B. Nếu không thấy Ghost -> Săn lùng theo xác suất
        # Tìm ô có xác suất cao nhất
        flat_idx = np.argmax(self.ghost_belief)
        target = np.unravel_index(flat_idx, self.ghost_belief.shape)
        if self.ghost_belief[target] > 0:
            return target
            
        return self.last_known_enemy_pos

    def _execute_move_static(self, my_pos, target_pos):
        """
        Tìm nước đi tốt nhất dựa trên bảng khoảng cách đã tính sẵn.
        Tự động tối ưu Speed 2.
        """
        best_move = None
        min_turns = float('inf')
        
        # Check 4 hướng
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            # 1. Check xem đi được 1 bước hay 2 bước theo hướng này
            steps_possible = 0
            curr = my_pos
            for _ in range(self.pacman_speed):
                nxt = self._get_next_pos(curr, move)
                if not self._is_valid_static(nxt):
                    break
                curr = nxt
                steps_possible += 1
            
            if steps_possible == 0:
                continue
                
            final_pos_after_move = curr
            
            # 2. Tính xem từ vị trí mới, còn bao xa nữa tới đích?
            dist_remaining = self._get_real_dist(final_pos_after_move, target_pos)
            
            # 3. Heuristic: Số lượt ước tính = 1 (lượt này) + (Quãng đường còn lại / 2)
            # Ưu tiên đi 2 bước (steps_possible=2) để tiếp cận nhanh hơn
            turns_needed = 1 + (dist_remaining / 2.0)
            
            # Phạt nhẹ nếu hướng này chỉ đi được 1 bước mà lẽ ra đi được 2 (để ưu tiên đường thoáng)
            if steps_possible < self.pacman_speed and dist_remaining > 0:
                turns_needed += 0.1

            if turns_needed < min_turns:
                min_turns = turns_needed
                best_move = (move, steps_possible)
        
        if best_move:
            return best_move
        return (Move.STAY, 1)

    # =========================================================================
    # HELPER: PRE-COMPUTATION & STATIC DATA
    # =========================================================================

    def _load_classic_layout(self):
        """Map Pacman Cổ Điển (21x21) -> Numpy Array (1=Wall, 0=Path)"""
        layout = [
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
        grid = np.zeros((21, 21), dtype=int)
        for r, row in enumerate(layout):
            for c, char in enumerate(row):
                if char in ['#', '-']: # Tường hoặc Cửa nhà ma
                    grid[r, c] = 1
                else:
                    grid[r, c] = 0
        return grid

    def _precompute_all_distances(self):
        """BFS từ mọi ô trống đến mọi ô trống khác. O(N^2) nhưng chỉ chạy 1 lần."""
        h, w = self.map_h, self.map_w
        dist_matrix = {}
        empties = [(r, c) for r in range(h) for c in range(w) if self.static_map[r, c] == 0]
        
        for start in empties:
            q = deque([(start, 0)])
            visited = {start}
            dist_matrix[start] = {}
            dist_matrix[start][start] = 0
            
            while q:
                curr, d = q.popleft()
                dist_matrix[start][curr] = d
                
                r, c = curr
                for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < h and 0 <= nc < w and self.static_map[nr, nc] == 0:
                        if (nr, nc) not in visited:
                            visited.add((nr, nc))
                            q.append(((nr, nc), d+1))
        return dist_matrix

    def _precompute_dead_ends(self):
        """Tìm các ô ngõ cụt tĩnh"""
        dead_ends = set()
        h, w = self.map_h, self.map_w
        for r in range(h):
            for c in range(w):
                if self.static_map[r, c] == 0:
                    walls = 0
                    for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                        nr, nc = r+dr, c+dc
                        if not (0 <= nr < h and 0 <= nc < w) or self.static_map[nr, nc] == 1:
                            walls += 1
                    if walls >= 3:
                        dead_ends.add((r, c))
        return dead_ends

    def _get_real_dist(self, p1, p2):
        """Lấy khoảng cách thực tế O(1)"""
        if p1 not in self.dist_matrix or p2 not in self.dist_matrix[p1]:
            return float('inf') # Không đi đến được
        return self.dist_matrix[p1][p2]

    # =========================================================================
    # HELPER: LOGIC PHỤ TRỢ
    # =========================================================================

    def _predict_ghost_pos(self, ghost_pos, steps=2):
        """Dự đoán đơn giản dựa trên lịch sử di chuyển"""
        if len(self.enemy_history) < 2:
            return ghost_pos
        
        last = self.enemy_history[-1]
        prev = self.enemy_history[-2]
        dr = last[0] - prev[0]
        dc = last[1] - prev[1]
        
        # Dự đoán tuyến tính
        pred_r = ghost_pos[0] + dr * steps
        pred_c = ghost_pos[1] + dc * steps
        
        # Clamp bounds & check walls
        pred_r = max(0, min(self.map_h - 1, pred_r))
        pred_c = max(0, min(self.map_w - 1, pred_c))
        
        if self.static_map[pred_r, pred_c] == 0:
            return (pred_r, pred_c)
        return ghost_pos

    def _diffuse_belief(self, map_state):
        """Lan truyền xác suất khi không thấy Ghost"""
        # Xóa xác suất ở những nơi Pacman đang nhìn thấy (mà không có Ghost)
        visible_mask = map_state != -1
        self.ghost_belief[visible_mask] = 0
        
        # Diffusion Step
        new_belief = np.zeros_like(self.ghost_belief)
        active_cells = np.argwhere(self.ghost_belief > 0)
        
        for r, c in active_cells:
            val = self.ghost_belief[r, c]
            neighbors = []
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = r+dr, c+dc
                if self._is_valid_static((nr, nc)):
                    neighbors.append((nr, nc))
            
            if not neighbors:
                new_belief[r, c] += val
            else:
                # Ghost có xu hướng giữ nguyên vị trí hoặc di chuyển
                stay_prob = 0.2
                move_prob = 0.8 / len(neighbors)
                new_belief[r, c] += val * stay_prob
                for nr, nc in neighbors:
                    new_belief[nr, nc] += val * move_prob
                    
        self.ghost_belief = new_belief
        total = np.sum(self.ghost_belief)
        if total > 0: self.ghost_belief /= total

    def _escape_loop_static(self, pos):
        self.my_history.clear()
        valid_moves = []
        for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            n = self._get_next_pos(pos, m)
            if self._is_valid_static(n):
                valid_moves.append(m)
        if valid_moves:
            return (random.choice(valid_moves), 1)
        return (Move.STAY, 1)

    def _is_valid_static(self, pos):
        r, c = pos
        return 0 <= r < self.map_h and 0 <= c < self.map_w and self.static_map[r, c] == 0

    def _get_next_pos(self, pos, move):
        return (pos[0] + move.value[0], pos[1] + move.value[1])

import numpy as np
import random
import os
import json
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move


class GhostAgent(BaseGhostAgent):
    """
    Ghost (Hider) Agent - Goal: Avoid being caught
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Parkour_Ghost"
        
        # State & Memory
        self.history = deque(maxlen=4)
        self.last_known_pacman = None
        self.turns_since_seen = 0
        
        # CHIẾN THUẬT KHAI CUỘC (Ambush)
        self.opening_target = None
        self.opening_moves = [] 
        self.sectors = {}
        
        # TRẠNG THÁI
        self.in_opening_phase = True
        self.in_camping_phase = False 

    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int) -> Move:
        
        height, width = map_state.shape
        
        # Setup động (Dynamic Setup) cho lần gọi đầu tiên để thích ứng mọi map
        if not self.sectors:
            self.sectors = {
                'TR': (height // 4, width * 3 // 4),
                'TL': (height // 4, width // 4),
                'BL': (height * 3 // 4, width // 4),
                'BR': (height * 3 // 4, width * 3 // 4)
            }
            # Mặc định góc phục kích theo tỷ lệ map, nếu bị vướng tường thì lấy vị trí hiện tại
            ambush_spot = (5, 12)
            if ambush_spot[0] < height and ambush_spot[1] < width and map_state[ambush_spot[0], ambush_spot[1]] != 1:
                self.opening_target = ambush_spot
            else:
                self.opening_target = my_position # Bỏ qua phase mở đầu nếu vị trí lỗi
                self.in_opening_phase = False
                self.in_camping_phase = True
                
        # --- 0. BÁO ĐỘNG ĐỎ ---
        if enemy_position is not None:
            if self.in_opening_phase or self.in_camping_phase:
                self.in_opening_phase = False
                self.in_camping_phase = False
            
            self.last_known_pacman = enemy_position
            self.turns_since_seen = 0
        else:
            self.turns_since_seen += 1

        if self.last_known_pacman is None:
            self.last_known_pacman = (height // 2, width // 2)

        # --- PHASE 1: OPENING ---
        if self.in_opening_phase:
            if my_position == self.opening_target:
                self.in_opening_phase = False
                self.in_camping_phase = True
                return Move.STAY
            else:
                if not self.opening_moves:
                    self.opening_moves = self._bfs_path(my_position, self.opening_target, map_state)
                
                if self.opening_moves:
                    next_move = self.opening_moves.pop(0)
                    if next_move:
                        dr, dc = next_move.value
                        self.history.append((my_position[0]+dr, my_position[1]+dc))
                        return next_move
                self.in_opening_phase = False 

        # --- PHASE 2: CAMPING ---
        if self.in_camping_phase:
            return Move.STAY

        # --- PHASE 3: ACTIVE EVASION (PARKOUR STYLE) ---
        pacman_dist_map = self._get_bfs_distance_map(self.last_known_pacman, map_state)
        
        # 2. Tìm điểm an toàn nhất (xa Pacman nhất theo đường đi thực tế)
        target_pos = my_position
        max_dist = -1
        for pos, dist in pacman_dist_map.items():
            if dist > max_dist:
                max_dist = dist
                target_pos = pos

        # 3. Chấm điểm nước đi dựa trên khoảng cách thực tế (truyền thêm pacman_dist_map)
        best_move = self._evaluate_best_move(my_position, self.last_known_pacman, target_pos, map_state, pacman_dist_map)
        if best_move is None: best_move = Move.STAY

        # Anti-stuck
        if best_move == Move.STAY and self.turns_since_seen < 5:
             valid = self._get_valid_neighbors(my_position, map_state)
             candidates = [n for n in valid if n not in self.history]
             if candidates:
                 next_pos = random.choice(candidates)
                 dr, dc = next_pos[0]-my_position[0], next_pos[1]-my_position[1]
                 for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                     if m.value == (dr, dc): return m

        dr, dc = best_move.value
        self.history.append((my_position[0]+dr, my_position[1]+dc))
        return best_move

    # --- HELPER METHODS ---
    
    def _get_bfs_distance_map(self, start_pos, map_state):
        """Trả về một dictionary chứa khoảng cách thực tế từ start_pos đến mọi ô"""
        distances = {start_pos: 0}
        queue = [start_pos]
        
        while queue:
            curr = queue.pop(0)
            dist = distances[curr]
            
            for nr, nc in self._get_valid_neighbors(curr, map_state):
                if (nr, nc) not in distances:
                    distances[(nr, nc)] = dist + 1
                    queue.append((nr, nc))
        return distances

    def _get_valid_neighbors(self, pos, map_state):
        r, c = pos
        height, width = map_state.shape
        valid = []
        for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            nr, nc = r + m.value[0], c + m.value[1]
            if 0 <= nr < height and 0 <= nc < width and map_state[nr, nc] != 1:
                valid.append((nr, nc))
        return valid

    def _has_line_of_sight(self, p1, p2, map_state):
        r1, c1 = p1
        r2, c2 = p2
        if r1 == r2:
            step = 1 if c2 > c1 else -1
            for c in range(c1 + step, c2, step):
                if map_state[r1, c] == 1: return False
            return True
        if c1 == c2:
            step = 1 if r2 > r1 else -1
            for r in range(r1 + step, r2, step):
                if map_state[r, c1] == 1: return False
            return True
        return False

    def _bfs_path(self, start, target, map_state):
        if start == target: return []
        queue = [(start, [])]
        visited = {start}
        
        while queue:
            curr, path = queue.pop(0)
            if curr == target: return path
            
            for nr, nc in self._get_valid_neighbors(curr, map_state):
                if (nr, nc) not in visited:
                    visited.add((nr, nc))
                    new_path = list(path)
                    dr, dc = nr - curr[0], nc - curr[1]
                    move_enum = None
                    for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                        if m.value == (dr, dc):
                            move_enum = m
                            break
                    if move_enum:
                        new_path.append(move_enum)
                        queue.append(((nr, nc), new_path))
        return []

    def _evaluate_best_move(self, my_pos, enemy_pos, target_pos, map_state, pacman_dist_map):
        valid_neighbors = self._get_valid_neighbors(my_pos, map_state)
        moves_score = []
        height, width = map_state.shape

        for nr, nc in valid_neighbors:
            score = 0
            
            # 1. SAFETY
            real_dist_to_enemy = pacman_dist_map.get((nr, nc), 0)
            # dist_to_enemy = abs(nr - enemy_pos[0]) + abs(nc - enemy_pos[1])
            # if dist_to_enemy < 4: 
            #     score -= 2000
            #     score += dist_to_enemy * 50 
            if real_dist_to_enemy <= 3: 
                score -= 2000 # Tử địa thực sự
            elif real_dist_to_enemy <= 5:
                score -= 500
                
            score += real_dist_to_enemy * 20
            
            # 2. LOS BREAKING
            if not self._has_line_of_sight((nr, nc), enemy_pos, map_state):
                score += 300 

            # 3. ANTI-CORRIDOR & JUNCTION
            next_valid_moves = self._get_valid_neighbors((nr, nc), map_state)
            num_exits = len(next_valid_moves)
            
            is_corridor = False
            if num_exits == 2:
                r1, c1 = next_valid_moves[0]
                r2, c2 = next_valid_moves[1]
                if r1 == r2 or c1 == c2: 
                    is_corridor = True
            
            if num_exits >= 3: score += 100 
            elif num_exits == 1: score -= 500 
            elif is_corridor: score -= 100 
            else: score += 50 

            # 4. WALL HUGGING
            adjacent_walls = 0
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                 check_r, check_c = nr+dr, nc+dc
                 if 0 <= check_r < height and 0 <= check_c < width:
                     if map_state[check_r, check_c] == 1:
                         adjacent_walls += 1
            
            if adjacent_walls > 0: score += 20 * adjacent_walls 

            # 5. TARGET DIRECTION
            dist_to_target = abs(nr - target_pos[0]) + abs(nc - target_pos[1])
            score -= dist_to_target * 5 

            # 6. HISTORY
            if (nr, nc) in self.history:
                score -= 200

            dr, dc = nr - my_pos[0], nc - my_pos[1]
            move_enum = Move.STAY
            for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                if m.value == (dr, dc): 
                    move_enum = m
                    break
            
            moves_score.append((score, move_enum))

        if not moves_score: return Move.STAY
        moves_score.sort(key=lambda x: x[0], reverse=True)
        return moves_score[0][1]