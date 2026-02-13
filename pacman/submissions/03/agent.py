import sys
from pathlib import Path
import numpy as np
import random
from collections import deque
import heapq
import itertools
import time

# Configure path for imports
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move

DELTAS_TO_MOVE = {
    (-1, 0): Move.UP,
    (1, 0): Move.DOWN,
    (0, -1): Move.LEFT,
    (0, 1): Move.RIGHT,
    (0, 0): Move.STAY
}

class PacmanAgent(BasePacmanAgent):
    """Agent Pacman dùng HMM, Minimax-lite và A*."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Pacman HMM"
        self.size = 21
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.mental_map = np.full((self.size, self.size), -1)
        self.belief = np.zeros((self.size, self.size))
        self.initialized = False
        self.alpha = 0.6
        self.beta = 0.9
        self.gamma = 0.9

    def _initialize_belief(self):
        """Khởi tạo belief đều trên các ô trống."""
        free = np.where(self.mental_map == 0)
        if len(free[0]) == 0:
            return
        p = 1.0 / len(free[0])
        self.belief[:] = 0
        for r, c in zip(free[0], free[1]):
            self.belief[r, c] = p
        self.initialized = True

    def _transition(self, pacman_pos):
        """Dự đoán chuyển động Ghost dựa trên belief hiện tại."""
        new_belief = np.zeros_like(self.belief)

        for r in range(self.size):
            for c in range(self.size):
                p = self.belief[r, c]
                if p == 0:
                    continue

                neighbors = []
                weights = []

                for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                    nr, nc = r + m.value[0], c + m.value[1]
                    if 0 <= nr < self.size and 0 <= nc < self.size:
                        if self.mental_map[nr, nc] == 0:
                            d = abs(nr - pacman_pos[0]) + abs(nc - pacman_pos[1])
                            w = np.exp(self.alpha * d)
                            neighbors.append((nr, nc))
                            weights.append(w)

                if neighbors:
                    weights = np.array(weights)
                    weights /= weights.sum()
                    for (nr, nc), w in zip(neighbors, weights):
                        new_belief[nr, nc] += p * w

        if new_belief.sum() > 0:
            new_belief /= new_belief.sum()

        return new_belief

    def _emission(self, belief, ghost_pos, visible_mask):
        """Cập nhật belief dựa trên quan sát."""
        if ghost_pos is not None:
            belief[:] = 0
            belief[ghost_pos] = 1.0
        else:
            belief[visible_mask] = 0
        return belief

    def _hmm_filter(self, pacman_pos, ghost_pos, visible_mask):
        """Thực hiện HMM filtering."""
        if not self.initialized:
            self._initialize_belief()

        predicted = self._transition(pacman_pos)
        updated = self._emission(predicted, ghost_pos, visible_mask)

        total = updated.sum()
        if total > 0:
            updated /= total

        self.belief = updated

    def _minimax_predict(self, pacman_pos):
        """Dự đoán di chuyển Ghost với Minimax-lite."""
        future = np.zeros_like(self.belief)

        for r in range(self.size):
            for c in range(self.size):
                p = self.belief[r, c]
                if p == 0:
                    continue

                best_dist = -1
                best_cells = []

                for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                    nr, nc = r + m.value[0], c + m.value[1]
                    if 0 <= nr < self.size and 0 <= nc < self.size:
                        if self.mental_map[nr, nc] == 0:
                            d = abs(nr - pacman_pos[0]) + abs(nc - pacman_pos[1])
                            if d > best_dist:
                                best_dist = d
                                best_cells = [(nr, nc)]
                            elif d == best_dist:
                                best_cells.append((nr, nc))

                if best_cells:
                    share = p / len(best_cells)
                    for cell in best_cells:
                        future[cell] += share

        if future.sum() > 0:
            future /= future.sum()

        return future

    def _select_intercept(self, pacman_pos, future_belief):
        """Chọn ô đích để chặn Ghost."""
        best_score = -1
        best = None

        for r in range(self.size):
            for c in range(self.size):
                if future_belief[r, c] > 0 and self.mental_map[r, c] == 0:
                    d = abs(r - pacman_pos[0]) + abs(c - pacman_pos[1])
                    score = future_belief[r, c] / (d + 1e-5) * self.gamma
                    if score > best_score:
                        best_score = score
                        best = (r, c)
        return best

    def _is_safe(self, pos):
        """Kiểm tra ô có an toàn để di chuyển."""
        r, c = pos
        return 0 <= r < self.size and 0 <= c < self.size and self.mental_map[r, c] != 1

    def _astar(self, start, goal, belief):
        """A* pathfinding với chi phí động."""
        open_set = [(0, start)]
        came_from = {}
        g = {start: 0}

        while open_set:
            _, curr = heapq.heappop(open_set)
            if curr == goal:
                path = []
                while curr != start:
                    path.append(curr)
                    curr = came_from[curr]
                return path[::-1]

            for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                nxt = (curr[0] + m.value[0], curr[1] + m.value[1])
                if not self._is_safe(nxt):
                    continue

                cost = 1 - self.beta * belief[nxt]
                ng = g[curr] + max(0.2, cost)

                if ng < g.get(nxt, float("inf")):
                    g[nxt] = ng
                    came_from[nxt] = curr
                    f = ng + abs(nxt[0] - goal[0]) + abs(nxt[1] - goal[1])
                    heapq.heappush(open_set, (f, nxt))
        return []

    def step(self, obs, my_pos, visible_enemy, step_number):
        """Quyết định chính."""
        visible_mask = (obs != -1)
        self.mental_map[visible_mask] = obs[visible_mask]
        self._hmm_filter(my_pos, visible_enemy, visible_mask)

        future_belief = self._minimax_predict(my_pos)
        target = self._select_intercept(my_pos, future_belief)

        if target is None:
            queue = deque([my_pos])
            visited = {my_pos}
            found_unknown = None
            while queue:
                curr = queue.popleft()
                if self.mental_map[curr] == -1:
                    found_unknown = curr
                    break
                for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                    nr, nc = curr[0] + m.value[0], curr[1] + m.value[1]
                    if 0 <= nr < self.size and 0 <= nc < self.size and (nr, nc) not in visited:
                        if self.mental_map[nr, nc] != 1:
                            visited.add((nr, nc))
                            queue.append((nr, nc))
            target = found_unknown

        if target is None:
            return (Move.STAY, 1)

        path = self._astar(my_pos, target, future_belief)
        if not path:
            return (Move.STAY, 1)

        delta = (path[0][0] - my_pos[0], path[0][1] - my_pos[1])
        move = DELTAS_TO_MOVE.get(delta, Move.STAY)

        steps = 1
        if self.pacman_speed > 1:
            for i in range(1, min(len(path), self.pacman_speed)):
                prev, curr = path[i - 1], path[i]
                if (curr[0] - prev[0], curr[1] - prev[1]) == delta:
                    steps += 1
                else:
                    break

        return (move, steps)


class GhostAgent(BaseGhostAgent):
    """Agent Ghost dùng Minimax và heuristic di động."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_known_pacman = None
        self.map_h, self.map_w = None, None
        self.walls = None
        self.moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]

    def _init_map(self, map_state):
        """Khởi tạo kích thước bản đồ và ma trận tường."""
        if self.map_h is None:
            self.map_h, self.map_w = map_state.shape
            self.walls = np.zeros((self.map_h, self.map_w), dtype=bool)

    def _get_valid_neighbors(self, pos):
        """Lấy danh sách ô kề hợp lệ."""
        r, c = pos
        valid_moves = []
        for m in self.moves:
            nr, nc = r + m.value[0], c + m.value[1]
            if 0 <= nr < self.map_h and 0 <= nc < self.map_w and not self.walls[nr, nc]:
                valid_moves.append(((nr, nc), m))
        return valid_moves

    def _evaluate(self, my_pos, pacman_pos):
        """Đánh giá trạng thái trò chơi."""
        dist = abs(my_pos[0]-pacman_pos[0]) + abs(my_pos[1]-pacman_pos[1])
        if dist < 2: 
            return -999999
        
        score = dist * 10 

        if my_pos[0] == pacman_pos[0] or my_pos[1] == pacman_pos[1]:
            score -= 50

        exits = len(self._get_valid_neighbors(my_pos))
        if exits <= 1: 
            score -= 2000
        elif exits == 2:
            score -= 100
        else:
            score += 20

        return score

    def _minimax(self, my_pos, pacman_pos, depth, is_max, alpha, beta):
        """Minimax với alpha-beta pruning."""
        if depth == 0: 
            return self._evaluate(my_pos, pacman_pos)
        
        dist = abs(my_pos[0]-pacman_pos[0]) + abs(my_pos[1]-pacman_pos[1])
        if dist < 2: 
            return -999999

        neighbors = self._get_valid_neighbors(my_pos if is_max else pacman_pos)
        
        if not neighbors: 
            return -999999 if is_max else 999999

        if is_max:
            val = -float('inf')
            for nxt, _ in neighbors:
                val = max(val, self._minimax(nxt, pacman_pos, depth-1, False, alpha, beta))
                alpha = max(alpha, val)
                if beta <= alpha: 
                    break
            return val
        else:
            val = float('inf')
            for nxt, _ in neighbors:
                val = min(val, self._minimax(my_pos, nxt, depth-1, True, alpha, beta))
                beta = min(beta, val)
                if beta <= alpha: 
                    break
            return val

    def step(self, map_state, my_pos, enemy_pos, step_number):
        """Quyết định chính."""
        self._init_map(map_state)
        
        visible_mask = (map_state != -1)
        new_walls = (map_state == 1)
        self.walls[visible_mask] = new_walls[visible_mask]

        if enemy_pos: 
            self.last_known_pacman = enemy_pos
        
        if not self.last_known_pacman:
            nb = self._get_valid_neighbors(my_pos)
            return random.choice(nb)[1] if nb else Move.STAY

        dist = abs(my_pos[0] - self.last_known_pacman[0]) + abs(my_pos[1] - self.last_known_pacman[1])

        if dist <= 8:
            best_val, best_move = -float('inf'), Move.STAY
            nb = self._get_valid_neighbors(my_pos)
            random.shuffle(nb)
            for nxt, move in nb:
                val = self._minimax(nxt, self.last_known_pacman, 4, False, -float('inf'), float('inf'))
                if val > best_val: 
                    best_val, best_move = val, move
            return best_move
            
        best_move = Move.STAY
        best_score = -float('inf')
        
        nb = self._get_valid_neighbors(my_pos)
        random.shuffle(nb)
        
        for nxt, move in nb:
            d = abs(nxt[0] - self.last_known_pacman[0]) + abs(nxt[1] - self.last_known_pacman[1])
            
            future_exits = sum(
                1 for m in self.moves
                if 0 <= nxt[0] + m.value[0] < self.map_h and 0 <= nxt[1] + m.value[1] < self.map_w and not self.walls[nxt[0] + m.value[0], nxt[1] + m.value[1]]
            )
            score = d + (future_exits * 2.5)
            
            if nxt[0] == self.last_known_pacman[0] or nxt[1] == self.last_known_pacman[1]:
                score -= 5

            if score > best_score:
                best_score = score
                best_move = move
                
        return best_move