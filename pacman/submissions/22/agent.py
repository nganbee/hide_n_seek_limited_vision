import sys
from pathlib import Path
import numpy as np
import random
import heapq
from collections import deque
import time

# Thêm src vào path để import interface
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move

class PacmanAgent(BasePacmanAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Advanced A* Pacman"
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        # Bộ nhớ bản đồ toàn cục: -1=ẩn, 0=trống, 1=tường
        self.memory_map = np.full((21, 21), -1)
        self.last_known_enemy_pos = None

    def _update_memory(self, obs):
        for r in range(21):
            for c in range(21):
                if obs[r, c] != -1:
                    self.memory_map[r, c] = obs[r, c]

    def _is_valid(self, pos):
        r, c = pos
        return 0 <= r < 21 and 0 <= c < 21 and self.memory_map[r, c] == 0

    def _astar(self, start, goal, start_time):
        """Thêm timeout vào A*"""
        def heuristic(p):
            return abs(p[0] - goal[0]) + abs(p[1] - goal[1])
        
        frontier = [(0, start, [])]
        visited = {start}
        
        while frontier:
            # Kiểm tra timeout
            if time.time() - start_time > 0.9:
                return None
                
            f, current, path = heapq.heappop(frontier)
            if current == goal: return path
            
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move.value
                nxt = (current[0] + dr, current[1] + dc)
                if self._is_valid(nxt) and nxt not in visited:
                    visited.add(nxt)
                    new_path = path + [move]
                    heapq.heappush(frontier, (len(new_path) + heuristic(nxt), nxt, new_path))
        return None

    def _find_nearest_unseen(self, start, start_time):
        queue = deque([start])
        visited = {start}
        while queue:
            if time.time() - start_time > 0.9:
                return None
                
            curr = queue.popleft()
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move.value
                nxt = (curr[0] + dr, curr[1] + dc)
                if 0 <= nxt[0] < 21 and 0 <= nxt[1] < 21:
                    if self.memory_map[nxt] == -1: return curr
                    if self.memory_map[nxt] == 0 and nxt not in visited:
                        visited.add(nxt)
                        queue.append(nxt)
        return None

    def step(self, map_state, my_position, enemy_position, step_number):
        start_time = time.time()  

        self._update_memory(map_state)

        if enemy_position: 
            self.last_known_enemy_pos = enemy_position
        
        if time.time() - start_time > 0.95:
            return self._get_fallback_move(my_position, map_state)
        
        target = enemy_position or self.last_known_enemy_pos     

        if target:
            if time.time() - start_time > 0.85:
                return self._get_fallback_move(my_position, map_state)
                
            if my_position == target: 
                self.last_known_enemy_pos = None
            else:
                path = self._astar(my_position, target, start_time)
                if time.time() - start_time > 0.95:
                    return self._get_fallback_move(my_position, map_state)
                if path and len(path) > 0:
                    steps = self._calculate_actual_steps(my_position, path[0], map_state)
                    return (path[0], steps)

        if time.time() - start_time > 0.85:
            return self._get_fallback_move(my_position, map_state)
            
        unseen_target = self._find_nearest_unseen(my_position, start_time)
        if unseen_target:
            if time.time() - start_time > 0.85:
                return self._get_fallback_move(my_position, map_state)
                
            path = self._astar(my_position, unseen_target, start_time)
            if path:
                if time.time() - start_time > 0.9:
                    return self._get_fallback_move(my_position, map_state)
                    
                steps = self._calculate_actual_steps(my_position, path[0], map_state)
                return (path[0], steps)

        if time.time() - start_time > 0.9:
            return self._get_fallback_move(my_position, map_state)
            
        return (Move.STAY, 1)

    def _calculate_actual_steps(self, pos, move, map_state):
        actual = 0
        curr = pos
        for _ in range(self.pacman_speed):
            dr, dc = move.value
            nxt = (curr[0] + dr, curr[1] + dc)
            if 0 <= nxt[0] < 21 and 0 <= nxt[1] < 21 and map_state[nxt] == 0:
                actual += 1
                curr = nxt
            else: 
                break
        return max(1, actual) if move != Move.STAY else 1
    
    def _get_fallback_move(self, my_pos, map_state):
        valid_moves = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dr, dc = move.value
            nxt = (my_pos[0] + dr, my_pos[1] + dc)
            if 0 <= nxt[0] < 21 and 0 <= nxt[1] < 21 and map_state[nxt] == 0:
                valid_moves.append(move)
        
        if valid_moves:
            move = random.choice(valid_moves)
            steps = self._calculate_actual_steps(my_pos, move, map_state)
            return (move, steps)
        
        return (Move.STAY, 1)



class GhostAgent(BaseGhostAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Minimax-MCTS Ghost"
        self.memory_map = np.full((21, 21), -1)
        self.last_pacman_pos = (15, 10)

    def _update_memory(self, obs):
        for r in range(21):
            for c in range(21):
                if obs[r, c] != -1: 
                    self.memory_map[r, c] = obs[r, c]

    def _is_valid(self, pos):
        r, c = pos
        return 0 <= r < 21 and 0 <= c < 21 and self.memory_map[r, c] == 0

    def _minimax(self, g_pos, p_pos, depth, is_ghost, start_time):
        if depth == 0 or g_pos == p_pos or time.time() - start_time > 0.9:
            return abs(g_pos[0] - p_pos[0]) + abs(g_pos[1] - p_pos[1]), Move.STAY

        moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]
        if is_ghost:
            val, best_m = float('-inf'), Move.STAY
            for m in moves:
                # Kiểm tra timeout
                if time.time() - start_time > 0.9:
                    return val, best_m
                    
                nxt = (g_pos[0] + m.value[0], g_pos[1] + m.value[1])
                if self._is_valid(nxt):
                    res, _ = self._minimax(nxt, p_pos, depth-1, False, start_time)
                    if res > val: 
                        val, best_m = res, m
            return val, best_m
        else:
            val = float('inf')
            for m in moves:
                # Kiểm tra timeout
                if time.time() - start_time > 0.9:
                    return val, Move.STAY
                    
                nxt = (p_pos[0] + m.value[0], p_pos[1] + m.value[1])
                if self._is_valid(nxt):
                    res, _ = self._minimax(g_pos, nxt, depth-1, True, start_time)
                    val = min(val, res)
            return val, Move.STAY

    def _mcts_simulation(self, my_pos, start_time):
        scores = {m: 0 for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]}
        
        for m in scores.keys():
            # Kiểm tra timeout
            if time.time() - start_time > 0.9:
                break
                
            nxt = (my_pos[0] + m.value[0], my_pos[1] + m.value[1])
            if not self._is_valid(nxt): 
                continue
            
            # Giảm số simulation để đảm bảo timeout
            max_simulations = 15 if time.time() - start_time < 0.5 else 5
            
            for _ in range(max_simulations):
                # Kiểm tra timeout trong mỗi simulation
                if time.time() - start_time > 0.9:
                    break
                    
                curr_sim = nxt
                max_depth = 8 if time.time() - start_time < 0.7 else 4
                
                for _ in range(max_depth):
                    if time.time() - start_time > 0.9:
                        break
                        
                    options = [mv for mv in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT] 
                              if self._is_valid((curr_sim[0]+mv.value[0], curr_sim[1]+mv.value[1]))]
                    if not options: 
                        break
                    curr_sim = (curr_sim[0]+random.choice(options).value[0], 
                                curr_sim[1]+random.choice(options).value[1])
                
                scores[m] += (abs(curr_sim[0] - self.last_pacman_pos[0]) + 
                             abs(curr_sim[1] - self.last_pacman_pos[1]))
        
        if not scores or all(v == 0 for v in scores.values()):
            return self._get_fallback_move(my_pos)
            
        return max(scores, key=scores.get)

    def step(self, map_state, my_position, enemy_position, step_number):
        start_time = time.time()  
        self._update_memory(map_state)
        if enemy_position: 
            self.last_pacman_pos = enemy_position        
        
        if time.time() - start_time > 0.95:
            return self._get_fallback_move(my_position)

        if enemy_position:
            if time.time() - start_time > 0.9:
                return self._get_fallback_move(my_position)
                
            _, move = self._minimax(my_position, enemy_position, depth=3, is_ghost=True, start_time=start_time)
            if move != Move.STAY:
                return move
            else:
                return self._get_fallback_move(my_position)

        if time.time() - start_time > 0.9:
            return self._get_fallback_move(my_position)
            
        return self._mcts_simulation(my_position, start_time)
    
    def _get_fallback_move(self, my_pos):
        valid_moves = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dr, dc = move.value
            nxt = (my_pos[0] + dr, my_pos[1] + dc)
            if self._is_valid(nxt):
                valid_moves.append(move)
        
        if valid_moves:
            return random.choice(valid_moves)
        
        return Move.STAY