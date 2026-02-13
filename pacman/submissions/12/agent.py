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
import time

class PacmanAgent(BasePacmanAgent):
    """
    Pacman (Seeker) Agent - Goal: Catch the Ghost
    
    Implement your search algorithm to find and catch the ghost.
    Suggested algorithms: BFS, DFS, A*, Greedy Best-First
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.name = "A* Pacman"
        
        # Initialize state variables
        self.last_known_enemy_pos = None
        self.current_path = []
        self.path_step = 0
    
    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int):
        """
        Decide the next move.
        
        Args:
            map_state: 2D numpy array where 1=wall, 0=empty, -1=unseen (fog)
            my_position: Your current (row, col) in absolute coordinates
            enemy_position: Ghost's (row, col) if visible, None otherwise
            step_number: Current step number (starts at 1)
            
        Returns:
            Move or (Move, steps): Direction to move (optionally with step count)
        """
                
        # Update memory if enemy is visible
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
        
        # Use current sighting, fallback to last known, or explore
        target = enemy_position or self.last_known_enemy_pos
        
        if target is None:
            moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
            random.shuffle(moves)
            # Use Template Helper _choose_action
            action = self._choose_action(my_position, moves, map_state, 1)
            return action if action else (Move.STAY, 1)

        # 3. Strategy: A* Pursuit
        
        # Re-plan if path is empty, finished, or enemy moved from where we thought
        should_replan = (
            not self.current_path or
            (enemy_position is not None and enemy_position != self.current_path[-1]) or
            self.path_step >= len(self.current_path)
        )

        if should_replan:
            # Call the A* algorithm logic from 12.py
            self.current_path = self._astar(my_position, target, map_state)
            self.path_step = 0
        
        # Execute the path
        if self.current_path and self.path_step < len(self.current_path):
            next_move = self.current_path[self.path_step]
            
            # Look ahead to see if we can sprint (using pacman_speed)
            desired_steps = 0
            for i in range(self.pacman_speed):
                idx = self.path_step + i
                if idx >= len(self.current_path) or self.current_path[idx] != next_move:
                    break
                desired_steps += 1
            
            # Use Template Helper to validate moves safely
            # This ensures we don't hit a wall even if A* was slightly off
            real_steps = self._max_valid_steps(my_position, next_move, map_state, desired_steps)
            
            if real_steps > 0:
                self.path_step += real_steps
                
                # Check if we reached the guess position but found nothing
                if enemy_position is None and self.path_step >= len(self.current_path):
                    self.last_known_enemy_pos = None
                    
                return (next_move, real_steps)

        # Fallback: Use Template Helper for a greedy move
        row_diff = target[0] - my_position[0]
        col_diff = target[1] - my_position[1]
        moves = []
        if abs(row_diff) >= abs(col_diff):
            moves.append(Move.DOWN if row_diff > 0 else Move.UP)
            moves.append(Move.RIGHT if col_diff > 0 else Move.LEFT)
        else:
            moves.append(Move.RIGHT if col_diff > 0 else Move.LEFT)
            moves.append(Move.DOWN if row_diff > 0 else Move.UP)
            
        return self._choose_action(my_position, moves, map_state, 1) or (Move.STAY, 1)
    
    # Helper methods (you can add more)
    
    def _choose_action(self, pos: tuple, moves, map_state: np.ndarray, desired_steps: int):
        for move in moves:
            max_steps = min(self.pacman_speed, max(1, desired_steps))
            steps = self._max_valid_steps(pos, move, map_state, max_steps)
            if steps > 0:
                return (move, steps)
        return None

    def _max_valid_steps(self, pos: tuple, move: Move, map_state: np.ndarray, max_steps: int) -> int:
        steps = 0
        current = pos
        for _ in range(max_steps):
            delta_row, delta_col = move.value
            next_pos = (current[0] + delta_row, current[1] + delta_col)
            if not self._is_valid_position(next_pos, map_state):
                break
            steps += 1
            current = next_pos
        return steps
    
    def _is_valid_move(self, pos: tuple, move: Move, map_state: np.ndarray) -> bool:
        """Check if a move from pos is valid for at least one step."""
        return self._max_valid_steps(pos, move, map_state, 1) == 1
    
    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        """Check if a position is valid (not a wall and within bounds)."""
        row, col = pos
        height, width = map_state.shape
        
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        
        return map_state[row, col] != 1
    
    def _astar(self, start, goal, map_state):
        if start == goal: return []
        
        # A* Logic adapted to use _is_valid_position
        frontier = [[0, 0, self._manhattan_distance(start, goal), start, None]]
        g_costs = {start: 0}
        came_from = {}
        closed_set = set()
        
        start_time = time.time()
        
        while frontier:
            if time.time() - start_time > 0.8: break # Timeout protection
            
            frontier.sort(key=lambda x: x[0])
            _, g_cost, _, current_pos, _ = frontier.pop(0)
            
            if current_pos in closed_set: continue
            closed_set.add(current_pos)
            
            if current_pos == goal:
                path = []
                while current_pos != start:
                    if current_pos not in came_from: break
                    parent, move = came_from[current_pos]
                    path.append(move)
                    current_pos = parent
                path.reverse()
                return path
            
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move.value
                next_pos = (current_pos[0] + dr, current_pos[1] + dc)
                
                # Use the class's helper method!
                if not self._is_valid_position(next_pos, map_state): continue
                if next_pos in closed_set: continue
                
                tentative_g = g_cost + 1
                if next_pos in g_costs and g_costs[next_pos] <= tentative_g: continue
                
                g_costs[next_pos] = tentative_g
                came_from[next_pos] = (current_pos, move)
                f = tentative_g + self._manhattan_distance(next_pos, goal)
                frontier.append([f, tentative_g, f-tentative_g, next_pos, current_pos])
        return []

    def _manhattan_distance(self, p1, p2):
        return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])
    


class GhostAgent(BaseGhostAgent):
    """
    Ghost AI Agent - Project 2 
    Strategy:
    - BFS Distance-to-Control when Pacman visible
    - Junction survival when hidden

    ESCAPE:
        score =
            w1 * distance_from_pacman +
            w2 * number_of_exits (1-4) +
            turn -
            penalties

    SURVIVAL
        score = 
            w1 * number_of_exists +
            w2 * vertical_bias -
            penalties
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.internal_map = None
        self.last_pos = None

    def step(self, map_state, my_position, enemy_position, step_number):
        self._update_map(map_state)

        if enemy_position:
            dist_map = self._bfs_distance(enemy_position)
            return self._escape_control(my_position, dist_map)

        return self._junction_survival(my_position)

    # ================= MAP =================

    def _update_map(self, map_state):
        if self.internal_map is None:
            self.internal_map = np.full(map_state.shape, -1)

        self.internal_map[map_state != -1] = map_state[map_state != -1]

    def _is_valid(self, r, c):
        return (0 <= r < self.internal_map.shape[0] and
                0 <= c < self.internal_map.shape[1] and
                self.internal_map[r, c] != 1)

    # ================= BFS =================

    def _bfs_distance(self, source):
        rows, cols = self.internal_map.shape
        dist = np.full((rows, cols), np.inf)

        queue = [source]
        head = 0
        dist[source] = 0

        while head < len(queue):
            r, c = queue[head]
            head += 1

            for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                nr, nc = r + dr, c + dc
                if self._is_valid(nr, nc) and dist[nr, nc] == np.inf:
                    dist[nr, nc] = dist[r, c] + 1
                    queue.append((nr, nc))

        return dist

    # ================= ESCAPE =================

    def _escape_control(self, my_pos, dist_map):
        best_move = Move.STAY
        best_score = -float('inf')

        cur_r, cur_c = my_pos
        
        prev_dir = None
        if self.last_pos:
            prev_dir = (cur_r - self.last_pos[0], cur_c - self.last_pos[1])

        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dr, dc = move.value
            nr, nc = cur_r + dr, cur_c + dc

            #remove dead-end
            if not self._is_valid(nr, nc):
                continue

            exits = self._count_exits(nr, nc)
            d = dist_map[nr, nc]

            if exits <= 1:
                continue

            score = 0

            #bfs distance
            score += (d - 2) * 40

            #exist count
            score += exits * 15

            #turn
            if prev_dir and prev_dir != (dr,dc):
                score += 50

            #avoid return before pos
            if self.last_pos == (nr, nc):
                score -= 70

            if score > best_score:
                best_score = score
                best_move = move

        self.last_pos = my_pos
        return best_move

    # ================= SURVIVAL =================

    def _junction_survival(self, my_pos):
        best_move = Move.STAY
        best_score = -float('inf')
        moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        safe_zone = 5

        for move in moves:
            nr, nc = my_pos[0] + move.value[0], my_pos[1] + move.value[1]
            if not self._is_valid(nr, nc):
                continue

            exits = self._count_exits(nr, nc)

            score = 0           
            score += exits * 30  

            vertical = my_pos[0] - nr
            # 1. Out of safe zone: up priority
            if my_pos[0] > safe_zone:
                if exits >= 3:
                    score += vertical * 10
                elif exits == 2:
                    score += vertical * 5
                    
            # 2. In safe zone: horizol direction or stay
            else:
                if nr > safe_zone:
                    score -= 50
                
                if vertical == 0: 
                    score += 10

            # 3. 
            if exits <= 1:
                score -= 1000

            if self.last_pos == (nr, nc): score -= 50

            if score > best_score:
                best_score = score            
                best_move = move

        self.last_pos = my_pos
        return best_move

    # ================= UTIL =================

    def _count_exits(self, r, c):
        count = 0
        for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
            if self._is_valid(r+dr, c+dc):
                count += 1
        return count
