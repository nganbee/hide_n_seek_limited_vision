"""
IMPROVED Hybrid Agent - Team 23120134
Approach: Q-Learning (trained vs smart opponents) + Advanced heuristics

Key Improvements:
- Q-table priority with better fallback
- Enemy position prediction when not visible
- Smarter pathfinding with obstacle avoidance
- Adaptive strategy based on game state
"""

import sys
from pathlib import Path

src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np
import pickle
from heapq import heappush, heappop
from collections import deque


class PacmanAgent(BasePacmanAgent):
    """Improved Pacman with Q-Learning + Smart heuristics."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.name = "Advanced Q-Pacman"
        
        # Memory - Enhanced with Belief Tracking
        self.known_map = None
        self.known_walls = None  # Explicitly track known walls
        self.known_free = None   # Explicitly track known free cells
        self.ghost_belief = None  # Probability distribution over ghost position
        self.last_known_enemy_pos = None
        self.enemy_last_move = None
        self.steps_without_seeing = 0
        self.visited_positions = set()
        self.position_history = deque(maxlen=10)  # Track last 10 positions
        self.move_history = deque(maxlen=20)  # Track last 20 moves for pattern detection
        self.stuck_counter = 0
        self.last_position = None
        self.last_move = None  # Track for turn penalty
        self.loop_detected = False
        self.perturbation_countdown = 0
        
        # Performance optimization
        self.path_cache = {}  # Cache A* results
        self.cache_ttl = 3  # Cache valid for 3 steps
        self.last_cache_step = -999
        
        # Dynamic exploration (NO hardcoded map knowledge)
        self.current_search_index = 0
        
        # Load Q-table
        model_dir = Path(__file__).parent
        try:
            with open(model_dir / 'pacman_qtable.pkl', 'rb') as f:
                data = pickle.load(f)
                self.q_table = data['q_table']
                print(f"[OK] Loaded Pacman Q-table: {len(self.q_table)} states")
        except:
            self.q_table = {}
            print("[WARNING] No Q-table found, using fallback only")
    
    def step(self, map_state, my_position, enemy_position, step_number):
        """Main decision with hybrid belief tracking."""
        # Init belief state on first step
        if self.known_map is None:
            self.known_map = np.full_like(map_state, -1)
            self.known_walls = (map_state == 1)
            self.known_free = (map_state == 0)
            self._init_belief(map_state)
        else:
            # Update known maps incrementally
            self.known_walls = np.logical_or(self.known_walls, (map_state == 1))
            self.known_free = np.logical_or(self.known_free, (map_state == 0))
        
        # Update memory
        self._update_memory(map_state, my_position, enemy_position)
        
        # LOOP DETECTION: Check for oscillation pattern
        if len(self.move_history) >= 6:  # Check earlier
            if self._detect_loop_pattern():
                self.loop_detected = True
                self.perturbation_countdown = 20  # Increased from 5 for stronger break
        
        if self.perturbation_countdown > 0:
            self.perturbation_countdown -= 1
        
        # EMERGENCY: If approaching timeout, use systematic sweep
        if step_number > 140 or (step_number > 100 and self.steps_without_seeing > 40):
            move, steps = self._systematic_sweep(my_position, map_state, step_number)
            if move != Move.STAY:
                self.last_move = move
                return (move, steps)
        
        # FAST PATH: Khi thấy Ghost - chase trực tiếp như example_student
        if enemy_position is not None:
            self.steps_without_seeing = 0
            self.last_known_enemy_pos = enemy_position
            
            # DIRECT CHASE - đơn giản và nhanh!
            move, steps = self._fast_chase(my_position, enemy_position, map_state, step_number)
        else:
            # INVISIBLE: Dùng belief tracking nhẹ
            self.steps_without_seeing += 1
            
            # ANTI-LOOP: Force random exploration if loop detected
            if self.perturbation_countdown > 0:
                move, steps = self._random_perturbation(my_position, None, map_state)
                if move != Move.STAY:
                    self.move_history.append(move)
                    self.last_move = move
                    return (move, steps)
            
            # Chỉ predict/correct khi cần thiết
            if step_number > 1 and self.ghost_belief is not None:
                self._predict_belief()
            self._correct_belief(map_state, my_position, enemy_position)
            
            target = self._choose_target_from_belief(my_position)
            
            if target:
                move, steps = self._fast_chase(my_position, target, map_state, step_number)
            else:
                move, steps = self._strategic_search(my_position, map_state, step_number)
        
        # Record last move for turn penalty
        if move != Move.STAY:
            self.move_history.append(move)
        self.last_move = move if move != Move.STAY else self.last_move
        return (move, steps)
    
    def _fast_chase(self, my_pos, target_pos, map_state, step):
        """Fast chase như example_student: A* + chain moves."""
        # A* đến target
        path = self._astar_cached(my_pos, target_pos, map_state, step)
        if not path:
            return Move.STAY, 1
        
        first_move = path[0]
        
        # Chain moves khi cùng hướng (như example_student)
        steps = 1
        for i in range(1, min(self.pacman_speed, len(path))):
            if path[i] == first_move:
                # Kiểm tra có thể đi tiếp không
                if self._max_steps(my_pos, first_move, map_state, i + 1) >= i + 1:
                    steps = i + 1
                else:
                    break
            else:
                break  # Đổi hướng - dừng chain
        
        next_pos = self._get_new_pos(my_pos, first_move, steps)
        self.position_history.append(next_pos)
        return (first_move, max(1, steps))
    
    def _predict_intercept_point(self, my_pos, enemy_pos):
        """Predict intercept point to cut off ghost path."""
        if not self.enemy_last_move:
            return None
        
        dr, dc = self.enemy_last_move.value
        
        # Predict ghost will continue moving in same direction
        # Find point where we can intercept
        h, w = self.known_map.shape
        for steps_ahead in range(3, 8):  # Look 3-7 steps ahead
            pred_r = enemy_pos[0] + dr * steps_ahead
            pred_c = enemy_pos[1] + dc * steps_ahead
            
            if not (0 <= pred_r < h and 0 <= pred_c < w):
                continue
            if self.known_walls[pred_r, pred_c]:
                break  # Ghost will hit wall
            
            # Check if we can reach this point in time
            my_dist = abs(my_pos[0] - pred_r) + abs(my_pos[1] - pred_c)
            if my_dist <= steps_ahead * self.pacman_speed:
                return (pred_r, pred_c)
        
        return None
    
    def _intelligent_chase(self, my_pos, enemy_pos, map_state, step):
        """Chase with intercept prediction and zone cutting."""
        # Calculate distance
        dist = abs(my_pos[0] - enemy_pos[0]) + abs(my_pos[1] - enemy_pos[1])
        
        # INTERCEPT MODE: Try to cut off ghost path when medium distance
        if 7 <= dist <= 15 and self.enemy_last_move:
            intercept_target = self._predict_intercept_point(my_pos, enemy_pos)
            if intercept_target:
                path = self._astar_cached(my_pos, intercept_target, map_state, step)
                if path:
                    move = path[0]
                    steps = self._max_steps(my_pos, move, map_state, self.pacman_speed)
                    if steps > 0:
                        self.position_history.append(self._get_new_pos(my_pos, move, steps))
                        return (move, max(1, steps))
        
        # AGGRESSIVE MODE: If close, use pure A* (no Q-learning hesitation)
        if dist <= 10:  # Increased from 6 to 10 - more aggressive
            path = self._astar_cached(my_pos, enemy_pos, map_state, step)
            if path:
                move = path[0]
                steps = self._max_steps(my_pos, move, map_state, self.pacman_speed)
                if steps > 0:
                    self.position_history.append(self._get_new_pos(my_pos, move, steps))
                    return (move, max(1, steps))
        
        # Check if stuck (very sensitive now)
        if my_pos in self.position_history and self.position_history.count(my_pos) > 0:
            self.stuck_counter += 2  # Faster escalation
        else:
            self.stuck_counter = max(0, self.stuck_counter - 1)  # Decay slowly
        
        # STRATEGY 1: Q-Learning (if trained and confident)
        state = self._encode_state(my_pos, enemy_pos, map_state, step)
        
        if state in self.q_table:
            q_values = self.q_table[state]
            
            # Get best valid moves with anti-stuck penalty
            valid_moves = []
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                if move in q_values:
                    steps_possible = self._max_steps(my_pos, move, map_state, self.pacman_speed)
                    if steps_possible > 0:
                        next_pos = self._get_new_pos(my_pos, move, steps_possible)
                        
                        # Anti-stuck: penalize recent positions
                        penalty = 0
                        if next_pos in self.position_history:
                            penalty = 50 * self.position_history.count(next_pos)
                        
                        valid_moves.append((q_values[move] - penalty, move, steps_possible, next_pos))
            
            if valid_moves:
                valid_moves.sort(key=lambda x: x[0], reverse=True)
                best_q, best_move, steps, next_pos = valid_moves[0]
                
                # Confidence check: if stuck or Q-values too similar → use A*
                # Very aggressive switching: lower threshold
                if self.stuck_counter > 2 or (len(valid_moves) > 1 and abs(best_q - valid_moves[1][0]) < 3):
                    path = self._astar_cached(my_pos, enemy_pos, map_state, step)
                    if path:
                        move = path[0]
                        steps = self._max_steps(my_pos, move, map_state, self.pacman_speed)
                        if steps > 0:
                            self.position_history.append(self._get_new_pos(my_pos, move, steps))
                            self.stuck_counter = 0  # Reset on successful A*
                            return (move, max(1, steps))
                
                self.position_history.append(next_pos)
                return (best_move, max(1, steps))
        
        # STRATEGY 2: A* with lookahead (use cache)
        path = self._astar_cached(my_pos, enemy_pos, map_state, step)
        if path:
            move = path[0]
            steps = self._max_steps(my_pos, move, map_state, self.pacman_speed)
            if steps > 0:
                self.position_history.append(self._get_new_pos(my_pos, move, steps))
                return (move, max(1, steps))
        
        # STRATEGY 3: Greedy with obstacle avoidance
        return self._greedy_move(my_pos, enemy_pos, map_state)
    
    def _strategic_search(self, my_pos, map_state, step):
        """Smart search with dynamic exploration - ANTI-TIMEOUT."""
        targets = []
        
        # CRITICAL: If lost too long, use systematic exploration
        if self.steps_without_seeing > 35:
            zone_target = self._get_next_search_zone(my_pos, map_state)
            if zone_target:
                path = self._astar_cached(my_pos, zone_target, map_state, step)
                if path:
                    move = path[0]
                    steps = self._max_steps(my_pos, move, map_state, self.pacman_speed)
                    if steps > 0:
                        return (move, max(1, steps))
        
        # 1. DYNAMIC SEARCH: Find open areas from current observation
        if self.steps_without_seeing > 2:
            zone_weight = min(120, 40 + self.steps_without_seeing * 8)
            
            # Search visible and nearby cells dynamically
            h, w = map_state.shape
            for r in range(max(0, my_pos[0] - 10), min(h, my_pos[0] + 11), 3):
                for c in range(max(0, my_pos[1] - 10), min(w, my_pos[1] + 11), 3):
                    if map_state[r, c] == 0 or map_state[r, c] == -1:
                        dist = abs(my_pos[0] - r) + abs(my_pos[1] - c)
                        priority = zone_weight - dist
                        targets.append((priority, (r, c)))
        
        # 2. TRAJECTORY-BASED: Predict from last known position
        if self.last_known_enemy_pos and self.steps_without_seeing <= 20:  # Extended more
            search_radius = min(15, 4 + self.steps_without_seeing // 2)  # Start bigger, expand faster
            r0, c0 = self.last_known_enemy_pos
            
            # Direction bias
            dr_bias, dc_bias = 0, 0
            if self.enemy_last_move:
                dr_bias, dc_bias = self.enemy_last_move.value
            
            for dr in range(-search_radius, search_radius + 1, 2):  # Skip every other to reduce checks
                for dc in range(-search_radius, search_radius + 1, 2):
                    r, c = r0 + dr, c0 + dc
                    
                    if 0 <= r < 21 and 0 <= c < 21 and map_state[r, c] == 0:
                        # Weighted by direction and distance
                        dist_from_last = abs(dr) + abs(dc)
                        direction_bonus = 0
                        if dr_bias != 0 or dc_bias != 0:
                            # Bonus if moving in same direction
                            if (dr * dr_bias > 0) or (dc * dc_bias > 0):
                                direction_bonus = 15
                        
                        priority = 50 - dist_from_last + direction_bonus
                        targets.append((priority, (r, c)))
        
        # 3. UNEXPLORED: Very aggressive when lost
        if len(targets) < 40:  # Increased from 30
            # Dense search immediately when can't find ghost - DYNAMIC
            h, w = map_state.shape
            step_size = 1 if self.steps_without_seeing > 5 else 2
            for r in range(1, h - 1, step_size):
                for c in range(1, w - 1, step_size):
                    if map_state[r, c] == 0 and (r, c) not in self.visited_positions:
                        dist = abs(my_pos[0] - r) + abs(my_pos[1] - c)
                        max_search_dist = int(min(h, w) * 0.85)
                        if dist <= max_search_dist:
                            # Diversity bonus: avoid recent positions
                            recent_positions = set(list(self.position_history)[-8:])
                            diversity_bonus = 0
                            if (r, c) not in recent_positions and self.perturbation_countdown > 0:
                                diversity_bonus = 20
                            
                            # Strong bonus for corners when lost - DYNAMIC
                            corner_bonus = 0
                            if self.steps_without_seeing > 8:
                                corner_threshold = min(h, w) // 3
                                if (r < corner_threshold or r > h - corner_threshold - 1) and \
                                   (c < corner_threshold or c > w - corner_threshold - 1):
                                    corner_bonus = 30
                            # Bonus for edges - DYNAMIC
                            edge_bonus = 0
                            if self.steps_without_seeing > 5:
                                edge_threshold = max(2, min(h, w) // 7)
                                if r < edge_threshold or r > h - edge_threshold - 1 or \
                                   c < edge_threshold or c > w - edge_threshold - 1:
                                    edge_bonus = 15
                            priority = 30 - dist + corner_bonus + edge_bonus + diversity_bonus
                            targets.append((priority, (r, c)))
        
        # 4. FAST TARGET SELECTION: Sort and try top candidates
        if targets:
            targets.sort(key=lambda x: x[0], reverse=True)
            
            # More aggressive: try more targets when lost
            num_targets = min(15 if self.steps_without_seeing > 8 else 8, len(targets))
            for _, target in targets[:num_targets]:
                path = self._astar_cached(my_pos, target, map_state, step)
                if path:
                    move = path[0]
                    steps = self._max_steps(my_pos, move, map_state, self.pacman_speed)
                    if steps > 0:
                        return (move, max(1, steps))
        
        # Fallback: intelligent exploration
        return self._explore_intelligent(my_pos, map_state)
    
    def _systematic_sweep(self, my_pos, map_state, step):
        """Emergency systematic sweep when approaching timeout."""
        # Generate dynamic search grid
        h, w = map_state.shape
        grid_step = 4
        search_zones = [(r, c) for r in range(2, h-2, grid_step) for c in range(2, w-2, grid_step)]
        
        # Go through zones systematically
        for zone in search_zones:
            if 0 <= zone[0] < h and 0 <= zone[1] < w:
                if map_state[zone] == 0 or map_state[zone] == -1:
                    # Haven't visited this zone recently
                    if zone not in list(self.position_history)[-20:]:
                        path = self._astar_cached(my_pos, zone, map_state, step)
                        if path:
                            move = path[0]
                            steps = self._max_steps(my_pos, move, map_state, self.pacman_speed)
                            if steps > 0:
                                return (move, max(1, steps))
        
        # Last resort: explore towards map edges/corners dynamically
        h, w = map_state.shape
        # Generate corner targets dynamically based on map size
        corner_offsets = [(2, 2), (2, -3), (-3, 2), (-3, -3)]  # Offsets from edges
        corners = []
        for dr, dc in corner_offsets:
            r = dr if dr >= 0 else h + dr
            c = dc if dc >= 0 else w + dc
            if 0 <= r < h and 0 <= c < w and (map_state[r, c] == 0 or map_state[r, c] == -1):
                corners.append((r, c))
        
        for corner in corners:
            path = self._astar_cached(my_pos, corner, map_state, step)
            if path:
                move = path[0]
                steps = self._max_steps(my_pos, move, map_state, self.pacman_speed)
                if steps > 0:
                    return (move, max(1, steps))
        
        return Move.STAY, 0
    
    def _get_next_search_zone(self, my_pos, map_state):
        """Get next systematic search zone."""
        # Try zones in order, skip visited ones
        for i in range(len(self.search_zones)):
            zone_idx = (self.current_search_zone + i) % len(self.search_zones)
            zone = self.search_zones[zone_idx]
            
            if 0 <= zone[0] < 21 and 0 <= zone[1] < 21:
                if (map_state[zone] == 0 or map_state[zone] == -1):
                    # Not recently visited
                    if zone not in list(self.position_history)[-30:]:
                        self.current_search_zone = (zone_idx + 1) % len(self.search_zones)
                        return zone
        
        return None
    
    def _detect_loop_pattern(self):
        """Detect if agent is stuck in a loop (oscillation)."""
        if len(self.move_history) < 6:  # Reduced from 8 for faster detection
            return False
        
        recent_moves = list(self.move_history)[-8:] if len(self.move_history) >= 8 else list(self.move_history)
        
        # Pattern 1: Simple 2-move oscillation (A-B-A-B) - check shorter sequence
        if len(recent_moves) >= 6:
            if (recent_moves[-6] == recent_moves[-4] == recent_moves[-2] and
                recent_moves[-5] == recent_moves[-3] == recent_moves[-1]):
                return True
        
        # Pattern 2: Position loop - visiting same positions repeatedly
        if len(self.position_history) >= 6:
            recent_pos = list(self.position_history)[-6:]
            unique_pos = set(recent_pos)
            if len(unique_pos) <= 2:  # Only 2 or fewer unique positions in last 6 moves
                return True
        
        # Pattern 3: 3-move loop (A-B-C-A-B-C)
        if len(self.move_history) >= 6:
            last_6 = list(self.move_history)[-6:]
            if (last_6[0] == last_6[3] and
                last_6[1] == last_6[4] and
                last_6[2] == last_6[5]):
                return True
        
        # Pattern 4: Same direction too many times
        if len(recent_moves) >= 5 and len(set(recent_moves[-5:])) == 1:
            return True
        
        return False
    
    def _random_perturbation(self, my_pos, enemy_pos, map_state):
        """Add random move to break loop pattern."""
        import random
        
        perpendicular_moves = []
        
        if enemy_pos is not None:
            # Try moves perpendicular to direct path
            dr = enemy_pos[0] - my_pos[0]
            dc = enemy_pos[1] - my_pos[1]
            
            if abs(dr) > abs(dc):  # Moving vertically, try horizontal
                perpendicular_moves = [Move.LEFT, Move.RIGHT]
            else:  # Moving horizontally, try vertical
                perpendicular_moves = [Move.UP, Move.DOWN]
        else:
            # No target: pick moves NOT in recent history
            if len(self.move_history) >= 3:
                recent = set(list(self.move_history)[-3:])
                all_moves = {Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT}
                perpendicular_moves = list(all_moves - recent)
            
            if not perpendicular_moves:
                perpendicular_moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        
        random.shuffle(perpendicular_moves)
        
        # Prefer moves to unvisited positions
        recent_positions = set(list(self.position_history)[-5:])
        for move in perpendicular_moves:
            steps = self._max_steps(my_pos, move, map_state, self.pacman_speed)
            if steps > 0:
                new_pos = self._get_new_pos(my_pos, move, steps)
                if new_pos not in recent_positions:
                    return (move, max(1, steps))
        
        # Fallback: any valid move
        for move in perpendicular_moves:
            steps = self._max_steps(my_pos, move, map_state, self.pacman_speed)
            if steps > 0:
                return (move, max(1, steps))
        
        # Last resort: try all moves
        all_moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        random.shuffle(all_moves)
        for move in all_moves:
            steps = self._max_steps(my_pos, move, map_state, self.pacman_speed)
            if steps > 0:
                return (move, max(1, steps))
        
        return Move.STAY, 0
    
    def _predict_enemy_position(self, step):
        """Predict with weighted probability - IMPROVED."""
        if not self.last_known_enemy_pos:
            return None
        
        # Weighted candidates: (position, probability_weight)
        candidates = [(self.last_known_enemy_pos, 20)]  # Base weight
        r0, c0 = self.last_known_enemy_pos
        
        if self.enemy_last_move:
            # Model 1: Linear continuation (HIGH PROBABILITY)
            dr, dc = self.enemy_last_move.value
            
            # Weighted by time unseen: recent = close, old = far
            for i, multiplier in enumerate([1, 2, 3, min(6, self.steps_without_seeing)]):
                pred = (r0 + dr * multiplier, c0 + dc * multiplier)
                if 0 <= pred[0] < 21 and 0 <= pred[1] < 21:
                    if self.known_map[pred] == 0 or self.known_map[pred] == -1:
                        # Weight decreases with distance
                        weight = 60 - i * 10 if i < self.steps_without_seeing else 30
                        candidates.append((pred, weight))
            
            # Model 2: Perpendicular (MEDIUM PROBABILITY)
            for perp_move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                if perp_move != self.enemy_last_move:
                    pr, pc = perp_move.value
                    for dist in [1, 2]:
                        pred = (r0 + pr * dist, c0 + pc * dist)
                        h, w = self.known_map.shape
                        if 0 <= pred[0] < h and 0 <= pred[1] < w:
                            if self.known_map[pred] == 0 or self.known_map[pred] == -1:
                                candidates.append((pred, 25))
        
        # Model 3: Dynamic exploration areas (increases with time)
        if self.steps_without_seeing > 5:
            zone_weight = min(80, 20 + self.steps_without_seeing * 5)
            h, w = self.known_map.shape
            # Search in grid pattern dynamically
            for r in range(2, h-2, 4):
                for c in range(2, w-2, 4):
                    if self.known_map[r, c] == 0 or self.known_map[r, c] == -1:
                        candidates.append(((r, c), zone_weight))
        
        # Select highest probability candidate
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]
        
        return None
    
    def _get_new_pos(self, pos, move, steps=1):
        """Calculate new position after move."""
        r, c = pos
        dr, dc = move.value
        return (r + dr * steps, c + dc * steps)
    
    def _astar_cached(self, start, goal, map_state, current_step):
        """A* with caching to avoid redundant searches."""
        # Check cache
        cache_key = (start, goal)
        if cache_key in self.path_cache:
            cached_path, cached_step = self.path_cache[cache_key]
            # Cache valid if within TTL
            if current_step - cached_step <= self.cache_ttl:
                return cached_path
        
        # Compute new path with turn penalty
        path = self._astar_with_turn_penalty(start, goal, map_state)
        
        # Store in cache (limit cache size)
        if len(self.path_cache) > 50:
            # Clear old entries
            self.path_cache = {k: v for k, v in list(self.path_cache.items())[-25:]}
        
        self.path_cache[cache_key] = (path, current_step)
        return path
    
    def _astar_with_turn_penalty(self, start, goal, map_state):
        """A* with turn penalty to prefer straight lines (optimize for speed > 1)."""
        if start == goal:
            return []
        
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
        # Convert Move to int for hashable state
        def move_to_int(m):
            if m is None: return -1
            if m == Move.UP: return 0
            if m == Move.DOWN: return 1
            if m == Move.LEFT: return 2
            if m == Move.RIGHT: return 3
            return -1
        
        # State: (row, col, last_move_int)
        start_state = (start[0], start[1], move_to_int(self.last_move))
        frontier = [(0, 0, 0, start_state, [])]
        visited = {}
        counter = 1
        
        while frontier:
            f, _, g, (r, c, prev_move_int), path = heappop(frontier)
            
            state_key = (r, c, prev_move_int)
            if state_key in visited:
                continue
            visited[state_key] = g
            
            if (r, c) == goal:
                return path
            
            # Increase max_depth for long-distance chases
            max_depth = min(50, heuristic(start, goal) + 25)  # Increased from 30 to 50
            if len(path) >= max_depth:
                continue
            
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move.value
                nr, nc = r + dr, c + dc
                
                if not (0 <= nr < map_state.shape[0] and 0 <= nc < map_state.shape[1]):
                    continue
                
                # Don't pathfind through walls, but allow unknown cells
                if self.known_walls[nr, nc]:
                    continue
                
                # Turn penalty: straight line = 0.5, turn = 1.0
                move_int = move_to_int(move)
                cost = 0.5 if (prev_move_int == move_int and prev_move_int != -1) else 1.0
                new_g = g + cost
                new_state = (nr, nc, move_int)
                
                if new_state not in visited or new_g < visited.get(new_state, float('inf')):
                    h = heuristic((nr, nc), goal)
                    counter += 1
                    heappush(frontier, (new_g + h, counter, new_g, new_state, path + [move]))
        
        # Fallback to regular A* if turn penalty failed
        return self._astar_advanced(start, goal, map_state)
    
    def _astar_advanced(self, start, goal, map_state):
        """A* with better heuristic - OPTIMIZED."""
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
        # Early exit if same position
        if start == goal:
            return []
        
        frontier = [(0, start, [])]
        visited = set()
        max_depth = min(25, heuristic(start, goal) + 10)  # Dynamic max depth
        
        while frontier:
            _, current, path = heappop(frontier)
            
            if current in visited:
                continue
            visited.add(current)
            
            if current == goal:
                return path
            
            if len(path) >= max_depth:  # Dynamic depth limit
                continue
            
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move.value
                next_pos = (current[0] + dr, current[1] + dc)
                
                if self._is_valid(next_pos, map_state) and next_pos not in visited:
                    new_path = path + [move]
                    # Lower cost for unexplored areas
                    cost_bonus = -2 if next_pos not in self.visited_positions else 0
                    priority = len(new_path) + heuristic(next_pos, goal) + cost_bonus
                    heappush(frontier, (priority, next_pos, new_path))
        
        return None
    
    def _greedy_move(self, my_pos, enemy_pos, map_state):
        """Greedy move with obstacle avoidance."""
        row_diff = enemy_pos[0] - my_pos[0]
        col_diff = enemy_pos[1] - my_pos[1]
        
        # Prioritize larger difference
        if abs(row_diff) > abs(col_diff):
            primary = Move.DOWN if row_diff > 0 else Move.UP
            secondary = Move.RIGHT if col_diff > 0 else Move.LEFT
        else:
            primary = Move.RIGHT if col_diff > 0 else Move.LEFT
            secondary = Move.DOWN if row_diff > 0 else Move.UP
        
        # Try primary
        steps = self._max_steps(my_pos, primary, map_state, self.pacman_speed)
        if steps > 0:
            return (primary, steps)
        
        # Try secondary
        steps = self._max_steps(my_pos, secondary, map_state, self.pacman_speed)
        if steps > 0:
            return (secondary, steps)
        
        # Try any valid move
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            steps = self._max_steps(my_pos, move, map_state, self.pacman_speed)
            if steps > 0:
                return (move, steps)
        
        return (Move.STAY, 1)
    
    def _explore_intelligent(self, my_pos, map_state):
        """Explore with frontier detection for unknown cells."""
        moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        
        # Prioritize unknown cells (frontier exploration)
        unknown_moves = []
        known_moves = []
        
        for move in moves:
            steps = self._max_steps(my_pos, move, map_state, self.pacman_speed)
            if steps > 0:
                dr, dc = move.value
                next_pos = (my_pos[0] + dr * steps, my_pos[1] + dc * steps)
                
                # Check if this reveals unknown cells
                is_unknown = (not self.known_free[next_pos]) and (not self.known_walls[next_pos])
                
                # Score
                score = 0
                if is_unknown:
                    score += 100  # High priority for unknown
                elif next_pos not in self.visited_positions:
                    score += 60
                
                # Center proximity
                dist_to_center = abs(next_pos[0] - 10) + abs(next_pos[1] - 10)
                score += (20 - dist_to_center)
                
                # Unseen reveal (optimized: sparse grid)
                unseen_count = 0
                for check_r in range(max(0, next_pos[0] - 3), min(21, next_pos[0] + 4), 2):
                    for check_c in range(max(0, next_pos[1] - 3), min(21, next_pos[1] + 4), 2):
                        if self.known_map[check_r, check_c] == -1:
                            unseen_count += 1
                score += unseen_count * 3
                
                # Recent visit penalty
                if next_pos in self.position_history:
                    score -= 15 * self.position_history.count(next_pos)
                
                # Map center proximity (dynamic)
                h, w = map_state.shape
                center_dist = abs(next_pos[0] - h//2) + abs(next_pos[1] - w//2)
                score += max(0, 10 - center_dist // 2)
                
                if is_unknown:
                    unknown_moves.append((score, move, steps))
                else:
                    known_moves.append((score, move, steps))
        
        # Prefer unknown, then known
        candidates = unknown_moves if unknown_moves else known_moves
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            _, move, steps = candidates[0]
            return (move, steps)
        
        return (Move.STAY, 1)
    
    def _encode_state(self, my_pos, enemy_pos, map_state, step):
        """Encode state for Q-table lookup - must match train_curriculum.py."""
        if enemy_pos is None:
            return None
        
        row_diff = enemy_pos[0] - my_pos[0]
        col_diff = enemy_pos[1] - my_pos[1]
        dist = abs(row_diff) + abs(col_diff)
        
        # Distance bucket
        dist_bucket = 0 if dist <= 2 else (1 if dist <= 5 else (2 if dist <= 10 else 3))
        
        # Direction (simplified)
        direction = 0  # Up
        if abs(col_diff) > abs(row_diff):
            direction = 2 if col_diff > 0 else 3  # Right or Left
        elif row_diff > 0:
            direction = 1  # Down
        
        # Wall information
        r, c = my_pos
        wall_up = 1 if (r == 0 or map_state[r-1, c] == 1) else 0
        wall_down = 1 if (r == 20 or map_state[r+1, c] == 1) else 0
        wall_left = 1 if (c == 0 or map_state[r, c-1] == 1) else 0
        wall_right = 1 if (c == 20 or map_state[r, c+1] == 1) else 0
        
        # Count exits
        exits = (1-wall_up) + (1-wall_down) + (1-wall_left) + (1-wall_right)
        
        # Step bucket
        step_bucket = min(step // 40, 5)
        
        return (dist_bucket, direction, wall_up, wall_down, wall_left, wall_right, 
                exits, step_bucket)
    
    def _update_memory(self, map_state, my_pos, enemy_pos):
        """Update known map and enemy tracking."""
        # Update known map
        visible_mask = (map_state != -1)
        self.known_map[visible_mask] = map_state[visible_mask]
        self.visited_positions.add(my_pos)
        
        # Track enemy if visible
        if enemy_pos is not None:
            if self.last_known_enemy_pos is not None:
                dr = enemy_pos[0] - self.last_known_enemy_pos[0]
                dc = enemy_pos[1] - self.last_known_enemy_pos[1]
                if dr != 0 or dc != 0:
                    if abs(dr) <= 1 and abs(dc) <= 1:
                        if dr == -1: self.enemy_last_move = Move.UP
                        elif dr == 1: self.enemy_last_move = Move.DOWN
                        elif dc == -1: self.enemy_last_move = Move.LEFT
                        elif dc == 1: self.enemy_last_move = Move.RIGHT
            self.last_known_enemy_pos = enemy_pos
    
    def _init_belief(self, map_state):
        """Initialize belief uniformly (no hardcoded spawn knowledge)."""
        h, w = map_state.shape
        self.ghost_belief = np.zeros((h, w), dtype=float)
        
        # Uniform prior across all free cells
        for r in range(h):
            for c in range(w):
                if not self.known_walls[r, c]:
                    self.ghost_belief[r, c] = 1.0
        
        total = self.ghost_belief.sum()
        if total > 0:
            self.ghost_belief /= total
        else:
            self.ghost_belief[h // 2, w // 2] = 1.0
    
    def _predict_belief(self):
        """Predict belief: ghost can move 1 step or stay."""
        if self.ghost_belief is None:
            return
        
        h, w = self.ghost_belief.shape
        new_belief = np.zeros_like(self.ghost_belief)
        moves = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]  # Stay + 4 directions
        
        for r in range(h):
            for c in range(w):
                p = self.ghost_belief[r, c]
                if p == 0.0:
                    continue
                
                # Find valid successors
                successors = []
                for dr, dc in moves:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        # Allow if not known wall (includes unknown)
                        if not self.known_walls[nr, nc]:
                            successors.append((nr, nc))
                
                if len(successors) == 0:
                    new_belief[r, c] += p  # Stuck, stays
                else:
                    share = p / len(successors)
                    for nr, nc in successors:
                        new_belief[nr, nc] += share
        
        self.ghost_belief = self._normalize_belief(new_belief)
    
    def _correct_belief(self, map_state, my_pos, enemy_pos):
        """Correct belief with observation (Bayes update)."""
        if enemy_pos is not None:
            # Direct observation: collapse to single position
            self.ghost_belief[:, :] = 0.0
            self.ghost_belief[enemy_pos] = 1.0
            return
        
        # Zero belief on observed cells (we see they're empty)
        observed_free = (map_state == 0)
        observed_wall = (map_state == 1)
        self.ghost_belief[observed_free] = 0.0
        self.ghost_belief[observed_wall] = 0.0
        
        # Ghost can't be where we are
        self.ghost_belief[my_pos] = 0.0
        
        self.ghost_belief = self._normalize_belief(self.ghost_belief)
    
    def _normalize_belief(self, belief):
        """Normalize belief to sum to 1.0."""
        total = float(belief.sum())
        if total <= 0.0:
            # Reset to uniform over non-walls
            h, w = belief.shape
            belief = np.zeros_like(belief)
            mask = ~self.known_walls
            free_count = int(mask.sum())
            if free_count > 0:
                belief[mask] = 1.0 / free_count
            else:
                belief[h // 2, w // 2] = 1.0
            return belief
        return belief / total
    
    def _choose_target_from_belief(self, my_pos):
        """Choose target using argmax + weighted centroid for broad distributions."""
        if self.ghost_belief is None:
            return self._predict_enemy_position(0)
        
        max_p = float(self.ghost_belief.max())
        if max_p == 0.0:
            return self._predict_enemy_position(0)
        
        coords = np.argwhere(self.ghost_belief == max_p)
        if len(coords) == 0:
            return None
        
        # Primary: nearest argmax
        best = min(coords, key=lambda p: abs(int(p[0]) - my_pos[0]) + abs(int(p[1]) - my_pos[1]))
        target = (int(best[0]), int(best[1]))
        
        # If distribution is broad (entropy high), use weighted centroid
        entropy_like = len(np.argwhere(self.ghost_belief > max_p * 0.5))
        if entropy_like > 8:  # Broad distribution
            total = float(self.ghost_belief.sum())
            if total > 0:
                rr, cc = np.indices(self.ghost_belief.shape)
                wr = float((rr * self.ghost_belief).sum()) / total
                wc = float((cc * self.ghost_belief).sum()) / total
                centroid = (int(round(wr)), int(round(wc)))
                # Use centroid if not wall
                if not self.known_walls[centroid]:
                    target = centroid
        
        return target
    
    def _get_new_pos(self, pos, move, steps=1):
        """Calculate new position after move."""
        r, c = pos
        dr, dc = move.value
        return (r + dr * steps, c + dc * steps)
    
    def _max_steps(self, pos, move, map_state, max_speed):
        """Calculate max steps: 1 if turning, max_speed if straight."""
        # Check if this is a TURN (different direction from last move)
        if self.last_move is not None and self.last_move != Move.STAY and self.last_move != move:
            # This is a TURN → only 1 step allowed
            r, c = pos
            dr, dc = move.value
            nr, nc = r + dr, c + dc
            if self._is_valid((nr, nc), map_state):
                return 1
            return 0
        
        # STRAIGHT line or first move → use max_speed
        steps = 0
        r, c = pos
        dr, dc = move.value
        
        for i in range(1, max_speed + 1):
            nr, nc = r + dr * i, c + dc * i
            if not self._is_valid((nr, nc), map_state):
                break
            steps = i
        
        return steps
    
    def _is_valid(self, pos, map_state):
        """Check if position is valid (not a wall)."""
        r, c = pos
        h, w = map_state.shape
        if r < 0 or r >= h or c < 0 or c >= w:
            return False
        # Allow movement to free (0) and unknown (-1) cells, but not walls (1)
        return map_state[r, c] != 1


class GhostAgent(BaseGhostAgent):
    """Improved Ghost with Q-Learning + Smart evasion."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Advanced Q-Ghost"
        
        # Memory
        self.known_map = None
        self.escape_routes = []
        self.danger_level = 0
        self.position_history = deque(maxlen=10)  # Anti-stuck
        self.stuck_counter = 0
        self.zigzag_counter = 0
        self.last_escape_dir = None
        self.panic_mode = False
        self.predicted_pacman_pos = None
        self.first_move_done = False
        self.last_move = None
        
        # Load Q-table
        model_dir = Path(__file__).parent
        try:
            with open(model_dir / 'ghost_qtable.pkl', 'rb') as f:
                data = pickle.load(f)
                self.q_table = data['q_table']
                print(f"[OK] Loaded Ghost Q-table: {len(self.q_table)} states")
        except:
            self.q_table = {}
            print("[WARNING] No Q-table found, using fallback only")
    
    def step(self, map_state, my_position, enemy_position, step_number):
        """Main decision - ĐƠN GIẢN HÓA theo example_student."""
        # Init
        if self.known_map is None:
            self.known_map = np.full_like(map_state, -1)
        
        self.known_map = np.maximum(self.known_map, map_state)
        
        # Track position
        self.position_history.append(my_position)
        
        # SIMPLIFIED DECISION: Dùng distance + mobility như example_student
        if enemy_position:
            # VISIBLE: Maximize distance + mobility (simple & effective!)
            move = self._maximize_distance_simple(my_position, enemy_position, map_state)
            self.last_move = move
            return move
        else:
            # INVISIBLE: Hide strategically
            move = self._hide_strategic(my_position, map_state)
            self.last_move = move
            return move
    
    def _escape_spawn_area(self, my_pos, map_state, step):
        """Escape spawn area smartly in first 3 steps."""
        # Maximize distance from center column (typically spawn column)
        h, w = map_state.shape
        center_col = w // 2
        
        # Priority: move AWAY from center column
        if abs(my_pos[1] - center_col) <= 2:  # Near center
            # Try to move horizontally away from center
            if my_pos[1] >= center_col:
                # Move RIGHT (away from center)
                if self._is_valid_move(my_pos, Move.RIGHT, map_state):
                    return Move.RIGHT
            else:
                # Move LEFT
                if self._is_valid_move(my_pos, Move.LEFT, map_state):
                    return Move.LEFT
        
        # Find moves that lead to more freedom
        best_move = Move.STAY
        best_score = -1
        
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if not self._is_valid_move(my_pos, move, map_state):
                continue
            
            dr, dc = move.value
            next_pos = (my_pos[0] + dr, my_pos[1] + dc)
            
            # Count total exits from next position
            total_exits = sum(1 for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT] 
                            if self._is_valid_move(next_pos, m, map_state))
            
            # Distance from center column
            col_dist = abs(next_pos[1] - center_col)
            
            score = total_exits * 10 + col_dist * 5
            
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move
    
    def _escape_corridor_early(self, my_pos, map_state):
        """Escape narrow corridors in first 3 steps (no enemy info needed)."""
        # Check horizontal freedom
        exits_h = sum(1 for m in [Move.LEFT, Move.RIGHT] if self._is_valid_move(my_pos, m, map_state))
        exits_v = sum(1 for m in [Move.UP, Move.DOWN] if self._is_valid_move(my_pos, m, map_state))
        
        # If trapped in vertical corridor (no horizontal exits)
        if exits_h == 0 and exits_v > 0:
            # Move to position with horizontal exits
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                if not self._is_valid_move(my_pos, move, map_state):
                    continue
                dr, dc = move.value
                next_pos = (my_pos[0] + dr, my_pos[1] + dc)
                h_exits = sum(1 for m in [Move.LEFT, Move.RIGHT] if self._is_valid_move(next_pos, m, map_state))
                if h_exits > 0:
                    return move
        
        # If trapped in horizontal corridor
        if exits_v == 0 and exits_h > 0:
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                if not self._is_valid_move(my_pos, move, map_state):
                    continue
                dr, dc = move.value
                next_pos = (my_pos[0] + dr, my_pos[1] + dc)
                v_exits = sum(1 for m in [Move.UP, Move.DOWN] if self._is_valid_move(next_pos, m, map_state))
                if v_exits > 0:
                    return move
        
        return Move.STAY
    
    def _dynamic_alignment_escape(self, my_pos, pacman_pos, map_state):
        """Escape column/row alignment if possible."""
        same_col = (my_pos[1] == pacman_pos[1])
        same_row = (my_pos[0] == pacman_pos[0])
        
        if same_col:
            # Try horizontal escape
            for move in [Move.LEFT, Move.RIGHT]:
                if self._is_valid_move(my_pos, move, map_state):
                    dr, dc = move.value
                    new_pos = (my_pos[0] + dr, my_pos[1] + dc)
                    exits = self._count_exits(new_pos, map_state)
                    if exits >= 2:
                        return move
            for move in [Move.LEFT, Move.RIGHT]:
                if self._is_valid_move(my_pos, move, map_state):
                    return move
        
        if same_row:
            for move in [Move.UP, Move.DOWN]:
                if self._is_valid_move(my_pos, move, map_state):
                    dr, dc = move.value
                    new_pos = (my_pos[0] + dr, my_pos[1] + dc)
                    exits = self._count_exits(new_pos, map_state)
                    if exits >= 2:
                        return move
            for move in [Move.UP, Move.DOWN]:
                if self._is_valid_move(my_pos, move, map_state):
                    return move
        
        return Move.STAY
    
    def _dynamic_first_move(self, my_pos, pacman_pos, map_state):
        """Smart first move: detect alignment and escape optimally."""
        same_col = (my_pos[1] == pacman_pos[1])
        same_row = (my_pos[0] == pacman_pos[0])
        
        # CRITICAL: Exit column/row immediately if aligned
        if same_col:
            # Try horizontal moves to break column alignment
            for move in [Move.LEFT, Move.RIGHT]:
                if self._is_valid_move(my_pos, move, map_state):
                    dr, dc = move.value
                    new_pos = (my_pos[0] + dr, my_pos[1] + dc)
                    # Prefer direction with more exits
                    exits = self._count_exits(new_pos, map_state)
                    if exits >= 2:
                        return move
            # Fallback: any horizontal move
            for move in [Move.LEFT, Move.RIGHT]:
                if self._is_valid_move(my_pos, move, map_state):
                    return move
        
        if same_row:
            # Try vertical moves to break row alignment
            for move in [Move.UP, Move.DOWN]:
                if self._is_valid_move(my_pos, move, map_state):
                    dr, dc = move.value
                    new_pos = (my_pos[0] + dr, my_pos[1] + dc)
                    exits = self._count_exits(new_pos, map_state)
                    if exits >= 2:
                        return move
            for move in [Move.UP, Move.DOWN]:
                if self._is_valid_move(my_pos, move, map_state):
                    return move
        
        # No alignment: use force-turn strategy
        return self._force_turn_move(my_pos, pacman_pos, map_state)
    
    def _force_turn_move(self, my_pos, pacman_pos, map_state):
        """Move perpendicular to Pacman's chase direction with lookahead."""
        row_diff = my_pos[0] - pacman_pos[0]
        col_diff = my_pos[1] - pacman_pos[1]
        
        # Detect Pacman's likely chase direction
        if abs(row_diff) > abs(col_diff):
            # Pacman will chase vertically → move HORIZONTALLY
            perpendicular = [Move.LEFT, Move.RIGHT]
            parallel = [Move.UP, Move.DOWN]
        else:
            # Pacman will chase horizontally → move VERTICALLY
            perpendicular = [Move.UP, Move.DOWN]
            parallel = [Move.LEFT, Move.RIGHT]
        
        # Score moves with advanced metrics
        best_move = Move.STAY
        best_score = -999
        
        for move in perpendicular + parallel:  # Try perpendicular first, then parallel
            if not self._is_valid_move(my_pos, move, map_state):
                continue
            
            dr, dc = move.value
            new_pos = (my_pos[0] + dr, my_pos[1] + dc)
            
            # Current distance
            new_dist = abs(new_pos[0] - pacman_pos[0]) + abs(new_pos[1] - pacman_pos[1])
            exits = self._count_exits(new_pos, map_state)
            
            # Predict Pacman's next position (with turn penalty consideration)
            predicted_pacman = self._predict_pacman_with_turns(pacman_pos, my_pos, self.last_move)
            predicted_dist = abs(new_pos[0] - predicted_pacman[0]) + abs(new_pos[1] - predicted_pacman[1])
            
            # Wall proximity bonus (force Pacman to navigate around)
            wall_bonus = self._wall_proximity_score(new_pos, map_state)
            
            # Lookahead: check if this path has future escape options
            future_score = self._lookahead_escape(new_pos, predicted_pacman, map_state, depth=2)
            
            # Perpendicular bonus
            perp_bonus = 15 if move in perpendicular else 0
            
            # Combined score: distance + exits + wall + future + perpendicular
            score = (new_dist * 3) + (exits * 8) + wall_bonus + future_score + perp_bonus + (predicted_dist * 2)
            
            # High weight on exits to avoid dead ends
            score = new_dist * 2 + exits * 5
            
            if score > best_score:
                best_score = score
                best_move = move
        
        if best_move != Move.STAY:
            return best_move
        
        # Fallback: maximize distance
        return self._maximize_distance_smart(my_pos, pacman_pos, map_state)
    
    def _wall_proximity_score(self, pos, map_state):
        """Score based on wall proximity - higher near walls."""
        h, w = map_state.shape
        walls_nearby = 0
        
        # Check adjacent cells for walls
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = pos[0] + dr, pos[1] + dc
            if 0 <= nr < h and 0 <= nc < w:
                if map_state[nr, nc] == 1:  # Wall
                    walls_nearby += 1
        
        # Being next to walls is good (creates obstacles for Pacman)
        return walls_nearby * 3
    
    def _lookahead_escape(self, start_pos, pacman_pos, map_state, depth=2):
        """Lookahead to check future escape options."""
        if depth == 0:
            return 0
        
        best_future = 0
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if not self._is_valid_move(start_pos, move, map_state):
                continue
            
            dr, dc = move.value
            next_pos = (start_pos[0] + dr, start_pos[1] + dc)
            
            # Distance improvement
            future_dist = abs(next_pos[0] - pacman_pos[0]) + abs(next_pos[1] - pacman_pos[1])
            exits = self._count_exits(next_pos, map_state)
            
            future_score = future_dist + exits * 2
            
            if depth > 1:
                # Recursive lookahead
                future_score += self._lookahead_escape(next_pos, pacman_pos, map_state, depth - 1) * 0.5
            
            best_future = max(best_future, future_score)
        
        return best_future
    
    def _predict_pacman_with_turns(self, pacman_pos, ghost_pos, pacman_last_move):
        """Predict Pacman position accounting for turn penalty."""
        row_diff = ghost_pos[0] - pacman_pos[0]
        col_diff = ghost_pos[1] - pacman_pos[1]
        
        # Determine Pacman's likely move direction
        if abs(row_diff) > abs(col_diff):
            # Move vertically
            move = Move.DOWN if row_diff > 0 else Move.UP
        else:
            # Move horizontally
            move = Move.RIGHT if col_diff > 0 else Move.LEFT
        
        # Check if this is a turn
        steps = 2  # Default straight move
        if pacman_last_move is not None and pacman_last_move != move:
            steps = 1  # Turn = only 1 step
        
        # Predict new position
        dr, dc = move.value
        predicted = (pacman_pos[0] + dr * steps, pacman_pos[1] + dc * steps)
        
        # Clamp to map bounds
        h, w = self.known_map.shape
        predicted = (max(0, min(h-1, predicted[0])), max(0, min(w-1, predicted[1])))
        
        return predicted
    
    def _predict_pacman_next_pos(self, pacman_pos, my_pos):
        """Predict where Pacman will move (toward Ghost)."""
        # Pacman will move toward Ghost
        row_diff = my_pos[0] - pacman_pos[0]
        col_diff = my_pos[1] - pacman_pos[1]
        
        predicted = list(pacman_pos)
        
        # Pacman moves toward ghost (greedy assumption)
        if abs(row_diff) > abs(col_diff):
            # Move vertically toward ghost
            predicted[0] += 2 if row_diff > 0 else -2  # Pacman speed = 2
        else:
            # Move horizontally toward ghost
            predicted[1] += 2 if col_diff > 0 else -2
        
        # Clamp to map bounds
        h, w = self.known_map.shape
        predicted[0] = max(0, min(h - 1, predicted[0]))
        predicted[1] = max(0, min(w - 1, predicted[1]))
        
        return tuple(predicted)
    
    def _evade_intelligent(self, my_pos, enemy_pos, map_state, step):
        """Enhanced evasion: Potential Fields + A* Escape Planning."""
        dist = abs(my_pos[0] - enemy_pos[0]) + abs(my_pos[1] - enemy_pos[1])
        
        # PRIORITY 1: Corridor escape - avoid getting trapped
        current_exits = self._count_exits(my_pos, map_state)
        if current_exits <= 2:  # In a corridor
            corridor_escape = self._escape_corridor(my_pos, enemy_pos, map_state)
            if corridor_escape != Move.STAY:
                return corridor_escape
        
        # PANIC MODE: Very close - emergency zigzag
        if dist <= 3:
            self.panic_mode = True
            return self._emergency_escape(my_pos, enemy_pos, map_state)
        
        # TACTICAL MODE: Medium distance (3 < dist <= 8)
        # Use MINIMAX to anticipate Pacman and plan escape
        if 3 < dist <= 8:
            self.panic_mode = False
            
            # Try Minimax planning (depth=2)
            minimax_move = self._minimax_escape(my_pos, enemy_pos, map_state, depth=2, maximizing=True, is_root=True)
            if minimax_move != Move.STAY:
                return minimax_move
            
            # Fallback: A* escape planning
            astar_move = self._astar_escape_path(my_pos, enemy_pos, map_state, depth=4)
            if astar_move != Move.STAY:
                return astar_move
            
            # Last resort: Potential Fields
            return self._potential_field_move(my_pos, enemy_pos, map_state)
        
        # STRATEGIC MODE: Far distance (dist > 8)
        # Use Potential Fields for smooth, intelligent movement
        self.panic_mode = False
        
        if dist > 8:
            # Potential Fields works well at long range
            pf_move = self._potential_field_move(my_pos, enemy_pos, map_state)
            if pf_move != Move.STAY:
                return pf_move
            
            # Q-Learning as fallback
            if dist > 10:
                state = self._encode_state(my_pos, enemy_pos, map_state, step)
                if state in self.q_table:
                    q_values = self.q_table[state]
                    valid_moves = []
                    for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]:
                        if move in q_values and self._is_valid_move(my_pos, move, map_state):
                            valid_moves.append((q_values[move], move))
                    if valid_moves:
                        valid_moves.sort(key=lambda x: x[0], reverse=True)
                        return valid_moves[0][1]
        
        # Final fallback: maximize distance
        return self._maximize_distance_smart(my_pos, enemy_pos, map_state)
    
    def _escape_corridor(self, my_pos, pacman_pos, map_state):
        """Escape corridors toward open areas with multiple exits."""
        best_move = Move.STAY
        best_score = -9999
        
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if not self._is_valid_move(my_pos, move, map_state):
                continue
            
            dr, dc = move.value
            next_pos = (my_pos[0] + dr, my_pos[1] + dc)
            
            # Exits from next position
            exits = self._count_exits(next_pos, map_state)
            
            # Distance from Pacman
            new_dist = abs(next_pos[0] - pacman_pos[0]) + abs(next_pos[1] - pacman_pos[1])
            old_dist = abs(my_pos[0] - pacman_pos[0]) + abs(my_pos[1] - pacman_pos[1])
            
            # 2-step lookahead: how open is the area ahead?
            future_openness = 0
            for move2 in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                if self._is_valid_move(next_pos, move2, map_state):
                    dr2, dc2 = move2.value
                    pos2 = (next_pos[0] + dr2, next_pos[1] + dc2)
                    future_openness += self._count_exits(pos2, map_state)
            
            # Scoring: prioritize open spaces (more exits = better)
            score = (exits * 25) + (future_openness * 8) + (new_dist * 4)
            
            # CRITICAL: Don't move closer to Pacman in a corridor
            if new_dist < old_dist and exits <= 2:
                score -= 200
            
            # Avoid recent positions
            if next_pos in list(self.position_history)[-3:]:
                score -= 50
            
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move
    
    def _potential_field_move(self, my_pos, pacman_pos, map_state):
        """Potential Fields: Pacman = repulsive force, open space = attractive."""
        best_move = Move.STAY
        best_potential = -999999
        
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if not self._is_valid_move(my_pos, move, map_state):
                continue
            
            dr, dc = move.value
            next_pos = (my_pos[0] + dr, my_pos[1] + dc)
            
            # Repulsive potential from Pacman (inverse square law)
            dist_to_pacman = max(1, abs(next_pos[0] - pacman_pos[0]) + abs(next_pos[1] - pacman_pos[1]))
            repulsive = 100.0 / (dist_to_pacman ** 1.5)  # Strong repulsion when close
            
            # Attractive potential to open spaces (more exits = more attractive)
            exits = self._count_exits(next_pos, map_state)
            attractive_open = exits * 15
            
            # Attractive potential to map edges/corners (safer zones)
            h, w = map_state.shape
            edge_dist = min(next_pos[0], next_pos[1], h - 1 - next_pos[0], w - 1 - next_pos[1])
            attractive_edge = (10 - edge_dist) * 8 if edge_dist < 10 else 0
            
            # Wall proximity bonus (Pacman has to navigate around)
            walls_nearby = sum(1 for dr2, dc2 in [(0,1), (0,-1), (1,0), (-1,0)]
                             if 0 <= next_pos[0]+dr2 < h and 0 <= next_pos[1]+dc2 < w
                             and map_state[next_pos[0]+dr2, next_pos[1]+dc2] == 1)
            wall_bonus = walls_nearby * 10
            
            # Combine potentials (repulsive is NEGATIVE, attractive is POSITIVE)
            total_potential = -repulsive + attractive_open + attractive_edge + wall_bonus
            
            # Penalty for backtracking
            if next_pos in list(self.position_history)[-3:]:
                total_potential -= 100
            
            # CRITICAL: Extra repulsion if moving toward Pacman
            old_dist = abs(my_pos[0] - pacman_pos[0]) + abs(my_pos[1] - pacman_pos[1])
            if dist_to_pacman < old_dist:
                total_potential -= 50  # Penalty for approaching
            
            if total_potential > best_potential:
                best_potential = total_potential
                best_move = move
        
        return best_move
    
    def _astar_escape_path(self, my_pos, pacman_pos, map_state, depth=5):
        """A* to find escape path that maximizes distance from Pacman."""
        from heapq import heappush, heappop
        
        # A* search for best escape route
        frontier = []
        # Priority: -distance (we want to MAXIMIZE distance, so negate it)
        # Start: (priority, steps, position, path)
        heappush(frontier, (0, 0, my_pos, []))
        visited = {my_pos}
        
        best_path = None
        best_final_dist = -1
        
        while frontier and len(visited) < 50:  # Limit search
            priority, steps, pos, path = heappop(frontier)
            
            # If reached depth, evaluate this path
            if steps >= depth:
                final_dist = abs(pos[0] - pacman_pos[0]) + abs(pos[1] - pacman_pos[1])
                if final_dist > best_final_dist:
                    best_final_dist = final_dist
                    best_path = path
                continue
            
            # Expand neighbors
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                if not self._is_valid_move(pos, move, map_state):
                    continue
                
                dr, dc = move.value
                next_pos = (pos[0] + dr, pos[1] + dc)
                
                if next_pos in visited:
                    continue
                
                visited.add(next_pos)
                
                # Calculate future Pacman position (he moves 2 steps per turn if straight)
                future_pacman = self._predict_pacman_next_pos(pacman_pos, next_pos)
                
                # Heuristic: negative distance (want to maximize)
                dist = abs(next_pos[0] - future_pacman[0]) + abs(next_pos[1] - future_pacman[1])
                exits = self._count_exits(next_pos, map_state)
                
                # Priority = -distance (lower is better for heap)
                priority = -(dist + exits * 2)
                
                new_path = path + [move]
                heappush(frontier, (priority, steps + 1, next_pos, new_path))
        
        # Return first move of best path
        if best_path and len(best_path) > 0:
            return best_path[0]
        
        return Move.STAY
    
    def _minimax_escape(self, ghost_pos, pacman_pos, map_state, depth, maximizing=True, is_root=False):
        """Minimax: Ghost maximizes distance, Pacman minimizes it."""
        # Base case
        if depth == 0:
            dist = abs(ghost_pos[0] - pacman_pos[0]) + abs(ghost_pos[1] - pacman_pos[1])
            exits = self._count_exits(ghost_pos, map_state)
            return dist * 10 + exits * 3
        
        if maximizing:  # Ghost's turn
            best_move = Move.STAY
            best_value = -99999
            
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                if not self._is_valid_move(ghost_pos, move, map_state):
                    continue
                
                dr, dc = move.value
                new_ghost = (ghost_pos[0] + dr, ghost_pos[1] + dc)
                
                # Simulate Pacman's response
                value = self._minimax_escape(new_ghost, pacman_pos, map_state, depth - 1, False, is_root=False)
                
                # Avoid backtracking
                if new_ghost in list(self.position_history)[-2:]:
                    value -= 40
                
                if value > best_value:
                    best_value = value
                    if is_root:  # Only track best_move at root
                        best_move = move
            
            return best_move if is_root else best_value
        
        else:  # Pacman's turn
            best_value = 99999
            
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move.value
                # Pacman moves 2 steps if straight
                steps = 2
                new_pacman = (pacman_pos[0] + dr * steps, pacman_pos[1] + dc * steps)
                
                # Validate
                h, w = map_state.shape
                if not (0 <= new_pacman[0] < h and 0 <= new_pacman[1] < w):
                    continue
                if map_state[new_pacman] == 1:
                    continue
                
                value = self._minimax_escape(ghost_pos, new_pacman, map_state, depth - 1, True, is_root=False)
                best_value = min(best_value, value)
            
            return best_value
    
    def _hug_wall_move(self, my_pos, enemy_pos, map_state):
        """Move along walls to create obstacles for Pacman."""
        h, w = self.known_map.shape
        best_move = Move.STAY
        best_score = -999
        
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if not self._is_valid_move(my_pos, move, map_state):
                continue
            
            dr, dc = move.value
            new_pos = (my_pos[0] + dr, my_pos[1] + dc)
            
            # Count walls adjacent to new position
            walls_adjacent = 0
            for wr, wc in [(-1,0), (1,0), (0,-1), (0,1)]:
                check_pos = (new_pos[0] + wr, new_pos[1] + wc)
                if 0 <= check_pos[0] < h and 0 <= check_pos[1] < w:
                    if self.known_map[check_pos] == 1:
                        walls_adjacent += 1
            
            # Only consider if hugging wall (at least 1 adjacent wall)
            if walls_adjacent == 0:
                continue
            
            # Distance from Pacman
            new_dist = abs(new_pos[0] - enemy_pos[0]) + abs(new_pos[1] - enemy_pos[1])
            exits = self._count_exits(new_pos, map_state)
            
            # Score: prioritize wall proximity + distance + exits
            score = walls_adjacent * 10 + new_dist * 4 + exits * 3
            
            # Avoid recent positions
            if new_pos in list(self.position_history)[-3:]:
                score -= 20
            
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move
    
    def _emergency_escape(self, my_pos, enemy_pos, map_state):
        """Emergency escape when very close - unpredictable movement."""
        import random
        
        # Calculate escape direction (opposite of enemy)
        row_diff = my_pos[0] - enemy_pos[0]
        col_diff = my_pos[1] - enemy_pos[1]
        
        # Primary escape moves (away from enemy)
        escape_moves = []
        if row_diff > 0:  # Enemy above
            escape_moves.append(Move.DOWN)
        elif row_diff < 0:  # Enemy below
            escape_moves.append(Move.UP)
        
        if col_diff > 0:  # Enemy left
            escape_moves.append(Move.RIGHT)
        elif col_diff < 0:  # Enemy right
            escape_moves.append(Move.LEFT)
        
        # Zigzag: alternate between direct escape and perpendicular
        self.zigzag_counter += 1
        if self.zigzag_counter % 2 == 1 and len(escape_moves) > 1:
            # Take perpendicular move for unpredictability
            perpendicular = [m for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT] 
                           if m not in escape_moves]
            if perpendicular:
                escape_moves = perpendicular + escape_moves
        
        # Try moves with distance + exit scoring
        scored_moves = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if not self._is_valid_move(my_pos, move, map_state):
                continue
            
            dr, dc = move.value
            new_pos = (my_pos[0] + dr, my_pos[1] + dc)
            
            new_dist = abs(new_pos[0] - enemy_pos[0]) + abs(new_pos[1] - enemy_pos[1])
            exits = self._count_exits(new_pos, map_state)
            
            # Avoid recent positions
            recent_penalty = 0
            if new_pos in list(self.position_history)[-3:]:
                recent_penalty = -10
            
            # Corner penalty (dynamic based on map size)
            h, w = map_state.shape
            is_corner = (new_pos[0] <= 2 or new_pos[0] >= h - 3) and \
                       (new_pos[1] <= 2 or new_pos[1] >= w - 3)
            corner_penalty = -8 if is_corner else 0
            
            # Prefer escape direction but not exclusively
            direction_bonus = 3 if move in escape_moves else 0
            
            score = new_dist * 3 + exits * 2 + direction_bonus + corner_penalty + recent_penalty
            scored_moves.append((score, move))
        
        if scored_moves:
            scored_moves.sort(key=lambda x: x[0], reverse=True)
            # Add randomness: pick from top 2
            top_moves = scored_moves[:2]
            selected = random.choice(top_moves)
            self.last_escape_dir = selected[1]
            return selected[1]
        
        return Move.STAY
    
    def _maximize_distance_smart(self, my_pos, enemy_pos, map_state):
        """Maximize distance with corner avoidance."""
        moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        scored_moves = []
        
        for move in moves:
            if not self._is_valid_move(my_pos, move, map_state):
                continue
            
            dr, dc = move.value
            new_pos = (my_pos[0] + dr, my_pos[1] + dc)
            
            # Distance score (higher weight)
            new_dist = abs(new_pos[0] - enemy_pos[0]) + abs(new_pos[1] - enemy_pos[1])
            
            # Mobility score (prefer open areas)
            exits = self._count_exits(new_pos, map_state)
            
            # Center bonus (prefer staying near center) - dynamic
            h, w = map_state.shape
            center_r, center_c = h // 2, w // 2
            center_dist = abs(new_pos[0] - center_r) + abs(new_pos[1] - center_c)
            center_bonus = max(0, min(h, w) // 2 - center_dist) * 0.5
            
            # Corner/edge penalty (stronger) - dynamic
            edge_threshold = 2
            is_corner = (new_pos[0] <= edge_threshold or new_pos[0] >= h - edge_threshold - 1) and \
                       (new_pos[1] <= edge_threshold or new_pos[1] >= w - edge_threshold - 1)
            is_edge = new_pos[0] <= edge_threshold or new_pos[0] >= h - edge_threshold - 1 or \
                     new_pos[1] <= edge_threshold or new_pos[1] >= w - edge_threshold - 1
            corner_penalty = -10 if is_corner else (-5 if is_edge else 0)
            
            # Avoid STAY unless necessary
            stay_penalty = -3 if move == Move.STAY else 0
            
            total_score = new_dist * 3 + exits * 2 + center_bonus + corner_penalty + stay_penalty
            scored_moves.append((total_score, move))
        
        if scored_moves:
            scored_moves.sort(key=lambda x: x[0], reverse=True)
            return scored_moves[0][1]
        
        return Move.STAY
    
    def _hide_strategic(self, my_pos, map_state):
        """Hide in safe areas when not visible - DYNAMIC (no hardcode)."""
        h, w = map_state.shape
        
        # Find safe spots dynamically: corners and edges with good exits
        safe_candidates = []
        
        # Check corners/edges dynamically
        for r in range(0, h, h // 10):  # Sample grid
            for c in range(0, w, w // 10):
                if 0 <= r < h and 0 <= c < w and map_state[r, c] == 0:
                    exits = self._count_exits((r, c), map_state)
                    if exits >= 2:  # Safe if has multiple exits
                        dist = abs(my_pos[0] - r) + abs(my_pos[1] - c)
                        safe_candidates.append((dist, (r, c)))
        
        # Find closest safe spot
        if safe_candidates:
            safe_candidates.sort(key=lambda x: x[0])
            closest = safe_candidates[0][1]
            
            # Move toward it
            row_diff = closest[0] - my_pos[0]
            col_diff = closest[1] - my_pos[1]
            
            if abs(row_diff) > abs(col_diff):
                move = Move.DOWN if row_diff > 0 else Move.UP
            else:
                move = Move.RIGHT if col_diff > 0 else Move.LEFT
            
            if self._is_valid_move(my_pos, move, map_state):
                return move
        
        # Fallback: move to position with most exits
        best_move = Move.STAY
        best_exits = -1
        
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if self._is_valid_move(my_pos, move, map_state):
                dr, dc = move.value
                next_pos = (my_pos[0] + dr, my_pos[1] + dc)
                exits = self._count_exits(next_pos, map_state)
                if exits > best_exits:
                    best_exits = exits
                    best_move = move
        
        return best_move
    
    def _encode_state(self, my_pos, enemy_pos, map_state, step):
        """Encode state for Q-table - must match train_curriculum.py."""
        if enemy_pos is None:
            return None
        
        row_diff = my_pos[0] - enemy_pos[0]
        col_diff = my_pos[1] - enemy_pos[1]
        dist = abs(row_diff) + abs(col_diff)
        
        # Distance bucket
        dist_bucket = 0 if dist <= 2 else (1 if dist <= 5 else (2 if dist <= 10 else 3))
        
        # Direction (simplified)
        direction = 0  # Up
        if abs(col_diff) > abs(row_diff):
            direction = 2 if col_diff < 0 else 3  # Left or Right (Ghost flees opposite)
        elif row_diff < 0:
            direction = 1  # Down
        
        # Wall information
        r, c = my_pos
        wall_up = 1 if (r == 0 or map_state[r-1, c] == 1) else 0
        wall_down = 1 if (r == 20 or map_state[r+1, c] == 1) else 0
        wall_left = 1 if (c == 0 or map_state[r, c-1] == 1) else 0
        wall_right = 1 if (c == 20 or map_state[r, c+1] == 1) else 0
        
        # Count exits
        exits = (1-wall_up) + (1-wall_down) + (1-wall_left) + (1-wall_right)
        
        # Danger level (closer = more danger)
        danger = 2 if dist <= 2 else (1 if dist <= 5 else 0)
        
        # Step bucket
        step_bucket = min(step // 40, 5)
        
        # Check if in corner (bad for Ghost)
        in_corner = 1 if exits <= 1 else 0
        
        return (dist_bucket, direction, wall_up, wall_down, wall_left, wall_right,
                exits, danger, in_corner, step_bucket, 1)
    
    def _maximize_distance_simple(self, my_pos, enemy_pos, map_state):
        """Maximize distance + mobility (ĐÚNG NHƯ example_student)."""
        r, c = my_pos
        best_move = Move.STAY
        best_score = -999999
        
        h, w = map_state.shape
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]:
            dr, dc = move.value
            nr, nc = r + dr, c + dc
            
            # Check valid
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if map_state[nr, nc] == 1:  # Wall
                continue
            
            # Calculate distance to Pacman
            dist = abs(nr - enemy_pos[0]) + abs(nc - enemy_pos[1])
            
            # Calculate mobility (number of exits)
            mobility = 0
            for dr2, dc2 in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ar, ac = nr + dr2, nc + dc2
                if 0 <= ar < 21 and 0 <= ac < 21 and map_state[ar, ac] == 0:
                    mobility += 1
            
            # SCORE FORMULA (GIỐNG example_student): distance * 10 + mobility
            score = dist * 10 + mobility
            
            # Anti-stuck: penalize recent positions
            if (nr, nc) in list(self.position_history)[-3:]:
                score -= 50
            
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move
    
    def _count_exits(self, pos, map_state):
        """Count valid exits from position."""
        r, c = pos
        count = 0
        
        h, w = map_state.shape
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and map_state[nr, nc] == 0:
                count += 1
        
        return count
    
    def _is_valid_move(self, pos, move, map_state):
        """Check if move is valid."""
        r, c = pos
        dr, dc = move.value
        nr, nc = r + dr, c + dc
        
        h, w = map_state.shape
        if nr < 0 or nr >= h or nc < 0 or nc >= w:
            return False
        return map_state[nr, nc] == 0
    
    def _get_new_pos(self, pos, move, steps=1):
        """Calculate new position after move."""
        r, c = pos
        dr, dc = move.value
        return (r + dr * steps, c + dc * steps)
