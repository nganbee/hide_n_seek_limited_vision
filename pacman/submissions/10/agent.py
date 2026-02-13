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


class PacmanConfig:
    """Configuration constants for Pacman agent - centralized for easy tuning."""

    # Map size classification
    SMALL_MAP_THRESHOLD = 11        # Maps ≤ 11x11 are "small"
    MEDIUM_MAP_THRESHOLD = 20       # Maps ≤ 20x20 are "medium"

    # Pursuit and sweep parameters
    PURSUIT_RANGE_MULTIPLIER = 1.0  # Multiplied by (2 * speed + 2)
    SWEEP_DURATION_SMALL = 12       # Steps to sweep in small maps
    SWEEP_DURATION_MEDIUM = 8       # Steps to sweep in medium maps
    SWEEP_DURATION_LARGE = 5        # Steps to sweep in large maps
    SWEEP_RADIUS = 4                # Cells around sweep center

    # Memory and history
    MAX_POSITION_HISTORY = 30       # Track last N positions
    STUCK_DETECTION_THRESHOLD = 3   # Stuck if same 2 positions in last 4 steps

    # Frontier scoring weights
    FRONTIER_DISTANCE_WEIGHT = -1.0         # Prefer closer frontiers
    FRONTIER_VISIBILITY_WEIGHT = 10.0       # Unknown neighbors bonus
    FRONTIER_INFORMATION_GAIN_WEIGHT = 15.0 # New information potential
    FRONTIER_VISIT_PENALTY = -50.0          # Recent visit penalty


class PacmanAgent(BasePacmanAgent):
    """
    Advanced Seek agent with partial observability support.
    Maintains belief map and uses A* for pursuit and frontier exploration.
    """
    
    def __init__(self, **kwargs):
        """Initialize the Seek agent with belief map and exploration state."""
        super().__init__(**kwargs)
        self.name = "Seek Agent with Partial Observability"
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        
        # Load configuration
        self.config = PacmanConfig()

        # Belief map: None until first observation
        # 1 = wall (permanent), 0 = empty (confirmed), -1 = unknown
        self.belief_map = None
        self.map_shape = None
        
        # Last known enemy position for tracking
        self.last_known_enemy_pos = None
        self.prev_enemy_pos = None  # For velocity tracking
        self.steps_since_last_sighting = 0
        
        # Exploration state
        self.recent_positions = []  # Track visited positions to avoid loops
        self.max_history = self.config.MAX_POSITION_HISTORY

        # Pattern detection
        self.exploration_targets = []  # Track failed exploration targets
        self.stuck_counter = 0  # Detect when stuck in loop

        # Small map adaptation with dynamic sweep
        self.sweep_steps_remaining = 0  # Track local sweep duration
        self.sweep_center = None  # Center of current sweep operation

    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int):
        """
        Main decision logic with belief map maintenance and adaptive strategy.
        """
        # Initialize belief map on first step
        if self.belief_map is None:
            self.map_shape = map_state.shape
            self.belief_map = np.full(self.map_shape, -1, dtype=int)
        
        # Update belief map with current observation
        self._update_belief_map(map_state)
        
        # Update enemy tracking
        if enemy_position is not None:
            self.prev_enemy_pos = self.last_known_enemy_pos  # Save previous before updating
            self.last_known_enemy_pos = enemy_position
            self.steps_since_last_sighting = 0
        else:
            self.steps_since_last_sighting += 1
        
        # Track position history for anti-loop
        self.recent_positions.append(my_position)
        if len(self.recent_positions) > self.max_history:
            self.recent_positions.pop(0)
        
        # Detect loop pattern (stuck in cycle)
        if len(self.recent_positions) >= 10:
            # Check if last 4 positions repeat
            last_4 = self.recent_positions[-4:]
            if len(set(last_4)) <= 2:  # Only 2 unique positions in last 4 steps
                self.stuck_counter += 1
            else:
                self.stuck_counter = 0

        # Choose strategy based on enemy visibility
        if enemy_position is not None:
            # PURSUIT MODE: Enemy visible, use A* to chase
            self.stuck_counter = 0  # Reset when enemy found
            move = self._pursuit_mode(my_position, enemy_position)
        else:
            # EXPLORATION MODE: Enemy not visible, explore frontiers
            move = self._exploration_mode(my_position)
        
        return move
    
    def _update_belief_map(self, observation: np.ndarray):
        """
        Update belief map with new observation.
        Walls (1) are permanent, empty cells (0) are confirmed traversable.
        Unknown cells (-1) remain unknown until observed.
        """
        for r in range(self.map_shape[0]):
            for c in range(self.map_shape[1]):
                obs_val = observation[r, c]
                if obs_val == 1:  # Wall - permanent knowledge
                    self.belief_map[r, c] = 1
                elif obs_val == 0:  # Empty - confirmed traversable
                    self.belief_map[r, c] = 0
                # obs_val == -1 means still unknown, keep belief_map unchanged
    
    def _pursuit_mode(self, my_pos: tuple, enemy_pos: tuple) -> tuple:
        """
        Chase enemy using A* pathfinding on belief map.
        Uses prediction to intercept rather than just chase.
        """
        # Predict where enemy will be
        predicted_enemy = self._predict_enemy_next_pos(enemy_pos)

        # Tính khoảng cách đến Ghost
        dist_to_enemy = self._manhattan_distance(my_pos, enemy_pos)

        # Chiến lược theo khoảng cách
        if dist_to_enemy <= 3:
            # Gần Ghost (<=3 ô): Cẩn thận, path trực tiếp đến Ghost
            target = enemy_pos
        else:
            # Xa Ghost: Dùng prediction để intercept
            target = predicted_enemy if predicted_enemy != enemy_pos else enemy_pos

        path = self._astar(my_pos, target, self.belief_map)

        # If prediction path doesn't work, use current position
        if not path or len(path) <= 1:
            path = self._astar(my_pos, enemy_pos, self.belief_map)

        if path and len(path) > 1:
            next_pos = path[1]  # path[0] is current position
            move = self._get_move_direction(my_pos, next_pos)
            
            # Calculate optimal steps - KHÔNG đi xuyên Ghost
            steps = self._calculate_steps_along_path(my_pos, move, path, enemy_pos)

            # Luôn tận dụng speed=2 khi có thể (bỏ điều kiện steps==1)
            return (move, steps)
        
        # Fallback: try to get closer with single step
        return self._greedy_approach(my_pos, enemy_pos)

    def _predict_enemy_next_pos(self, enemy_pos: tuple) -> tuple:
        """
        Predict where enemy will move based on velocity.
        WITH FALLBACK: If prediction fails, immediately use current position.
        """
        if self.prev_enemy_pos is None:
            # No velocity info - return current position (safest bet)
            return enemy_pos

        # Calculate velocity
        dr = enemy_pos[0] - self.prev_enemy_pos[0]
        dc = enemy_pos[1] - self.prev_enemy_pos[1]

        # Predict next position
        predicted = (enemy_pos[0] + dr, enemy_pos[1] + dc)

        # VALIDATE prediction - if unsafe, IMMEDIATELY use current position
        if self._in_bounds(predicted[0], predicted[1]) and self.belief_map[predicted[0], predicted[1]] == 0:
            return predicted

        # FALLBACK: Prediction failed, use current enemy position
        return enemy_pos

    def _exploration_mode(self, my_pos: tuple) -> tuple:
        """
        Explore map by targeting frontier cells (visible empty adjacent to unknown).

        ADAPTIVE STRATEGY FOR SMALL MAPS:
        - In small maps (≤11x11) with speed advantage (speed=2 > Ghost=1),
          prioritize CAPTURE PROBABILITY over pure exploration.
        - Maintain pressure on last_known_enemy_pos instead of abandoning it.
        - Ghost cannot escape far in small maps → last known position stays informative.
        """
        # STUCK DETECTION: If stuck in loop, force different strategy
        if self.stuck_counter >= 3:
            return self._break_pattern(my_pos)

        # CHECK IF MAP IS SMALL
        is_small_map = self._is_small_map()

        # Calculate distance to last known enemy position if it exists
        dist_to_last_known = None
        if self.last_known_enemy_pos is not None:
            dist_to_last_known = self._manhattan_distance(my_pos, self.last_known_enemy_pos)

        # PRIORITY 1: SMALL MAP + RECENT SIGHTING + CLOSE ENOUGH → SEARCH-AND-SWEEP
        # In small maps, aggressively pursue last known position when within reach
        if (is_small_map and
            self.last_known_enemy_pos is not None and
            self.steps_since_last_sighting < 40 and  # Longer memory for small maps
            dist_to_last_known is not None):

            # Within pursuit range (can reach in ~2-3 turns with speed=2)
            pursuit_range = 2 * self.pacman_speed + 2  # ~6 cells

            if dist_to_last_known <= pursuit_range:
                # Enter SEARCH-AND-SWEEP mode
                return self._search_and_sweep_mode(my_pos)

        # PRIORITY 2: NORMAL PURSUIT (any map size, recent sighting)
        # If we have a last known enemy position, go there
        if self.last_known_enemy_pos is not None and self.steps_since_last_sighting < 30:
            path = self._astar(my_pos, self.last_known_enemy_pos, self.belief_map)
            if path and len(path) > 1:
                next_pos = path[1]
                move = self._get_move_direction(my_pos, next_pos)
                steps = self._calculate_steps_along_path(my_pos, move, path, self.last_known_enemy_pos)
                return (move, steps)

        # PRIORITY 3: Find and go to frontiers (standard exploration)
        frontiers = self._find_frontiers()

        if frontiers:
            # Score frontiers and choose best
            best_frontier = self._select_best_frontier(my_pos, frontiers)

            if best_frontier:
                path = self._astar(my_pos, best_frontier, self.belief_map)
                if path and len(path) > 1:
                    next_pos = path[1]
                    move = self._get_move_direction(my_pos, next_pos)
                    steps = self._calculate_steps_along_path(my_pos, move, path)
                    return (move, steps)

        # PRIORITY 4: Explore unvisited safe cells
        fallback = self._explore_unvisited(my_pos)
        if fallback != (Move.STAY, 1):
            return fallback

        # PRIORITY 5: Just move to any safe cell
        return self._greedy_any_safe_move(my_pos)
    
    def _find_frontiers(self) -> list:
        """
        Find frontier cells: confirmed empty cells (0) adjacent to unknown cells (-1).
        """
        frontiers = []
        for r in range(self.map_shape[0]):
            for c in range(self.map_shape[1]):
                if self.belief_map[r, c] == 0:  # Confirmed empty
                    # Check if adjacent to unknown
                    for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                        nr, nc = r + dr, c + dc
                        if self._in_bounds(nr, nc) and self.belief_map[nr, nc] == -1:
                            frontiers.append((r, c))
                            break
        return frontiers
    
    def _select_best_frontier(self, my_pos: tuple, frontiers: list) -> tuple:
        """
        Select frontier that maximizes: proximity + visibility potential + information gain.

        IMPROVED: Information gain metric considers how many NEW unknown cells
        will become visible by moving to this frontier.
        """
        best_score = -float('inf')
        best_frontier = None
        
        for frontier in frontiers:
            # Distance penalty (prefer closer)
            dist = self._manhattan_distance(my_pos, frontier)
            distance_score = dist * self.config.FRONTIER_DISTANCE_WEIGHT

            # Visibility potential (count adjacent unknown cells)
            unknown_neighbors = self._count_unknown_neighbors(frontier)
            visibility_score = unknown_neighbors * self.config.FRONTIER_VISIBILITY_WEIGHT

            # INFORMATION GAIN: Estimate how many unknown cells will be revealed
            # by moving to this frontier (cross-shaped observation)
            information_gain = self._estimate_information_gain(frontier)
            information_score = information_gain * self.config.FRONTIER_INFORMATION_GAIN_WEIGHT

            # Recent visit penalty
            visit_penalty = self.config.FRONTIER_VISIT_PENALTY if frontier in self.recent_positions[-5:] else 0

            total_score = distance_score + visibility_score + information_score + visit_penalty

            if total_score > best_score:
                best_score = total_score
                best_frontier = frontier
        
        return best_frontier
    
    def _estimate_information_gain(self, pos: tuple) -> int:
        """
        NEW: Estimate information gain by counting unknown cells that would
        become visible from this position.

        Uses cross-shaped observation model (Manhattan rays up to 5 cells).
        """
        if self.belief_map is None:
            return 0

        information_gain = 0

        # Check 4 directions (cross-shaped observation)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            # Ray up to 5 cells (observation radius)
            for dist in range(1, 6):
                check_r = pos[0] + dr * dist
                check_c = pos[1] + dc * dist

                # Out of bounds
                if not self._in_bounds(check_r, check_c):
                    break

                cell_val = self.belief_map[check_r, check_c]

                # Wall blocks vision
                if cell_val == 1:
                    break

                # Unknown cell - information gain!
                if cell_val == -1:
                    information_gain += 1
                # Already known - continue ray

        return information_gain

    def _count_unknown_neighbors(self, pos: tuple) -> int:
        """Count how many unknown cells are adjacent to position."""
        count = 0
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = pos[0] + dr, pos[1] + dc
            if self._in_bounds(nr, nc) and self.belief_map[nr, nc] == -1:
                count += 1
        return count

    def _break_pattern(self, my_pos: tuple) -> tuple:
        """
        Break out of stuck loop pattern.
        Find furthest safe position that hasn't been visited recently.
        """
        # Reset stuck counter after breaking
        self.stuck_counter = 0

        # Find all safe positions not recently visited
        candidate_targets = []
        for r in range(self.map_shape[0]):
            for c in range(self.map_shape[1]):
                pos = (r, c)
                if self.belief_map[r, c] == 0 and pos not in self.recent_positions[-15:]:
                    # Calculate distance
                    dist = self._manhattan_distance(my_pos, pos)
                    # Prefer far positions with high mobility
                    mobility = self._count_safe_neighbors_belief(pos)
                    score = dist * 10 + mobility * 5
                    candidate_targets.append((score, pos))

        if candidate_targets:
            # Sort by score (furthest + most mobile)
            candidate_targets.sort(reverse=True)
            best_target = candidate_targets[0][1]

            # Path to furthest position
            path = self._astar(my_pos, best_target, self.belief_map)
            if path and len(path) > 1:
                next_pos = path[1]
                move = self._get_move_direction(my_pos, next_pos)
                steps = self._calculate_steps_along_path(my_pos, move, path)
                return (move, steps)

        # Ultimate fallback: random valid move
        valid_moves = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dr, dc = move.value
            next_pos = (my_pos[0] + dr, my_pos[1] + dc)
            if self._is_safe_in_belief(next_pos, self.belief_map):
                valid_moves.append(move)

        if valid_moves:
            return (random.choice(valid_moves), 1)

        return (Move.STAY, 1)

    def _count_safe_neighbors_belief(self, pos: tuple) -> int:
        """Count safe neighbors for a position in belief map."""
        count = 0
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            neighbor = (pos[0] + dr, pos[1] + dc)
            if self._is_safe_in_belief(neighbor, self.belief_map):
                count += 1
        return count

    def _astar(self, start: tuple, goal: tuple, belief_map: np.ndarray) -> list:
        """
        A* pathfinding under partial observability.
        - Wall (1): not traversable
        - Empty (0): low cost
        - Unknown (-1): high cost but allowed
        Returns list of positions from start to goal.
        """
        from heapq import heappush, heappop

        ROWS, COLS = belief_map.shape

        def in_bounds(pos):
            return 0 <= pos[0] < ROWS and 0 <= pos[1] < COLS

        def cell_cost(pos):
            val = belief_map[pos[0], pos[1]]
            if val == 1:
                return float("inf")  # wall
            if val == -1:
                return 5  # unknown = risky
            return 1  # confirmed empty

        open_set = []
        heappush(open_set, (0, 0, start, [start]))
        best_g = {start: 0}

        while open_set:
            f, g, current, path = heappop(open_set)

            if current == goal:
                return path

            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move.value
                neighbor = (current[0] + dr, current[1] + dc)

                if not in_bounds(neighbor):
                    continue

                cost = cell_cost(neighbor)
                if cost == float("inf"):
                    continue  # wall

                new_g = g + cost

                if neighbor in best_g and new_g >= best_g[neighbor]:
                    continue

                best_g[neighbor] = new_g
                h = self._manhattan_distance(neighbor, goal)
                new_f = new_g + h

                heappush(open_set, (new_f, new_g, neighbor, path + [neighbor]))

        return []

    def _is_safe_in_belief(self, pos: tuple, belief_map: np.ndarray) -> bool:
        """Check if position is confirmed safe (not wall, not unknown)."""
        if not self._in_bounds(pos[0], pos[1]):
            return False
        return belief_map[pos[0], pos[1]] == 0
    
    def _calculate_steps_along_path(self, my_pos: tuple, move: Move, path: list, enemy_pos: tuple = None) -> int:
        """
        Calculate how many CONSECUTIVE steps in the SAME direction we can take.
        Stops when:
        1. Path changes direction (need to turn)
        2. Reach pacman_speed limit
        3. Reach end of path
        4. NEW: About to collide with Ghost (stop 1 cell before)
        """
        if len(path) <= 1:
            return 1

        steps = 0
        dr, dc = move.value  # Direction of first move

        # Count consecutive steps in same direction starting from current position
        for i in range(1, len(path)):
            # Calculate direction from path[i-1] to path[i]
            current_dr = path[i][0] - path[i-1][0]
            current_dc = path[i][1] - path[i-1][1]

            # Check if still moving in same direction
            if current_dr != dr or current_dc != dc:
                break  # Path turns here, stop

            # Tăng steps TRƯỚC khi kiểm tra điều kiện dừng
            steps += 1

            # CRITICAL: Stop if NEXT position would be Ghost (prevent phasing)
            if enemy_pos and path[i] == enemy_pos:
                # Đã đi đến ô trước Ghost, dừng lại
                break

            # Reached speed limit - check AFTER incrementing
            if steps >= self.pacman_speed:
                break

        return max(1, steps)
    
    def _get_move_direction(self, from_pos: tuple, to_pos: tuple) -> Move:
        """Convert position delta to Move enum."""
        dr = to_pos[0] - from_pos[0]
        dc = to_pos[1] - from_pos[1]
        
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if move.value == (dr, dc):
                return move
        return Move.STAY
    
    def _manhattan_distance(self, pos1: tuple, pos2: tuple) -> int:
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _in_bounds(self, row: int, col: int) -> bool:
        """Check if position is within map bounds."""
        return 0 <= row < self.map_shape[0] and 0 <= col < self.map_shape[1]
    
    def _greedy_approach(self, my_pos: tuple, target: tuple) -> tuple:
        """
        Greedy approach toward target with multi-step support.
        Try to move closer on confirmed safe cells.
        """
        best_move = Move.STAY
        best_dist = self._manhattan_distance(my_pos, target)
        best_steps = 1

        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dr, dc = move.value
            next_pos = (my_pos[0] + dr, my_pos[1] + dc)
            
            if self._is_safe_in_belief(next_pos, self.belief_map):
                dist = self._manhattan_distance(next_pos, target)
                if dist < best_dist:
                    best_dist = dist
                    best_move = move
                    # Calculate how many steps we can take in this direction
                    best_steps = self._calculate_straight_steps(my_pos, move)

        # Luôn trả về tuple
        return (best_move, best_steps)

    def _explore_unvisited(self, my_pos: tuple) -> tuple:
        """Move to nearest unvisited safe cell with multi-step support."""
        all_safe = []
        for r in range(self.map_shape[0]):
            for c in range(self.map_shape[1]):
                if self.belief_map[r, c] == 0 and (r, c) not in self.recent_positions[-10:]:
                    all_safe.append((r, c))
        
        if all_safe:
            # Find closest unvisited
            closest = min(all_safe, key=lambda p: self._manhattan_distance(my_pos, p))
            path = self._astar(my_pos, closest, self.belief_map)
            if path and len(path) > 1:
                next_pos = path[1]
                move = self._get_move_direction(my_pos, next_pos)
                steps = self._calculate_steps_along_path(my_pos, move, path)
                return (move, steps)

        # Ultimate fallback
        return self._greedy_any_safe_move(my_pos)
    
    def _greedy_any_safe_move(self, my_pos: tuple) -> tuple:
        """Move to any safe adjacent cell with multi-step support, preferring unvisited and maximizing options."""
        moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        
        # Score each move based on: not recently visited, mobility (neighbors), distance from walls
        best_move = None
        best_score = -float('inf')
        best_steps = 1

        for move in moves:
            dr, dc = move.value
            next_pos = (my_pos[0] + dr, my_pos[1] + dc)

            if not self._is_safe_in_belief(next_pos, self.belief_map):
                continue

            score = 0

            # Prefer unvisited positions
            if next_pos not in self.recent_positions[-10:]:
                score += 100
            elif next_pos not in self.recent_positions[-5:]:
                score += 50

            # Prefer positions with more mobility (more safe neighbors)
            mobility = sum(1 for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
                          if self._is_safe_in_belief((next_pos[0] + m.value[0], next_pos[1] + m.value[1]), self.belief_map))
            score += mobility * 10

            # Add some randomness to avoid deterministic loops
            score += random.random() * 5

            if score > best_score:
                best_score = score
                best_move = move
                # Calculate how many steps in this direction
                best_steps = self._calculate_straight_steps(my_pos, move)

        if best_move:
            return (best_move, best_steps)

        return (Move.STAY, 1)

    def _calculate_straight_steps(self, pos: tuple, move: Move) -> int:
        """
        Calculate how many consecutive steps can be taken in the SAME direction.
        Stops when hitting wall, unknown cell, or reaching pacman_speed limit.
        """
        steps = 0
        current = pos
        dr, dc = move.value

        for _ in range(self.pacman_speed):
            next_pos = (current[0] + dr, current[1] + dc)

            # Stop if not safe
            if not self._is_safe_in_belief(next_pos, self.belief_map):
                break

            # Optional: avoid immediate revisit (can remove if causes issues)
            # if next_pos in self.recent_positions[-3:]:
            #     break

            steps += 1
            current = next_pos

        return max(1, steps)

    def _is_small_map(self) -> bool:
        """
        Detect if current map is "small" based on dimensions.

        Small maps: width ≤ 11 OR height ≤ 11
        In small maps with speed advantage, Ghost cannot escape far.
        """
        if self.map_shape is None:
            return False

        return (self.map_shape[0] <= self.config.SMALL_MAP_THRESHOLD or
                self.map_shape[1] <= self.config.SMALL_MAP_THRESHOLD)

    def _get_map_size_category(self) -> str:
        """Classify map size for adaptive strategy."""
        if self.map_shape is None:
            return "unknown"

        max_dim = max(self.map_shape[0], self.map_shape[1])

        if max_dim <= self.config.SMALL_MAP_THRESHOLD:
            return "small"
        elif max_dim <= self.config.MEDIUM_MAP_THRESHOLD:
            return "medium"
        else:
            return "large"

    def _get_dynamic_sweep_duration(self) -> int:
        """
        Calculate sweep duration based on map size.
        Smaller maps need longer sweeps (Ghost can't escape far).
        """
        category = self._get_map_size_category()

        if category == "small":
            return self.config.SWEEP_DURATION_SMALL
        elif category == "medium":
            return self.config.SWEEP_DURATION_MEDIUM
        else:
            return self.config.SWEEP_DURATION_LARGE

    def _search_and_sweep_mode(self, my_pos: tuple) -> tuple:
        """
        SEARCH-AND-SWEEP mode for small maps.

        Strategy:
        1. Move toward last_known_enemy_pos
        2. Upon reaching it, perform local sweep of nearby corridors/hubs
        3. Limit sweep duration to avoid overcommitment
        4. Return to exploration if Ghost not found

        This exploits speed asymmetry: with Pacman speed=2 and Ghost speed=1,
        the Ghost cannot have moved far in a small map.
        """
        target = self.last_known_enemy_pos
        dist_to_target = self._manhattan_distance(my_pos, target)

        # PHASE 1: Moving toward last known position
        if dist_to_target > 2:
            # Still approaching - path to target
            path = self._astar(my_pos, target, self.belief_map)
            if path and len(path) > 1:
                next_pos = path[1]
                move = self._get_move_direction(my_pos, next_pos)
                steps = self._calculate_steps_along_path(my_pos, move, path, target)

                # Reset sweep when moving toward target
                self.sweep_steps_remaining = 0
                return (move, steps)

        # PHASE 2: Reached vicinity - perform local sweep
        # Initialize sweep if not already sweeping
        if self.sweep_steps_remaining == 0:
            # Start sweep - duration adapts to map size
            self.sweep_steps_remaining = self._get_dynamic_sweep_duration()
            self.sweep_center = target

        # Perform sweep move
        if self.sweep_steps_remaining > 0:
            sweep_move = self._local_sweep_move(my_pos, self.sweep_center)
            self.sweep_steps_remaining -= 1
            return sweep_move

        # PHASE 3: Sweep completed without finding Ghost
        # Return to normal exploration (fallback to frontier mode)
        # This will be handled by caller (exploration_mode will continue to Priority 3)
        return self._explore_unvisited(my_pos)

    def _local_sweep_move(self, my_pos: tuple, sweep_center: tuple) -> tuple:
        """
        Perform local sweep around sweep_center.

        Strategy:
        - Visit high-connectivity positions (hubs, junctions)
        - Check corridors branching from sweep_center
        - Prefer unvisited cells within sweep radius
        - Maintain coverage pattern to maximize Ghost detection
        """
        sweep_radius = self.config.SWEEP_RADIUS  # Use config value

        # Find candidate sweep positions: confirmed safe cells near sweep_center
        candidates = []

        for r in range(self.map_shape[0]):
            for c in range(self.map_shape[1]):
                pos = (r, c)

                # Must be confirmed safe
                if self.belief_map[r, c] != 0:
                    continue

                # Must be within sweep radius
                dist_to_center = self._manhattan_distance(pos, sweep_center)
                if dist_to_center > sweep_radius:
                    continue

                # Score this position for sweeping
                score = 0.0

                # Prefer positions with high mobility (junctions, hubs)
                mobility = sum(1 for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
                              if self._is_safe_in_belief((pos[0] + m.value[0], pos[1] + m.value[1]), self.belief_map))
                score += mobility * 40

                # Strongly prefer unvisited positions
                if pos not in self.recent_positions[-15:]:
                    score += 150
                elif pos not in self.recent_positions[-8:]:
                    score += 80

                # Prefer positions closer to sweep center (systematic coverage)
                score -= dist_to_center * 5

                # Distance from current position (prefer reachable)
                dist_from_current = self._manhattan_distance(my_pos, pos)
                score -= dist_from_current * 3

                candidates.append((score, pos))

        # Select best sweep target
        if candidates:
            candidates.sort(reverse=True, key=lambda x: x[0])
            best_target = candidates[0][1]

            # Path to best sweep target
            path = self._astar(my_pos, best_target, self.belief_map)
            if path and len(path) > 1:
                next_pos = path[1]
                move = self._get_move_direction(my_pos, next_pos)
                steps = self._calculate_steps_along_path(my_pos, move, path)
                return (move, steps)

        # Fallback: move to any safe adjacent cell
        return self._greedy_any_safe_move(my_pos)


class GhostConfig:
    """Configuration constants for Ghost agent - centralized for easy tuning."""

    # Speed and threat modeling
    PACMAN_SPEED_ESTIMATE = 2           # Pacman's multi-step capability
    THREAT_RADIUS_MULTIPLIER = 2.5      # Threat radius = speed × multiplier

    # Position robustness thresholds
    MIN_ESCAPE_DEPTH = 4                # Minimum safe escape depth (strict for speed=2)
    ROBUST_MOBILITY_THRESHOLD = 3       # Mobility needed for robust position
    PANIC_MOBILITY_THRESHOLD = 2        # Trigger panic mode if mobility ≤ this

    # Scoring weights - Evasion mode
    EVASION_DISTANCE_WEIGHT_CLOSE = 500      # Distance bonus when in threat radius
    EVASION_DISTANCE_WEIGHT_FAR = 180        # Distance bonus when outside threat radius
    EVASION_DISTANCE_ABS_CLOSE = 120         # Absolute distance when close
    EVASION_DISTANCE_ABS_FAR = 35            # Absolute distance when far
    EVASION_DEAD_END_PENALTY = -900          # Dead-end penalty in panic
    EVASION_CORRIDOR_PENALTY = -280          # Corridor toward Pacman penalty

    # Scoring weights - Stealth mode (early avoidance)
    STEALTH_ROBUST_BONUS = 300               # Bonus for reaching robust position
    STEALTH_MOBILITY_IMPROVEMENT = 200       # Bonus for improving mobility
    STEALTH_DEAD_END_PENALTY = -400          # Dead-end penalty
    STEALTH_CORRIDOR_PENALTY = -150          # Corridor penalty

    # Unknown cell exploration
    UNKNOWN_EXPLORATION_BONUS = 80      # Bonus for moving toward unknown when trapped

    # Anti-prediction behavior
    ANTI_PREDICTION_FREQUENCY = 7       # Steps between anti-prediction moves
    ANTI_PREDICTION_VELOCITY_THRESHOLD = 3  # Consecutive steps to consider Pacman predictable

    # Performance optimization
    ESCAPE_DEPTH_SEARCH_LIMIT = 30      # Max cells to search in BFS
    CONNECTIVITY_SEARCH_RADIUS = 2      # Radius for connectivity evaluation
    ESCAPE_DEPTH_CACHE_SIZE = 100       # Max cached escape depth calculations


class GhostAgent(BaseGhostAgent):
    """
    Strategic Hide agent with belief-map intelligence.
    Uses topology analysis, multi-step threat modeling, and adaptive behavior.
    """

    def __init__(self, **kwargs):
        """Initialize the Hide agent with belief map and strategic state."""
        super().__init__(**kwargs)
        self.name = "Strategic Hide Agent (Belief-Map Intelligence)"

        # Load configuration
        self.config = GhostConfig()

        # Belief map: None until first observation
        # 1 = wall (permanent), 0 = empty (confirmed), -1 = unknown
        self.belief_map = None
        self.map_shape = None

        # Enemy tracking
        self.last_known_enemy_pos = None
        self.prev_enemy_pos = None  # For velocity prediction
        self.steps_since_last_sighting = 0
        self.velocity_consistency_counter = 0  # Initialize properly
        self.pacman_velocity_consistent = False

        # Movement history to avoid loops
        self.recent_positions = []
        self.max_history = 15
        self.stay_counter = 0  # Track consecutive STAY moves

        # Strategic state
        self.step_count = 0
        self.last_anti_prediction_step = -10

        # Performance optimization - cache escape depth calculations
        self.escape_depth_cache = {}  # {(pos, threat): depth}
        self.escape_depth_cache_hits = 0
        self.escape_depth_cache_misses = 0

    def step(self, map_state: np.ndarray,
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int) -> Move:
        """
        Strategic evasion with belief-map intelligence.
        """
        # Initialize belief map on first step
        if self.belief_map is None:
            self.map_shape = map_state.shape
            self.belief_map = np.full(self.map_shape, -1, dtype=int)

        # Update belief map with current observation
        self._update_belief_map(map_state)

        # Update enemy tracking and velocity consistency
        if enemy_position is not None:
            # Track velocity consistency for anti-prediction
            # Only consider Pacman predictable if velocity is stable over SEVERAL steps
            if self.prev_enemy_pos and self.last_known_enemy_pos:
                old_vel = (self.last_known_enemy_pos[0] - self.prev_enemy_pos[0],
                          self.last_known_enemy_pos[1] - self.prev_enemy_pos[1])
                new_vel = (enemy_position[0] - self.last_known_enemy_pos[0],
                          enemy_position[1] - self.last_known_enemy_pos[1])
                # Velocity is consistent if non-zero and same direction
                if old_vel == new_vel and old_vel != (0, 0):
                    self.velocity_consistency_counter += 1
                    # Only consider truly consistent after threshold
                    self.pacman_velocity_consistent = (
                        self.velocity_consistency_counter >= self.config.ANTI_PREDICTION_VELOCITY_THRESHOLD
                    )
                else:
                    self.velocity_consistency_counter = 0
                    self.pacman_velocity_consistent = False
            else:
                self.pacman_velocity_consistent = False

            self.prev_enemy_pos = self.last_known_enemy_pos
            self.last_known_enemy_pos = enemy_position
            self.steps_since_last_sighting = 0
        else:
            self.steps_since_last_sighting += 1
            # Reset consistency when Pacman not visible
            self.velocity_consistency_counter = 0
            self.pacman_velocity_consistent = False

        # Track position history for anti-loop
        self.recent_positions.append(my_position)
        if len(self.recent_positions) > self.max_history:
            self.recent_positions.pop(0)

        self.step_count += 1

        # Choose strategy based on enemy visibility
        if enemy_position is not None:
            # EVASION MODE: Enemy visible
            move = self._evasion_mode(my_position, enemy_position)
        else:
            # STEALTH MODE: Enemy not visible
            move = self._stealth_mode(my_position)

        # Track STAY moves
        if move == Move.STAY:
            self.stay_counter += 1
        else:
            self.stay_counter = 0

        return move

    def _update_belief_map(self, observation: np.ndarray):
        """
        Update belief map with new observation.
        Walls (1) are permanent, empty cells (0) are confirmed traversable.
        Unknown cells (-1) remain unknown until observed.
        """
        for r in range(self.map_shape[0]):
            for c in range(self.map_shape[1]):
                obs_val = observation[r, c]
                if obs_val == 1:  # Wall - permanent knowledge
                    self.belief_map[r, c] = 1
                elif obs_val == 0:  # Empty - confirmed traversable
                    self.belief_map[r, c] = 0
                # obs_val == -1 means still unknown, keep belief_map unchanged

    def _evasion_mode(self, my_pos: tuple, enemy_pos: tuple) -> Move:
        """
        Flee from visible enemy using intelligent escape strategies.
        Prioritize: distance, mobility, cover, avoiding dead ends.
        Uses counter-prediction to anticipate enemy movement.
        """
        # Predict where Pacman will be next (counter-prediction)
        predicted_pacman = self._predict_pacman_position(enemy_pos)

        # Calculate all valid moves
        candidate_moves = []

        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dr, dc = move.value
            next_pos = (my_pos[0] + dr, my_pos[1] + dc)

            # Only consider confirmed safe cells
            if not self._is_safe_in_belief(next_pos):
                continue

            # Score this move for evasion (use predicted position)
            score = self._evaluate_evasion_move(my_pos, next_pos, predicted_pacman)
            candidate_moves.append((score, move, next_pos))

        # Sort by score (higher is better)
        candidate_moves.sort(reverse=True, key=lambda x: x[0])

        if candidate_moves:
            best_score, best_move, best_pos = candidate_moves[0]
            return best_move

        # No valid move - stay
        return Move.STAY

    def _predict_pacman_position(self, pacman_pos: tuple) -> tuple:
        """
        Predict where Pacman will move (toward Ghost).
        Assume Pacman uses greedy approach or A*.
        """
        if self.prev_enemy_pos is None:
            # No velocity info - assume Pacman moves toward Ghost
            # Find direction from Pacman to Ghost
            dr = self.last_known_enemy_pos[0] - pacman_pos[0] if self.last_known_enemy_pos else 0
            dc = self.last_known_enemy_pos[1] - pacman_pos[1] if self.last_known_enemy_pos else 0

            # Normalize to single step
            if dr != 0:
                dr = 1 if dr > 0 else -1
            if dc != 0:
                dc = 1 if dc > 0 else -1

            # Pacman will likely move toward Ghost's current position
            predicted = (pacman_pos[0] + dr, pacman_pos[1] + dc)
        else:
            # Use velocity
            dr = pacman_pos[0] - self.prev_enemy_pos[0]
            dc = pacman_pos[1] - self.prev_enemy_pos[1]
            predicted = (pacman_pos[0] + dr, pacman_pos[1] + dc)

        # Validate and return
        if self._in_bounds(predicted[0], predicted[1]) and self.belief_map[predicted[0], predicted[1]] == 0:
            return predicted

        return pacman_pos

    def _stealth_mode(self, my_pos: tuple) -> Move:
        """
        Proactive stealth with CONDITIONAL EARLY AVOIDANCE.

        Key principle: Apply early avoidance ONLY when position is NOT robust.

        Strategies:
        1. Check position robustness
        2. If NOT robust: move toward stronger positions (early avoidance)
        3. If robust: normal patrol/positioning
        4. Anti-prediction when appropriate
        """
        # Anti-prediction behavior
        if self._should_use_anti_prediction():
            self.last_anti_prediction_step = self.step_count
            # Debug log
            if hasattr(self, '_debug_mode') and self._debug_mode:
                print(f"[Ghost DEBUG] Step {self.step_count}: Anti-prediction triggered!")
                print(f"  Velocity consistent for {self.velocity_consistency_counter} steps")
                print(f"  Steps since last anti-pred: {self.step_count - self.last_anti_prediction_step}")
            return self._anti_prediction_move(my_pos)

        # POSITION ROBUSTNESS CHECK
        # Estimate threat position for robustness evaluation
        estimated_threat = self.last_known_enemy_pos if self.last_known_enemy_pos else (self.map_shape[0] // 2, self.map_shape[1] // 2)
        is_robust = self._is_position_robust(my_pos, estimated_threat)

        # CRITICAL: Early avoidance must activate MORE AGGRESSIVELY
        # With Pacman speed=2, even if Pacman is far now, weak positions are lethal
        # Activate early avoidance when:
        # - Pacman was seen recently (within 30 steps), AND
        # - Current position is NOT robust
        if self.last_known_enemy_pos is not None and self.steps_since_last_sighting < 30:
            predicted_enemy = self._predict_enemy_position()

            candidate_moves = []

            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move.value
                next_pos = (my_pos[0] + dr, my_pos[1] + dc)

                if not self._is_safe_in_belief(next_pos):
                    continue

                # CONDITIONAL EARLY AVOIDANCE
                if is_robust:
                    # Position is ROBUST - normal stealth scoring
                    score = self._evaluate_stealth_move_normal(my_pos, next_pos, predicted_enemy)
                else:
                    # Position is NOT ROBUST - apply early avoidance
                    score = self._evaluate_stealth_move_early_avoidance(my_pos, next_pos, predicted_enemy)

                candidate_moves.append((score, move, next_pos))

            candidate_moves.sort(reverse=True, key=lambda x: x[0])

            if candidate_moves:
                return candidate_moves[0][1]

        # No recent sighting
        if is_robust:
            # Robust position - active patrol
            return self._active_patrol(my_pos)
        else:
            # Not robust - find stronger position
            return self._move_to_robust_position(my_pos)

    def _anti_prediction_move(self, my_pos: tuple) -> Move:
        """
        Execute anti-prediction behavior.

        Options:
        1. Pause (STAY) - breaks velocity prediction
        2. Move to high-connectivity hub
        3. Brief move toward unknown (confuse Pacman)

        Purpose: Counter Pacman's A* prediction assumptions.
        """
        # Option 1: Random pause (30% chance)
        if random.random() < 0.3:
            return Move.STAY

        # Option 2 & 3: Move to best strategic position
        candidate_moves = []

        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dr, dc = move.value
            next_pos = (my_pos[0] + dr, my_pos[1] + dc)

            if not self._is_safe_in_belief(next_pos):
                continue

            score = 0.0

            # Prefer high-connectivity positions (hubs)
            connectivity = self._evaluate_local_connectivity(next_pos)
            score += connectivity * 50

            # Bonus for positions near unknown (confusing)
            unknown_neighbors = self._count_unknown_neighbors(next_pos)
            score += unknown_neighbors * 30

            # Mobility bonus
            mobility = self._count_safe_neighbors(next_pos)
            score += mobility * 40

            # Avoid recent positions
            if next_pos in self.recent_positions[-5:]:
                score -= 100

            candidate_moves.append((score, move))

        if candidate_moves:
            candidate_moves.sort(reverse=True, key=lambda x: x[0])
            return candidate_moves[0][1]

        return Move.STAY

    def _evaluate_stealth_move_normal(self, current_pos: tuple, next_pos: tuple, predicted_enemy: tuple) -> float:
        """
        Normal stealth evaluation when position IS ROBUST.

        Focus:
        - Strategic positioning (hubs, choke points)
        - Moderate distance maintenance
        - Active patrol behavior
        """
        score = 0.0

        # 1. Distance scoring (maintain safe distance, not overly cautious)
        next_dist = self._manhattan_distance(next_pos, predicted_enemy)
        current_dist = self._manhattan_distance(current_pos, predicted_enemy)
        score += (next_dist - current_dist) * 80
        score += next_dist * 15

        # 2. Mobility and connectivity
        mobility = self._count_safe_neighbors(next_pos)
        connectivity = self._evaluate_local_connectivity(next_pos)

        score += mobility * 35
        score += connectivity * 25

        # 3. Choke-point control (proactive positioning)
        if mobility == 2:
            map_center = (self.map_shape[0] // 2, self.map_shape[1] // 2)
            dist_to_center = self._manhattan_distance(next_pos, map_center)
            pacman_to_center = self._manhattan_distance(predicted_enemy, map_center)

            if dist_to_center < pacman_to_center:
                score += 60

        # 4. Avoid recent positions
        if next_pos in self.recent_positions[-5:]:
            score -= 100

        score += random.random() * 15

        return score

    def _evaluate_stealth_move_early_avoidance(self, current_pos: tuple, next_pos: tuple, predicted_enemy: tuple) -> float:
        """
        Early avoidance evaluation when position is NOT ROBUST.

        Goal: Move toward STRONGER positions BEFORE Pacman becomes visible.

        CRITICAL FOR SPEED ASYMMETRY:
        - With Pacman speed=2, weak positions can become lethal in 1-2 turns
        - Must aggressively prioritize robustness improvement
        - Treat escape_depth == 3 as WEAK when mobility <= 2

        Prioritize:
        - Increasing mobility (move toward hubs)
        - Increasing escape depth (to ≥4 if possible, or ≥3 with high mobility)
        - Moving closer to unknown cells (potential exits, but not deep dive)
        - STRONGLY AVOID staying in corridors/dead-ends
        """
        score = 0.0

        # 1. Distance - still important but lower weight than robustness
        next_dist = self._manhattan_distance(next_pos, predicted_enemy)
        current_dist = self._manhattan_distance(current_pos, predicted_enemy)
        score += (next_dist - current_dist) * 60
        score += next_dist * 12

        # 2. ROBUSTNESS IMPROVEMENT (primary goal - STRENGTHENED)
        next_mobility = self._count_safe_neighbors(next_pos)
        current_mobility = self._count_safe_neighbors(current_pos)
        next_escape_depth = self._calculate_escape_depth(next_pos, predicted_enemy)
        current_escape_depth = self._calculate_escape_depth(current_pos, predicted_enemy)

        # Strong bonus for improving robustness
        if next_mobility > current_mobility:
            score += 200  # Moving toward higher mobility

        if next_escape_depth > current_escape_depth:
            score += 100  # Improving escape depth

        # AGGRESSIVE ROBUSTNESS SCORING
        if next_mobility >= 3 and next_escape_depth >= 3:
            score += self.config.STEALTH_ROBUST_BONUS  # Reaching truly robust position
        elif next_escape_depth >= 4:
            score += 280  # Deep escape compensates for mobility
        elif next_mobility >= 3:
            score += 150  # High mobility but shallow escape
        elif next_mobility == 2 and next_escape_depth >= 3:
            score += 80   # Corridor but has some depth
        elif next_mobility == 2:
            score -= 100  # Corridor with shallow escape - risky
        else:
            score += self.config.STEALTH_DEAD_END_PENALTY  # Dead-end or very weak - dangerous

        # Escape depth bonus (stricter thresholds)
        if next_escape_depth >= 4:
            score += 100
        elif next_escape_depth >= 3:
            score += 50
        else:
            score -= 80  # Shallow escape is risky

        # 3. Connectivity (move toward hubs) - STRENGTHENED
        connectivity = self._evaluate_local_connectivity(next_pos)
        score += connectivity * 50

        # 4. Unknown cells as potential exits
        # Move CLOSER to unknown (not into unknown)
        unknown_neighbors = self._count_unknown_neighbors(next_pos)
        if next_mobility <= 2:
            # Low mobility - unknown cells are valuable escape potential
            score += unknown_neighbors * 60

        # 5. STRONGLY PENALIZE staying in weak positions
        if current_mobility <= 1 and next_mobility <= 1:
            score += self.config.STEALTH_DEAD_END_PENALTY  # Staying in dead-end - very bad

        if current_mobility == 2 and next_mobility == 2:
            # Both corridor - penalize staying in corridor
            score += self.config.STEALTH_CORRIDOR_PENALTY

        # Additional: penalize moving INTO weaker positions
        if next_mobility < current_mobility:
            score -= 180

        # 6. Context-aware urgency
        # If Pacman was seen VERY recently, be even more aggressive
        if self.steps_since_last_sighting < 10:
            # High urgency - multiply penalties and bonuses
            if next_mobility <= 2 and next_escape_depth <= 3:
                score -= 200  # Extra penalty for weak positions when Pacman nearby
            if next_mobility >= 3 and next_escape_depth >= 3:
                score += 150  # Extra bonus for strong positions

        # 7. Avoid recent positions (but less important than robustness)
        if next_pos in self.recent_positions[-5:]:
            score -= 100
        elif next_pos in self.recent_positions[-10:]:
            score -= 50

        score += random.random() * 8

        return score

    def _move_to_robust_position(self, my_pos: tuple) -> Move:
        """
        Find and move toward a robust position when current position is weak.

        Strategy:
        - Scan nearby confirmed cells
        - Find positions with mobility ≥3 or escape_depth ≥3
        - Move toward closest robust position
        """
        candidate_moves = []

        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dr, dc = move.value
            next_pos = (my_pos[0] + dr, my_pos[1] + dc)

            if not self._is_safe_in_belief(next_pos):
                continue

            score = 0.0

            # Check if next position is robust
            estimated_threat = self.last_known_enemy_pos if self.last_known_enemy_pos else (self.map_shape[0] // 2, self.map_shape[1] // 2)

            if self._is_position_robust(next_pos, estimated_threat):
                score += 300  # Strong bonus for reaching robust position

            # Mobility improvement
            next_mobility = self._count_safe_neighbors(next_pos)
            current_mobility = self._count_safe_neighbors(my_pos)

            score += (next_mobility - current_mobility) * 100
            score += next_mobility * 50

            # Connectivity
            connectivity = self._evaluate_local_connectivity(next_pos)
            score += connectivity * 30

            # Avoid recent
            if next_pos in self.recent_positions[-5:]:
                score -= 100

            candidate_moves.append((score, move))

        if candidate_moves:
            candidate_moves.sort(reverse=True, key=lambda x: x[0])
            return candidate_moves[0][1]

        # Fallback
        return Move.STAY

    def _evaluate_evasion_move(self, current_pos: tuple, next_pos: tuple, enemy_pos: tuple) -> float:
        """
        Belief-map aware evasion evaluation.

        ADJUSTED FOR SPEED ASYMMETRY:
        - Pacman speed=2 means threat radius is larger
        - Weak positions (mobility<=2, escape_depth<=3) are MORE dangerous
        - Must be more aggressive about escape depth requirements

        Considers:
        1. Multi-step Pacman threat radius (speed=2)
        2. Escape depth through confirmed cells (stricter thresholds)
        3. Strategic use of unknown cells
        4. Local connectivity topology
        5. Panic mode when cornered
        """
        score = 0.0

        # 1. Multi-step threat modeling
        # Pacman can reach threat_radius in one turn
        threat_radius = self.config.PACMAN_SPEED_ESTIMATE * self.config.THREAT_RADIUS_MULTIPLIER
        next_dist = self._manhattan_distance(next_pos, enemy_pos)
        current_dist = self._manhattan_distance(current_pos, enemy_pos)

        # CRITICAL: If within threat radius, heavily prioritize distance
        if next_dist < threat_radius:
            score += (next_dist - current_dist) * self.config.EVASION_DISTANCE_WEIGHT_CLOSE
            score += next_dist * self.config.EVASION_DISTANCE_ABS_CLOSE
        else:
            # Outside threat radius - normal distance scoring
            score += (next_dist - current_dist) * self.config.EVASION_DISTANCE_WEIGHT_FAR
            score += next_dist * self.config.EVASION_DISTANCE_ABS_FAR

        # 2. Belief-map topology: Local connectivity
        local_connectivity = self._evaluate_local_connectivity(next_pos)
        mobility = self._count_safe_neighbors(next_pos)

        # 3. Panic mode detection (adjusted for speed asymmetry)
        # Panic if close to Pacman AND in weak position
        is_panic = (next_dist < threat_radius and
                   mobility <= self.config.PANIC_MOBILITY_THRESHOLD)

        if is_panic:
            # PANIC MODE: Ignore normal heuristics, focus on survival
            escape_depth = self._calculate_escape_depth(next_pos, enemy_pos)

            # Stricter escape depth requirements
            if escape_depth >= self.config.MIN_ESCAPE_DEPTH:
                score += 700  # Strong bonus for deep escape routes
            elif escape_depth >= 3:
                score += 400  # Moderate bonus for decent escape
            else:
                # Shallow escape - try unknown cells as last resort
                unknown_neighbors = self._count_unknown_neighbors(next_pos)
                if unknown_neighbors > 0:
                    score += self.config.UNKNOWN_EXPLORATION_BONUS * unknown_neighbors
                    # In panic, unknown is better than trapped
                else:
                    score -= 200  # No escape options

            # In panic, heavily penalize dead ends
            if mobility <= 1:
                score += self.config.EVASION_DEAD_END_PENALTY
            else:
                score += mobility * 70
        else:
            # NORMAL MODE: Balance multiple factors

            # Escape depth scoring (stricter)
            escape_depth = self._calculate_escape_depth(next_pos, enemy_pos)
            if escape_depth >= 4:
                score += 80
            elif escape_depth >= 3:
                score += 40
            else:
                score -= 50  # Penalize shallow escape

            # Mobility scoring (STRICTER - reflect speed asymmetry)
            if mobility <= 1:
                score -= 600  # Dead end penalty (increased)
            elif mobility == 2:
                # Corridor - evaluate carefully
                if next_dist < current_dist:
                    score += self.config.EVASION_CORRIDOR_PENALTY  # Moving into corridor toward Pacman
                elif escape_depth >= 3:
                    score += mobility * 50  # Corridor OK if has depth
                else:
                    score -= 100  # Corridor with shallow escape - risky
            else:
                score += mobility * 60  # High mobility is good

            # 4. Strategic unknown cell usage
            # Unknown cells can be escape routes when mobility is low
            if mobility <= 2 and next_dist < threat_radius + 2:
                unknown_neighbors = self._count_unknown_neighbors(next_pos)
                if unknown_neighbors > 0:
                    score += 50 * unknown_neighbors  # Potential escape

        # 5. Connectivity bonus (prefer high-degree nodes in belief graph)
        score += local_connectivity * 25

        # 6. Avoid recent positions (anti-loop)
        if next_pos in self.recent_positions[-3:]:
            score -= 180
        elif next_pos in self.recent_positions[-8:]:
            score -= 100

        # 7. Line of sight breaking
        if self._breaks_line_of_sight(current_pos, next_pos, enemy_pos):
            score += 90

        # 8. NEVER move toward enemy unless trapped (stricter)
        if next_dist < current_dist and not is_panic:
            score -= 400  # Increased penalty

        # 9. Small randomness
        score += random.random() * 8

        return score

    def _evaluate_stealth_move(self, current_pos: tuple, next_pos: tuple, predicted_enemy: tuple) -> float:
        """
        Evaluate move quality for stealth mode (enemy not visible).
        """
        score = 0.0

        # 1. Distance from predicted enemy position
        dist = self._manhattan_distance(next_pos, predicted_enemy)
        score += dist * 20

        # 2. Mobility (maintain escape options)
        mobility = self._count_safe_neighbors(next_pos)
        score += mobility * 25

        # 3. Avoid frontier cells (don't expose position)
        if self._is_frontier_cell(next_pos):
            score -= 100

        # 4. Avoid recently visited
        if next_pos in self.recent_positions[-8:]:
            score -= 60

        # 5. Prefer central safe areas (not edges)
        centrality = self._evaluate_centrality(next_pos)
        score += centrality * 15

        # 6. Small randomness
        score += random.random() * 3

        return score

    def _predict_enemy_position(self) -> tuple:
        """Predict enemy position based on last known position and velocity."""
        if self.prev_enemy_pos is None or self.last_known_enemy_pos is None:
            # Return last known or fallback to center
            if self.last_known_enemy_pos:
                return self.last_known_enemy_pos
            else:
                return (self.map_shape[0] // 2, self.map_shape[1] // 2)

        # Calculate velocity
        dr = self.last_known_enemy_pos[0] - self.prev_enemy_pos[0]
        dc = self.last_known_enemy_pos[1] - self.prev_enemy_pos[1]

        # Extrapolate based on time since last sighting
        steps_ahead = min(self.steps_since_last_sighting, 5)
        predicted = (
            self.last_known_enemy_pos[0] + dr * steps_ahead,
            self.last_known_enemy_pos[1] + dc * steps_ahead
        )

        return predicted

    def _count_safe_neighbors(self, pos: tuple) -> int:
        """Count how many confirmed safe neighbors a position has."""
        count = 0
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            neighbor = (pos[0] + dr, pos[1] + dc)
            if self._is_safe_in_belief(neighbor):
                count += 1
        return count

    def _count_unknown_neighbors(self, pos: tuple) -> int:
        """
        Count unknown cells adjacent to position.

        Purpose: Unknown cells may provide escape routes when cornered.
        """
        count = 0
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            neighbor = (pos[0] + dr, pos[1] + dc)
            if self._in_bounds(neighbor[0], neighbor[1]):
                if self.belief_map[neighbor[0], neighbor[1]] == -1:
                    count += 1
        return count

    def _calculate_escape_depth(self, start: tuple, threat: tuple) -> int:
        """
        Calculate escape depth: how many moves can be made through
        confirmed safe cells while increasing distance from threat.

        Uses BFS limited to confirmed cells (belief_map == 0).
        WITH CACHING for performance optimization.

        Purpose: Measure quality of escape route through known topology.
        """
        # Check cache first
        cache_key = (start, threat)
        if cache_key in self.escape_depth_cache:
            self.escape_depth_cache_hits += 1
            return self.escape_depth_cache[cache_key]

        self.escape_depth_cache_misses += 1

        from collections import deque

        visited = {start}
        queue = deque([(start, 0)])
        max_depth = 0

        # Limit search using config
        while queue and len(visited) < self.config.ESCAPE_DEPTH_SEARCH_LIMIT:
            pos, depth = queue.popleft()
            max_depth = max(max_depth, depth)

            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                neighbor = (pos[0] + dr, pos[1] + dc)

                if neighbor in visited:
                    continue

                # Only traverse confirmed safe cells
                if not self._is_safe_in_belief(neighbor):
                    continue

                # Only continue if moving away from threat
                if self._manhattan_distance(neighbor, threat) >= self._manhattan_distance(pos, threat):
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))

        # Cache result (limit cache size)
        if len(self.escape_depth_cache) < self.config.ESCAPE_DEPTH_CACHE_SIZE:
            self.escape_depth_cache[cache_key] = max_depth
        elif len(self.escape_depth_cache) >= self.config.ESCAPE_DEPTH_CACHE_SIZE:
            # Clear oldest 20% of cache when full
            keys_to_remove = list(self.escape_depth_cache.keys())[:self.config.ESCAPE_DEPTH_CACHE_SIZE // 5]
            for key in keys_to_remove:
                del self.escape_depth_cache[key]
            self.escape_depth_cache[cache_key] = max_depth

        return max_depth

    def _evaluate_local_connectivity(self, pos: tuple) -> int:
        """
        Evaluate local connectivity in belief-map graph.

        Returns: Number of reachable safe cells within configured radius.

        Purpose: High connectivity = more strategic options.
        Identifies hubs vs. dead-end branches in topology.
        """
        from collections import deque

        visited = {pos}
        queue = deque([(pos, 0)])

        while queue:
            current, dist = queue.popleft()

            if dist >= self.config.CONNECTIVITY_SEARCH_RADIUS:
                continue

            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                neighbor = (current[0] + dr, current[1] + dc)

                if neighbor in visited:
                    continue

                if self._is_safe_in_belief(neighbor):
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

        return len(visited)

    def _should_use_anti_prediction(self) -> bool:
        """
        Decide if Ghost should break Pacman's prediction assumptions.

        Triggers when:
        - Pacman has CONSISTENT velocity over threshold steps (truly predictable)
        - Enough steps since last anti-prediction move
        - NOT in immediate danger (distance > threat_radius)

        Purpose: Make Ghost unpredictable to counter Pacman's A* planning.
        Only trigger when Pacman's behavior is actually predictable.
        """
        if not self.pacman_velocity_consistent:
            return False

        steps_since_last = self.step_count - self.last_anti_prediction_step

        if steps_since_last < self.config.ANTI_PREDICTION_FREQUENCY:
            return False

        # Don't use anti-prediction if Pacman is very close (focus on evasion)
        if self.last_known_enemy_pos:
            # Approximate current position (since we're in stealth mode, Pacman not visible)
            # Anti-prediction only makes sense when we have some breathing room
            return True

        return True

    def _is_position_robust(self, pos: tuple, estimated_threat_pos: tuple = None) -> bool:
        """
        Evaluate if Ghost can survive sudden Pacman appearance at this position.

        DEFINITION OF ROBUST POSITION (ADJUSTED FOR SPEED ASYMMETRY):
        - (High mobility ≥3 AND escape_depth ≥3), OR
        - Deep escape routes (≥4 steps away from threat)

        CRITICAL: With Pacman speed=2, positions that appear "barely safe"
        are actually lethal. Stricter thresholds required.

        Purpose: Determine if early avoidance is needed BEFORE Pacman becomes visible.

        A robust position means Ghost has good survival chances even if
        Pacman appears suddenly within vision range (≤5 Manhattan distance).

        Returns:
            True if position is ROBUST (no early avoidance needed)
            False if position is WEAK (apply early avoidance)
        """
        # Check mobility (local topology strength)
        mobility = self._count_safe_neighbors(pos)

        # Check escape depth (strategic depth through confirmed cells)
        if estimated_threat_pos:
            escape_depth = self._calculate_escape_depth(pos, estimated_threat_pos)
        else:
            # No known threat - estimate based on last known or map center
            if self.last_known_enemy_pos:
                escape_depth = self._calculate_escape_depth(pos, self.last_known_enemy_pos)
            else:
                # Fallback: use map center as reference
                map_center = (self.map_shape[0] // 2, self.map_shape[1] // 2)
                escape_depth = self._calculate_escape_depth(pos, map_center)

        # STRICTER ROBUSTNESS CRITERIA FOR SPEED ASYMMETRY
        # Strong position: high mobility AND good escape depth
        if mobility >= self.config.ROBUST_MOBILITY_THRESHOLD and escape_depth >= 3:
            return True

        # Alternative: very deep escape route compensates for lower mobility
        if escape_depth >= self.config.MIN_ESCAPE_DEPTH:
            return True

        # TREAT AS WEAK: mobility <= 2 AND escape_depth <= 3
        # This is dangerous under speed asymmetry
        if mobility <= 2 and escape_depth <= 3:
            return False

        # Edge case: mobility >= 3 but shallow escape
        # Still somewhat risky, err on the side of caution
        return False

    def _is_frontier_cell(self, pos: tuple) -> bool:
        """Check if position is adjacent to unknown cells (frontier)."""
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            neighbor = (pos[0] + dr, pos[1] + dc)
            if self._in_bounds(neighbor[0], neighbor[1]):
                if self.belief_map[neighbor[0], neighbor[1]] == -1:
                    return True
        return False

    def _breaks_line_of_sight(self, current: tuple, next_pos: tuple, enemy: tuple) -> bool:
        """Check if moving to next_pos breaks line of sight with enemy."""
        # Simple heuristic: check if a wall is between next_pos and enemy
        # in the direction we're moving
        dr = next_pos[0] - current[0]
        dc = next_pos[1] - current[1]

        # Check position behind next_pos
        check_pos = (next_pos[0] + dr, next_pos[1] + dc)
        if self._in_bounds(check_pos[0], check_pos[1]):
            if self.belief_map[check_pos[0], check_pos[1]] == 1:
                return True  # Wall provides cover

        return False

    def _calculate_path_depth(self, start: tuple, avoid: tuple) -> int:
        """
        Calculate how many moves we can make from start while moving away from avoid.
        Returns depth of escape route.
        """
        from collections import deque

        visited = {start}
        queue = deque([(start, 0)])
        max_depth = 0

        while queue and len(visited) < 20:  # Limit search
            pos, depth = queue.popleft()
            max_depth = max(max_depth, depth)

            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                neighbor = (pos[0] + dr, pos[1] + dc)

                if neighbor in visited:
                    continue

                if not self._is_safe_in_belief(neighbor):
                    continue

                # Only continue if moving away or maintaining distance
                if self._manhattan_distance(neighbor, avoid) >= self._manhattan_distance(pos, avoid):
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))

        return max_depth

    def _evaluate_centrality(self, pos: tuple) -> float:
        """
        Evaluate how central a position is within known safe area.
        Higher = more central (safer).
        """
        # Count safe cells in a 3x3 neighborhood
        safe_count = 0
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                neighbor = (pos[0] + dr, pos[1] + dc)
                if self._is_safe_in_belief(neighbor):
                    safe_count += 1

        return safe_count

    def _evaluate_position_safety(self, pos: tuple, threat: tuple) -> float:
        """Evaluate overall safety of a position."""
        score = 0.0
        score += self._manhattan_distance(pos, threat) * 20
        score += self._count_safe_neighbors(pos) * 25
        if not self._is_frontier_cell(pos):
            score += 50
        return score

    def _forced_repositioning(self, my_pos: tuple) -> Move:
        """Force movement when stayed too long."""
        best_move = None
        best_mobility = -1

        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dr, dc = move.value
            next_pos = (my_pos[0] + dr, my_pos[1] + dc)

            if not self._is_safe_in_belief(next_pos):
                continue

            mobility = self._count_safe_neighbors(next_pos)
            if mobility > best_mobility and next_pos not in self.recent_positions[-3:]:
                best_mobility = mobility
                best_move = move

        return best_move if best_move else Move.STAY

    def _find_safer_position(self, my_pos: tuple) -> Move:
        """Find a safer adjacent position."""
        best_move = Move.STAY
        best_safety = self._count_safe_neighbors(my_pos)

        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dr, dc = move.value
            next_pos = (my_pos[0] + dr, my_pos[1] + dc)

            if not self._is_safe_in_belief(next_pos):
                continue

            safety = self._count_safe_neighbors(next_pos)
            if safety > best_safety and next_pos not in self.recent_positions[-5:]:
                best_safety = safety
                best_move = move

        return best_move

    def _active_patrol(self, my_pos: tuple) -> Move:
        """
        Active patrol mode - keep moving to maintain unpredictability.
        Prefer high-mobility areas far from edges.
        """
        candidate_moves = []

        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dr, dc = move.value
            next_pos = (my_pos[0] + dr, my_pos[1] + dc)

            if not self._is_safe_in_belief(next_pos):
                continue

            score = 0.0

            # 1. Strongly prefer unvisited positions
            if next_pos not in self.recent_positions[-10:]:
                score += 200
            elif next_pos not in self.recent_positions[-5:]:
                score += 100
            else:
                score += 50  # Still consider recently visited

            # 2. Prefer high mobility positions (more escape routes)
            mobility = self._count_safe_neighbors(next_pos)
            score += mobility * 40

            # 3. Prefer central positions (avoid edges)
            centrality = self._evaluate_centrality(next_pos)
            score += centrality * 20

            # 4. Avoid frontier cells (don't reveal unexplored areas)
            if self._is_frontier_cell(next_pos):
                score -= 80

            # 5. Add randomness for unpredictability
            score += random.random() * 30

            candidate_moves.append((score, move, next_pos))

        if candidate_moves:
            # Sort and pick best move
            candidate_moves.sort(reverse=True, key=lambda x: x[0])
            best_score, best_move, best_pos = candidate_moves[0]
            return best_move

        # Ultimate fallback - stay if no valid moves
        return Move.STAY

    def _is_safe_in_belief(self, pos: tuple) -> bool:
        """Check if position is confirmed safe (not wall, not unknown)."""
        if not self._in_bounds(pos[0], pos[1]):
            return False
        return self.belief_map[pos[0], pos[1]] == 0

    def _manhattan_distance(self, pos1: tuple, pos2: tuple) -> int:
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _in_bounds(self, row: int, col: int) -> bool:
        """Check if position is within map bounds."""
        return 0 <= row < self.map_shape[0] and 0 <= col < self.map_shape[1]
