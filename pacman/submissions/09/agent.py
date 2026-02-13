import sys
from pathlib import Path
from collections import deque
from typing import Tuple, List, Set
import numpy as np
import random
from enum import Enum
import heapq

src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move

Pos = Tuple[int, int]

# =====================================================
# AgentUtils - Utilities for Ghost Agent
# =====================================================
class AgentUtils:
    @staticmethod
    def get_valid_moves(pos, map_state, grid_shape):
        """Trả về các nước đi hợp lệ từ vị trí pos."""
        h, w = grid_shape
        r, c = pos
        moves = []
        directions = [
            (Move.UP, (-1, 0)),
            (Move.DOWN, (1, 0)),
            (Move.LEFT, (0, -1)),
            (Move.RIGHT, (0, 1))
        ]
        
        for move_enum, (dr, dc) in directions:
            nr, nc = r + dr, c + dc
            # Check bounds and ensure cell is not a wall
            if 0 <= nr < h and 0 <= nc < w and map_state[nr, nc] != 1:
                moves.append((move_enum, (nr, nc)))
        return moves

    @staticmethod
    def get_safe_distance_map(enemy_pos, map_state, grid_shape, max_dist=15):
        """Tính khoảng cách từ mọi ô đến địch bằng BFS."""
        if enemy_pos is None:
            return None
        
        # Initialize distance map with -1
        dist_map = np.full(grid_shape, -1, dtype=int)
        queue = deque([(enemy_pos, 0)])
        dist_map[enemy_pos] = 0
        
        # BFS to calculate Manhattan distances from enemy
        while queue:
            pos, dist = queue.popleft()
            if dist >= max_dist:
                continue
                
            for _, next_pos in AgentUtils.get_valid_moves(pos, map_state, grid_shape):
                if dist_map[next_pos] == -1:
                    dist_map[next_pos] = dist + 1
                    queue.append((next_pos, dist + 1))
        
        return dist_map

# =====================================================
# FSM States for Pacman
# =====================================================
class PacmanState(Enum):
    # Define operational states for Pacman behavior logic
    HUNT = "hunt"                    # Actively chasing visible ghost
    INTERCEPT = "intercept"          # Predicting and intercepting ghost path
    SEARCH = "search"                # Systematic search when ghost is lost
    GUARD_CHOKEPOINT = "guard"       # Guard strategic positions
    RETREAT = "retreat"              # Emergency retreat from trap

# =====================================================
# Utilities
# =====================================================
class U:
    @staticmethod
    def manhattan(a: Pos, b: Pos) -> int:
        # Calculate standard Manhattan distance between two points
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    @staticmethod
    def is_free(p: Pos, grid: np.ndarray) -> bool:
        # Check if position is within bounds and walkable (0)
        return 0 <= p[0] < 21 and 0 <= p[1] < 21 and grid[p] == 0

    @staticmethod
    def apply(p: Pos, mv: Move) -> Pos:
        return (p[0] + mv.value[0], p[1] + mv.value[1])

    @staticmethod
    def neighbors(p: Pos, grid: np.ndarray) -> List[Pos]:
        # Get list of valid adjacent coordinates
        res = []
        for mv in (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT):
            np_ = U.apply(p, mv)
            if U.is_free(np_, grid):
                res.append(np_)
        return res

    @staticmethod
    def a_star(start: Pos, goal: Pos, grid: np.ndarray, max_steps: int = 100) -> List[Pos]:
        """A* pathfinding algorithm"""
        from heapq import heappush, heappop
        
        if start == goal:
            return [start]
        
        open_set = []
        heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: U.manhattan(start, goal)}
        
        steps = 0
        while open_set and steps < max_steps:
            steps += 1
            _, current = heappop(open_set)
            
            # Reconstruct path if goal is reached
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return list(reversed(path))
            
            for neighbor in U.neighbors(current, grid):
                tentative_g = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + U.manhattan(neighbor, goal)
                    heappush(open_set, (f_score[neighbor], neighbor))
        
        return [start]

    @staticmethod
    def is_dead_end(p: Pos, grid: np.ndarray, depth: int = 3) -> bool:
        """Check if position leads to a dead end"""
        visited = {p}
        q = deque([(p, 0)])
        
        while q:
            pos, dist = q.popleft()
            if dist >= depth:
                return False
            
            # If junction found, it is not a dead end
            neighbors = U.neighbors(pos, grid)
            if len(neighbors) > 2:
                return False
            
            for nb in neighbors:
                if nb not in visited:
                    visited.add(nb)
                    q.append((nb, dist + 1))
        
        return True

    @staticmethod
    def get_move_towards(from_pos: Pos, to_pos: Pos, grid: np.ndarray) -> Move:
        """Get the best move to go from from_pos towards to_pos"""
        best_mv = None
        best_dist = float('inf')
        
        # Greedy check for immediate move decreasing distance
        for mv in (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT):
            np_ = U.apply(from_pos, mv)
            if U.is_free(np_, grid):
                dist = U.manhattan(np_, to_pos)
                if dist < best_dist:
                    best_dist = dist
                    best_mv = mv
        
        return best_mv if best_mv else Move.UP

    @staticmethod
    def predict_ghost_position(ghost_pos: Pos, ghost_history: deque, pacman_pos: Pos, grid: np.ndarray, steps_ahead: int = 3) -> Pos:
        """Predict where ghost will be in N steps - ghost tries to escape"""
        if not ghost_pos:
            return None
        
        predicted = ghost_pos
        
        # Simulate ghost logic maximizing distance from Pacman
        for _ in range(steps_ahead):
            best_move_pos = predicted
            best_score = -1e9
            
            for nb in U.neighbors(predicted, grid):
                # Ghost wants to maximize distance and mobility
                score = U.manhattan(nb, pacman_pos) * 10
                score += len(U.neighbors(nb, grid)) * 3
                
                # Avoid dead ends
                if U.is_dead_end(nb, grid):
                    score -= 100
                
                if score > best_score:
                    best_score = score
                    best_move_pos = nb
            
            predicted = best_move_pos
        
        return predicted

    @staticmethod
    def find_interception_point(pacman_pos: Pos, ghost_pos: Pos, ghost_history: deque, grid: np.ndarray) -> Pos:
        """Find the best point to intercept the ghost"""
        # Predict ghost movement
        predicted_ghost = U.predict_ghost_position(ghost_pos, ghost_history, pacman_pos, grid, steps_ahead=4)
        
        if not predicted_ghost:
            return ghost_pos
        
        # Calculate weighted midpoint between current and predicted ghost pos
        intercept_x = int(ghost_pos[0] * 0.3 + predicted_ghost[0] * 0.7)
        intercept_y = int(ghost_pos[1] * 0.3 + predicted_ghost[1] * 0.7)
        intercept = (intercept_x, intercept_y)
        
        # Ensure intercept point is valid; otherwise find nearest neighbor
        if not U.is_free(intercept, grid):
            min_dist = float('inf')
            best_pos = predicted_ghost
            
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    pos = (intercept_x + dx, intercept_y + dy)
                    if U.is_free(pos, grid):
                        dist = U.manhattan(pos, intercept)
                        if dist < min_dist:
                            min_dist = dist
                            best_pos = pos
            
            intercept = best_pos
        
        return intercept

    @staticmethod
    def find_cutoff_positions(ghost_pos: Pos, pacman_pos: Pos, grid: np.ndarray) -> List[Pos]:
        """Find positions to cut off ghost's escape routes"""
        cutoff_positions = []
        
        # Find all positions that would block ghost's best escape paths
        ghost_neighbors = U.neighbors(ghost_pos, grid)
        
        for nb in ghost_neighbors:
            # Check positions that are farther from pacman (escape direction)
            if U.manhattan(nb, pacman_pos) > U.manhattan(ghost_pos, pacman_pos):
                # Find positions between this escape route and pacman
                path = U.a_star(pacman_pos, nb, grid, max_steps=30)
                if len(path) > 2:
                    cutoff_positions.append(path[len(path)//2])
        
        return cutoff_positions

    @staticmethod
    def is_ghost_trapped(ghost_pos: Pos, pacman_pos: Pos, grid: np.ndarray) -> bool:
        """Check if ghost is in a trappable position"""
        escape_routes = 0
        
        for nb in U.neighbors(ghost_pos, grid):
            # Count positions that lead away from pacman and are not dead ends
            if U.manhattan(nb, pacman_pos) >= U.manhattan(ghost_pos, pacman_pos):
                if not U.is_dead_end(nb, grid, depth=4):
                    escape_routes += 1
        
        return escape_routes <= 1

    @staticmethod
    def find_exploration_target(current_pos: Pos, grid: np.ndarray, visited_recently: Set[Pos]) -> Pos:
        """Find a good exploration target when ghost location is unknown"""
        best_target = current_pos
        best_score = -1e9
        
        visited = {current_pos}
        q = deque([(current_pos, 0)])
        max_depth = 40
        
        # BFS search for highest value exploration node
        while q:
            pos, depth = q.popleft()
            if depth > max_depth:
                break
            
            score = 0
            if pos not in visited_recently:
                score += 50
            score += len(U.neighbors(pos, grid)) * 5
            score -= depth  # Prefer closer targets
            
            # Avoid dead ends
            if U.is_dead_end(pos, grid):
                score -= 100
            
            if score > best_score:
                best_score = score
                best_target = pos
            
            for nb in U.neighbors(pos, grid):
                if nb not in visited:
                    visited.add(nb)
                    q.append((nb, depth + 1))
        
        return best_target

    @staticmethod
    def find_priority_zones(grid: np.ndarray) -> List[Pos]:
        """Find priority zones: junctions and high-mobility areas where ghost might be"""
        priority_zones = []
        
        for x in range(21):
            for y in range(21):
                pos = (x, y)
                if not U.is_free(pos, grid):
                    continue
                
                # Junctions (3+ neighbors) are good hiding spots
                neighbors = U.neighbors(pos, grid)
                if len(neighbors) >= 3:
                    priority_zones.append(pos)
        
        return priority_zones

    @staticmethod
    def get_spiral_search_path(center: Pos, grid: np.ndarray, max_radius: int = 15) -> List[Pos]:
        """Generate a spiral search pattern from center position"""
        search_path = []
        visited = set()
        
        # Spiral outward from center
        for radius in range(1, max_radius + 1):
            # Check positions at this radius
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    # Only check positions roughly at this radius (ring-based)
                    if abs(dx) == radius or abs(dy) == radius:
                        pos = (center[0] + dx, center[1] + dy)
                        if pos not in visited and U.is_free(pos, grid):
                            visited.add(pos)
                            # Prioritize junctions in spiral
                            if len(U.neighbors(pos, grid)) >= 3:
                                search_path.insert(0, pos)  # High priority
                            else:
                                search_path.append(pos)
        
        return search_path

    @staticmethod
    def find_chokepoints(grid: np.ndarray) -> List[Pos]:
        """Find strategic chokepoints - narrow passages where ghost must pass"""
        chokepoints = []
        
        for x in range(1, 20):
            for y in range(1, 20):
                pos = (x, y)
                if not U.is_free(pos, grid):
                    continue
                
                neighbors = U.neighbors(pos, grid)
                # Chokepoint: exactly 2 neighbors and they're opposite
                if len(neighbors) == 2:
                    dx = abs(neighbors[0][0] - neighbors[1][0])
                    dy = abs(neighbors[0][1] - neighbors[1][1])
                    # If neighbors are opposite (corridor)
                    if dx == 2 or dy == 2:
                        chokepoints.append(pos)
        
        return chokepoints

    @staticmethod
    def get_corner_cut_move(current: Pos, target: Pos, grid: np.ndarray) -> Pos:
        """Get position that cuts corner to intercept target faster"""
        # Try to move diagonally in grid terms (combined horizontal + vertical)
        dx = 1 if target[0] > current[0] else (-1 if target[0] < current[0] else 0)
        dy = 1 if target[1] > current[1] else (-1 if target[1] < current[1] else 0)
        
        # Try diagonal-like movement (prefer moving in both directions)
        if dx != 0 and dy != 0:
            # Try horizontal first
            h_pos = (current[0] + dx, current[1])
            if U.is_free(h_pos, grid):
                # Check if next vertical move is also free (corner cutting)
                v_next = (h_pos[0], h_pos[1] + dy)
                if U.is_free(v_next, grid):
                    return h_pos
            
            # Try vertical first
            v_pos = (current[0], current[1] + dy)
            if U.is_free(v_pos, grid):
                h_next = (v_pos[0] + dx, v_pos[1])
                if U.is_free(h_next, grid):
                    return v_pos
        
        # Fallback to A*
        path = U.a_star(current, target, grid, max_steps=20)
        return path[1] if len(path) > 1 else current

    @staticmethod
    def analyze_ghost_velocity(ghost_history: deque) -> Tuple[float, float]:
        """Analyze ghost's movement velocity (dx/dt, dy/dt)"""
        if len(ghost_history) < 2:
            return (0.0, 0.0)
        
        # Calculate average velocity over recent history
        velocities = []
        for i in range(1, min(5, len(ghost_history))):
            dx = ghost_history[-i][0] - ghost_history[-i-1][0]
            dy = ghost_history[-i][1] - ghost_history[-i-1][1]
            velocities.append((dx, dy))
        
        if not velocities:
            return (0.0, 0.0)
        
        avg_vx = sum(v[0] for v in velocities) / len(velocities)
        avg_vy = sum(v[1] for v in velocities) / len(velocities)
        
        return (avg_vx, avg_vy)

    @staticmethod
    def predict_with_velocity(pos: Pos, velocity: Tuple[float, float], steps: int, grid: np.ndarray) -> Pos:
        """Predict position using velocity vector"""
        predicted_x = int(pos[0] + velocity[0] * steps)
        predicted_y = int(pos[1] + velocity[1] * steps)
        predicted = (predicted_x, predicted_y)
        
        # Clamp to grid bounds
        predicted = (max(0, min(20, predicted[0])), max(0, min(20, predicted[1])))
        
        # If predicted position is not free, find nearest free position
        if not U.is_free(predicted, grid):
            min_dist = float('inf')
            best = pos
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    test_pos = (predicted[0] + dx, predicted[1] + dy)
                    if U.is_free(test_pos, grid):
                        dist = U.manhattan(test_pos, predicted)
                        if dist < min_dist:
                            min_dist = dist
                            best = test_pos
            predicted = best
        
        return predicted

    @staticmethod
    def predict_pacman_path(pacman_pos: Pos, ghost_pos: Pos, grid: np.ndarray) -> List[Pos]:
        """Predict Pacman's likely path using A* (Pacman uses A* to chase)"""
        # Pacman will use A* to chase ghost, so we can predict it
        path = U.a_star(pacman_pos, ghost_pos, grid, max_steps=30)
        return path if len(path) > 1 else [pacman_pos]

    @staticmethod
    def find_escape_routes(ghost_pos: Pos, pacman_pos: Pos, grid: np.ndarray, count: int = 3) -> List[Pos]:
        """Find multiple escape routes away from Pacman"""
        routes = []
        
        # BFS to find distant positions in different directions
        visited = {ghost_pos}
        q = deque([(ghost_pos, 0)])
        candidates = []
        
        while q and len(candidates) < count * 5:
            pos, depth = q.popleft()
            if depth > 15:
                break
            
            if depth >= 8:  # Far enough to be escape target
                dist = U.manhattan(pos, pacman_pos)
                mob = len(U.neighbors(pos, grid))
                # Score: far from Pacman + high mobility
                score = dist * 10 + mob * 5
                if not U.is_dead_end(pos, grid):
                    score += 50
                candidates.append((score, pos))
            
            for nb in U.neighbors(pos, grid):
                if nb not in visited:
                    visited.add(nb)
                    q.append((nb, depth + 1))
        
        # Sort by score and pick diverse routes (different directions)
        candidates.sort(reverse=True)
        for score, pos in candidates:
            if len(routes) >= count:
                break
            # Check if this route is in a different direction from existing routes
            is_diverse = True
            for existing in routes:
                if U.manhattan(pos, existing) < 5:  # Too close to existing
                    is_diverse = False
                    break
            if is_diverse:
                routes.append(pos)
        
        return routes if routes else [ghost_pos]

    @staticmethod
    def get_perpendicular_moves(from_pos: Pos, to_pos: Pos, grid: np.ndarray) -> List[Move]:
        """Get moves perpendicular to direct path (for zigzag)"""
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        
        perp_moves = []
        # If moving horizontally, try vertical; if vertical, try horizontal
        if abs(dx) > abs(dy):  # Mostly horizontal
            for mv in [Move.UP, Move.DOWN]:
                np_ = U.apply(from_pos, mv)
                if U.is_free(np_, grid):
                    perp_moves.append(mv)
        else:  # Mostly vertical
            for mv in [Move.LEFT, Move.RIGHT]:
                np_ = U.apply(from_pos, mv)
                if U.is_free(np_, grid):
                    perp_moves.append(mv)
        
        return perp_moves


# =====================================================
# PACMAN – NON-CAPTURABLE AWARE FINAL
# =====================================================
class PacmanAgent(BasePacmanAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Pacman Master Hunter"

        self.speed = max(1, int(kwargs.get("pacman_speed", 1)))
        # Initialize memory with -1 (unseen)
        self.memory = np.full((21, 21), -1, dtype=int)

        # Track ghost belief using set of possible positions
        self.ghost_belief: Set[Pos] = set()
        self.last_seen_ghost = None
        self.ghost_history = deque(maxlen=10)

        # Stuck detection variables
        self.last_move = None
        self.dist_history = deque(maxlen=15)
        self.stuck_counter = 0
        self.last_pos = None
        
        # Strategy tracking
        self.pursuit_mode = "aggressive"  # aggressive, intercept, trap
        self.failed_pursuits = 0
        self.visited_positions = deque(maxlen=50)  # Track recent positions
        
        # Search strategy
        self.search_coverage = {}  # Map position -> last_step_searched
        self.search_pattern = []  # Current search path
        self.search_pattern_index = 0
        self.steps_since_ghost_seen = 0
        self.priority_zones = []  # High-value search targets
        self.chokepoints = []  # Strategic narrow passages
        
        # Ghost behavior analysis
        self.ghost_velocity = (0.0, 0.0)  # Track ghost movement velocity
        self.consecutive_approaches = 0  # Track how many steps we're getting closer
        
        # Advanced strategies
        self.ghost_at_junction = False  # Track if ghost is at junction
        self.ghost_preferred_routes = []  # Track ghost's escape preferences
        self.trap_attempts = 0  # Count trap forcing attempts

    def _update_memory(self, obs):
        # Update internal map with visible area
        self.memory[obs != -1] = obs[obs != -1]

    def _update_belief(self, enemy_pos):
        # If enemy visible, collapse belief to single point
        if enemy_pos is not None:
            self.ghost_belief = {enemy_pos}
            self.last_seen_ghost = enemy_pos
            self.ghost_history.append(enemy_pos)
            
            # Update ghost velocity analysis
            self.ghost_velocity = U.analyze_ghost_velocity(self.ghost_history)
            return

        # Keep using last seen ghost for longer if belief becomes empty
        if not self.ghost_belief and self.last_seen_ghost:
            self.ghost_belief = {self.last_seen_ghost}

        # Expand belief based on possible ghost movements
        new = set()
        for g in self.ghost_belief:
            neighbors = U.neighbors(g, self.memory)
            if len(neighbors) > 0:
                new.update(neighbors)
        
        # If belief expansion fails, keep last known position
        if not new and self.last_seen_ghost:
            new = {self.last_seen_ghost}
        
        # Limit belief set size for performance
        if len(new) > 25:
            # Keep closest positions to last known
            if self.last_seen_ghost:
                new = set(sorted(new, key=lambda p: U.manhattan(p, self.last_seen_ghost))[:25])
        
        self.ghost_belief = new if new else self.ghost_belief

    def _mobility(self, p):
        return len(U.neighbors(p, self.memory))

    # ---------------- RETREAT TARGET ----------------
    def _best_open_cell(self, pos, ghost_pos=None):
        """Find safest open position - high mobility and far from ghost"""
        best, best_score = pos, -1e9
        q = deque([pos])
        visited = {pos}
        max_search = 50

        while q and len(visited) < max_search:
            cur = q.popleft()
            
            # Skip dead ends
            if U.is_dead_end(cur, self.memory):
                continue
            
            mob = self._mobility(cur)
            score = mob * 10
            
            # Prefer positions far from ghost
            if ghost_pos:
                dist_to_ghost = U.manhattan(cur, ghost_pos)
                score += dist_to_ghost * 5
            
            if score > best_score:
                best, best_score = cur, score
            
            for nb in U.neighbors(cur, self.memory):
                if nb not in visited:
                    visited.add(nb)
                    q.append(nb)
        
        return best

    # ---------------- SPEED ----------------
    def _apply_speed(self, mv, pos):
        # Apply speed boost if agent settings allow (>1)
        if self.speed <= 1:
            return mv

        # Adaptive speed: use double speed when pursuing and path is clear
        steps = 2 if mv == self.last_move else 1
        
        # Check if double speed is safe (no walls immediately ahead)
        if steps == 2:
            test_pos = pos
            for _ in range(2):
                test_pos = U.apply(test_pos, mv)
                if not U.is_free(test_pos, self.memory):
                    steps = 1
                    break
        
        cur = pos
        real = 0
        for _ in range(steps):
            nxt = U.apply(cur, mv)
            if not U.is_free(nxt, self.memory):
                break
            cur = nxt
            real += 1

        self.last_move = mv if real > 0 else None
        return (mv, real) if real > 1 else mv

    def step(self, obs, my_pos, enemy_pos, step_number):
        self._update_memory(obs)
        self._update_belief(enemy_pos)
        
        self.visited_positions.append(my_pos)
        self.search_coverage[my_pos] = step_number
        
        # Track time since last saw ghost
        if enemy_pos is not None:
            self.steps_since_ghost_seen = 0
        else:
            self.steps_since_ghost_seen += 1

        # Detect if stuck
        if my_pos == self.last_pos:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        self.last_pos = my_pos

        ghost_guess = None
        if self.ghost_belief:
            # Estimate ghost position based on belief
            ghost_guess = min(self.ghost_belief, key=lambda g: U.manhattan(my_pos, g))
            current_dist = U.manhattan(my_pos, ghost_guess)
            self.dist_history.append(current_dist)
            
            # Track if we're successfully approaching
            if len(self.dist_history) >= 2:
                if current_dist < self.dist_history[-2]:
                    self.consecutive_approaches += 1
                else:
                    self.consecutive_approaches = max(0, self.consecutive_approaches - 1)
            
            # Analyze ghost position
            ghost_mobility = len(U.neighbors(ghost_guess, self.memory))
            self.ghost_at_junction = (ghost_mobility >= 3)

        if self.stuck_counter >= 8:
            # Force ANY different move if stuck for too long
            for mv in (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT):
                if mv != self.last_move:
                    np_ = U.apply(my_pos, mv)
                    if U.is_free(np_, self.memory):
                        return self._apply_speed(mv, my_pos)


        
        # =================================================
        # SYSTEMATIC SEARCH MODE - When ghost is lost
        # =================================================
        if not ghost_guess or self.steps_since_ghost_seen > 20:
            # Initialize priority zones if not done
            if not self.priority_zones:
                self.priority_zones = U.find_priority_zones(self.memory)
            
            # Generate new search pattern if needed
            if not self.search_pattern or self.search_pattern_index >= len(self.search_pattern):
                # Use spiral search from last known position or center
                search_center = self.last_seen_ghost if self.last_seen_ghost else (10, 10)
                self.search_pattern = U.get_spiral_search_path(search_center, self.memory)
                
                # Filter out recently searched areas
                current_time = step_number
                self.search_pattern = [
                    pos for pos in self.search_pattern 
                    if pos not in self.search_coverage or 
                    (current_time - self.search_coverage[pos]) > 30
                ]
                
                # Add priority zones to beginning
                unsearched_priority = [
                    pos for pos in self.priority_zones
                    if pos not in self.search_coverage or
                    (current_time - self.search_coverage[pos]) > 20
                ]
                self.search_pattern = unsearched_priority + self.search_pattern
                self.search_pattern_index = 0
            
            if self.search_pattern and self.search_pattern_index < len(self.search_pattern):
                target = self.search_pattern[self.search_pattern_index]
                
                if U.manhattan(my_pos, target) <= 2:
                    self.search_pattern_index += 1
                    if self.search_pattern_index < len(self.search_pattern):
                        target = self.search_pattern[self.search_pattern_index]
                
                path = U.a_star(my_pos, target, self.memory, max_steps=80)
                if len(path) > 1:
                    next_pos = path[1]
                    for mv in (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT):
                        if U.apply(my_pos, mv) == next_pos:
                            return self._apply_speed(mv, my_pos)
        
        # =================================================
        # HUNT MODE - Choose strategy based on distance
        # =================================================
        if ghost_guess:
            dist_to_ghost = U.manhattan(my_pos, ghost_guess)
            target = ghost_guess
            use_intercept = False
            
            # STRATEGY 1: Speed Burst (Close Range)
            if dist_to_ghost <= 3:
                if self.ghost_velocity != (0.0, 0.0):
                    target = U.predict_with_velocity(ghost_guess, self.ghost_velocity, 1, self.memory)
            
            # STRATEGY 2: Cut-off at junctions (Mid Range)
            elif 4 <= dist_to_ghost <= 10 and self.ghost_at_junction:
                ghost_neighbors = U.neighbors(ghost_guess, self.memory)
                
                best_ghost_escape = None
                best_ghost_score = -1e9
                
                for g_nb in ghost_neighbors:
                    g_score = U.manhattan(g_nb, my_pos) * 100  
                    g_score += len(U.neighbors(g_nb, self.memory)) * 50 
                    if U.is_dead_end(g_nb, self.memory):
                        g_score -= 500
                    
                    if g_score > best_ghost_score:
                        best_ghost_score = g_score
                        best_ghost_escape = g_nb
                
                # Try to cut off that escape
                if best_ghost_escape:
                    # Position ourselves between ghost and escape
                    intercept_x = int(ghost_guess[0] * 0.4 + best_ghost_escape[0] * 0.6)
                    intercept_y = int(ghost_guess[1] * 0.4 + best_ghost_escape[1] * 0.6)
                    target = (intercept_x, intercept_y)
                    
                    if not U.is_free(target, self.memory):
                        target = best_ghost_escape
                    
                    use_intercept = True
            
            # STRATEGY 3: Trap or Predict (Long Range)
            elif dist_to_ghost > 10:
                dead_ends = []
                for dx in range(-8, 9):
                    for dy in range(-8, 9):
                        test_pos = (ghost_guess[0] + dx, ghost_guess[1] + dy)
                        if U.is_free(test_pos, self.memory) and U.is_dead_end(test_pos, self.memory):
                            dead_ends.append(test_pos)
                
                if dead_ends and self.trap_attempts < 3:
                    # Try to push ghost toward nearest dead end
                    nearest_trap = min(dead_ends, key=lambda p: U.manhattan(ghost_guess, p))
                    
                    # Position to block escapes AWAY from trap
                    push_x = int(ghost_guess[0] * 0.6 + nearest_trap[0] * 0.4)
                    push_y = int(ghost_guess[1] * 0.6 + nearest_trap[1] * 0.4)
                    target = (push_x, push_y)
                    
                    if U.is_free(target, self.memory):
                        use_intercept = True
                        self.trap_attempts += 1
                else:
                    # Direct prediction
                    if self.ghost_velocity != (0.0, 0.0):
                        target = U.predict_with_velocity(ghost_guess, self.ghost_velocity, 3, self.memory)
            
            # Execute movement
            path = U.a_star(my_pos, target, self.memory, max_steps=100)
            if len(path) > 1:
                next_pos = path[1]
                
                can_double_speed = False
                if len(path) > 2 and self.speed >= 2:
                    dir1 = (path[1][0] - my_pos[0], path[1][1] - my_pos[1])
                    dir2 = (path[2][0] - path[1][0], path[2][1] - path[1][1])
                    if dir1 == dir2: 
                        can_double_speed = True
                
                for mv in (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT):
                    if U.apply(my_pos, mv) == next_pos:
                        return self._apply_speed(mv, my_pos)
        
        # Fallback: Choose best move - ALWAYS move towards ghost if known
        best_mv, best_score = None, -1e9
        for mv in (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT):
            np_ = U.apply(my_pos, mv)
            if not U.is_free(np_, self.memory):
                continue
            
            score = self._mobility(np_) * 2
            
            if ghost_guess:
                score -= U.manhattan(np_, ghost_guess) * 10 
            else:
                # EXPLORATION MODE - move towards unexplored areas
                visited_set = set(self.visited_positions)
                if np_ not in visited_set:
                    score += 20
                else:
                    score -= 5
                
                if np_ not in self.search_coverage or \
                   (step_number - self.search_coverage.get(np_, 0)) > 25:
                    score += 15
                
                if len(U.neighbors(np_, self.memory)) >= 3:
                    score += 10
                
                if self.last_seen_ghost:
                    score -= U.manhattan(np_, self.last_seen_ghost) * 2
            
            if not ghost_guess and U.is_dead_end(np_, self.memory):
                score -= 30
            
            if mv == self.last_move:
                score += 3
            
            if self.stuck_counter >= 2 and mv != self.last_move:
                score += 15
            
            if score > best_score:
                best_score, best_mv = score, mv

        if best_mv:
            return self._apply_speed(best_mv, my_pos)

        for mv in (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT):
            if U.is_free(U.apply(my_pos, mv), self.memory):
                return mv

        return Move.UP

class GhostAgent(BaseGhostAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Survivalist Ghost"
        self.internal_map = None
        self.last_pacman_pos = None
        self.last_pacman_timestamp = 0
        self.escape_route = []
        self.safe_zones = []

    def _init_memory(self, shape):
        if self.internal_map is None:
            self.internal_map = np.full(shape, -1, dtype=int)

    def _find_safe_zones(self, map_state):
        """Tìm các khu vực an toàn: xa Pacman, nhiều lối thoát."""
        h, w = map_state.shape
        safe_zones = []
        
        for r in range(h):
            for c in range(w):
                if map_state[r, c] == 0:
                    exits = len(AgentUtils.get_valid_moves((r, c), map_state, map_state.shape))
                    
                    if exits >= 3:
                        dist_to_pacman = float('inf')
                        if self.last_pacman_pos:
                            dist_to_pacman = abs(r - self.last_pacman_pos[0]) + \
                                           abs(c - self.last_pacman_pos[1])
                        
                        if dist_to_pacman > 5:
                            safe_zones.append((dist_to_pacman, exits, (r, c)))
        
        safe_zones.sort(reverse=True)
        return [pos for _, _, pos in safe_zones[:3]]

    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int) -> Move:
        
        self._init_memory(map_state.shape)
        
        # Merge visible observations into internal memory
        visible_mask = (map_state != -1)
        self.internal_map[visible_mask] = map_state[visible_mask]
        
        if enemy_position:
            self.last_pacman_pos = enemy_position
            self.last_pacman_timestamp = step_number
        
        valid_moves = AgentUtils.get_valid_moves(my_position, self.internal_map, map_state.shape)
        if not valid_moves:
            return Move.STAY

        dist_map = None
        if self.last_pacman_pos:
            dist_map = AgentUtils.get_safe_distance_map(
                self.last_pacman_pos, 
                self.internal_map, 
                map_state.shape
            )

        # HEURISTIC SCORING SYSTEM
        best_score = -float('inf')
        best_move = Move.STAY
        
        # Evaluate each valid move using heuristics
        for move_enum, next_pos in valid_moves:
            score = 0
            
            if dist_map is not None and dist_map[next_pos] != -1:
                dist_to_pacman = dist_map[next_pos]
                
                # Heavily penalize getting too close to Pacman
                if enemy_position:  
                    score += dist_to_pacman * 100
                    
                    if dist_to_pacman <= 2:
                        score -= 10000
                    elif dist_to_pacman <= 4:
                        score -= 1000
                else:
                    score += dist_to_pacman * 30
            
            future_moves = AgentUtils.get_valid_moves(next_pos, self.internal_map, map_state.shape)
            num_exits = len(future_moves)
            
            # Penalize dead ends, reward high mobility
            if num_exits <= 1:  
                score -= 500
            else:
                score += num_exits * 50
            
            # Reward staying near walls (stealth)
            walls_nearby = 0
            r, c = next_pos
            h, w = map_state.shape
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < h and 0 <= nc < w and self.internal_map[nr, nc] == 1:
                    walls_nearby += 1
            
            score += walls_nearby * 15
            
            # Reward exploring unknown cells
            if self.internal_map[next_pos] == -1:
                score += 10
            
            # Penalize backtracking to immediate history
            if len(self.escape_route) > 0 and next_pos == self.escape_route[-1]:
                score -= 100

            if score > best_score:
                best_score = score
                best_move = move_enum
        
        self.escape_route.append(my_position)
        if len(self.escape_route) > 5:
            self.escape_route.pop(0)
                
        return best_move