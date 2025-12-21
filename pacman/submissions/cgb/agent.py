"""
Advanced Agent Implementation with Bayesian Estimation and Smart Tactics

Features:
- Ghost: Bayesian threat tracking, safe zone identification, strategic evasion
- Pacman: Pattern recognition, cut-off strategy, intelligent pursuit
"""

import sys
from pathlib import Path
import numpy as np
from collections import deque
import time

src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move

UNKNOWN = -1
EMPTY = 0
WALL = 1


# =====================================================
# GHOST AGENT (HIDE) - PROACTIVE Strategy
# =====================================================
from collections import deque
import numpy as np

class GhostAgent(BaseGhostAgent):
    """
    Ghost (Hider) - PROACTIVE hiding strategy
    
    KEY DIFFERENCE FROM REACTIVE:
    1. Maintains a GOAL (safe zone to reach)
    2. Plans PATHS in advance, not just next move
    3. Only replans when goal becomes unsafe
    4. Uses "potential field" for long-term positioning
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.global_map = np.full((21, 21), UNKNOWN, dtype=int)
        
        # Pacman tracking
        self.last_seen_pacman = None
        self.steps_since_seen = 0
        
        # PROACTIVE components
        self.current_goal = None  # Target safe position
        self.planned_path = []    # Committed path to goal
        self.path_step = 0        # Progress on path
        self.goal_validity_steps = 0  # How long goal has been valid
        
        # Strategic map (updated infrequently)
        self.safe_zones = []      # Pre-computed safe areas
        self.danger_zones = []    # Pre-computed danger areas
        self.last_strategy_update = -100

    # =====================================================
    # MAIN STEP - PROACTIVE DECISION
    # =====================================================
    def step(self, map_state, my_pos, enemy_pos, step_number):
        
        self._update_global_map(map_state)
        
        # Update strategic map periodically (not every step)
        if step_number - self.last_strategy_update > 20:
            self._update_strategic_map()
            self.last_strategy_update = step_number
        
        # Update Pacman tracking
        if enemy_pos is not None:
            self.last_seen_pacman = enemy_pos
            self.steps_since_seen = 0
        else:
            self.steps_since_seen += 1
        
        # === PROACTIVE DECISION LOGIC ===
        
        # 1. Check if current goal is still valid
        goal_valid = self._is_goal_still_valid(my_pos, enemy_pos)
        
        # 2. If goal invalid OR no goal, create new goal
        if not goal_valid or self.current_goal is None:
            self._create_new_goal(my_pos, enemy_pos, step_number)
            self.goal_validity_steps = 0
        else:
            self.goal_validity_steps += 1
        
        # 3. Execute plan toward goal (not just next move!)
        move = self._execute_plan(my_pos, enemy_pos)
        
        return move

    # =====================================================
    # MAP UPDATE
    # =====================================================
    def _update_global_map(self, local_map):
        for i in range(21):
            for j in range(21):
                if local_map[i, j] != UNKNOWN:
                    self.global_map[i, j] = local_map[i, j]

    # =====================================================
    # STRATEGIC MAP (PROACTIVE PLANNING)
    # =====================================================
    def _update_strategic_map(self):
        """
        Pre-compute safe and danger zones
        This is PROACTIVE: plan before threat arrives
        """
        self.safe_zones = []
        self.danger_zones = []
        
        for x in range(21):
            for y in range(21):
                if self.global_map[x, y] != EMPTY:
                    continue
                
                pos = (x, y)
                safety_score = self._calculate_strategic_safety(pos)
                
                if safety_score > 50:
                    self.safe_zones.append((pos, safety_score))
                elif safety_score < 20:
                    self.danger_zones.append((pos, safety_score))
        
        # Sort by safety score
        self.safe_zones.sort(key=lambda x: x[1], reverse=True)
        self.danger_zones.sort(key=lambda x: x[1])
    
    def _calculate_strategic_safety(self, pos):
        """
        Calculate long-term safety of position
        NOT based on current Pacman position (proactive!)
        """
        x, y = pos
        score = 0
        
        # 1. Structural safety (independent of Pacman)
        exits = self._count_exits(pos)
        score += exits * 15
        
        if exits <= 1:
            score -= 100  # Dead ends always bad
        
        # 2. Wall coverage (occlusion potential)
        adjacent_walls = sum(
            1 for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]
            if self._is_wall((x+dx, y+dy))
        )
        score += adjacent_walls * 10
        
        # 3. Depth in map (avoid edges)
        depth = min(x, 20-x, y, 20-y)
        score += depth * 3
        
        # 4. Connectivity (can reach many places)
        reachable = len(self._flood_fill(pos, max_dist=4))
        score += reachable * 0.5
        
        # 5. Check for nearby corners (tactical advantage)
        nearby_corners = 0
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                check_pos = (x+dx, y+dy)
                if self._is_valid_pos(check_pos) and self._is_corner(check_pos):
                    nearby_corners += 1
        score += nearby_corners * 5
        
        # 6. Open space penalty (exposed)
        open_space = 0
        for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
            consecutive = 0
            for step in range(1, 6):
                check_pos = (x+dx*step, y+dy*step)
                if not self._is_walkable(check_pos):
                    break
                consecutive += 1
            open_space += consecutive
        score -= open_space * 2
        
        return score

    # =====================================================
    # GOAL MANAGEMENT (PROACTIVE)
    # =====================================================
    def _is_goal_still_valid(self, my_pos, pacman_pos):
        """
        Check if current goal is still safe
        PROACTIVE: Don't wait until we're in danger
        """
        if self.current_goal is None:
            return False
        
        # If we reached goal, it's "invalid" (need new goal)
        if my_pos == self.current_goal:
            return False
        
        # If Pacman visible and close to goal, goal is compromised
        if pacman_pos is not None:
            goal_dist_to_pacman = abs(self.current_goal[0] - pacman_pos[0]) + \
                                  abs(self.current_goal[1] - pacman_pos[1])
            
            if goal_dist_to_pacman < 4:
                return False  # Goal too close to threat
        
        # If goal in danger zone (based on last known Pacman)
        if self.last_seen_pacman is not None:
            predicted_pacman = self._predict_pacman_position()
            
            goal_dist_to_predicted = abs(self.current_goal[0] - predicted_pacman[0]) + \
                                     abs(self.current_goal[1] - predicted_pacman[1])
            
            if goal_dist_to_predicted < 5:
                return False
        
        # If path is blocked
        if self.planned_path:
            if not self._is_walkable(self.planned_path[0]):
                return False
        
        # Goal still valid if survived this long
        if self.goal_validity_steps > 15:
            return True
        
        return True
    
    def _create_new_goal(self, my_pos, pacman_pos, step_number):
        """
        Create new goal and plan path
        PROACTIVE: Choose goal based on strategic value, not just immediate safety
        """
        # Emergency mode: Pacman very close
        if pacman_pos is not None:
            dist_to_pacman = abs(my_pos[0] - pacman_pos[0]) + abs(my_pos[1] - pacman_pos[1])
            
            if dist_to_pacman <= 4:
                # EMERGENCY: Pick closest safe zone away from Pacman
                self.current_goal = self._find_emergency_goal(my_pos, pacman_pos)
            else:
                # TACTICAL: Pick best strategic safe zone
                self.current_goal = self._find_strategic_goal(my_pos, pacman_pos)
        else:
            # NO THREAT: Pick optimal long-term position
            self.current_goal = self._find_optimal_position(my_pos)
        
        # Plan path to goal
        if self.current_goal:
            self.planned_path = self._plan_path(my_pos, self.current_goal)
            self.path_step = 0
    
    def _find_emergency_goal(self, my_pos, pacman_pos):
        """Emergency: find closest safe position away from Pacman"""
        best_goal = None
        best_score = -1e9
        
        # Check safe zones
        for safe_pos, safety_score in self.safe_zones[:10]:
            dist_to_me = abs(safe_pos[0] - my_pos[0]) + abs(safe_pos[1] - my_pos[1])
            dist_to_pacman = abs(safe_pos[0] - pacman_pos[0]) + abs(safe_pos[1] - pacman_pos[1])
            
            # Want: close to me, far from Pacman
            score = dist_to_pacman * 20 - dist_to_me * 5 + safety_score
            
            # Must be away from Pacman
            if dist_to_pacman < 5:
                continue
            
            if score > best_score:
                best_score = score
                best_goal = safe_pos
        
        return best_goal if best_goal else my_pos
    
    def _find_strategic_goal(self, my_pos, pacman_pos):
        """Tactical: find best strategic position"""
        best_goal = None
        best_score = -1e9
        
        # Predict where Pacman will be
        predicted_pacman = self._predict_pacman_position()
        
        for safe_pos, safety_score in self.safe_zones[:15]:
            dist_to_me = abs(safe_pos[0] - my_pos[0]) + abs(safe_pos[1] - my_pos[1])
            dist_to_predicted = abs(safe_pos[0] - predicted_pacman[0]) + \
                               abs(safe_pos[1] - predicted_pacman[1])
            
            # Balance: strategic value, distance from predicted threat, reachability
            score = safety_score * 2 + dist_to_predicted * 10 - dist_to_me * 2
            
            # Don't go too far
            if dist_to_me > 10:
                score -= 50
            
            if score > best_score:
                best_score = score
                best_goal = safe_pos
        
        return best_goal if best_goal else my_pos
    
    def _find_optimal_position(self, my_pos):
        """No threat: find best long-term position"""
        if self.safe_zones:
            # Go to highest value safe zone within reasonable distance
            for safe_pos, safety_score in self.safe_zones[:20]:
                dist = abs(safe_pos[0] - my_pos[0]) + abs(safe_pos[1] - my_pos[1])
                if dist <= 8:
                    return safe_pos
            
            # Otherwise just pick best
            return self.safe_zones[0][0]
        
        return my_pos
    
    def _predict_pacman_position(self):
        """Predict Pacman's likely position"""
        if self.last_seen_pacman is None:
            return (10, 10)  # Center of map
        
        # Estimate based on time since seen
        # Pacman could move 2 tiles/step on straights
        max_movement = self.steps_since_seen * 2
        
        # Assume Pacman is searching (moving toward center or toward us)
        # Simple heuristic: stays near last seen
        return self.last_seen_pacman

    # =====================================================
    # PATH PLANNING & EXECUTION
    # =====================================================
    def _plan_path(self, start, goal):
        """Plan path from start to goal using BFS (returns list of positions, not moves)"""
        if start == goal:
            return []
        
        queue = deque([(start, [start])])
        visited = {start}
        
        max_iterations = 500
        iterations = 0
        
        while queue and iterations < max_iterations:
            iterations += 1
            current, path = queue.popleft()
            
            if current == goal:
                # Return path excluding start position
                return path[1:]
            
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                next_pos = self._apply_move(current, move)
                
                if not self._is_walkable(next_pos):
                    continue
                
                if next_pos not in visited:
                    visited.add(next_pos)
                    queue.append((next_pos, path + [next_pos]))
        
        return []
    
    def _execute_plan(self, my_pos, pacman_pos):
        """
        Execute planned path
        PROACTIVE: Follow plan unless emergency
        """
        # Emergency override: Pacman very close
        if pacman_pos is not None:
            dist = abs(my_pos[0] - pacman_pos[0]) + abs(my_pos[1] - pacman_pos[1])
            if dist <= 2:
                # EMERGENCY: Override plan
                return self._emergency_escape(my_pos, pacman_pos)
        
        # Follow planned path (path is list of positions)
        if self.planned_path and self.path_step < len(self.planned_path):
            next_pos = self.planned_path[self.path_step]
            
            # Verify move is still valid
            if self._is_walkable(next_pos):
                # Calculate move direction from current to next position
                move = self._get_move_to_position(my_pos, next_pos)
                
                if move is not None:
                    self.path_step += 1
                    return move
            
            # Path blocked or invalid, replan
            self.current_goal = None
            return Move.STAY
        
        # Reached goal or path exhausted
        return Move.STAY
    
    def _get_move_to_position(self, current_pos, target_pos):
        """Get move direction from current to target position"""
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        
        if dx == 1 and dy == 0:
            return Move.DOWN
        elif dx == -1 and dy == 0:
            return Move.UP
        elif dx == 0 and dy == 1:
            return Move.RIGHT
        elif dx == 0 and dy == -1:
            return Move.LEFT
        
        return None
    
    def _emergency_escape(self, my_pos, pacman_pos):
        """Emergency: immediate escape (reactive fallback)"""
        best_move = Move.STAY
        best_score = -1e9
        
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            new_pos = self._apply_move(my_pos, move)
            if not self._is_walkable(new_pos):
                continue
            
            # Distance from Pacman
            dist = abs(new_pos[0] - pacman_pos[0]) + abs(new_pos[1] - pacman_pos[1])
            
            # Exits
            exits = self._count_exits(new_pos)
            
            score = dist * 30 + exits * 10
            
            if exits <= 1:
                score -= 200
            
            # Prefer perpendicular to Pacman's line
            if new_pos[0] != pacman_pos[0] and new_pos[1] != pacman_pos[1]:
                score += 50
            
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move

    # =====================================================
    # HELPERS
    # =====================================================
    def _apply_move(self, pos, move):
        dx, dy = move.value
        return (pos[0] + dx, pos[1] + dy)

    def _is_walkable(self, pos):
        x, y = pos
        if not (0 <= x < 21 and 0 <= y < 21):
            return False
        return self.global_map[x, y] == EMPTY
    
    def _is_valid_pos(self, pos):
        x, y = pos
        return 0 <= x < 21 and 0 <= y < 21

    def _is_wall(self, pos):
        x, y = pos
        if not (0 <= x < 21 and 0 <= y < 21):
            return True
        return self.global_map[x, y] == WALL

    def _count_exits(self, pos):
        count = 0
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if self._is_walkable(self._apply_move(pos, move)):
                count += 1
        return count
    
    def _is_corner(self, pos):
        """Check if position is a corner"""
        if not self._is_walkable(pos):
            return False
        
        exits = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if self._is_walkable(self._apply_move(pos, move)):
                exits.append(move)
        
        if len(exits) != 2:
            return False
        
        # Check if perpendicular
        dx1, dy1 = exits[0].value
        dx2, dy2 = exits[1].value
        return (dx1 == 0 and dy2 == 0) or (dy1 == 0 and dx2 == 0)
    
    def _flood_fill(self, start, max_dist):
        """Flood fill to find reachable positions"""
        visited = set([start])
        queue = deque([(start, 0)])
        
        while queue:
            pos, dist = queue.popleft()
            if dist >= max_dist:
                continue
            
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                new_pos = self._apply_move(pos, move)
                if new_pos not in visited and self._is_walkable(new_pos):
                    visited.add(new_pos)
                    queue.append((new_pos, dist + 1))
        
        return visited

# =====================================================
# PACMAN AGENT (SEEK) - Fixed Oscillation Issues
# =====================================================
class PacmanAgent(BasePacmanAgent):

    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        
        # Memory
        self.global_map = np.full((21, 21), UNKNOWN, dtype=int)
        self.last_known_ghost = None
        self.ghost_history = []
        
        # Committed path and target
        self.committed_path = []
        self.target_position = None
        self.steps_on_current_path = 0
        self.min_commitment_steps = 3  # Minimum steps before reconsidering
        
    def step(self, map_state, my_position, enemy_position, step_number):
        """Main decision logic with single decision point"""
        
        # Update map
        self._update_global_map(map_state)
        
        # Update Ghost tracking
        ghost_moved_significantly = False
        if enemy_position is not None:
            if self.last_known_ghost is not None:
                dist = abs(enemy_position[0] - self.last_known_ghost[0]) + \
                       abs(enemy_position[1] - self.last_known_ghost[1])
                ghost_moved_significantly = dist > 3
            
            self.last_known_ghost = enemy_position
            self.ghost_history.append(enemy_position)
            if len(self.ghost_history) > 10:
                self.ghost_history.pop(0)
        
        # Check if we should replan
        should_replan = (
            not self.committed_path or  # No current path
            ghost_moved_significantly or  # Ghost moved significantly
            self.steps_on_current_path >= 10 or  # Path is stale
            (self.target_position and enemy_position and 
             self.target_position != enemy_position and
             self.steps_on_current_path >= self.min_commitment_steps)  # Target changed
        )
        
        # Continue on committed path if we shouldn't replan
        if not should_replan and self.committed_path:
            self.steps_on_current_path += 1
            return self._execute_committed_path(my_position)
        
        # Need to create new plan
        self.steps_on_current_path = 0
        
        if enemy_position is not None:
            # Direct pursuit
            self.target_position = enemy_position
            return self._create_pursuit_plan(my_position, enemy_position)
        
        if self.last_known_ghost is not None:
            # Search last known area
            self.target_position = self.last_known_ghost
            return self._create_search_plan(my_position, self.last_known_ghost)
        
        # Explore systematically
        self.target_position = None
        return self._create_exploration_plan(my_position)
    
    # ===== MAP UPDATE =====
    
    def _update_global_map(self, local_map):
        """Merge observation into global memory"""
        for i in range(21):
            for j in range(21):
                if local_map[i, j] != UNKNOWN:
                    self.global_map[i, j] = local_map[i, j]
    
    # ===== PLAN CREATION =====
    
    def _create_pursuit_plan(self, my_pos, ghost_pos):
        """Create committed pursuit plan"""
        path = self._astar(my_pos, ghost_pos)
        
        if path and len(path) > 1:
            self.committed_path = path[1:]  # Exclude current position
            return self._execute_committed_path(my_pos)
        
        # Fallback: greedy move
        return self._greedy_move(my_pos, ghost_pos)
    
    def _create_search_plan(self, my_pos, last_known):
        """Create search plan around last known position"""
        # Find best search target in area
        search_target = self._find_search_target(my_pos, last_known)
        
        if search_target:
            path = self._astar(my_pos, search_target)
            if path and len(path) > 1:
                self.committed_path = path[1:]
                return self._execute_committed_path(my_pos)
        
        return self._greedy_move(my_pos, last_known)
    
    def _create_exploration_plan(self, my_pos):
        """Create systematic exploration plan"""
        target = self._find_closest_unknown_cluster(my_pos)
        
        if target:
            path = self._astar(my_pos, target)
            if path and len(path) > 1:
                self.committed_path = path[1:]
                return self._execute_committed_path(my_pos)
        
        # Random valid move as fallback
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if self._is_valid_move(my_pos, move):
                return (move, 1)
        
        return (Move.STAY, 1)
    
    def _find_search_target(self, my_pos, last_known):
        """Find best position to search near last known ghost location"""
        search_radius = 6
        best_score = -1
        best_target = None
        
        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                x, y = last_known[0] + dx, last_known[1] + dy
                
                if not (0 <= x < 21 and 0 <= y < 21):
                    continue
                
                if self.global_map[x, y] != EMPTY:
                    continue
                
                # Score based on distance from last known and from Pacman
                dist_from_last = abs(dx) + abs(dy)
                dist_from_pacman = abs(x - my_pos[0]) + abs(y - my_pos[1])
                
                if dist_from_last > search_radius or dist_from_pacman == 0:
                    continue
                
                # Prefer positions slightly away from last known
                score = 10 - dist_from_last - dist_from_pacman * 0.1
                
                if score > best_score:
                    best_score = score
                    best_target = (x, y)
        
        return best_target
    
    def _find_closest_unknown_cluster(self, my_pos):
        """Find closest unknown area"""
        min_dist = float('inf')
        target = None
        
        for x in range(21):
            for y in range(21):
                if self.global_map[x, y] == UNKNOWN:
                    # Count unknown neighbors
                    unknown_neighbors = sum(
                        1 for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]
                        if 0 <= x+dx < 21 and 0 <= y+dy < 21
                        and self.global_map[x+dx, y+dy] == UNKNOWN
                    )
                    
                    if unknown_neighbors >= 2:
                        dist = abs(x - my_pos[0]) + abs(y - my_pos[1])
                        if dist < min_dist:
                            min_dist = dist
                            target = (x, y)
        
        return target
    
    # ===== PATH EXECUTION =====
    
    def _execute_committed_path(self, my_pos):
        """Execute committed path with proper multi-step handling"""
        
        if not self.committed_path:
            return (Move.STAY, 1)
        
        # Remove any positions we've already reached
        while self.committed_path and self.committed_path[0] == my_pos:
            self.committed_path.pop(0)
        
        if not self.committed_path:
            return (Move.STAY, 1)
        
        # Determine move direction from current position to next position
        next_pos = self.committed_path[0]
        dx = next_pos[0] - my_pos[0]
        dy = next_pos[1] - my_pos[1]
        
        # Find the move direction
        move_direction = None
        if dx > 0 and dy == 0:
            move_direction = Move.DOWN
        elif dx < 0 and dy == 0:
            move_direction = Move.UP
        elif dy > 0 and dx == 0:
            move_direction = Move.RIGHT
        elif dy < 0 and dx == 0:
            move_direction = Move.LEFT
        
        if move_direction is None:
            # Path is invalid, clear it
            self.committed_path = []
            return (Move.STAY, 1)
        
        # Count consecutive steps in same direction along path
        move_dx, move_dy = move_direction.value
        steps = 0
        current_pos = my_pos
        
        for i in range(min(len(self.committed_path), self.pacman_speed)):
            expected_next = (current_pos[0] + move_dx, current_pos[1] + move_dy)
            
            # Check if path continues in same direction
            if i < len(self.committed_path) and self.committed_path[i] == expected_next:
                if self._is_valid_position(expected_next):
                    steps += 1
                    current_pos = expected_next
                else:
                    break
            else:
                break
        
        steps = max(1, steps)
        
        # Remove consumed positions from path
        self.committed_path = self.committed_path[steps:]
        
        return (move_direction, steps)
    
    # ===== PATHFINDING =====
    
    def _astar(self, start, goal):
        """A* pathfinding"""
        from heapq import heappush, heappop
        
        frontier = []
        heappush(frontier, (0, start, [start]))
        visited = {start: 0}
        
        max_iterations = 500  # Prevent infinite loops
        iterations = 0
        
        while frontier and iterations < max_iterations:
            iterations += 1
            f_cost, current, path = heappop(frontier)
            
            if current == goal:
                return path
            
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dx, dy = move.value
                neighbor = (current[0] + dx, current[1] + dy)
                
                if not self._is_valid_position(neighbor):
                    continue
                
                g_cost = len(path)
                
                if neighbor not in visited or g_cost < visited[neighbor]:
                    visited[neighbor] = g_cost
                    h_cost = abs(neighbor[0] - goal[0]) + abs(neighbor[1] - goal[1])
                    f = g_cost + h_cost
                    heappush(frontier, (f, neighbor, path + [neighbor]))
        
        return []
    
    def _greedy_move(self, start, goal):
        """Greedy movement towards goal"""
        dx = goal[0] - start[0]
        dy = goal[1] - start[1]
        
        # Prefer the axis with larger distance
        moves_to_try = []
        if abs(dx) > abs(dy):
            moves_to_try = [
                Move.DOWN if dx > 0 else Move.UP,
                Move.RIGHT if dy > 0 else Move.LEFT
            ]
        else:
            moves_to_try = [
                Move.RIGHT if dy > 0 else Move.LEFT,
                Move.DOWN if dx > 0 else Move.UP
            ]
        
        # Try preferred moves first
        for move in moves_to_try:
            steps = self._max_valid_steps(start, move, self.pacman_speed)
            if steps > 0:
                return (move, steps)
        
        # Try all other moves
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if move not in moves_to_try:
                steps = self._max_valid_steps(start, move, self.pacman_speed)
                if steps > 0:
                    return (move, steps)
        
        return (Move.STAY, 1)
    
    # ===== HELPERS =====
    
    def _is_valid_move(self, pos, move):
        """Check if move is valid"""
        dx, dy = move.value
        new_pos = (pos[0] + dx, pos[1] + dy)
        return self._is_valid_position(new_pos)
    
    def _is_valid_position(self, pos):
        """Check if position is valid"""
        x, y = pos
        if not (0 <= x < 21 and 0 <= y < 21):
            return False
        return self.global_map[x, y] == EMPTY
    
    def _max_valid_steps(self, pos, move, max_steps):
        """Calculate maximum valid steps in direction"""
        steps = 0
        current = pos
        dx, dy = move.value
        
        for _ in range(max_steps):
            next_pos = (current[0] + dx, current[1] + dy)
            if not self._is_valid_position(next_pos):
                break
            steps += 1
            current = next_pos
        
        return steps