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

from functools import cmp_to_key
from collections import deque


class PacmanAgent(BasePacmanAgent):
    """
    The Pacman agent uses a search strategy based on the potential frontier set
    and captures the Ghost through tracing.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the Pacman agent.
        Parameters:
        -   last_known_enemy_pos: the enemy’s position since the most recent sighting.

        -   recent_visited: list of positions recently visited.

        -   recent_visited_frontier: list of frontier positions recently visited.

        -   not_recently_explored_frontiers: list of frontier positions detected but not yet explored.

        -   last_seen_distance: distance to the Ghost since the most recent sighting.

        -   map: backup copy of the map.

        -   belief: belief map indicating the probability that the Ghost is at a given position.

        -   belief_active: flag to activate the use of “belief.”
        """
        super().__init__(**kwargs)
        self.name = "Pacman"
        self.last_known_enemy_pos = None

        self.recent_visited = deque([])
        self.recent_visited_frontier = deque([])
        self.not_recently_explored_frontiers: deque[tuple] = deque([])

        self.last_seen_distance = -1
        self.map = None
        self.belief = None
        self.belief_active = False
    
    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int):
        """
        Pacman primarily explores directions through the potential frontier set (the frontier values
        from which the most information can be extracted).
        
        When encountering a Ghost, Pacman will pursue it with full effort; if the Ghost is lost during
        the chase, Pacman relies on the most recent known position of the Ghost to infer and track.
        
        The frontier values are probed sequentially according to their information weight. Pacman
        will not revisit a frontier set within a short time; if no
        frontier set satisfies the condition, Pacman proceeds to unexplored frontier values before
        considering returning to previously visited ones.

        Args:
            map_state (np.ndarray): The current map layout.
            my_position (tuple): Pacman's current position (row, column).
            enemy_position (tuple): Ghost's current position (row, column).
            step_number (int): The current step number (maximum 200).

        Returns:
            Move: The next move Pacman should take. Returns Move.STAY if no valid path is found.
        """
        self.map = map_state

        if self.belief is None:
            self.belief = np.zeros(map_state.shape)

        if (self.recent_visited and self.recent_visited[-1] == enemy_position and self.last_known_enemy_pos == my_position):
            self.recent_visited.append(my_position)
            self.not_recently_explored_frontiers = deque([nef for nef in self.not_recently_explored_frontiers if nef != my_position])
            return (Move.STAY, 1)
        
        self.recent_visited.append(my_position)
        self.not_recently_explored_frontiers = deque([nef for nef in self.not_recently_explored_frontiers if nef != my_position])

        if (len(self.recent_visited) > 50):
            self.recent_visited.popleft()

        if enemy_position is not None:
            self.belief.fill(0)
            self.belief_active = False
            self.belief[enemy_position] = 100.0
            self.last_known_enemy_pos = enemy_position
            self.last_seen_distance = self._manhattan(my_position, enemy_position)

            path = self._bfs(my_position, enemy_position, map_state)
            return self._make_decision(path)
        
        if self.last_known_enemy_pos is not None:
            if not self.belief_active:
                self.belief.fill(0)
                self.belief[self.last_known_enemy_pos] = 1.0
                self.belief_active = True

            if self._manhattan(my_position, self.last_known_enemy_pos) != 0:
                path = self._bfs(my_position, self.last_known_enemy_pos, map_state)
                return self._make_decision(path)
            
            self.last_known_enemy_pos = None
            

        if self.belief_active:
            self._update_belief(self.belief, map_state)

        for i in range(len(self.recent_visited_frontier) - 1, max(len(self.recent_visited_frontier) - 31, -1), -1):
            map_state[self.recent_visited_frontier[i]] = 1
            
        frontiers = self._find_frontiers(map_state)
        frontiers.sort(key=lambda p: (self._score(p, map_state, my_position) + self._belief_gain(p, self.belief, map_state)), reverse=True)

        for i in range(len(frontiers)):
            distance = self._manhattan(my_position, frontiers[i])
            if self.last_seen_distance != -1 and distance >= self.last_seen_distance:
                continue
            path = self._bfs(my_position, frontiers[i], map_state)
            self.last_seen_distance = -1
            decision = self._make_decision(path)
            dx, dy = decision[0].value
            nx, ny = my_position[0] + dx, my_position[1] + dy
            
            if ((nx, ny) in self.recent_visited):
                continue
            for j in range(len(frontiers) - 1, i, -1):
                nef = frontiers[j]
                if nef not in self.not_recently_explored_frontiers:
                    self.not_recently_explored_frontiers.appendleft(nef)

            self.last_known_enemy_pos = frontiers[i]
            self.recent_visited_frontier.append(frontiers[i])
            if len(self.recent_visited_frontier) > 30:
                self.recent_visited_frontier.popleft()
           
            return decision
        
        tmp = list(self.not_recently_explored_frontiers)
        tmp.sort(key=cmp_to_key(self._custom_comparator))
        self.not_recently_explored_frontiers = deque(tmp)

        target = self.not_recently_explored_frontiers[0] if self.not_recently_explored_frontiers else frontiers[0]
        self.last_known_enemy_pos = target
        path = self._bfs(my_position, target, map_state)
        self.not_recently_explored_frontiers.popleft()
        self.recent_visited_frontier.append(target)
        if len(self.recent_visited_frontier) > 15:
            self.recent_visited_frontier.popleft()
        return self._make_decision(path)

    def _get_neighbors(self, position, map_state: np.ndarray) -> list[tuple[tuple, Move]]:
        """
        Find all valid neighboring cells from a given position, along with the moves required to reach them.

        Args:
            position (tuple): The current position, represented as (row, column).
            map_state (np.ndarray): The map state.

        Returns:
            (list[tuple[tuple, Move]]): A list of tuples, each containing a valid neighboring position
            and the corresponding move to reach it.
        """

        x, y = position
        height, width = map_state.shape
        directions = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        neighbors = []
        for direction in directions:
            dx, dy = direction.value
            nx, ny = x + dx, y + dy
            if 0 <= nx < height and 0 <= ny < width:
                if (map_state[nx, ny] != 1):
                    neighbors.append(((nx, ny), direction))
        
        return neighbors

    def _bfs(self, start: tuple, goal: tuple, map_state: np.ndarray) -> list[Move]:
        """
        Find the shortest path from start to goal using Breadth-First Search (BFS).

        This function performs BFS on a grid-based map to find the shortest path between two positions.
        It returns the first move that agent should take to follow that path.

        Args:
            start (tuple): The starting position (row, column).
            goal (tuple): The target position (row, column).
            map_state (np.ndarray): The current map layout.

        Returns:
            Move: The list of moves along the shortest path to the goal.
                Returns [Move.STAY] if no path is found.
        """
        queue = deque([(start, [])])
        visited = {start}

        while queue:
            current_pos, path = queue.popleft()

            if current_pos == goal:
                return path

            for next_pos, move in self._get_neighbors(current_pos, map_state):
                if next_pos not in visited:
                    visited.add(next_pos)
                    queue.append((next_pos, path + [move]))

        return [Move.STAY]
    
    def _astar(self, start: tuple, goal: tuple, map_state: np.ndarray) -> list[Move]:
        """
        Find the optimal path from start to goal using the A-star algorithm.

        This implementation uses the Manhattan distance as the heuristic function h(n), which estimates 
        the cost from the current position to the goal. The total cost function is defined as:
            f(n) = g(n) + h(n)

        where:
            - g(n) is the actual cost from the start to the current position (path length so far),
            - h(n) is the estimated cost from the current postion to the goal.
            - f(n) is the total estimated cost of the path through the current node.
        
        The search frontier is maintained as a min-heap priority queue, where each entry is a tuple 
        containing the f-cost, a unique counter (to break ties), the current position, and the path of 
        moves taken to reach that position.

        Args:
            start (tuple): The starting position on the map (row, column).
            goal (tuple): The target position to reach (row, column).
            map_state (np.ndarray): The map state.

        Returns:
            (list[Move]): The list of moves along the optimal path to the goal.
                Returns [Move.STAY] if no path is found.
        """
                
        from heapq import heappush, heappop
        frontier = [(0, start, [])]
        visited = set()

        while frontier:
            f_cost, current_pos, path = heappop(frontier)

            if current_pos == goal:
                return path
            
            if current_pos in visited:
                continue

            visited.add(current_pos)

            for next_pos, move in self._get_neighbors(current_pos, map_state):
                if next_pos not in visited:
                    new_path = path + [move]
                    g_cost = len(new_path)
                    h_cost = self._manhattan(next_pos, goal)
                    f_cost = g_cost + h_cost
                    heappush(frontier, (f_cost, next_pos, new_path))

        return [Move.STAY]

    def _manhattan(self, first_position: tuple, second_position: tuple) -> int:
        """
        Calculate the Manhattan distance between two grid positions.

        Args:
            first_position (tuple): The first position, represented as (row, column).
            second_position (tuple): The second position, represented as (row, column).

        Returns:
            int: The Manhattan distance between the two positions.
        """
        return abs(first_position[0] - second_position[0]) + abs(first_position[1] - second_position[1])
    
    def _count_neighbors(self, position: tuple, map_state: np.ndarray, target_cell: int | None = None) -> int:
        """
        Count all valid neighboring cells from a given position.

        Args:
            position (tuple): The current position, represented as (row, column).
            map_state (np.ndarray): The map state.

        Returns:
            int: The number of valid neighboring cells around the given position.
        """
        
        x, y = position
        height, width = map_state.shape
        directions = [Move.UP.value, Move.DOWN.value, Move.LEFT.value, Move.RIGHT.value]
        count = 0
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < height and 0 <= ny < width:
                count += (map_state[nx, ny] != 1 and map_state[nx, ny] != target_cell)
        
        return count

    def _find_frontiers(self, map_state: np.ndarray) -> list[tuple]:
        """
        Identify frontier positions within Pacman’s observable range; these are positions 
        with at least one surrounding unknown cell (-1).
        
        This serves Pacman in evaluating which frontier position is optimal based on the
        potential to extract information from unexplored areas.

        :param map_state: The current map layout.
        :type map_state: np.ndarray
        :return: List of frontier positions from Pacman’s observable range.
        :rtype: list[tuple]
        """

        
        frontiers = []

        height, width = map_state.shape
        directions = [Move.UP.value, Move.DOWN.value, Move.LEFT.value, Move.RIGHT.value]
        for i in range(height):
            for j in range(width):
                if map_state[i, j] == 0:
                    for drow, dcol in directions:
                        nrow, ncol = i + drow, j + dcol
                        if self._in_bounds((nrow, ncol), map_state) and map_state[nrow, ncol] == -1:
                            frontiers.append((i, j))
                            break
        
        return frontiers
    
    def _in_bounds(self, position: tuple, map_state: np.ndarray) -> bool:
        """
        Check whether a position lies within the valid range of the map.

        :param position: The position that need to be checked
        :type position: tuple
        :param map_state: The current map layout.
        :type map_state: np.ndarray
        :return: Return True if the position is within the valid range of the map; 
        otherwise, return False.
        :rtype: bool
        """

        row, col = position
        height, width = map_state.shape
        if (row < 0 or row >= height or col < 0 or col >= width):
            return False
        
        return map_state[row, col] != 1

    def _custom_comparator(self, a: tuple, b: tuple) -> float:
        """
        The comparator is used to sort the list of unexplored frontier positions.
        
        These frontier positions are evaluated based on their score together with
        the 'belief' value.
        
        :param a: The position that we want to prioritize.
        :type a: tuple
        :param b: The position that we want to rank later.
        :type b: tuple
        :return: The evaluation score distance between two positions.
        :rtype: float
        """
        my_position = self.recent_visited[-1]
        da = self._manhattan(a, my_position)
        db = self._manhattan(b, my_position)
        if (da != db):
            return da - db
        sa = self._score(a, self.map, my_position) + self.belief[a]
        sb = self._score(b, self.map, my_position) + self.belief[b]

        return sb - sa

    def _make_decision(self, path: list[Move]) -> tuple[Move, int]:
        """
        Provide the move along with the number of steps based on the path 
        list to the target.

        :param path: The list of moves to reach the target
        :type path: list[Move]
        :return: A tuple consisting of the first move of the path to the target 
        along with the number of steps.
        :rtype: tuple[Move, int]
        """
        if len(path) == 1:
            return (path[0], 1)
        
        steps = 1
        for i in range(1, len(path)):
            if path[i] == path[0]:
                steps += 1
            else:
                break
        
        return (path[0], min(2, steps))

    def _update_belief(self, belief: np.ndarray, map_state: np.ndarray):
        """
        Update the belief values of positions. The belief value of a position can spread to surrounding
        cells, implying that if the Ghost is not currently in this cell, it is very likely
        to be in the neighboring ones. At the same time, set a threshold to prevent these
        values from continuously increasing.
        
        :param belief: Belief value map for each cell.
        :type belief: np.ndarray
        :param map_state: The current map layout.
        :type map_state: np.ndarray
        :return: None
        :rtype: None
        """
        
        new_belief = np.zeros_like(belief)
        height, width = map_state.shape
        for i in range(height):
            for j in range(width):
                if belief[i, j] > 0:
                    for next_pos, move in self._get_neighbors((i, j), map_state):
                        new_belief[next_pos] += belief[i, j]
                if map_state[i, j] == 0:
                    new_belief[i, j] = 0
        
        belief = new_belief * 0.9

    def _belief_gain(self, frontier: tuple, belief: np.ndarray, map_state: np.ndarray) -> float:
        """
        Calculate the belief value of a frontier position by summing the belief values within 
        its cross-shaped neighborhood.
        
        :param frontier: The frontier position that need to be calculated.
        :type frontier: tuple
        :param belief: Belief value map for each cell.
        :type belief: np.ndarray
        :param map_state: The current map layout.
        :type map_state: np.ndarray
        :return: The belief gain of the given frontier position.
        :rtype: float
        """
        return sum(belief[x, y] for x, y in self._fov(frontier, map_state))
    
    def _fov(self, position: tuple, map_state: np.ndarray) -> list[tuple]:
        """
        "Find the positions within the cross-shaped range that can be observed 
        if Pacman is placed at a given specific position.

        :param position: The position at the center of the cross-shaped area.
        :type position: tuple
        :param map_state: The current map layout.
        :type map_state: np.ndarray
        :return: List of observable positions.
        :rtype: list[tuple]
        """
        
        height, width = map_state.shape
        fov = []
        for k in range(1, 6):
            nx, ny = position[0] + k, position[1]
            if nx >= height or map_state[nx, ny] == 1:
                break
            fov.append((nx, ny))

        for k in range(1, 6):
            nx, ny = position[0] - k, position[1]
            if nx < 0 or map_state[nx, ny] == 1:
                break
            fov.append((nx, ny))

        for k in range(1, 6):
            nx, ny = position[0], position[1] + k
            if ny >= width or map_state[nx, ny] == 1:
                break
            fov.append((nx, ny))

        for k in range(1, 6):
            nx, ny = position[0], position[1] - k
            if ny < 0 or map_state[nx, ny] == 1:
                break
            fov.append((nx, ny))

        return fov

    def _estimate_information_gain(self, frontier: tuple, map_state: np.ndarray) -> int:
        """
        Estimate the information value of a frontier position, which is defined as the 
        number of unknown cells (-1) within its cross-shaped range (excluding cells behind 
        walls). This serves to evaluate whether a frontier position can yield the most 
        information.

        :param frontier: The frontier position that need to be evaluated.
        :type frontier: tuple
        :param map_state: The current map layout.
        :type map_state: np.ndarray
        :return: The estimated information gain of the given frontier position.
        :rtype: int
        """

        
        x, y = frontier
        gain = 0
        directions = [Move.UP.value, Move.DOWN.value, Move.LEFT.value, Move.RIGHT.value]
        for dx, dy in directions:
            for k in range(1, 6):
                nx, ny = x + k * dx, y + k * dy
                if not self._in_bounds((nx, ny), map_state):
                    break
                if map_state[nx, ny] == 1:
                    break
                if map_state[nx, ny] == -1:
                    gain += 1
        
        return gain
    
    def _score(self, frontier: tuple, map_state: np.ndarray, current_position: tuple) -> float:
        """
        Evaluate the score of a frontier position based on the following criteria:

        -   Information gain

        -   Penalty if the position has already been visited

        -   Number of escape positions

        -   Number of possible directions

        :param frontier: The frontier position that need to be evaluated.
        :type frontier: tuple
        :param map_state: The current map layout.
        :type map_state: np.ndarray
        :param current_position: The Pacman's current position
        :type current_position: tuple
        :return: The evaluated score of the frontier position
        :rtype: float
        """

        gain = self._estimate_information_gain(frontier, map_state)
        visited_penalty = 0
        distance = self._manhattan(current_position, frontier)
        line_of_sight = self._count_neighbors(current_position, map_state)
        escape = self._count_neighbors(current_position, map_state, -1)

        if frontier in self.recent_visited:
            visited_penalty = max(0, 50 - distance)

        IG_norm = 3.0 * gain
        L_norm = 1.0 * line_of_sight
        E_norm = 0.5 * escape
        D_norm = 2.0 * distance
        R_norm = 1.5 * visited_penalty
        
        return (IG_norm + L_norm + E_norm - D_norm - R_norm)


class GhostAgent(BaseGhostAgent):
    """
    The Ghost agent uses a four-state strategy: turning at corners when detected,
    seeking safe intersections, repositioning at a distance from the enemy, and
    hiding when not yet detected.
    """

    def __init__(self, **kwargs):
        """
        Initialize the Pacman agent.
        Parameters:
        -   last_known_enemy_pos: the enemy’s position since the most recent sighting.

        -   last_seen_step: the most recent step where the enemy was seen.

        -   intersections: the set of intersections with three or more branches.

        -   target: the destination the Ghost needs to reach.

        -   recent_visited: list of positions recently visited.

        -   danger_zones: list of dangerous positions (near Pacman).

        -   safe_zones: list of safe positions that can be reached for hiding.

        -   map_corners: list of the map’s corners.

        -   strategy_mode: the current strategic mode of the Ghost.

        -   panic_threshold: the distance threshold to Pacman at which the Ghost starts corner-running.

        -   danger_intersections: intersections where Pacman can easily detect the Ghost.

        -   ttl: threshold value at which Pacman will not stay at a corner too long 
            and will move to a new corner in the cycle.           
        """

        super().__init__(**kwargs)
        self.name = "Ghost"
        self.last_known_enemy_pos = None
        self.last_seen_step = -1
        self.intersections = None
        self.target = None
        self.recent_visisted = deque(maxlen=30)

        self.danger_zones = set()
        self.safe_zones = []
        self.map_corners = []

        self.strategy_mode = ""
        self.panic_threshold = 5
        self.danger_intersections = []
        self.ttl = [-1, 0]
    
    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int) -> Move:
        """
        Initially, the Ghost initializes the necessary data for the computation process.
        
        Based on its position and distance to the enemy, the Ghost selects one of the 
        following strategies:

        - PANIC: if the distance to the enemy is within the threshold, it searches 
        for the escape move with the highest evaluation score, considering factors 
        such as distance to the enemy, number of walls, danger zones, etc.

        - TACTICAL FLEE: when the enemy is within about 5–10 steps, the Ghost turns 
        at a corner if currently at an intersection; otherwise, it searches for the 
        nearest and safest intersection, and finally considers the safest corner of the map.

        - REPOSITION: if the Ghost has shaken off the enemy and is about 5 steps away from 
        the most recent sighting position, it resets its own position. The Ghost evaluates 
        the distance to the last known enemy position and then seeks a path toward safe 
        zones or intersections.

        - HIDE: when Pacman has not detected the Ghost or none of the above conditions apply, 
        the Ghost adopts a hiding strategy. If it is already in a safe zone or at a map corner, 
        it may stay put or move randomly a few steps within a limited range. Afterwards, 
        it must recalculate and move to another safe zone or another map corner to avoid 
        Pacman’s search.
                
        Args:
            map_state (np.ndarray): The current map layout.
            my_position (tuple[int, int]): Ghost's current position (row, column).
            enemy_position (tuple[int, int]): Pacman's current position (row, column).
            step_number (int): The current step number (maximum 200).

        Returns:
            Move: The next move Ghost should take. Returns Move.STAY if no valid path is found.
        """



        if not self.intersections:
            self._initialize_map(map_state)

        self.recent_visisted.append(my_position)

        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
            self.last_seen_step = step_number
            self._update_danger_zones(enemy_position, map_state)

        enemy_distance = self._manhattan(my_position, enemy_position) if enemy_position else float("inf")

        # Mode 1: PANIC
        if enemy_distance <= self.panic_threshold:
            self.strategy_mode = "FLEE"
            return self._panic_flee(my_position, enemy_position, map_state)
        
        # Mode 2: TATICAl FLEE
        if 5 < enemy_distance <= 10:
            self.strategy_mode = "FLEE"
            return self._tactical_flee(my_position, enemy_position, map_state)
        
        # Mode 3: REPOSITION
        if enemy_position is None and (step_number - self.last_seen_step) < 5:
            self.strategy_mode = "REPOSITION"
            return self._reposition_after_escape(my_position, map_state, step_number)

        # Mode 4: HIDE
        self.strategy_mode = "HIDE"
        return self._strategic_hiding(my_position, map_state)
    
    def _initialize_map(self, map_state: np.ndarray) -> None:
        """
        Initialize several necessary parameters during the computation 
        process. These include the positions of intersections 
        (T-junctions, crossroads), the corner positions of the map, and 
        the safe positions in the central area surrounded by multiple walls.

        :param map_state: The current map layout.
        :type map_state: np.ndarray
        """
        
        height, width = map_state.shape

        self.intersections = []
        for i in range(height):
            for j in range(width):
                if map_state[i, j] == 0:
                    neighbors = self._count_neighbors((i, j), map_state)
                    if neighbors >= 3:
                        self.intersections.append((i, j))

        corners = [
            (1, 1), (1, width - 2), (height - 2, 1), (height - 2, width - 2)
        ]

        self.danger_intersections = [
            (3, 5), (3, 15), (9, 5), (9, 15), (13, 5), (13, 15)
        ]

        for corner in corners:
            if map_state[corner[0], corner[1]] != 1:
                self.map_corners.append(corner)

        center = (height // 2, width // 2)
        for i in range(height):
            for j in range(width):
                if map_state[i, j] == 0:
                    if self._manhattan((i, j), center) > min(height, width) // 3:
                        wall_count = self._count_nearby_walls((i, j), map_state, radius=3)
                        if wall_count >= 8:
                            self.safe_zones.append((i, j))

    def _panic_flee(self, my_position: tuple, enemy_position: tuple, map_state: np.ndarray) -> Move:
        """
        Determine the escape direction from Pacman when the distance is too close 
        (within visible range). The escape path is based on the walls separating 
        Pacman and the Ghost, the number of surrounding walls, whether it is in a dead-end, 
        and whether it is in a dangerous area.

        :param my_position: The current position of Ghost.
        :type my_position: tuple
        :param enemy_position: The current position of Pacman.
        :type enemy_position: tuple
        :param map_state: The current map layout.
        :type map_state: np.ndarray
        :return: The best direction to escape from Pacman.
        :rtype: Move
        """
        
        neighbors = self._get_neighbors(my_position, map_state)
        
        if not neighbors:
            return Move.STAY
        
        best_move = None
        best_score = -float("inf")

        for next_pos, move in neighbors:
            score = 0

            if self._has_wall_between(next_pos, enemy_position, map_state):
                score += 200

            wall_density = self._count_nearby_walls(next_pos, map_state)
            score += wall_density * 10

            if self._is_dead_end(next_pos, map_state):
                score -= 500

            if next_pos in self.danger_zones:
                score -= 300

            if score > best_score:
                best_score = score
                best_move = move

        return best_move if best_move else Move.STAY

    def _tactical_flee(self, my_position: tuple, enemy_position: tuple, map_state: np.ndarray) -> Move:
        """
        When the distance to Pacman is not too close, proceed to find a turn if at an intersection. 
        Otherwise, search for the safest available intersection. If no safe intersection exists, 
        corner positions will be the reasonable choice.

        :param my_position: The current position of Ghost.
        :type my_position: tuple
        :param enemy_position: The current position of Pacman.
        :type enemy_position: tuple
        :param map_state: The current map layout.
        :type map_state: np.ndarray
        :return: Movement direction to the designated safe position.
        :rtype: Move
        """
        
        if my_position in self.intersections:
            return self._choose_best_exit(my_position, enemy_position, map_state)
    
        safe_intersections = self._find_safe_intersections(my_position, enemy_position, map_state)

        if not safe_intersections:
            target = self._find_best_corner(my_position, enemy_position)
        else:
            target = safe_intersections[0]

        path = self._bfs(my_position, target, map_state)
        return path[0] if path else Move.STAY
    
    def _reposition_after_escape(self, my_position: tuple, map_state: np.ndarray, step_number: int) -> Move:
        """
        Reset the position after successfully evading Pacman. If the Ghost is in a safe intersection, it may 
        temporarily remain still. Otherwise, proceed to find a path to the nearest safe area or the farthest intersection.

        :param my_position: The current position of Ghost.
        :type my_position: tuple
        :param map_state: The current map layout.
        :type map_state: np.ndarray
        :param step_number: The current step number of the game.
        :type step_number: int
        :return: The best move to reset the position.
        """
        
        if my_position in self.intersections:
           
            neighbor_count = self._count_neighbors(my_position, map_state)

            estimated_pacman_dist = self._estimate_enemy_distance(my_position, step_number)

            if (neighbor_count >= 4
                and estimated_pacman_dist > 8
                and len(self.recent_visisted) >= 2
                and self.recent_visisted[-1] == self.recent_visisted[-2]):
                pass
            elif neighbor_count >= 4 and estimated_pacman_dist > 8:
                return Move.STAY
            
        if self.safe_zones:
            target = min(self.safe_zones, key=lambda z: self._manhattan(my_position, z))
            path = self._bfs(my_position, target, map_state)
            return path[0] if path else Move.STAY
        
        if self.intersections:
            
            farthest = max(self.intersections, key=lambda i: (self._manhattan((15, 10), i), i[1]))
            path = self._bfs(my_position, farthest, map_state)
            return path[0] if path else Move.STAY
        
        return Move.STAY
    
    def _strategic_hiding(self, my_position: tuple, map_state: np.ndarray) -> Move:
        """
        A position is considered safe if it is in a safe zone or at the corners of the map. At the 
        same time, the Ghost will not remain too long in a single safe position but needs to move 
        to a new one. The early stage of the game is the golden time for the Ghost to hide from 
        Pacman, and the map corners can be exploited for more effective concealment.

        :param my_position: The current position of Ghost.
        :type my_position: tuple
        :param map_state: The current map layout.
        :type map_state: np.ndarray
        :return: The move to the target hiding position.
        :rtype: Move
        """
        
        is_safe = (my_position in self.safe_zones or my_position in self.map_corners)

        if is_safe:
            
            neighbors = self._get_neighbors(my_position, map_state)
            self.ttl[0] -= 1
            if self.ttl[0] > 0:
                if random.random() < 0.7:
                    return Move.STAY
                else:
                    return random.choice([move for _, move in neighbors])
            else:
                self.ttl[1] += 1
            
        if self.safe_zones:
            
            target = min(self.safe_zones, key=lambda z: self._manhattan(my_position, z))
            path = self._bfs(my_position, target, map_state)
            return path[0] if path else Move.STAY
        
        if self.map_corners:
            for intersection in self.danger_intersections:
                if self._count_neighbors(intersection, map_state) == 4:
                    map_state[intersection] = 1

            self.ttl[1] %= len(self.map_corners)
            target = self.map_corners[self.ttl[1]]
            if self.ttl[0] <= 0:
                self.ttl[0] = 6
                target = None
                while not target or self.map_corners[self.ttl[1]] == self.map_corners.index(target):
                    target = random.choice(self.map_corners)
                self.ttl[1] = self.map_corners.index(target)
            
            
            path = self._bfs(my_position, target, map_state)
            return path[0] if path else Move.STAY
        
        return Move.STAY
    
    def _find_safe_intersections(self, my_position: tuple, enemy_position: tuple, map_state: np.ndarray) -> list[tuple]:
        """
        Find the list of safe intersections from the set of intersections we have identified. 
        For each intersection, calculate the Ghost’s distance (using BFS) as well as Pacman’s Manhattan distance to the intersection. 
        An intersection is considered safe if the Ghost’s distance to it is less than half of Pacman’s Manhattan distance minus 2, 
        since Pacman moves twice as fast as the Ghost and therefore requires such a minimum distance. 
        Then, rank the intersections in priority order: first by the number of neighboring cells, and next by the Ghost’s distance to that intersection.

        :param my_position: The current position of Ghost.
        :type my_position: tuple
        :param enemy_position: The most recently visible position of Pacman.
        :type enemy_position: tuple
        :param map_state: The current map layout.
        :type map_state: np.ndarray
        :return: The list of safe intersections found.
        :rtype: list[tuple]
        """
        
        safe = []

        for intersection in self.intersection:
            ghost_dist = len(self._bfs(my_position, intersection, map_state))

            pacman_dist = self._manhattan(enemy_position, intersection) / 2

            if ghost_dist < pacman_dist - 2:
                neighbor_count = self._count_neighbors(intersection, map_state)
                safe.append((intersection, neighbor_count, ghost_dist))

        safe.sort(key=lambda x: (-x[1], x[2]))
        return [i[0] for i in safe]

    def _choose_best_exit(self, my_position: tuple, enemy_position: tuple, map_state: np.ndarray) -> Move:
        """
        Find the best escape route at an intersection, used when the Ghost itself is at an intersection near Pacman. 
        The escape routes will be evaluated based on the following scoring criteria:

        - The farther the distance from Pacman, the higher the score.

        - If the row distance is greater than the column distance, moves to the left or right will have higher scores.

        - If the column distance is greater than the row distance, moves upward or downward will have higher scores.

        - If a move results in a wall blocking between the Ghost and Pacman, the score will be higher.
        
        Based on these scoring criteria, choose the move with the highest score.

        :param my_position: The current position of Ghost.
        :type my_position: tuple
        :param enemy_position: The most recently visible position of Pacman.
        :type enemy_position: tuple
        :param map_state: The current map layout.
        :type map_state: np.ndarray
        :return: The move to the best escape route.
        :rtype: Move
        """
        
        neighbors = self._get_neighbors(my_position, map_state)

        best_move = None
        best_score = -float("inf")

        ex, ey = enemy_position
        mx, my = my_position

        for next_pos, move in neighbors:
            nx, ny = next_pos
            score = 0

            new_dist = self._manhattan(next_pos, enemy_position)
            score += new_dist * 50

            pacman_dx = mx - ex
            pacman_dy = my - ey

            if abs(pacman_dx) > abs(pacman_dy):
                if move in [Move.LEFT, Move.RIGHT]:
                    score += 150
            else:
                if move in [Move.UP, Move.DOWN]:
                    score += 150

            if self._has_wall_between(next_pos, enemy_position, map_state):
                score += 100

            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move if best_move else Move.STAY
            
    def _has_wall_between(self, pos1: tuple, pos2: tuple, map_state: np.ndarray) -> bool:
        """
        Check whether there is a wall blocking between two positions on the map. 
        This applies to two positions that are either in the same row or in the same column.

        :param pos1: The first position
        :type pos2: tuple
        :param pos2: The second position
        :type pos2: tuple
        :param map_state: The current map layout.
        :type map_state: np.ndarray
        :return: Return True if there is a wall blocking, otherwise return False.
        :rtype: bool
        """
        
        x1, y1 = pos1
        x2, y2 = pos2

        if x1 == x2:
            for y in range(min(y1, y2), max(y1, y2) + 1):
                if map_state[x1, y] == 1:
                    return True
                
        if y1 == y2:
            for x in range(min(x1, x2), max(x1, x2) + 1):
                if map_state[x, y1] == 1:
                    return True
                
        return False

    def _count_nearby_walls(self, position: tuple, map_state: np.ndarray, radius: int = 2) -> int:
        """
        Count the number of walls around within the specified radius.
        
        :param position: The given position.
        :type position: tuple
        :param map_state: The current map layout.
        :type map_state: np.ndarray
        :param radius: The size of the radius. (default 2 cell)
        :type radius: int
        :return: The number of walls around the given position within the specified radius.
        :rtype: int
        """
        
        x, y = position
        count = 0

        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                if (0 <= nx < map_state.shape[0]
                    and 0 <= ny < map_state.shape[1]
                    and map_state[nx, ny] == 1):
                    count += 1

        return count

    def _estimate_enemy_distance(self, my_position: tuple, step_number: int) -> float:
        """
        Estimate the distance from the Ghost to Pacman when Pacman is out of sight. 
        It is calculated as the Manhattan distance between the Ghost’s position and the nearest position 
        where Pacman was last seen, minus twice the number of steps taken since Pacman was last visible, because Pacman moves twice as fast as the Ghost. 
        If Pacman has never been seen, the distance is temporarily considered infinite.

        :param my_position: The current position of Ghost.
        :type my_position: tuple
        :param step_number: The current step number of Ghost.
        :type step_number: int
        :return: Estimated distance between Ghost and Pacman.
        :rtype: float
        """
        
        
        if not self.last_known_enemy_pos:
            return float("inf")
        
        steps_passed = step_number - self.last_seen_step

        min_possible_dist = max(0, self._manhattan(my_position, self.last_known_enemy_pos) - 2 * steps_passed)

        return min_possible_dist

    def _is_dead_end(self, position: tuple, map_state: np.ndarray) -> bool:
        """
        Check whether a position is a dead-end; a dead-end is a position that has no more than one neighbor.
        
        :param position: The given position.
        :type position: tuple
        :param map_state: The current map layout.
        :type map_state: np.ndarray
        :return: Return True if the position is a dead-end; otherwise, return False.
        :rtype: bool
        """
        
        return self._count_neighbors(position, map_state) <= 1

    def _find_best_corner(self, my_position: tuple, enemy_position: tuple) -> tuple:
        """
        Find the list of the best corner positions. 
        A corner position is considered the best if it is farther away from Pacman

        :param  my_position: The current position of Ghost.
        :type  my_position: tuple
        :param enemy_position: The most recently visible position of Pacman.
        :type enemy_position: tuple
        :return: None
        :rtype: tuple
        """
        
        if not self.map_corners:
            return my_position
        
        return max(self.map_corners, key=lambda c: self._manhattan(c, enemy_position))

    def _update_danger_zones(self, enemy_position: tuple, map_state: np.ndarray):
        """
        Update the list of dangerous zones. 
        A zone is considered dangerous if it is traversable within a 5x5 area around Pacman.

        :param enemy_position: The most recently visible position of Pacman.
        :type enemy_position: tuple
        :param map_state: The current map layout.
        :type map_state: np.ndarray
        :return: None
        :rtype: None
        """
        
        self.danger_zones.clear()
        height, width = map_state.shape
        ex, ey = enemy_position
        for dx in range(-5, 6):
            for dy in range(-5, 6):
                if abs(dx) + abs(dy) <= 5 \
                and 0 <= ex + dx < height \
                and 0 <= ey + dy < width \
                and map_state[ex + dx, ey + dy] != 1:
                    self.danger_zones.add((ex + dx, ey + dy))

    def _manhattan(self, first_position: tuple, second_position: tuple) -> int:
        """
        Calculate the Manhattan distance between two grid positions.

        Args:
            first_position (tuple): The first position, represented as (row, column).
            second_position (tuple): The second position, represented as (row, column).

        Returns:
            int: The Manhattan distance between the two positions.
        """
        
        return abs(first_position[0] - second_position[0]) + abs(first_position[1] - second_position[1])
    
    def _count_neighbors(self, position: tuple, map_state: np.ndarray) -> int:
        """
        Count all valid neighboring cells from a given position.

        Args:
            position (tuple): The current position, represented as (row, column).
            map_state (np.ndarray): The map state.

        Returns:
            int: The number of valid neighboring cells around the given position.
        """
        
        x, y = position
        height, width = map_state.shape
        count = 0

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if (0 <= nx < height 
                and 0 <= ny < width 
                and map_state[nx, ny] != 1):
                count += 1

        return count
    
    def _get_neighbors(self, position: tuple, map_state: np.ndarray) -> list[tuple[tuple, Move]]:
        """
        Find all valid neighboring cells from a given position, along with the moves required to reach them.

        Args:
            position (tuple): The current position, represented as (row, column).
            map_state (np.ndarray): The map state.

        Returns:
            (list[tuple[tuple, Move]]): A list of tuples, each containing a valid neighboring position
            and the corresponding move to reach it.
        """
        
        x, y = position
        height, width = map_state.shape
        neighbors = []

        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dx, dy = move.value
            nx, ny = x + dx, y + dy
            if (0 <= nx < height 
                and 0 <= ny < width 
                and map_state[nx, ny] != 1):
                neighbors.append(((nx, ny), move))

        return neighbors
    
    def _bfs(self, start: tuple, goal: tuple, map_state: np.ndarray) -> list[Move]:
        """
        Find the shortest path from start to goal using Breadth-First Search (BFS).

        This function performs BFS on a grid-based map to find the shortest path between two positions.
        It returns the first move that agent should take to follow that path.

        Args:
            start (tuple): The starting position (row, column).
            goal (tuple): The target position (row, column).
            map_state (np.ndarray): The current map layout.

        Returns:
            Move: The list of moves along the shortest path to the goal.
                Returns [Move.STAY] if no path is found.
        """
        
        queue = deque([(start, [])])
        visited = {start}

        while queue:
            current, path = queue.popleft()

            if current == goal:
                return path if path else [Move.STAY]
            
            for next_pos, move in self._get_neighbors(current, map_state):
                if next_pos not in visited:
                    visited.add(next_pos)
                    queue.append((next_pos, path + [move]))
        
        return [Move.STAY]
