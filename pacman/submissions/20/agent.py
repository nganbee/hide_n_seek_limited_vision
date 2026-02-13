# agent.py
import sys
import random
import numpy as np
from pathlib import Path
from collections import deque
from heapq import heappush, heappop

src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move

_DIRS = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]


def _apply(pos, move: Move):
    dr, dc = move.value
    return (pos[0] + dr, pos[1] + dc)


def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _move_from_to(a, b):
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    for m in _DIRS:
        if m.value == (dr, dc):
            return m
    return Move.STAY


class PacmanAgent(BasePacmanAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.speed = int(kwargs.get("pacman_speed", 1))
        self.obs_r = int(kwargs.get("pacman_obs_radius", 5))

        self.visited_counts = {}
        self.recent_pos = deque(maxlen=32)
        self.tabu_edges = deque(maxlen=12)  # (from,to) trong vài bước gần đây

        self.last_known_enemy = None
        self.last_enemy_step = None
        self.track_ttl = 24

        self.target = None
        self.escape_target = None
        self.escape_steps_left = 0

        self.assumed_walls = set()
        self._prev_pos = None
        self._intended_pos = None
        self._intended_was_fog = False
        self._stuck_count = 0

    def step(self, map_state, my_position, enemy_position, step_number):
        # --- infer fog-wall nếu lượt trước định đi vào -1 mà không di chuyển ---
        if (
            self._prev_pos is not None
            and self._intended_pos is not None
            and my_position == self._prev_pos
            and self._intended_pos != self._prev_pos
            and self._intended_was_fog
        ):
            self.assumed_walls.add(self._intended_pos)

        # stuck detection
        if self._prev_pos is not None and my_position == self._prev_pos:
            self._stuck_count += 1
        else:
            self._stuck_count = 0

        self._prev_pos = my_position
        self._intended_pos = None
        self._intended_was_fog = False

        # --- heatmap + recent ---
        self.visited_counts[my_position] = self.visited_counts.get(my_position, 0) + 1
        self.recent_pos.append(my_position)

        # --- loop detect -> enter escape mode ---
        if (self._stuck_count >= 2) or self._detect_loop():
            if self.escape_steps_left <= 0:
                self.escape_steps_left = 14
                self.escape_target = None
                self.target = None

        # --- update enemy memory ---
        if enemy_position:
            self.last_known_enemy = enemy_position
            self.last_enemy_step = step_number
            self.target = None
            self.escape_target = None
            self.escape_steps_left = 0

        # --- 1) HUNT ---
        if enemy_position:
            path = self._astar_path(my_position, enemy_position, map_state, mode="hunt")
            if path:
                action = self._dash_if_good(my_position, path, map_state)
                self._record_intent(my_position, action, map_state)
                return action

        # --- 2) TRACK ---
        if self.last_known_enemy is not None and self.last_enemy_step is not None:
            if (step_number - self.last_enemy_step) <= self.track_ttl:
                path = self._astar_path(my_position, self.last_known_enemy, map_state, mode="track")
                if path:
                    action = self._dash_if_good(my_position, path, map_state)
                    self._record_intent(my_position, action, map_state)
                    return action
            else:
                self.last_known_enemy = None
                self.last_enemy_step = None

        # --- 3) ESCAPE (thoát vùng lặp) ---
        if self.escape_steps_left > 0:
            self.escape_steps_left -= 1
            if self.escape_target is None or my_position == self.escape_target:
                self.escape_target = self._pick_escape_target(map_state, my_position)
            if self.escape_target is not None:
                path = self._astar_path(my_position, self.escape_target, map_state, mode="escape")
                if path:
                    action = self._dash_if_good(my_position, path, map_state)
                    self._record_intent(my_position, action, map_state)
                    return action
            self.escape_target = None

        # --- 4) EXPLORE: frontier tốt nhất ---
        if self.target is None or my_position == self.target:
            self.target = self._pick_frontier_target(map_state, my_position)

        if self.target is not None:
            path = self._astar_path(my_position, self.target, map_state, mode="explore")
            if path:
                action = self._dash_if_good(my_position, path, map_state)
                self._record_intent(my_position, action, map_state)
                return action
            # nếu target là fog mà fail liên tục -> tránh về sau
            if self._in_bounds(self.target, map_state) and map_state[self.target] == -1:
                self.assumed_walls.add(self.target)
            self.target = None

        # --- 5) fallback: local best move ---
        mv = self._best_local_move(my_position, map_state)
        self._record_intent(my_position, mv, map_state)
        return mv

    # =========================
    # Loop detection
    # =========================
    def _detect_loop(self):
        if len(self.recent_pos) < 10:
            return False

        recent = list(self.recent_pos)

        # A-B-A-B
        a, b, c, d = recent[-4], recent[-3], recent[-2], recent[-1]
        if a == c and b == d and a != b:
            return True

        # vùng nhỏ
        if len(self.recent_pos) == self.recent_pos.maxlen and len(set(self.recent_pos)) <= 7:
            return True

        return False

    # =========================
    # Map helpers
    # =========================
    def _in_bounds(self, pos, map_state):
        return 0 <= pos[0] < map_state.shape[0] and 0 <= pos[1] < map_state.shape[1]

    def _is_walkable(self, pos, map_state):
        if not self._in_bounds(pos, map_state):
            return False
        if pos in self.assumed_walls:
            return False
        return map_state[pos] != 1  # allow 0 and -1

    def _neighbors(self, pos, map_state, goal_hint=None):
        out = []
        for mv in _DIRS:
            np_pos = _apply(pos, mv)
            if self._is_walkable(np_pos, map_state):
                # (tabu?, visited, heuristic)
                tabu = 1 if (pos, np_pos) in self.tabu_edges else 0
                visited = self.visited_counts.get(np_pos, 0)
                h = 0 if goal_hint is None else _manhattan(np_pos, goal_hint)
                out.append((tabu, visited, h, np_pos, mv))
        out.sort(key=lambda x: (x[0], x[1], x[2]))
        return [(x[3], x[4]) for x in out]

    # =========================
    # Intent + tabu
    # =========================
    def _record_intent(self, my_position, action, map_state):
        if isinstance(action, tuple):
            mv, k = action
        else:
            mv, k = action, 1

        p = my_position
        for _ in range(max(1, k)):
            nxt = _apply(p, mv)
            if not self._in_bounds(nxt, map_state):
                break
            self.tabu_edges.append((p, nxt))
            p = nxt

        self._intended_pos = p
        self._intended_was_fog = self._in_bounds(p, map_state) and map_state[p] == -1

    # =========================
    # A* (allow fog with penalty)
    # =========================
    def _astar_path(self, start, goal, map_state, mode="explore"):
        if start == goal:
            return []
        if not self._is_walkable(start, map_state) or not self._is_walkable(goal, map_state):
            return None

        recent_set = set(self.recent_pos)

        # mode-dependent fog penalty: hunt/track thấp để bám nhanh; explore/escape cao hơn
        fog_pen = 1 if mode in ("hunt", "track") else 3
        tabu_pen = 4 if mode != "hunt" else 2

        def step_cost(p):
            c = 1
            if map_state[p] == -1:
                c += fog_pen
            v = self.visited_counts.get(p, 0)
            c += min(4, v)
            if p in recent_set:
                c += 2
            return c

        openh = []
        heappush(openh, (0, 0, start))
        g = {start: 0}
        parent = {start: None}

        while openh:
            _, _, cur = heappop(openh)
            if cur == goal:
                break

            for np_pos, _ in self._neighbors(cur, map_state, goal_hint=goal):
                ng = g[cur] + step_cost(np_pos)
                if (cur, np_pos) in self.tabu_edges:
                    ng += tabu_pen
                if np_pos not in g or ng < g[np_pos]:
                    g[np_pos] = ng
                    parent[np_pos] = cur
                    f = ng + _manhattan(np_pos, goal)
                    heappush(openh, (f, ng, np_pos))

        if goal not in parent:
            return None

        # reconstruct -> moves
        rev = []
        cur = goal
        while cur is not None:
            rev.append(cur)
            cur = parent[cur]
        rev.reverse()

        moves = []
        for i in range(len(rev) - 1):
            moves.append(_move_from_to(rev[i], rev[i + 1]))
        return moves

    def _dash_if_good(self, my_position, path_moves, map_state):
        if not path_moves:
            return Move.STAY
        m1 = path_moves[0]
        if self.speed <= 1 or len(path_moves) < 2:
            return m1
        m2 = path_moves[1]
        if m1 != m2:
            return m1
        p1 = _apply(my_position, m1)
        p2 = _apply(p1, m1)
        if self._is_walkable(p1, map_state) and self._is_walkable(p2, map_state):
            return (m1, 2)
        return m1

    # =========================
    # Target selection
    # =========================
    def _is_frontier(self, pos, map_state):
        if not self._is_walkable(pos, map_state):
            return False
        for mv in _DIRS:
            np_pos = _apply(pos, mv)
            if self._in_bounds(np_pos, map_state) and map_state[np_pos] == -1 and np_pos not in self.assumed_walls:
                return True
        return False

    def _unknown_gain(self, pos, map_state):
        h, w = map_state.shape
        r0, c0 = pos
        r = self.obs_r
        rmin = max(0, r0 - r)
        rmax = min(h - 1, r0 + r)
        cmin = max(0, c0 - r)
        cmax = min(w - 1, c0 + r)
        cnt = 0
        for rr in range(rmin, rmax + 1):
            for cc in range(cmin, cmax + 1):
                p = (rr, cc)
                if map_state[p] == -1 and p not in self.assumed_walls:
                    cnt += 1
        return cnt

    def _bfs_dist_walkable(self, start, map_state):
        if not self._is_walkable(start, map_state):
            return None
        q = deque([start])
        dist = {start: 0}
        while q:
            cur = q.popleft()
            for np_pos, _ in self._neighbors(cur, map_state):
                if np_pos not in dist:
                    dist[np_pos] = dist[cur] + 1
                    q.append(np_pos)
        return dist

    def _pick_frontier_target(self, map_state, my_position):
        dist = self._bfs_dist_walkable(my_position, map_state)
        if dist is None:
            return None

        recent_set = set(self.recent_pos)

        best = None
        best_score = -10**18

        for p, d in dist.items():
            if not self._is_frontier(p, map_state):
                continue
            gain = self._unknown_gain(p, map_state)
            v = self.visited_counts.get(p, 0)
            recent_pen = 120 if p in recent_set else 0
            fog_pen = 25 if map_state[p] == -1 else 0

            # ưu tiên gain, nhưng vẫn ép thoát local (d vừa phải) và tránh lặp
            score = gain * 14 - d * 5 - v * 4 - recent_pen - fog_pen
            if score > best_score:
                best_score = score
                best = p

        if best is not None:
            return best

        # không có frontier: chọn cell xa + ít visited (đẩy ra khỏi vùng lặp)
        far = None
        far_score = -10**18
        for p, d in dist.items():
            v = self.visited_counts.get(p, 0)
            recent_pen = 90 if p in recent_set else 0
            gain = self._unknown_gain(p, map_state)
            score = d * 10 + gain * 2 - v * 6 - recent_pen
            if score > far_score:
                far_score = score
                far = p
        return far

    def _pick_escape_target(self, map_state, my_position):
        dist = self._bfs_dist_walkable(my_position, map_state)
        if dist is None:
            return None
        recent_set = set(self.recent_pos)

        best = None
        best_score = -10**18
        for p, d in dist.items():
            v = self.visited_counts.get(p, 0)
            gain = self._unknown_gain(p, map_state)
            recent_pen = 160 if p in recent_set else 0
            # escape: cực ưu tiên xa + có gain, tránh recent mạnh
            score = d * 14 + gain * 4 - v * 7 - recent_pen
            if score > best_score:
                best_score = score
                best = p
        return best

    # =========================
    # Local fallback move
    # =========================
    def _best_local_move(self, my_position, map_state):
        neigh = self._neighbors(my_position, map_state)
        if not neigh:
            return Move.STAY

        recent_set = set(self.recent_pos)

        best_mv = Move.STAY
        best_score = -10**18

        for np_pos, mv in neigh:
            v = self.visited_counts.get(np_pos, 0)
            score = 0
            score -= v * 12
            if np_pos in recent_set:
                score -= 120
            if (my_position, np_pos) in self.tabu_edges:
                score -= 150
            if map_state[np_pos] == -1:
                score -= 10
            if score > best_score:
                best_score = score
                best_mv = mv

        return best_mv


class GhostAgent(BaseGhostAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.pos_history = {}
        self.pacman_speed = int(kwargs.get("pacman_speed", 1))
        self.capture_dist = int(kwargs.get("capture_dist", 1))

        self.memory_pacman = None
        self.memory_step = None
        self.memory_ttl = 22

        # suy luận tường trong fog (-1) nếu cố bước vào mà đứng im
        self.assumed_walls = set()
        self._prev_pos = None
        self._intended_pos = None
        self._intended_was_fog = False

        # heatmap nguy hiểm (decay)
        self.threat_heat = {}  # pos -> float
        self._heat_decay = 0.90
        self._heat_add = 6.0
        self._heat_radius = 6

    def step(self, map_state, my_position, enemy_position, step_number):
        h, w = map_state.shape

        # infer fog-wall nếu lượt trước định đi vào -1 mà không di chuyển
        if (
            self._prev_pos is not None
            and self._intended_pos is not None
            and my_position == self._prev_pos
            and self._intended_pos != self._prev_pos
            and self._intended_was_fog
        ):
            self.assumed_walls.add(self._intended_pos)

        self._prev_pos = my_position
        self._intended_pos = None
        self._intended_was_fog = False

        self.pos_history[my_position] = self.pos_history.get(my_position, 0) + 1

        # update pacman memory
        if enemy_position:
            self.memory_pacman = enemy_position
            self.memory_step = step_number

        pac_pos = None
        if enemy_position:
            pac_pos = enemy_position
        elif self.memory_pacman is not None and self.memory_step is not None:
            if (step_number - self.memory_step) <= self.memory_ttl:
                pac_pos = self.memory_pacman
            else:
                self.memory_pacman = None
                self.memory_step = None

        # decay heatmap
        if self.threat_heat:
            to_del = []
            for k, v in self.threat_heat.items():
                nv = v * self._heat_decay
                if nv < 0.05:
                    to_del.append(k)
                else:
                    self.threat_heat[k] = nv
            for k in to_del:
                self.threat_heat.pop(k, None)

        # cập nhật heat nếu có pacman info
        if pac_pos is not None:
            self._add_threat_heat(pac_pos, map_state)

        neighbors = self._get_neighbors(my_position, map_state)
        if not neighbors:
            return Move.STAY

        # không có thông tin pacman: roam theo freedom + tránh loop + tránh heat
        if pac_pos is None:
            scored = []
            for np_pos, mv in neighbors:
                freedom = self._freedom(np_pos, map_state, depth=4)
                visited = self.pos_history.get(np_pos, 0)
                heat = self.threat_heat.get(np_pos, 0.0)
                deg = self._degree(np_pos, map_state)
                junction_bonus = 80 if deg >= 3 else 0
                score = freedom * 220 + junction_bonus - visited * 170 - heat * 120
                scored.append((score, np_pos, mv))
            scored.sort(key=lambda x: x[0], reverse=True)
            best = self._choose_from_top(scored, k=3, delta=180.0)
            self._record_intent(best[1], map_state)
            return best[2]

        # ===== có pacman info: build distance fields =====
        dist_from_pac = self._bfs_dist_from_sources([pac_pos], map_state)  # maze distance now
        pac_reach = self._pacman_reachable_set(pac_pos, map_state, max_steps=max(1, self.pacman_speed))
        dist_to_pac_next = self._bfs_dist_from_sources(list(pac_reach), map_state)  # min maze dist to next-turn pac

        d_now_here = dist_from_pac[my_position[0], my_position[1]]
        if d_now_here < 0:
            d_now_here = _manhattan(my_position, pac_pos)

        # panic theo maze distance (hoặc fallback)
        panic_dist = max(8, self.capture_dist + 2 * self.pacman_speed + 2)
        is_panic = (d_now_here <= panic_dist)

        scored_safe = []
        scored_all = []

        for np_pos, mv in neighbors:
            # dist now
            d_now = dist_from_pac[np_pos[0], np_pos[1]]
            if d_now < 0:
                d_now = _manhattan(np_pos, pac_pos)

            # min dist after pacman moves this turn
            d_min_next = dist_to_pac_next[np_pos[0], np_pos[1]]
            if d_min_next < 0:
                d_min_next = 10**6

            unsafe_next = (d_min_next <= self.capture_dist)
            unsafe_now = (d_now <= self.capture_dist)

            freedom = self._freedom(np_pos, map_state, depth=4 if is_panic else 3)
            visited = self.pos_history.get(np_pos, 0)
            heat = self.threat_heat.get(np_pos, 0.0)

            deg = self._degree(np_pos, map_state)
            junction_bonus = 120 if deg >= 3 else 0

            # corridor trap penalty
            corridor_pen = 0
            if is_panic:
                if deg <= 1:
                    corridor_pen += 9000
                elif deg == 2:
                    clen = self._corridor_length(np_pos, map_state, max_len=12)
                    if clen >= 5:
                        corridor_pen += 1800 + clen * 220

            # axis break penalty (dash line)
            axis_pen = 0
            if is_panic:
                same_axis = (np_pos[0] == pac_pos[0]) or (np_pos[1] == pac_pos[1])
                if same_axis:
                    axis_pen += 1400

            # border penalty when panic (avoid cornering)
            border_pen = 0
            if is_panic:
                bd = min(np_pos[0], np_pos[1], h - 1 - np_pos[0], w - 1 - np_pos[1])
                if bd <= 1:
                    border_pen += 2500
                elif bd == 2:
                    border_pen += 900

            # hard avoid stepping into pacman immediate zone
            hard_bad = 0
            if unsafe_now:
                hard_bad += 10**12
            if unsafe_next:
                hard_bad += 10**9

            # final score: prioritize d_min_next (worst-case), then d_now, then freedom
            score = 0.0
            score += d_min_next * (1200.0 if is_panic else 900.0)
            score += d_now * (240.0 if is_panic else 160.0)
            score += freedom * (240.0 if is_panic else 110.0)
            score += junction_bonus
            score -= visited * 170.0
            score -= heat * (220.0 if is_panic else 140.0)
            score -= axis_pen
            score -= corridor_pen
            score -= border_pen
            score -= hard_bad

            item = (score, np_pos, mv, unsafe_next)
            scored_all.append(item)
            if not unsafe_next:
                scored_safe.append(item)

        # chọn trong tập an toàn nếu có, không thì chọn best trong all
        if scored_safe:
            scored_safe.sort(key=lambda x: x[0], reverse=True)
            best = self._choose_from_top(scored_safe, k=3, delta=220.0)
        else:
            scored_all.sort(key=lambda x: x[0], reverse=True)
            best = scored_all[0]

        self._record_intent(best[1], map_state)
        return best[2]

    # ================= helpers =================
    def _in_bounds(self, pos, map_state):
        return 0 <= pos[0] < map_state.shape[0] and 0 <= pos[1] < map_state.shape[1]

    def _is_walkable(self, pos, map_state):
        if not self._in_bounds(pos, map_state):
            return False
        if pos in self.assumed_walls:
            return False
        return map_state[pos] != 1  # allow 0 and -1

    def _get_neighbors(self, pos, map_state):
        moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        random.shuffle(moves)
        out = []
        for mv in moves:
            np_pos = _apply(pos, mv)
            if self._is_walkable(np_pos, map_state):
                out.append((np_pos, mv))
        # anti-loop nhẹ
        out.sort(key=lambda x: self.pos_history.get(x[0], 0))
        return out

    def _record_intent(self, intended_pos, map_state):
        self._intended_pos = intended_pos
        self._intended_was_fog = self._in_bounds(intended_pos, map_state) and map_state[intended_pos] == -1

    def _bfs_dist_from_sources(self, sources, map_state):
        h, w = map_state.shape
        dist = np.full((h, w), -1, dtype=np.int32)
        q = deque()

        for s in sources:
            if self._is_walkable(s, map_state):
                if dist[s[0], s[1]] == -1:
                    dist[s[0], s[1]] = 0
                    q.append(s)

        while q:
            cur = q.popleft()
            cd = dist[cur[0], cur[1]]
            for mv in _DIRS:
                np_pos = _apply(cur, mv)
                if self._is_walkable(np_pos, map_state) and dist[np_pos[0], np_pos[1]] == -1:
                    dist[np_pos[0], np_pos[1]] = cd + 1
                    q.append(np_pos)
        return dist

    def _pacman_walkable(self, pos, map_state):
        # worst-case: Pacman đi được qua fog, chỉ chặn wall=1
        if not self._in_bounds(pos, map_state):
            return False
        return map_state[pos] != 1

    def _pacman_reachable_set(self, pac_pos, map_state, max_steps):
        q = deque([(pac_pos, 0)])
        seen = {pac_pos}
        while q:
            cur, d = q.popleft()
            if d >= max_steps:
                continue
            for mv in _DIRS:
                np_pos = _apply(cur, mv)
                if self._pacman_walkable(np_pos, map_state) and np_pos not in seen:
                    seen.add(np_pos)
                    q.append((np_pos, d + 1))
        return seen

    def _freedom(self, pos, map_state, depth):
        q = deque([(pos, 0)])
        seen = {pos}
        cnt = 0
        while q:
            cur, d = q.popleft()
            if d >= depth:
                continue
            for mv in _DIRS:
                np_pos = _apply(cur, mv)
                if self._is_walkable(np_pos, map_state) and np_pos not in seen:
                    seen.add(np_pos)
                    q.append((np_pos, d + 1))
                    cnt += 1
        return cnt

    def _degree(self, pos, map_state):
        deg = 0
        for mv in _DIRS:
            np_pos = _apply(pos, mv)
            if self._is_walkable(np_pos, map_state):
                deg += 1
        return deg

    def _corridor_length(self, pos, map_state, max_len=12):
        best = 0
        for mv in _DIRS:
            cur = pos
            length = 0
            for _ in range(max_len):
                nxt = _apply(cur, mv)
                if not self._is_walkable(nxt, map_state):
                    break
                length += 1
                cur = nxt
                if self._degree(cur, map_state) != 2:
                    break
            if length > best:
                best = length
        return best

    def _add_threat_heat(self, pac_pos, map_state):
        q = deque([(pac_pos, 0)])
        seen = {pac_pos}
        while q:
            cur, d = q.popleft()
            # ghi heat cả fog để tránh chạy vào vùng nguy hiểm
            self.threat_heat[cur] = self.threat_heat.get(cur, 0.0) + self._heat_add / (1.0 + d)
            if d >= self._heat_radius:
                continue
            for mv in _DIRS:
                np_pos = _apply(cur, mv)
                if self._pacman_walkable(np_pos, map_state) and np_pos not in seen:
                    seen.add(np_pos)
                    q.append((np_pos, d + 1))

    def _choose_from_top(self, scored, k=3, delta=200.0):
        if not scored:
            return None
        top = scored[: max(1, min(k, len(scored)))]
        best_score = top[0][0]
        cand = [x for x in top if (best_score - x[0]) <= delta]
        return random.choice(cand) if cand else top[0]
