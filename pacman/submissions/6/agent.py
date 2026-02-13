import sys
from pathlib import Path

# Add src to path to import the interface
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
from heapq import heappush, heappop
import numpy as np
from collections import deque
import heapq
import random

class PacmanAgent(BasePacmanAgent):
    """
    Pacman (Seek) agent cho Project 2 - Limited Vision

    Chiến lược:
    - Nếu thấy Ghost: đi 1 bước để giảm Manhattan distance
    - Nếu không thấy Ghost: khám phá an toàn (đi các ô nhìn thấy, không phải tường)
    - KHÔNG giả định ô -1 là đi được
    - Mỗi step chỉ trả về 1 Move
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = kwargs.get("name", "LimitedVisionPacman")

        # tránh lặp vị trí
        self.recent_positions = []
        self.max_recent = 5

        self.moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]

    # -------------------------
    # Helper functions
    # -------------------------
    def manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def apply_move(self, pos, move):
        dr, dc = move.value
        return (pos[0] + dr, pos[1] + dc)

    def is_valid(self, pos, map_state):
        """
        Ô hợp lệ nếu:
        - trong bản đồ
        - KHÔNG phải tường
        - KHÔNG phải unknown (-1)
        """
        r, c = pos
        h, w = map_state.shape
        if not (0 <= r < h and 0 <= c < w):
            return False
        return map_state[r, c] == 0

    # -------------------------
    # Main decision function
    # -------------------------
    def step(self, map_state: np.ndarray, my_position: tuple,
             enemy_position: tuple, step_number: int):

        # =========================
        # CASE 1: THẤY GHOST → ĐUỔI
        # =========================
        if enemy_position is not None:
            best_move = Move.STAY
            best_dist = self.manhattan(my_position, enemy_position)

            for move in self.moves:
                nxt = self.apply_move(my_position, move)
                if self.is_valid(nxt, map_state):
                    d = self.manhattan(nxt, enemy_position)
                    if d < best_dist:
                        best_dist = d
                        best_move = move

            self._remember(my_position, best_move)
            return best_move

        # ==================================
        # CASE 2: KHÔNG THẤY GHOST → EXPLORE
        # ==================================
        for move in self.moves:
            nxt = self.apply_move(my_position, move)
            if self.is_valid(nxt, map_state) and nxt not in self.recent_positions:
                self._remember(my_position, move)
                return move

        # ==================================
        # CASE 3: FALLBACK – đi bất kỳ ô hợp lệ
        # ==================================
        for move in self.moves:
            nxt = self.apply_move(my_position, move)
            if self.is_valid(nxt, map_state):
                self._remember(my_position, move)
                return move

        # ==================================
        # CASE 4: BÍ → STAY
        # ==================================
        return Move.STAY

    # -------------------------
    # Memory helper
    # -------------------------
    def _remember(self, cur_pos, move):
        nxt = self.apply_move(cur_pos, move)
        self.recent_positions.append(nxt)
        if len(self.recent_positions) > self.max_recent:
            self.recent_positions.pop(0)


class GhostAgent(BaseGhostAgent):
    """
    Ghost sử dụng chiến lược né Pacman theo kiểu Greedy cục bộ.

    Ý tưởng chính:
    - Ở mỗi lượt, Ghost chỉ xem xét các ô kề xung quanh
    - Với mỗi hướng đi, Ghost đánh giá mức độ an toàn
      thông qua khoảng cách đến Pacman
    - Chọn hướng làm Ghost ở xa Pacman nhất
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Lưu vị trí Pacman lần cuối Ghost nhìn thấy
        # Dùng khi Ghost bị hạn chế tầm nhìn (fog)
        self.last_known_enemy_pos = None

    # ================= MAIN =================

    def step(self, map_state, my_position, enemy_position, step_number):

        # Nếu đang nhìn thấy Pacman thì cập nhật trí nhớ
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position

        # Mối đe dọa:
        # - Pacman hiện tại nếu nhìn thấy
        # - Hoặc vị trí Pacman đã thấy gần nhất
        threat = enemy_position or self.last_known_enemy_pos

        best_move = Move.STAY
        best_dist = -1  # dùng để tìm khoảng cách lớn nhất

        # Thử tất cả các hướng di chuyển có thể
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            nxt = (my_position[0] + move.value[0],
                   my_position[1] + move.value[1])

            # Bỏ qua hướng không hợp lệ (tường hoặc ra ngoài bản đồ)
            if not self._is_walkable(nxt, map_state):
                continue

            # Đánh giá độ an toàn của ô kế tiếp
            # Ghost càng xa Pacman thì càng an toàn
            dist = self._manhattan(nxt, threat) if threat else 0

            # Chọn hướng làm khoảng cách tới Pacman lớn nhất
            if dist > best_dist:
                best_dist = dist
                best_move = move

        # Ghost chỉ di chuyển 1 bước mỗi lượt
        return best_move

    # ================= HÀM HỖ TRỢ =================

    def _is_walkable(self, pos, map_state):
        r, c = pos
        h, w = map_state.shape
        return 0 <= r < h and 0 <= c < w and map_state[r, c] != 1

    def _manhattan(self, a, b):
        if b is None:
            return 0
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
