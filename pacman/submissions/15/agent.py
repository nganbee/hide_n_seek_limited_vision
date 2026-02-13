import sys
from pathlib import Path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))
from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np
import random
import time

from heapq import heappush, heappop
from collections import deque, defaultdict

# Các hướng di chuyển cơ bản
DIRS = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
DIR_VECTORS = {
    Move.UP: (-1, 0),
    Move.DOWN: (1, 0),
    Move.LEFT: (0, -1),
    Move.RIGHT: (0, 1),
    Move.STAY: (0, 0)
}
# ============================================================================
# PACMAN AGENT 
# ============================================================================
class PacmanAgent(BasePacmanAgent):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Tốc độ mặc định của Pacman (thường là 2)
        self.speed = max(1, int(kwargs.get("pacman_speed", 2)))
        # Bản đồ ghi nhớ các ô đã đi qua (-1: chưa biết, 0: trống, 1: tường)
        self.beliefMap = np.full((21, 21), -1, dtype=np.int8)
        # Bản đồ xác suất vị trí Ghost
        self.ghostBelief = np.ones((21, 21), dtype=np.float32) / 441
        
        # Theo dõi các thông số
        self.lastEnemy = None
        self.lastSeenStep = -999
        self.lastMove = None
        self.myHistory = deque(maxlen=15)
        self.ghostHistory = deque(maxlen=20)
        
        # Phân tích cấu trúc bản đồ
        self.deadEnds = set()
        self.corridors = set()
        self.junctions = {}
        self.mapAnalyzed = False
        
        self.totalSteps = 0
        self.chaseMode = "aggressive"
    
    def step(self, map_state, my_position, enemy_position, step_number):
        """Correct interface"""
        t0 = time.time()
        self.totalSteps = step_number
        deadline = t0 + 0.95
        
        self.updateBelief(map_state)
        self.updateGhostBelief(map_state, my_position, enemy_position, step_number)
        self.myHistory.append(my_position)
        
        if not self.mapAnalyzed and step_number > 5:
            self.analyzeMapStructure()
            self.mapAnalyzed = True
        
        if enemy_position:
            self.lastEnemy = enemy_position
            self.lastSeenStep = step_number
            self.ghostHistory.append((enemy_position, step_number))
            
            if self.manhattan(my_position, enemy_position) <= 1:
                return (Move.STAY, 1)
        
        action = None
        
        # 1. Direct chase
        if enemy_position:
            timeBudget = min(0.85, deadline - time.time())
            if timeBudget > 0.1:
                action = self.iterativeDeepeningAstar(my_position, enemy_position, timeBudget)
        
        # 2. Recently lost
        elif (step_number - self.lastSeenStep) <= 20:  
            timeBudget = min(0.80, deadline - time.time())
            if timeBudget > 0.15:
                action = self.multiHypothesisSearch(my_position, timeBudget)
        
        # 3. Long lost
        else:
            action = self.strategicExploration(my_position, deadline)
        
        # 4. Endgame cutting
        if step_number > 150 and not action:
            action = self.endgameCutting(my_position, enemy_position)
        
        if action:
            self.lastMove = action[0]
        
        # Anti-loop
        if len(self.myHistory) >= 10:
            recent = list(self.myHistory)[-10:]
            if recent.count(my_position) >= 3:
                for move in DIRS:
                    for steps in ([2, 1] if self.speed >= 2 else [1]):
                        nextPos = self.advance(my_position, move, steps)
                        if nextPos and nextPos not in recent:
                            action = (move, steps)
                            break
                    if action and action != (Move.STAY, 1):
                        break
        
        # Fallback
        if not action or time.time() > deadline:
            target = enemy_position or self.bestBeliefPos()
            action = self.instantGreedy(my_position, target)
        
        # Final safety
        if action == (Move.STAY, 1) or (action and action[0] == Move.STAY):
            for move in DIRS:
                for steps in ([2, 1] if self.speed >= 2 else [1]):
                    nextPos = self.advance(my_position, move, steps)
                    if nextPos:
                        action = (move, steps)
                        break
                if action and action[0] != Move.STAY:
                    break
        
        return action
    
    # ------------------------------------------------------------------------
    # CÁC HÀM CẤP 1: TRUY ĐUỔI TRỰC DIỆN
    # ------------------------------------------------------------------------

    def iterativeDeepeningAstar(self, start, goal, timeBudget):
        """Tìm đường A* tăng dần độ sâu để đảm bảo thời gian xử lý"""
        deadline = time.time() + timeBudget
        dist = self.manhattan(start, goal)
        
        if dist <= 2:
            return self.instantGreedy(start, goal)
        
        # Thử tìm đường với độ sâu tăng dần
        for maxDepth in [5, 10, 15, 20, 30, 50]:
            if time.time() >= deadline:
                break
            result = self.boundedAstar(start, goal, maxDepth, deadline)
            if result:
                return result
        
        return self.instantGreedy(start, goal)
    
    def boundedAstar(self, start, goal, maxDepth, deadline):
        """Thuật toán A* giới hạn số bước đi"""
        pq = []
        gScore = {}
        parent = {}
        counter = 0
        
        # Thêm các nước đi đầu tiên vào hàng đợi ưu tiên
        for move in DIRS:
            for steps in ([1, 2] if self.speed >= 2 else [1]):
                pos = self.advance(start, move, steps)
                if not pos:
                    continue
                h = self.optimalHeuristic(pos, goal, move)
                if h > maxDepth:
                    continue
                state = (pos, move)
                gScore[state] = 1
                parent[state] = None
                heappush(pq, (1 + h, h, 1, counter, pos, move))
                counter += 1
        
        visited = set()
        while pq and time.time() < deadline:
            f, h, turns, _, pos, prevMove = heappop(pq)
            state = (pos, prevMove)
            if state in visited:
                continue
            visited.add(state)
            
            if self.manhattan(pos, goal) <= 1:
                return self.reconstructFirstAction(state, parent, start)
            
            if turns >= maxDepth:
                continue
                
            for move in DIRS:
                # Nếu đi thẳng thì được dùng tốc độ tối đa, nếu rẽ thì chỉ đi 1 ô
                maxSteps = self.speed if move == prevMove else 1
                for steps in range(1, maxSteps + 1):
                    nextPos = self.advance(pos, move, steps)
                    if not nextPos:
                        break
                    nextState = (nextPos, move)
                    nextTurns = turns + 1
                    if nextState in visited:
                        continue
                    if nextState not in gScore or nextTurns < gScore[nextState]:
                        gScore[nextState] = nextTurns
                        parent[nextState] = state
                        h = self.optimalHeuristic(nextPos, goal, move)
                        if nextTurns + h <= maxDepth:
                            heappush(pq, (nextTurns + h, h, nextTurns, counter, nextPos, move))
                            counter += 1
        return None

    def optimalHeuristic(self, pos, goal, currentDir):
        """
        Tính toán khoảng cách ước lượng TỐI ƯU cho Pacman tốc độ 2
        """
        dr = goal[0] - pos[0]
        dc = goal[1] - pos[1]
        
        if dr == 0 and dc == 0:
            return 0
        
        # Trường hợp đi thẳng
        if dr == 0 or dc == 0:
            dist = abs(dr + dc)
            targetDir = (Move.DOWN if dr > 0 else Move.UP if dr < 0 else
                        Move.RIGHT if dc > 0 else Move.LEFT)
            
            if currentDir == targetDir:
                # Đã align, dùng speed-2
                return (dist + self.speed - 1) // self.speed
            else:
                # Cần rẽ 1 turn trước
                return 1 + (dist + self.speed - 1) // self.speed
        
        # Trường hợp hình chữ L
        absdr = abs(dr)
        absdc = abs(dc)
        longSide = max(absdr, absdc)
        shortSide = min(absdr, absdc)
        
        # Tính turns cho cạnh dài
        longTurns = (longSide + self.speed - 1) // self.speed
        # Tính turns cho cạnh ngắn
        shortTurns = (shortSide + self.speed - 1) // self.speed
        
        # Kiểm tra xem direction hiện tại có thuận lợi không
        if currentDir in [Move.UP, Move.DOWN]:
            # Đang đi dọc
            if absdr >= absdc:
                # Nên tiếp tục dọc (aligned)
                return longTurns + 1 + shortTurns
            else:
                # Nên rẽ ngang trước
                return 1 + longTurns + shortTurns
        else:  # LEFT or RIGHT
            # Đang đi ngang
            if absdc >= absdr:
                # Nên tiếp tục ngang (aligned)
                return longTurns + 1 + shortTurns
            else:
                # Nên rẽ dọc trước
                return 1 + longTurns + shortTurns

    def reconstructFirstAction(self, finalState, parent, start):
        """Truy ngược lại bước đi đầu tiên từ kết quả A*"""
        path = []
        current = finalState
        while current and parent.get(current):
            path.append(current)
            current = parent[current]
        if not path:
            pos, move = finalState
            steps = self.countSteps(start, pos, move)
            return (move, min(steps, self.speed))
        firstState = path[-1]
        pos, move = firstState
        steps = self.countSteps(start, pos, move)
        return (move, min(steps, self.speed))

    def countSteps(self, start, end, move):
        """Đếm số ô thực tế giữa 2 điểm theo một hướng"""
        dr = end[0] - start[0]
        dc = end[1] - start[1]
        if move == Move.UP or move == Move.DOWN:
            return abs(dr)
        return abs(dc)

    # ------------------------------------------------------------------------
    # CÁC HÀM CẤP 2: TÌM KIẾM KHI MẤT DẤU
    # ------------------------------------------------------------------------

    def multiHypothesisSearch(self, myPos, timeBudget):
        """Tìm kiếm dựa trên nhiều giả thuyết vị trí của Ghost"""
        hypotheses = self.getWeightedHypotheses(k=12)  # Tăng từ 8 lên 12
        if not hypotheses:
            return self.strategicExploration(myPos, time.time() + timeBudget)
        
        actionScores = defaultdict(float)
        for move in DIRS:
            for steps in ([1, 2] if self.speed >= 2 else [1]):
                nextPos = self.advance(myPos, move, steps)
                if not nextPos:
                    continue
                action = (move, steps)
                score = 0
                for prob, ghostPos in hypotheses:
                    oldD = self.manhattan(myPos, ghostPos)
                    newD = self.manhattan(nextPos, ghostPos)
                    progress = (oldD - newD) * prob * 150
                    info = self.infoGain(nextPos) * 2
                    if newD > oldD:
                        progress *= 0.2
                    score += progress + info
                score += self.cuttingValue(nextPos, hypotheses) * 35  
        if actionScores:
            return max(actionScores.items(), key=lambda x: x[1])[0]
        return self.instantGreedy(myPos, hypotheses[0][1])

    def getWeightedHypotheses(self, k):
        """Lấy danh sách các vị trí có khả năng cao là Ghost đang ở đó"""
        candidates = []
        for r in range(21):
            for c in range(21):
                prob = self.ghostBelief[r, c]
                if prob > 0.005:
                    candidates.append((prob, (r, c)))
        candidates.sort(reverse=True)
        return candidates[:k]

    def cuttingValue(self, pacmanPos, hypotheses):
        """Tính điểm cho việc chặn đường thoát của Ghost"""
        value = 0
        for prob, ghostPos in hypotheses[:3]:
            for escapeDir in DIRS:
                escapePos = (ghostPos[0] + escapeDir.value[0], ghostPos[1] + escapeDir.value[1])
                if self.valid(escapePos):
                    ghostToEscape = self.manhattan(ghostPos, escapePos)
                    pacToEscape = self.manhattan(pacmanPos, escapePos)
                    if pacToEscape <= ghostToEscape:
                        value += prob * 10
        return value

    # ------------------------------------------------------------------------
    # CÁC HÀM CẤP 3: THĂM DÒ VÀ CHIẾN THUẬT CUỐI
    # ------------------------------------------------------------------------

    def strategicExploration(self, myPos, deadline):
        """Di chuyển đến những nơi có nhiều thông tin nhất"""
        bestAction = None
        bestScore = -1e9
        for move in DIRS:
            for steps in ([1, 2] if self.speed >= 2 else [1]):
                if time.time() >= deadline:
                    break
                nextPos = self.advance(myPos, move, steps)
                if not nextPos:
                    continue
                score = 0
                score += self.infoGain(nextPos) * 25
                score += self.probabilityCoverage(nextPos) * 30
                score += self.centroidScore(nextPos) * 15
                if nextPos in self.deadEnds:
                    score -= 50
                if nextPos not in self.myHistory:
                    score += 25
                else:
                    score -= 20
                if nextPos in self.junctions:
                    score += 15
                if score > bestScore:
                    bestScore = score
                    bestAction = (move, steps)
        return bestAction or (Move.STAY, 1)

    def probabilityCoverage(self, pos):
        """Tính xem từ vị trí này nhìn được bao nhiêu vùng xác suất của Ghost"""
        visibleProb = 0
        for direction in DIRS:
            for dist in range(1, 6):
                check = (pos[0] + direction.value[0] * dist, pos[1] + direction.value[1] * dist)
                if not (0 <= check[0] < 21 and 0 <= check[1] < 21):
                    break
                if self.beliefMap[check[0], check[1]] == 1:
                    break
                visibleProb += self.ghostBelief[check[0], check[1]]
        return visibleProb * 100

    def centroidScore(self, pos):
        """Điểm số dựa trên khoảng cách tới trung tâm của vùng nghi vấn"""
        centerR, centerC, totalProb = 0, 0, 0
        for r in range(21):
            for c in range(21):
                prob = self.ghostBelief[r, c]
                if prob > 0.01:
                    centerR += r * prob
                    centerC += c * prob
                    totalProb += prob
        if totalProb > 0:
            centerR /= totalProb
            centerC /= totalProb
            dist = abs(pos[0] - centerR) + abs(pos[1] - centerC)
            return max(0, 30 - dist)
        return 0

    def endgameCutting(self, myPos, enemyPos):
        """Chiến thuật ép góc Ghost vào giai đoạn cuối trận"""
        if not enemyPos:
            enemyPos = self.bestBeliefPos()
        escapeRoutes = []
        for direction in DIRS:
            escapePos = (enemyPos[0] + direction.value[0] * 3, enemyPos[1] + direction.value[1] * 3)
            if self.valid(escapePos):
                distToEscape = self.manhattan(myPos, escapePos)
                escapeRoutes.append((distToEscape, escapePos, direction))
        if escapeRoutes:
            escapeRoutes.sort()
            _, target, _ = escapeRoutes[0]
            return self.instantGreedy(myPos, target)
        return None
    
    # ------------------------------------------------------------------------
    # HỆ THỐNG CẬP NHẬT XÁC SUẤT (BAYESIAN)
    # ------------------------------------------------------------------------

    def updateGhostBelief(self, obs, myPos, enemyPos, step):
        """Cập nhật bản đồ xác suất vị trí Ghost theo thời gian"""
        if enemyPos:
            self.ghostBelief.fill(0)
            self.ghostBelief[enemyPos[0], enemyPos[1]] = 1.0
        else:
            stepsHidden = step - self.lastSeenStep
            self.ghostBelief = self.propagateBelief(self.ghostBelief, myPos, stepsHidden)
            self.applyNegativeEvidence(obs, myPos)
            total = self.ghostBelief.sum()
            if total > 1e-9:
                self.ghostBelief /= total
            else:
                self.ghostBelief.fill(0)
                valid = (self.beliefMap == 0) | (self.beliefMap == -1)
                self.ghostBelief[valid] = 1.0
                self.ghostBelief /= self.ghostBelief.sum()

    def propagateBelief(self, belief, pacmanPos, stepsHidden):
        """Dự đoán các hướng Ghost có thể đã đi khi bị khuất tầm nhìn"""
        newBelief = np.zeros_like(belief)
        # Trọng số ưu tiên Ghost chạy xa khỏi Pacman
        awayW, neutralW, closerW = 0.65, 0.25, 0.10
        if stepsHidden > 5:
            awayW, neutralW, closerW = 0.50, 0.30, 0.20
        
        for r in range(21):
            for c in range(21):
                if belief[r, c] < 1e-6:
                    continue
                ghostPos = (r, c)
                currentDist = self.manhattan(ghostPos, pacmanPos)
                moves = []
                for dr, dc in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
                    newPos = (r + dr, c + dc)
                    if not self.valid(newPos):
                        continue
                    newDist = self.manhattan(newPos, pacmanPos)
                    weight = awayW if newDist > currentDist else neutralW if newDist == currentDist else closerW
                    moves.append((newPos, weight))
                totalWeight = sum(w for _, w in moves)
                if totalWeight > 0:
                    for newPos, weight in moves:
                        newBelief[newPos[0], newPos[1]] += belief[r, c] * (weight / totalWeight)
        return newBelief

    def applyNegativeEvidence(self, obs, myPos):
        """Loại bỏ xác suất ở những ô trống mà Pacman nhìn thấy"""
        for direction in DIRS:
            for dist in range(1, 6):
                check = (myPos[0] + direction.value[0] * dist, myPos[1] + direction.value[1] * dist)
                if not (0 <= check[0] < 21 and 0 <= check[1] < 21):
                    break
                if obs[check[0], check[1]] == 0:
                    self.ghostBelief[check[0], check[1]] = 0
                if obs[check[0], check[1]] == 1:
                    break

    def bestBeliefPos(self):
        """Vị trí có xác suất Ghost cao nhất"""
        maxIdx = self.ghostBelief.argmax()
        r, c = maxIdx // 21, maxIdx % 21
        if self.ghostBelief[r, c] < 0.01:
            return self.lastEnemy if self.lastEnemy else (10, 10)
        return (r, c)

    # ------------------------------------------------------------------------
    # CÁC HÀM TIỆN ÍCH HỆ THỐNG
    # ------------------------------------------------------------------------

    def analyzeMapStructure(self):
        """Phân tích cấu trúc mê cung để tìm ngõ cụt và ngã rẽ"""
        for r in range(21):
            for c in range(21):
                if self.beliefMap[r, c] != 0:
                    continue
                pos = (r, c)
                exits = self.countExits(pos)
                if exits == 1:
                    self.deadEnds.add(pos)
                elif exits == 2:
                    self.corridors.add(pos)
                elif exits >= 3:
                    self.junctions[pos] = exits

    def countExits(self, pos):
        """Đếm số lối thoát từ một ô"""
        return sum(1 for d in DIRS if self.valid((pos[0] + d.value[0], pos[1] + d.value[1])))

    def infoGain(self, pos):
        """Đếm số ô 'mù' sẽ được soi sáng từ vị trí này"""
        count = 0
        for direction in DIRS:
            for dist in range(1, 6):
                check = (pos[0] + direction.value[0] * dist, pos[1] + direction.value[1] * dist)
                if not (0 <= check[0] < 21 and 0 <= check[1] < 21):
                    break
                if self.beliefMap[check[0], check[1]] == -1:
                    count += 1
                if self.beliefMap[check[0], check[1]] == 1:
                    break
        return count

    def instantGreedy(self, pos, target):
        """Thuật toán di chuyển tham lam siêu nhanh"""
        dr = target[0] - pos[0]
        dc = target[1] - pos[1]
        primary = (Move.DOWN if dr > 0 else Move.UP) if abs(dr) >= abs(dc) else (Move.RIGHT if dc > 0 else Move.LEFT)
        secondary = (Move.RIGHT if dc > 0 else Move.LEFT) if abs(dr) >= abs(dc) else (Move.DOWN if dr > 0 else Move.UP)
        
        for move in [primary, secondary]:
            if move == self.lastMove and self.speed >= 2:
                if self.advance(pos, move, 2):
                    return (move, 2)
            if self.advance(pos, move, 1):
                return (move, 1)
        
        for move in DIRS:
            if self.advance(pos, move, 1):
                return (move, 1)
        return (Move.STAY, 1)

    def advance(self, pos, move, steps):
        """Kiểm tra việc đi tới n bước có hợp lệ không"""
        newPos = pos
        for _ in range(steps):
            candidate = (newPos[0] + move.value[0], newPos[1] + move.value[1])
            if not self.valid(candidate):
                return None
            newPos = candidate
        return newPos

    def valid(self, pos):
        """Kiểm tra ô có nằm trong bản đồ và không phải là tường"""
        r, c = pos
        return 0 <= r < 21 and 0 <= c < 21 and self.beliefMap[r, c] != 1

    def manhattan(self, a, b):
        """Tính khoảng cách Manhattan giữa 2 điểm"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def updateBelief(self, obs):
        """Cập nhật bản đồ ghi nhớ từ quan sát hiện tại"""
        mask = obs != -1
        self.beliefMap[mask] = obs[mask]

# ============================================================================
# GHOST AGENT 
# ============================================================================

class GhostAgent(BaseGhostAgent):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.beliefMap = np.full((21, 21), -1, dtype=np.int8)
        self.lastPacman = None
        self.lastSeenStep = -999
        self.pacmanHistory = deque(maxlen=25)
        self.myHistory = deque(maxlen=12)
        self.currentStrategy = "evade"
        self.strategyTimer = 0
        self.movePatterns = defaultdict(int)
        self.lastMoves = deque(maxlen=5)
        self.randomness = 0.12 
        self.safeZones = []
        self.hidingSpots = []
        self.mapLearned = False
        self.survivalMode = False
    
    def step(self, map_state, my_position, enemy_position, step_number):
        """Correct interface"""
        self.updateBeliefMap(map_state)
        self.trackHistory(my_position, enemy_position, step_number)
        
        if not self.mapLearned and step_number > 8:
            self.learnMap()
            self.mapLearned = True
            
        self.strategyTimer += 1
        if self.strategyTimer >= 12:
            self.selectStrategy(my_position, enemy_position, step_number)
            self.strategyTimer = 0
        
        if enemy_position:
            threat = self.assessThreat(my_position, enemy_position)
            if threat >= 5:
                return self.emergencyEscape(my_position, enemy_position)
            elif threat >= 4:
                return self.tacticalRetreat(my_position, enemy_position)
            elif threat >= 3:
                return self.evasiveMovement(my_position, enemy_position)
        
        if self.currentStrategy == "evade":
            return self.activeEvasion(my_position, enemy_position, step_number)
        elif self.currentStrategy == "hide":
            return self.strategicHiding(my_position, enemy_position, step_number)
        elif self.currentStrategy == "deceive":
            return self.deceptiveTactics(my_position, enemy_position)
        else:
            return self.safeRandomMove(my_position, enemy_position)
    
    def selectStrategy(self, myPos, enemyPos, step):
        """Chọn chiến thuật dựa trên trạng thái game"""
        if not enemyPos:
            timeHidden = step - self.lastSeenStep
            if timeHidden > 25:
                self.currentStrategy = "hide"
            elif timeHidden > 12:
                self.currentStrategy = random.choice(["hide", "deceive"])
            else:
                self.currentStrategy = "evade"
        else:
            dist = self.manhattan(myPos, enemyPos)
            if dist < 8:
                self.currentStrategy = "evade"
            elif dist < 15:
                self.currentStrategy = random.choice(["evade", "deceive"])
            else:
                self.currentStrategy = "hide"
        
        if step > 160:
            self.survivalMode = True
            self.currentStrategy = "evade"
    
    def assessThreat(self, myPos, pacPos):
        """Đánh giá mức độ đe dọa từ Pacman"""
        dist = self.manhattan(myPos, pacPos)
        effDist = dist / 2.2
        if myPos[0] == pacPos[0] or myPos[1] == pacPos[1]:
            effDist *= 0.6
        if effDist <= 0.5: return 5
        if effDist <= 1.5: return 4
        if effDist <= 3: return 3
        if effDist <= 6: return 2
        if effDist <= 12: return 1
        return 0
    
    def emergencyEscape(self, myPos, pacPos):
        """ULTRA Emergency Escape với Minimax prediction"""
        bestMove = Move.STAY
        bestScore = -1e10
        
        for move in DIRS:
            nxt = self.movePos(myPos, move)
            if not self.valid(nxt):
                continue
            dist = self.manhattan(nxt, pacPos)
            pathLen = self.quickBfsDist(nxt, pacPos, limit=15)
            score = pathLen * 100 if pathLen else dist * 80
            if nxt[0] != pacPos[0] and nxt[1] != pacPos[1]:
                score += 200
            score += self.countExits(nxt) * 30
            if score > bestScore:
                bestScore = score
                bestMove = move
        return bestMove

    def tacticalRetreat(self, myPos, pacPos):
        """Rút lui chiến thuật, ưu tiên đường thông thoáng"""
        bestMove = Move.STAY
        bestScore = -1e9
        for move in DIRS:
            nxt = self.movePos(myPos, move)
            if not self.valid(nxt):
                continue
            dist = self.manhattan(nxt, pacPos)
            exits = self.countExits(nxt)
            score = dist * 70
            if exits == 0: score -= 5000
            elif exits == 1: score -= 1000
            else: score += exits * 40
            score += self.futureMobility(nxt) * 8
            if nxt[0] != pacPos[0] and nxt[1] != pacPos[1]:
                score += 100
            score += self.nearbyWalls(nxt) * 12
            if score > bestScore:
                bestScore = score
                bestMove = move
        return bestMove

    def evasiveMovement(self, myPos, pacPos):
        """Né tránh ở mức độ nguy hiểm trung bình"""
        return self.evaluateMoves(myPos, pacPos, {
            'distance': 50, 'exits': 35, 'walls': 15, 'avoid_line': 80, 'future_mobility': 10
        })

    def activeEvasion(self, myPos, enemyPos, step):
        """Né tránh chủ động, làm cho hướng đi khó đoán"""
        if len(self.lastMoves) >= 3:
            pattern = tuple(self.lastMoves)[-3:]
            self.movePatterns[pattern] += 1
            if self.movePatterns[pattern] > 2:
                return self.breakPattern(myPos, enemyPos, pattern)
        if random.random() < self.randomness:
            return self.safeRandomMove(myPos, enemyPos)
        pacPos = enemyPos or self.lastPacman
        if pacPos:
            return self.evaluateMoves(myPos, pacPos, {
                'distance': 60, 'exits': 30, 'walls': 10, 'avoid_line': 100, 'complexity': 20
            })
        return self.safeRandomMove(myPos, None)

    def strategicHiding(self, myPos, enemyPos, step):
        """Đi tìm các chỗ trốn kín đáo"""
        if not self.hidingSpots or step % 20 == 0:
            self.hidingSpots = self.findHidingSpots(myPos, enemyPos)
        if self.hidingSpots:
            target = self.hidingSpots[0]
            if self.manhattan(myPos, target) <= 1:
                self.hidingSpots.pop(0)
                if self.hidingSpots: target = self.hidingSpots[0]
            if random.random() < 0.15:
                return self.safeRandomMove(myPos, enemyPos)
            return self.moveToward(myPos, target)
        return self.safeRandomMove(myPos, enemyPos)

    def deceptiveTactics(self, myPos, enemyPos):
        """Chiến thuật đánh lạc hướng"""
        if enemyPos and random.random() < 0.20:
            return self.moveToward(myPos, enemyPos)
        if len(self.lastMoves) >= 2:
            perp = self.perpendicular(self.lastMoves[-1])
            valid = [m for m in perp if self.valid(self.movePos(myPos, m))]
            if valid and random.random() < 0.7:
                return random.choice(valid)
        return self.safeRandomMove(myPos, enemyPos)

    def evaluateMoves(self, myPos, pacPos, weights):
        """Chấm điểm các nước đi khả thi"""
        bestMove = Move.STAY
        bestScore = -1e9
        for move in DIRS:
            nxt = self.movePos(myPos, move)
            if not self.valid(nxt):
                continue
            exits = self.countExits(nxt)
            score = self.manhattan(nxt, pacPos) * weights.get('distance', 50)
            if exits == 0: score -= 10000
            elif exits == 1: score -= 2000
            else: score += exits * weights.get('exits', 30)
            score += self.nearbyWalls(nxt) * weights.get('walls', 10)
            if nxt[0] != pacPos[0] and nxt[1] != pacPos[1]:
                score += weights.get('avoid_line', 80)
            if nxt in self.myHistory: score -= 40
            if score > bestScore:
                bestScore = score
                bestMove = move
        self.lastMoves.append(bestMove)
        return bestMove

    def breakPattern(self, myPos, pacPos, pattern):
        """Cố tình đi khác đi để không bị lặp lại thói quen"""
        diff = [m for m in DIRS if m not in pattern]
        valid = [m for m in diff if self.valid(self.movePos(myPos, m))]
        if valid:
            res = max(valid, key=lambda m: self.manhattan(self.movePos(myPos, m), pacPos)) if pacPos else random.choice(valid)
            self.lastMoves.append(res)
            return res
        return self.safeRandomMove(myPos, pacPos)

    def learnMap(self):
        """Học các vùng an toàn (nhiều tường, nhiều lối thoát)"""
        for r in range(21):
            for c in range(21):
                if self.beliefMap[r, c] == 0:
                    if self.countExits((r, c)) >= 3 and self.nearbyWalls((r, c)) >= 4:
                        self.safeZones.append((r, c))

    def findHidingSpots(self, myPos, enemyPos):
        """Tìm các điểm ẩn nấp tốt nhất"""
        pacPos = enemyPos or self.lastPacman or (10, 10)
        spots = []
        for r in range(1, 20):
            for c in range(1, 20):
                if self.beliefMap[r, c] == 0 and self.countExits((r, c)) >= 2:
                    score = self.manhattan((r, c), pacPos) * 12 + self.countExits((r, c)) * 18 + self.nearbyWalls((r, c)) * 10
                    spots.append((score, (r, c)))
        spots.sort(reverse=True)
        return [pos for _, pos in spots[:12]]

    def moveToward(self, myPos, target):
        """Di chuyển một bước về phía mục tiêu"""
        dr, dc = target[0] - myPos[0], target[1] - myPos[1]
        moves = [Move.DOWN if dr > 0 else Move.UP, Move.RIGHT if dc > 0 else Move.LEFT] if abs(dr) > abs(dc) else [Move.RIGHT if dc > 0 else Move.LEFT, Move.DOWN if dr > 0 else Move.UP]
        for m in moves:
            if self.valid(self.movePos(myPos, m)):
                self.lastMoves.append(m)
                return m
        return self.safeRandomMove(myPos, None)

    def safeRandomMove(self, myPos, enemyPos):
        """Di chuyển ngẫu nhiên nhưng vẫn đảm bảo an toàn"""
        valid = []
        for move in DIRS:
            nxt = self.movePos(myPos, move)
            if self.valid(nxt) and self.countExits(nxt) >= 2:
                if not enemyPos or self.manhattan(nxt, enemyPos) >= self.manhattan(myPos, enemyPos):
                    valid.append(move)
        res = random.choice(valid) if valid else Move.STAY
        self.lastMoves.append(res)
        return res

    def perpendicular(self, move):
        """Lấy các hướng vuông góc với hướng hiện tại"""
        return [Move.LEFT, Move.RIGHT] if move in [Move.UP, Move.DOWN] else [Move.UP, Move.DOWN]

    def futureMobility(self, pos):
        """Đếm số ô có thể đi tới trong 2 bước nữa"""
        reach = set()
        for m1 in DIRS:
            p1 = self.movePos(pos, m1)
            if self.valid(p1):
                reach.add(p1)
                for m2 in DIRS:
                    p2 = self.movePos(p1, m2)
                    if self.valid(p2): reach.add(p2)
        return len(reach)

    def countExits(self, pos):
        """Đếm số lối thoát ngay lập tức"""
        return sum(1 for m in DIRS if self.valid(self.movePos(pos, m)))

    def nearbyWalls(self, pos):
        """Đếm số tường bao quanh trong vùng 3x3"""
        count = 0
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0: continue
                r, c = pos[0] + dr, pos[1] + dc
                if 0 <= r < 21 and 0 <= c < 21 and self.beliefMap[r, c] == 1:
                    count += 1
        return count

    def quickBfsDist(self, start, goal, limit):
        """Tìm khoảng cách ngắn nhất bằng BFS nhanh"""
        q, visited = deque([(start, 0)]), {start}
        while q:
            p, d = q.popleft()
            if p == goal: return d
            if d < limit:
                for m in DIRS:
                    nxt = self.movePos(p, m)
                    if nxt not in visited and self.valid(nxt):
                        visited.add(nxt); q.append((nxt, d + 1))
        return None

    def trackHistory(self, myPos, enemyPos, step):
        """Lưu lại lịch sử di chuyển"""
        self.myHistory.append(myPos)
        if enemyPos:
            self.lastPacman, self.lastSeenStep = enemyPos, step
            self.pacmanHistory.append((enemyPos, step))

    def movePos(self, pos, move):
        """Tính vị trí mới sau khi di chuyển"""
        return (pos[0] + move.value[0], pos[1] + move.value[1])

    def valid(self, pos):
        """Kiểm tra ô có hợp lệ để đi vào không"""
        return 0 <= pos[0] < 21 and 0 <= pos[1] < 21 and self.beliefMap[pos] != 1

    def manhattan(self, a, b):
        """Tính khoảng cách Manhattan"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def updateBeliefMap(self, obs):
        """Cập nhật bản đồ quan sát của Ghost"""
        mask = obs != -1
        self.beliefMap[mask] = obs[mask]
