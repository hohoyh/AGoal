# src/navigation/dstar_lite.py
import math
import heapq

INF = float('inf')

class DStarLite:
    def __init__(self, grid, start, goal, heuristic=None):
        # grid: 2D numpy array: -1 obstacle, 0 unknown, 1 free
        self.grid = grid
        self.rows = grid.shape[0]
        self.cols = grid.shape[1]
        self.km = 0
        self.start = start
        self.goal = goal
        self.rhs = {}
        self.g = {}
        self.U = []  # priority queue of (key, node)
        self.parent = {}
        self.heuristic = heuristic if heuristic else (lambda a,b: abs(a[0]-b[0]) + abs(a[1]-b[1]))
        # initialize
        for r in range(self.rows):
            for c in range(self.cols):
                self.g[(r,c)] = INF
                self.rhs[(r,c)] = INF
        self.rhs[self.goal] = 0
        self._push(self.goal, self._calculate_key(self.goal))

    def _calculate_key(self, node):
        g = self.g[node]
        rhs = self.rhs[node]
        val = (min(g, rhs) + self.heuristic(self.start, node) + self.km, min(g, rhs))
        return val

    def _push(self, node, key):
        heapq.heappush(self.U, (key, node))

    def _pop(self):
        if not self.U:
            return None, None
        key, node = heapq.heappop(self.U)
        return key, node

    def _neighbors(self, node):
        r, c = node
        nbs = []
        for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
            nr, nc = r+dr, c+dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                if self.grid[nr, nc] != -1:  # not obstacle
                    nbs.append((nr, nc))
        return nbs

    def cost(self, a, b):
        # assume uniform cost 1 for neighbor
        return 1

    def update_vertex(self, u):
        if u != self.goal:
            min_rhs = INF
            for s in self._neighbors(u):
                min_rhs = min(min_rhs, self.g[s] + self.cost(u, s))
            self.rhs[u] = min_rhs
        # remove u from queue if present -- lazy removal: we will check keys when popping
        self._push(u, self._calculate_key(u))

    def _key_less(self, a, b):
        # a, b 都是 (k1, k2)
        return a[0] < b[0] or (a[0] == b[0] and a[1] < b[1])

    def compute_shortest_path(self, max_iters=500000):
        iters = 0
        while self.U and (
            self._key_less(self.U[0][0], self._calculate_key(self.start)) or
            self.rhs[self.start] != self.g[self.start]
        ):
            key_old, u = self._pop()
            if u is None:
                break
            key_new = self._calculate_key(u)

            # ✅ 新增早停
            if self.rhs[self.start] == self.g[self.start]:
                print(f"[D*-Lite] 早停 at iter={iters}, path ready.")
                break

            if self._key_less(key_old, key_new):
                self._push(u, key_new)
            elif self.g[u] > self.rhs[u]:
                self.g[u] = self.rhs[u]
                for s in self._neighbors(u):
                    self.update_vertex(s)
            else:
                g_old = self.g[u]
                self.g[u] = INF
                self.update_vertex(u)
                for s in self._neighbors(u):
                    self.update_vertex(s)

            iters += 1
            if iters > max_iters:
                print(f"[D*-Lite] 超过最大迭代 ({iters})，提前终止。")
                break


    def get_path(self):
        # build path from start by greedy following min(g + cost)
        path = []
        cur = self.start
        if self.g[cur] == INF:
            return []
        while cur != self.goal:
            nbs = self._neighbors(cur)
            if not nbs:
                return []
            minn = None
            minval = INF
            for s in nbs:
                val = self.g[s] + self.cost(cur, s)
                if val < minval:
                    minval = val
                    minn = s
            if minn is None or self.g[minn] == INF:
                return []
            path.append(minn)
            cur = minn
            if len(path) > (self.rows*self.cols):
                break
        return path
