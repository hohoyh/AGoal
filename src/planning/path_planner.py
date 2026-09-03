# src/planning/path_planner.py
import numpy as np
import heapq
from scipy.ndimage import binary_dilation, distance_transform_edt

class PathPlanner:
    """
    基于占据栅格的A*规划器：
    - grid: 2D numpy，-1=障碍, 1=自由
    - resolution: 米/格
    - safety_margin: 安全边（米），通过膨胀实现
    """
    def __init__(self, occupancy_grid, resolution=0.5, safety_margin=2.0):
        self.resolution = float(resolution)
        self.safety_margin = float(safety_margin)
        self._set_grid(occupancy_grid)
        
        # ✅ 修复: 为 'nearest_free' 预先计算距离场
        self._precompute_distance_transform()

    def _set_grid(self, grid):
        # 统一为 0/1：1=障碍
        g = np.array(grid, dtype=np.int8)
        
        # ✅ 修复: -1 是障碍 (Obstacle), 1 是自由 (Free)
        # g = (g > 0).astype(np.uint8) # <-- 错误: 这会把 'Free' 设为障碍
        g = (g < 0).astype(np.uint8)  # <-- 正确: 只有 -1 (Obstacle) 才是障碍
        
        # 膨胀障碍确保安全边
        safety_cells = max(1, int(self.safety_margin / self.resolution))
        if safety_cells > 0:
            print(f"[PathPlanner] 膨胀障碍物: {safety_cells} 格 ({self.safety_margin}m)")
            structure = np.ones((safety_cells * 2 + 1, safety_cells * 2 + 1), dtype=np.uint8)
            inflated = binary_dilation(g == 1, structure=structure)
            self.grid = (inflated > 0).astype(np.uint8)  # 1=障碍, 0=可行
        else:
            self.grid = g
            
    def _precompute_distance_transform(self):
        """✅ 新增: 预计算距离场, 用于快速查找 'nearest_free'"""
        print("[PathPlanner] 预计算自由空间距离场...")
        # (self.grid == 0) 是自由空间
        self.dist_transform, self.idx_transform = distance_transform_edt(
            self.grid == 0, return_indices=True
        )

    # ——坐标换算与可行性——
    def world_to_grid(self, x, y):
        H, W = self.grid.shape
        # ✅ 修正: 确保 (x,y) 落在 (0, W-1) 和 (0, H-1) 范围内
        gx = int(W / 2 + x / self.resolution)
        gy = int(H / 2 - y / self.resolution)
        gx = max(0, min(W - 1, gx))
        gy = max(0, min(H - 1, gy))
        return gx, gy

    def grid_to_world(self, gx, gy):
        H, W = self.grid.shape
        x = (gx - W / 2) * self.resolution
        y = (H / 2 - gy) * self.resolution
        return float(x), float(y)

    def _in_bounds(self, p):
        gx, gy = p
        H, W = self.grid.shape
        return 0 <= gx < W and 0 <= gy < H

    def _is_free(self, p):
        gx, gy = p
        # ✅ 修复: 检查边界, 并且 grid[gy, gx] == 0 (0是可行)
        return self._in_bounds(p) and self.grid[gy, gx] == 0

    def _nearest_free(self, p, max_radius=30):
        """✅ 修复: 使用预计算的距离场快速查找"""
        gx, gy = p
        
        # 1. 如果点在界外, 返回 None
        if not self._in_bounds(p):
            return None
        
        # 2. 如果点本身是自由的, 直接返回
        if self._is_free(p):
            return p
            
        # 3. 如果点在障碍物内
        # dist_transform 在障碍物处的值 > 0, 表示离最近的自由空间有多远
        dist_to_free = self.dist_transform[gy, gx]
        
        if dist_to_free * self.resolution > max_radius:
            print(f"  [A* 警告] 最近的自由点在 {max_radius}m 之外, 放弃。")
            return None
            
        # 4. 直接从 idx_transform 查表得到最近的自由点坐标
        nearest_gy = self.idx_transform[0, gy, gx]
        nearest_gx = self.idx_transform[1, gy, gx]
        
        nearest_p = (int(nearest_gx), int(nearest_gy))
        
        # print(f"    [A* 调试] 点 {p} 在障碍中, 找到最近的自由点 {nearest_p}")
        return nearest_p

    # ——A*主体——
    def _astar(self, start_g, goal_g):
        def h(a, b):
            return np.hypot(a[0]-b[0], a[1]-b[1])

        # ✅ 修复: 使用新的 _nearest_free
        start_g = self._nearest_free(start_g)
        goal_g = self._nearest_free(goal_g)
        
        if start_g is None or goal_g is None:
            print("  [A* 错误] 无法找到起点或终点的有效自由空间。")
            return None

        # (安全检查, _nearest_free 应该已经保证了这一点)
        if not self._is_free(start_g) or not self._is_free(goal_g):
            print("  [A* 错误] 起点或终点仍在障碍物中。")
            return None

        dirs = [(-1,0),(1,0),(0,-1),(0,1), (-1,-1),(-1,1),(1,-1),(1,1)]
        openq = []
        heapq.heappush(openq, (0.0, start_g))
        came = {}
        gscore = {start_g: 0.0}
        fscore = {start_g: h(start_g, goal_g)}

        while openq:
            _, cur = heapq.heappop(openq)
            if cur == goal_g:
                path = [cur]
                while cur in came:
                    cur = came[cur]
                    path.append(cur)
                return path[::-1]

            for dx, dy in dirs:
                nxt = (cur[0]+dx, cur[1]+dy)
                if not self._is_free(nxt):
                    continue
                cost = 1.414 if dx != 0 and dy != 0 else 1.0
                tg = gscore[cur] + cost
                if nxt not in gscore or tg < gscore[nxt]:
                    came[nxt] = cur
                    gscore[nxt] = tg
                    fscore[nxt] = tg + h(nxt, goal_g)
                    heapq.heappush(openq, (fscore[nxt], nxt))
        return None

    def plan_path(self, start_world_xy, goal_world_xy, stride=6):
        """返回世界坐标路径: [[x,y], ...]"""
        sg = self.world_to_grid(*start_world_xy)
        gg = self.world_to_grid(*goal_world_xy)
        path_g = self._astar(sg, gg)
        if not path_g:
            return []

        # 简化：步进抽样
        if stride < 2:
            stride = 2
        sampled = path_g[::stride]
        if sampled[-1] != path_g[-1]:
            sampled.append(path_g[-1])

        waypoints = [self.grid_to_world(gx, gy) for gx, gy in sampled]
        return waypoints

    # ——可选：用局部深度临时更新——
    def update_with_local_depth_grid(self, local_grid):
        """允许传入一张局部占据网格（0/1），镶嵌到全局（简单合并：OR）"""
        if local_grid.shape != self.grid.shape:
            # 尺寸不一致就跳过（也可以自己写对齐）
            return
        self.grid = np.clip(self.grid | (local_grid > 0).astype(np.uint8), 0, 1)