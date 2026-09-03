# src/mapping/stmr_builder.py
# STMR: Semantic Top-down Map Representation —— 由 AirSim 分割图增量构建的俯视语义栅格

import numpy as np
import cv2

class STMRBuilder:
    """
    使用 AirSim segmentation 图自动建图
    支持高质量可视化输出
    """

    def __init__(self, map_size=100, grid_size=5):
        print("\n[STMR] 初始化 AirSim Segmentation 自适应版...")
        self.map_size = map_size
        self.grid_size = grid_size
        self.grid_num = map_size // grid_size
        self.topdown_map = np.zeros((self.grid_num, self.grid_num), dtype=np.int32)
        self.color_to_id = {}  # 自动生成颜色-ID 映射
        self.next_id = 1       # 从1开始编号
        self.trajectory = []   # 轨迹记录
        
        # ✅ 预定义美观的颜色映射
        self.predefined_colors = {
            0: [240, 240, 240],   # 未探索 - 浅灰
            1: [169, 169, 169],   # 建筑 - 深灰
            2: [144, 238, 144],   # 植被 - 浅绿
            3: [139, 119, 101],   # 道路 - 褐色
            4: [100, 149, 237],   # 车辆 - 浅蓝
            5: [255, 228, 196],   # 天空 - 浅橙
            6: [34, 139, 34],     # 树木 - 深绿
            7: [210, 180, 140],   # 地面 - 卡其色
            -1: [255, 69, 0],     # 轨迹 - 橙红色
        }
        
        print("  ✓ STMR 初始化完成 (使用 AirSim Segmentation 自动颜色映射)")

    # -----------------------------------------------------
    def update_map(self, seg_image, drone_pos):
        """根据 AirSim segmentation 图更新语义地图"""
        H, W = seg_image.shape[:2]
        updated_count = 0
        unique_colors = set()

        for y in range(0, H, 2):
            for x in range(0, W, 2):
                color = tuple(seg_image[y, x])  # (R,G,B)
                if color == (0, 0, 0):
                    continue

                unique_colors.add(color)

                if color not in self.color_to_id:
                    self.color_to_id[color] = self.next_id
                    self.next_id += 1

                cls = self.color_to_id[color]

                # ✅ 坐标转换
                gx = int((drone_pos[0] + self.map_size/2) / self.grid_size)
                gy = int((self.map_size/2 - drone_pos[1]) / self.grid_size)
                
                if 0 <= gx < self.grid_num and 0 <= gy < self.grid_num:
                    self.topdown_map[gy, gx] = cls
                    updated_count += 1

        print(f"  [STMR] 本帧更新: {updated_count} 个网格, 发现 {len(unique_colors)} 种颜色")
        print(f"  [STMR] 地图非零元素数: {np.count_nonzero(self.topdown_map)}")

        self.update_trajectory(drone_pos)

    # -----------------------------------------------------
    def update_trajectory(self, drone_pos):
        """更新轨迹点"""
        self.trajectory.append([drone_pos[0], drone_pos[1]])
        
        gx = int((drone_pos[0] + self.map_size/2) / self.grid_size)
        gy = int((self.map_size/2 - drone_pos[1]) / self.grid_size)
        if 0 <= gx < self.grid_num and 0 <= gy < self.grid_num:
            if self.topdown_map[gy, gx] == 0:
                self.topdown_map[gy, gx] = -1

    # -----------------------------------------------------
    def world_to_grid(self, world_pos):
        """世界坐标 -> 网格坐标"""
        gx = int((world_pos[0] + self.map_size/2) / self.grid_size)
        gy = int((self.map_size/2 - world_pos[1]) / self.grid_size)
        return [gx, gy]

    # -----------------------------------------------------
    def get_local_map(self, drone_pos, local_size=20):
        """获取局部地图"""
        cx = int((drone_pos[0] + self.map_size/2) / self.grid_size)
        cy = int((self.map_size/2 - drone_pos[1]) / self.grid_size)
        half = local_size // 2
        xs, xe = max(0, cx - half), min(self.grid_num, cx + half)
        ys, ye = max(0, cy - half), min(self.grid_num, cy + half)
        local = self.topdown_map[ys:ye, xs:xe]
        pad = np.zeros((local_size, local_size), dtype=np.int32)
        pad[:local.shape[0], :local.shape[1]] = local
        return pad, [half, half]

    # =====================================================
    # ✅ 高质量可视化方法
    # =====================================================
    def visualize_topdown_map(self, save_path, drone_pos=None, high_quality=True):
        """生成高质量语义占据地图"""
        h, w = self.topdown_map.shape
        
        color_map = self._generate_color_map()
        
        if high_quality:
            scale = 4
            rgb_map = np.zeros((h * scale, w * scale, 3), dtype=np.uint8)
            
            for semantic_id, color in color_map.items():
                mask = (self.topdown_map == semantic_id)
                mask_scaled = cv2.resize(
                    mask.astype(np.uint8), 
                    (w * scale, h * scale), 
                    interpolation=cv2.INTER_NEAREST
                )
                rgb_map[mask_scaled > 0] = color
            
            rgb_map = cv2.GaussianBlur(rgb_map, (3, 3), 0)
        else:
            scale = 1
            rgb_map = np.zeros((h, w, 3), dtype=np.uint8)
            for semantic_id, color in color_map.items():
                mask = (self.topdown_map == semantic_id)
                rgb_map[mask] = color
        
        # 绘制轨迹
        if len(self.trajectory) > 1:
            traj_grid = [self.world_to_grid(p) for p in self.trajectory]
            if high_quality:
                traj_grid = [(int(x*scale), int(y*scale)) for x, y in traj_grid]
            
            points = np.array(traj_grid, dtype=np.int32)
            cv2.polylines(rgb_map, [points], False, (255, 69, 0), 
                         thickness=3 if high_quality else 2)
        
        # 当前位置标记
        if drone_pos is not None:
            gx, gy = self.world_to_grid(drone_pos)
            if high_quality:
                gx, gy = int(gx * scale), int(gy * scale)
                radius = 8
            else:
                radius = 4
            
            cv2.circle(rgb_map, (int(gx), int(gy)), radius+3, (255, 255, 255), -1)
            cv2.circle(rgb_map, (int(gx), int(gy)), radius, (30, 144, 255), -1)
        
        rgb_map = self._add_scale_and_legend(rgb_map, color_map, scale)
        
        cv2.imwrite(save_path, cv2.cvtColor(rgb_map, cv2.COLOR_RGB2BGR))
        print(f"✅ 语义地图已保存: {save_path} (发现 {len(color_map)} 种语义类别)")

    # -----------------------------------------------------
    def _generate_color_map(self):
        """动态生成颜色映射"""
        unique_ids = np.unique(self.topdown_map)
        color_map = {}
        
        for sid in unique_ids:
            if sid in self.predefined_colors:
                color_map[int(sid)] = self.predefined_colors[sid]
            else:
                np.random.seed(int(sid) * 42)
                color_map[int(sid)] = [
                    np.random.randint(100, 200),
                    np.random.randint(100, 200),
                    np.random.randint(100, 200)
                ]
        
        return color_map

    # -----------------------------------------------------
    def _add_scale_and_legend(self, img, color_map, scale):
        """添加图例和比例尺"""
        h, w = img.shape[:2]
        
        canvas = np.ones((h, w + 220, 3), dtype=np.uint8) * 250
        canvas[:h, :w] = img
        
        cv2.putText(canvas, "Semantic Map", (w+15, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50,50,50), 2)
        
        y_offset = 60
        cv2.putText(canvas, "Legend:", (w+15, y_offset-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80,80,80), 1)
        
        legend_labels = {
            0: "Unexplored",
            1: "Building", 
            2: "Vegetation",
            3: "Road",
            4: "Vehicle",
            5: "Sky",
            6: "Tree",
            7: "Ground",
            -1: "Trajectory"
        }
        
        for sid in sorted(color_map.keys()):
            if sid not in color_map:
                continue
            color = tuple(color_map[sid])
            label = legend_labels.get(sid, f"Class {sid}")
            
            cv2.circle(canvas, (w+25, y_offset), 7, color, -1)
            cv2.circle(canvas, (w+25, y_offset), 7, (120,120,120), 1)
            
            cv2.putText(canvas, label, (w+40, y_offset+4), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60,60,60), 1)
            y_offset += 28
        
        # 比例尺
        scale_y = h - 50
        scale_length_m = 10
        scale_length_px = int(scale_length_m / self.grid_size * scale)
        
        cv2.rectangle(canvas, (15, scale_y-25), (30+scale_length_px, scale_y+10), 
                     (255,255,255), -1)
        cv2.rectangle(canvas, (15, scale_y-25), (30+scale_length_px, scale_y+10), 
                     (100,100,100), 1)
        
        cv2.line(canvas, (20, scale_y), (20 + scale_length_px, scale_y), 
                (0,0,0), 3)
        cv2.line(canvas, (20, scale_y-8), (20, scale_y+8), (0,0,0), 2)
        cv2.line(canvas, (20+scale_length_px, scale_y-8), 
                (20+scale_length_px, scale_y+8), (0,0,0), 2)
        
        cv2.putText(canvas, f"{scale_length_m}m", (25, scale_y-12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        
        return canvas

    # -----------------------------------------------------
    def visualize_matrix(self, local_matrix, save_path):
        """可视化局部语义矩阵(用于调试)"""
        h, w = local_matrix.shape
        
        cell_size = 20
        vis_map = np.zeros((h * cell_size, w * cell_size, 3), dtype=np.uint8)
        
        color_map = self._generate_color_map()
        
        for i in range(h):
            for j in range(w):
                cls_id = int(local_matrix[i, j])
                color = color_map.get(cls_id, [200, 200, 200])
                
                y1, y2 = i * cell_size, (i + 1) * cell_size
                x1, x2 = j * cell_size, (j + 1) * cell_size
                
                vis_map[y1:y2, x1:x2] = color
                cv2.rectangle(vis_map, (x1, y1), (x2, y2), (150, 150, 150), 1)
                
                if cls_id != 0:
                    text = str(cls_id) if cls_id != -1 else "T"
                    cv2.putText(vis_map, text, 
                               (x1 + cell_size//4, y1 + cell_size//2 + 4),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (50, 50, 50), 1)
        
        # 标注中心点
        center_y, center_x = h // 2, w // 2
        cv2.circle(vis_map, 
                  (center_x * cell_size + cell_size//2, 
                   center_y * cell_size + cell_size//2),
                  8, (255, 0, 0), -1)
        cv2.circle(vis_map, 
                  (center_x * cell_size + cell_size//2, 
                   center_y * cell_size + cell_size//2),
                  10, (255, 255, 255), 2)
        
        title_height = 40
        canvas = np.ones((vis_map.shape[0] + title_height, vis_map.shape[1], 3), 
                         dtype=np.uint8) * 255
        canvas[title_height:, :] = vis_map
        
        cv2.putText(canvas, "Local Semantic Matrix", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)
        
        cv2.imwrite(save_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        print(f"  ✓ 局部矩阵已保存: {save_path}")

    # -----------------------------------------------------
    def matrix_to_text(self, local_matrix, center):
        """将局部矩阵转换为文本表示(供LLM使用)"""
        h, w = local_matrix.shape
        
        id_to_name = {
            0: "?", 1: "B", 2: "V", 3: "R", 4: "C", 
            5: "S", 6: "T", 7: "G", -1: "*",
        }
        
        lines = ["=== Local Semantic Matrix ==="]
        lines.append(f"Center: ({center[0]}, {center[1]})")
        lines.append(f"Size: {h}x{w}")
        lines.append("")
        
        for i in range(h):
            row_text = ""
            for j in range(w):
                cls_id = int(local_matrix[i, j])
                symbol = id_to_name.get(cls_id, str(cls_id))
                
                if i == center[1] and j == center[0]:
                    symbol = f"[{symbol}]"
                else:
                    symbol = f" {symbol} "
                
                row_text += symbol
            
            lines.append(row_text)
        
        lines.append("")
        lines.append("Legend: ? = Unexplored, B = Building, V = Vegetation,")
        lines.append("        R = Road, C = Car, * = Trajectory, [...] = Current Pos")
        
