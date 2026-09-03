# map_post_processor.py - 地图后处理器

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter, binary_dilation, binary_opening
import yaml
import matplotlib
matplotlib.use("Agg")              # 无显示环境
import matplotlib.pyplot as plt


class MapPostProcessor:
    """地图后处理 - 去噪、边界锐化、特征增强"""
    
    def __init__(self, config_path='configs/mapping_config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.kernel_size = self.config['postprocess']['morphology_kernel_size']
        self.smooth_sigma = self.config['postprocess']['smooth_sigma']
        
        # ✅ 将调色板设为公共属性，以便其他文件导入
        self.palette = self._get_color_palette()
        
        print(f"✓ 地图后处理器已加载")
    
    def enhance_map(self, semantic_grid, occupancy_grid):
        """
        完整的地图增强流程
        """
        # ... (此函数保持不变) ...
        print("\n[后处理] 开始地图增强...")
        
        # Step 1: 形态学去噪
        semantic_grid = self._morphology_denoise(semantic_grid)
        print("  ✓ Step 1: 形态学去噪完成")
        
        # Step 2: 置信度加权平滑
        semantic_grid = self._confidence_smoothing(semantic_grid)
        print("  ✓ Step 2: 置信度平滑完成")
        
        # Step 3: 建筑物轮廓增强
        semantic_grid = self._enhance_buildings(semantic_grid, occupancy_grid)
        print("  ✓ Step 3: 建筑物增强完成")
        
        # Step 4: 道路推断
        semantic_grid = self._infer_roads(semantic_grid, occupancy_grid)
        print("  ✓ Step 4: 道路推断完成")
        
        # Step 5: 边界锐化
        semantic_grid = self._sharpen_boundaries(semantic_grid)
        print("  ✓ Step 5: 边界锐化完成")
        
        return semantic_grid
    
    # ... ( _morphology_denoise, _confidence_smoothing, ... )
    # ... ( _enhance_buildings, _infer_roads, _sharpen_boundaries ... )
    # ... (这些函数保持不变) ...
    
    def _morphology_denoise(self, semantic_grid):
        """形态学去噪 - 移除孤立噪点"""
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        
        class_layer = semantic_grid[:, :, 1].astype(np.uint8)
        
        # 对每个类别单独处理
        for class_id in np.unique(class_layer):
            if class_id == 0:
                continue
            
            mask = (class_layer == class_id).astype(np.uint8)
            
            # 开运算: 先腐蚀后膨胀,去除小噪点
            mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # 更新回语义网格
            semantic_grid[:, :, 1][mask_clean > 0] = class_id
            semantic_grid[:, :, 1][(mask > 0) & (mask_clean == 0)] = 0
        
        return semantic_grid
    
    def _confidence_smoothing(self, semantic_grid):
        """置信度加权平滑 - 减少边界锯齿"""
        conf_layer = semantic_grid[:, :, 2]
        
        # 高斯平滑置信度
        conf_smooth = gaussian_filter(conf_layer, sigma=self.smooth_sigma)
        
        # 只更新高置信度区域
        high_conf_mask = conf_layer > 0.5
        semantic_grid[high_conf_mask, 2] = conf_smooth[high_conf_mask]
        
        return semantic_grid
    
    def _enhance_buildings(self, semantic_grid, occupancy_grid):
        """增强建筑物轮廓 - 利用占据网格"""
        building_classes = [8]  # building class_id (来自 config)
        min_area = self.config['postprocess']['building_min_area']
        
        class_layer = semantic_grid[:, :, 1]
        
        # 提取障碍物区域
        obstacle_mask = (occupancy_grid == -1).astype(np.uint8)
        
        # 查找连通区域
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            obstacle_mask, connectivity=8
        )
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            
            # 大面积障碍物 -> 可能是建筑
            if area > min_area:
                region_mask = (labels == i)
                
                # 检查该区域是否已有语义标签
                has_label = np.any(class_layer[region_mask] > 0)
                
                if not has_label:
                    # 标记为建筑物
                    semantic_grid[region_mask, 0] = 1.0
                    semantic_grid[region_mask, 1] = 8  # building
                    semantic_grid[region_mask, 2] = 0.7
        
        return semantic_grid
    
    def _infer_roads(self, semantic_grid, occupancy_grid):
        """道路推断 - 连接空白的自由空间"""
        # 提取自由空间
        free_mask = (occupancy_grid == 1).astype(np.uint8)
        
        # 去除已有语义标签的区域
        labeled_mask = (semantic_grid[:, :, 1] > 0).astype(np.uint8)
        free_unlabeled = free_mask * (1 - labeled_mask)
        
        # 膨胀连接断开的道路
        kernel = np.ones((5, 5), np.uint8)
        road_mask = cv2.dilate(free_unlabeled, kernel, 
                                iterations=self.config['postprocess']['road_dilation_iter'])
        
        # 查找连通区域
        num_labels, labels = cv2.connectedComponents(road_mask)
        
        for i in range(1, num_labels):
            region_mask = (labels == i)
            area = np.sum(region_mask)
            
            # 大面积自由空间 -> 道路
            if area > 200:
                # 不覆盖已有的高置信度标签
                low_conf_mask = semantic_grid[:, :, 2] < 0.3
                road_candidate = region_mask & low_conf_mask
                
                # 标记为道路 (用class_id=12表示)
                semantic_grid[road_candidate, 0] = 0.5
                semantic_grid[road_candidate, 1] = 12  # road
                semantic_grid[road_candidate, 2] = 0.5
        
        return semantic_grid
    
    def _sharpen_boundaries(self, semantic_grid):
        """边界锐化 - 提升物体边缘清晰度"""
        class_layer = semantic_grid[:, :, 1].astype(np.uint8)
        
        # 拉普拉斯边缘检测
        edges = cv2.Laplacian(class_layer, cv2.CV_64F)
        edges = np.abs(edges)
        
        # 在边界处提升置信度
        edge_mask = edges > 0.5
        semantic_grid[edge_mask, 2] = np.minimum(
            semantic_grid[edge_mask, 2] * 1.2, 1.0
        )
        
        return semantic_grid

    def visualize_comparison(self, before, after, save_path):
        """可视化前后对比"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # ✅ 使用 self.palette
        palette = self.palette
        
        # Before
        vis_before = self._apply_palette(before[:, :, 1], palette)
        axes[0].imshow(vis_before)
        axes[0].set_title('Before Post-processing')
        axes[0].axis('off')
        
        # After
        vis_after = self._apply_palette(after[:, :, 1], palette)
        axes[1].imshow(vis_after)
        axes[1].set_title('After Post-processing')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"  可视化对比已保存: {save_path}")
    
    def _get_color_palette(self):
        """
        ✅ 获取类别颜色映射 (适配 DownTown)
        这是调色板的唯一来源
        """
        return {
            0: (245, 245, 250),    # 背景 - 浅灰蓝
            1: (220, 50, 50),      # car - 鲜红
            # 2: (255, 140, 0),      # truck - 深橙
            # 3: (255, 215, 0),      # bus - 金黄
            # 4: (255, 20, 147),     # person - 深粉
            # 5: (135, 206, 250),    # bicycle - (已删除)
            7:(34,139,34),         # tree
            # 6: (0, 191, 255),      # motorcycle - (已删除)
            8: (70, 70, 90),       # building - 深蓝灰
            # 9: (160, 82, 45),      # bench - 褐色
            # 10: (255, 165, 0),     # traffic light - 橙色
            # 11: (220, 20, 60),     # stop sign - 猩红
            12: (211, 211, 211),   # road - 浅灰
            13: (0, 100, 255),     # ✅ lake - 亮蓝色
            14: (124, 252, 0),     # ✅ grass - 草绿色
        }
    
    def _apply_palette(self, class_layer, palette):
        """应用颜色映射"""
        h, w = class_layer.shape
        vis = np.zeros((h, w, 3), dtype=np.uint8)
        
        for class_id, color in palette.items():
            mask = (class_layer == class_id)
            vis[mask] = color
        
        return vis