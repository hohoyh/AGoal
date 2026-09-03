# semantic_map_optimizer.py - 语义地图优化器

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
import yaml
import airsim # 确保 airsim 被导入

class SemanticMapOptimizer:
    """语义地图优化器 - 贝叶斯融合与分层"""
    
    def __init__(self, config_path='configs/mapping_config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # 类别分层
        self.layer_mapping = {}
        for layer in ['ground', 'mid', 'background']:
            for cls in self.config['classes'][layer]:
                self.layer_mapping[cls] = layer
        
        print(f"✓ 语义地图优化器已加载")
        print(f"  - 分层策略: {list(self.config['classes'].keys())}")
    
    def create_layered_grids(self, map_size):
        """创建分层语义网格"""
        layers = {}
        for layer_name in ['ground', 'mid', 'background']:
            # 每层: [occupancy, class_id, confidence, obs_count]
            layers[layer_name] = np.zeros((map_size, map_size, 4), dtype=np.float32)
        return layers
    
    def get_layer_for_class(self, label):
        """获取类别对应的层"""
        return self.layer_mapping.get(label, 'ground')
    
    # ❌ 移除错误的投影函数
    # def correct_camera_projection(...):
    
    def bayesian_update(self, prior_grid, class_id, confidence, mx, my):
        """
        贝叶斯置信度更新
        """
         # === 按类别调整融合权重 ===
        label_weights = self.config.get("fusion_weights", {
            "lake": 1.2,
            "grass": 1.2,
            "building": 1.0,
            "road": 1.0,
            "tree": 1.0
        })

        # 从 class_id 反查类别名
        label_name = None
        for name, idx in self.config.get("class_mapping", {}).items():
            if idx == class_id:
                label_name = name
                break

        # 取对应的权重
        weight_factor = label_weights.get(label_name, 1.0)
        obs_weight = self.config['fusion']['observation_weight'] * weight_factor

        
        # 累积观测次数
        prior_grid[my, mx, 3] += 1
        
        # 如果是相同类别,加权平均置信度
        if prior_grid[my, mx, 1] == class_id:
            old_conf = prior_grid[my, mx, 2]
            new_conf = old_conf * (1 - obs_weight) + confidence * obs_weight
            prior_grid[my, mx, 2] = new_conf
        
        # 如果是不同类别,比较置信度
        elif confidence > prior_grid[my, mx, 2]:
            prior_grid[my, mx, 0] = 1.0
            prior_grid[my, mx, 1] = class_id
            prior_grid[my, mx, 2] = confidence
        
        return prior_grid
    
    def merge_layers(self, layered_grids, map_size):
        """
        合并分层网格到最终语义地图
        """
        # ... (此函数保持不变) ...
        final_grid = np.zeros((map_size, map_size, 3), dtype=np.float32)
        
        # 按优先级从低到高叠加
        for layer_name in ['background', 'mid', 'ground']:
            layer = layered_grids[layer_name]
            mask = layer[:, :, 0] > 0  # 有内容的地方
            
            final_grid[mask, 0] = layer[mask, 0]  # occupancy
            final_grid[mask, 1] = layer[mask, 1]  # class_id
            final_grid[mask, 2] = layer[mask, 2]  # confidence
        
        return final_grid
    
    def filter_by_observations(self, layered_grids):
        """过滤观测次数不足的像素"""
        # ... (此函数保持不变) ...
        min_obs = self.config['fusion']['min_observations']
        
        for layer_name in layered_grids:
            layer = layered_grids[layer_name]
            insufficient_mask = layer[:, :, 3] < min_obs
            layer[insufficient_mask] = 0  # 清零
        
        return layered_grids


class EulerAngleHelper:
    """✅ 辅助函数: 从四元数创建旋转矩阵"""
    
    @staticmethod
    def quaternion_to_rotation_matrix(q):
        """
        从 AirSim 四元数创建 3x3 旋转矩阵 (NED)
        """
        # (w, x, y, z)
        q_w, q_x, q_y, q_z = q.w_val, q.x_val, q.y_val, q.z_val
        
        return np.array([
            [1 - 2*q_y*q_y - 2*q_z*q_z, 2*q_x*q_y - 2*q_z*q_w, 2*q_x*q_z + 2*q_y*q_w],
            [2*q_x*q_y + 2*q_z*q_w, 1 - 2*q_x*q_x - 2*q_z*q_z, 2*q_y*q_z - 2*q_x*q_w],
            [2*q_x*q_z - 2*q_y*q_w, 2*q_y*q_z + 2*q_x*q_w, 1 - 2*q_x*q_x - 2*q_y*q_y]
        ])

    @staticmethod
    def quaternion_to_euler(q):
        """四元数 -> 欧拉角 (ZYX顺序)"""
        # ... (此函数保持不变) ...
        sinr_cosp = 2 * (q.w_val * q.x_val + q.y_val * q.z_val)
        cosr_cosp = 1 - 2 * (q.x_val * q.x_val + q.y_val * q.y_val)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        sinp = 2 * (q.w_val * q.y_val - q.z_val * q.x_val)
        if abs(sinp) >= 1:
            pitch = np.sign(sinp) * np.pi / 2
        else:
            pitch = np.arcsin(sinp)
        
        siny_cosp = 2 * (q.w_val * q.z_val + q.x_val * q.y_val)
        cosy_cosp = 1 - 2 * (q.y_val * q.y_val + q.z_val * q.z_val)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw