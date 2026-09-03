# run_text_goal.py - 文本目标驱动的在线导航（预建语义地图 + YOLO 视觉验证）
import logging
logging.getLogger('tornado.application').setLevel(logging.ERROR)
logging.getLogger('asyncio').setLevel(logging.ERROR)

import argparse
import yaml
from types import SimpleNamespace
import numpy as np
import torch
import time
import cv2
import airsim 
import math 
import pickle
from pathlib import Path
from scipy.spatial import KDTree

from src.envs.airsim_env import AirSimEnv
from src.graph.graph import Graph
from src.map.bev_mapping import BEV_Map
from src.detectors.yolo_detector import FastYOLODetector
from src.utils.config import resolve_api_key


# ============================================================
# 预建地图导航器
# ============================================================

class PrebuiltMapNavigator:
    """基于预建地图的导航器"""
    
    def __init__(self, map_file):
        print(f"[地图] 加载预建地图: {map_file}")
        
        with open(map_file, 'rb') as f:
            map_data = pickle.load(f)
        
        self.semantic_map_grid = map_data['semantic_map']
        self.occupancy_grid = map_data['occupancy_grid']
        self.objects = map_data['objects']
        self.map_size = map_data['map_size']
        self.resolution = map_data['resolution']
        
        # 构建物体索引（快速查询）
        object_positions = [obj['position'][:2] for obj in self.objects]
        self.object_kdtree = KDTree(object_positions) if object_positions else None
        
        print(f"✓ 加载完成:")
        print(f"  物体数量: {len(self.objects)}")
        print(f"  地图大小: {self.map_size}x{self.map_size}")
        print(f"  覆盖率: {map_data['metadata']['coverage']:.1%}")
    
    def query_object(self, label, current_pos=None):
        """
        查询物体位置
        
        Args:
            label: 物体标签（如 "car"）
            current_pos: 当前位置 [x, y]，用于返回最近的
        
        Returns:
            {'label', 'position', 'confidence'} 或 None
        """
        candidates = [
            obj for obj in self.objects 
            if label.lower() in obj['label'].lower()
        ]
        
        if not candidates:
            return None
        
        if current_pos is None:
            # 返回置信度最高的
            return max(candidates, key=lambda x: x['confidence'])
        else:
            # 返回最近的
            distances = [
                np.linalg.norm(
                    np.array(obj['position'][:2]) - np.array(current_pos)
                )
                for obj in candidates
            ]
            return candidates[np.argmin(distances)]
    
    def world_to_map(self, x, y):
        mx = int(self.map_size / 2 + x / self.resolution)
        my = int(self.map_size / 2 - y / self.resolution)
        return mx, my
    
    def map_to_world(self, mx, my):
        x = (mx - self.map_size / 2) * self.resolution
        y = (self.map_size / 2 - my) * self.resolution
        return x, y


# ============================================================
# 改进的接近策略
# ============================================================

def approach_target_safe(env, yolo, target_det, main_target, obs, camera_params, stop_distance=8.0):
    """
    安全接近目标（避免飞到正上方）
    
    策略：
    1. 先飞到目标前方 8米处（保持相机视野内）
    2. 调整偏航角，让目标在图像中心
    3. 缓慢前进至 5米
    4. 连续验证
    """
    print("\n[接近阶段] 安全接近目标...")
    
    # ========== 阶段1：估算目标位置 ==========
    
    depth = obs['depth'][:, :, 0]
    current_state = env.client.getMultirotorState()
    
    target_world_pos = yolo.estimate_3d_position(
        target_det, depth, current_state, camera_params, env.altitude
    )
    
    if target_world_pos is None:
        print("  ✗ 无法估算位置")
        return False
    
    print(f"  ✓ 目标位置: ({target_world_pos[0]:.1f}, {target_world_pos[1]:.1f})")
    
    # ========== 阶段2：飞到目标前方 stop_distance 米 ==========
    
    current_pos = current_state.kinematics_estimated.position
    dx = target_world_pos[0] - current_pos.x_val
    dy = target_world_pos[1] - current_pos.y_val
    dist = np.hypot(dx, dy)
    
    print(f"  当前距离: {dist:.1f}m")
    
    if dist > stop_distance:
        # 计算 stop_distance 米外的点
        ratio = (dist - stop_distance) / dist
        approach_x = current_pos.x_val + dx * ratio
        approach_y = current_pos.y_val + dy * ratio
        
        print(f"  → 飞向前方 {stop_distance}m 处: ({approach_x:.1f}, {approach_y:.1f})")
        
        env.client.moveToPositionAsync(
            approach_x, approach_y, -env.altitude, velocity=2
        ).join()
        time.sleep(1)
    
    # ========== 阶段3：调整偏航角（让目标在中心） ==========
    
    print("  [对准] 调整朝向...")
    
    for adjust_iter in range(3):
        obs_now, _, _ = env.get_observation()
        if obs_now is None:
            continue
            
        dets_now = yolo.detect(obs_now['rgb'])
        
        target_now = next(
            (d for d in dets_now if main_target in d['label'].lower()),
            None
        )
        
        if not target_now:
            print("  ✗ 目标丢失")
            return False
        
        # 计算目标在图像中的偏移
        bbox = target_now['bbox']
        img_w = obs_now['rgb'].shape[1]
        bbox_center_x = (bbox[0] + bbox[2]) / 2
        error_x = bbox_center_x - img_w / 2
        
        print(f"    偏移: {error_x:.0f} 像素")
        
        if abs(error_x) < 30:  # 已对准
            print("    ✓ 已对准")
            break
        
        # 调整偏航角
        current_state = env.client.getMultirotorState()
        current_yaw = airsim.to_eularian_angles(
            current_state.kinematics_estimated.orientation
        )[2]
        
        adjust_angle = (error_x / img_w) * 20  # 最多调整20度
        new_yaw = current_yaw + np.radians(adjust_angle)
        new_yaw_deg = np.degrees(new_yaw) % 360
        
        env.client.rotateToYawAsync(new_yaw_deg, 10).join()
        time.sleep(0.5)
    
    # ========== 阶段4：缓慢前进至5米 ==========
    
    print("  [精进] 缓慢接近...")
    
    final_distance = 5.0
    
    for fine_iter in range(5):
        obs_fine, _, _ = env.get_observation()
        if obs_fine is None:
            continue
            
        dets_fine = yolo.detect_with_depth(obs_fine['rgb'], obs_fine['depth'][:,:,0])
        
        target_fine = next(
            (d for d in dets_fine if main_target in d['label'].lower()),
            None
        )
        
        if not target_fine:
            print("  ✗ 目标丢失")
            break
        
        # 更新目标位置
        current_state = env.client.getMultirotorState()
        target_pos_updated = yolo.estimate_3d_position(
            target_fine, obs_fine['depth'][:,:,0], 
            current_state, camera_params, env.altitude
        )
        
        if not target_pos_updated:
            continue
        
        current_pos = current_state.kinematics_estimated.position
        dx = target_pos_updated[0] - current_pos.x_val
        dy = target_pos_updated[1] - current_pos.y_val
        current_dist = np.hypot(dx, dy)
        
        print(f"    距离: {current_dist:.1f}m")
        
        if current_dist <= final_distance:
            print("    ✓ 已到达理想距离")
            break
        
        # 前进一小步
        step_size = min(2.0, current_dist - final_distance)
        ux, uy = dx / current_dist, dy / current_dist
        
        new_x = current_pos.x_val + ux * step_size
        new_y = current_pos.y_val + uy * step_size
        
        env.client.moveToPositionAsync(
            new_x, new_y, -env.altitude, velocity=1
        ).join()
        time.sleep(0.5)
    
    # ========== 阶段5：最终验证 ==========
    
    print("\n[验证阶段] 连续确认...")
    
    confirm_count = 0
    
    for verify_i in range(5):
        time.sleep(0.3)
        obs_verify, _, _ = env.get_observation()
        if obs_verify is None:
            continue
            
        dets_verify = yolo.detect(obs_verify['rgb'])
        
        if any(main_target in d['label'].lower() for d in dets_verify):
            confirm_count += 1
            print(f"  验证 {verify_i+1}/5: ✓")
        else:
            print(f"  验证 {verify_i+1}/5: ✗")
    
    if confirm_count >= 3:
        print("\n✓✓✓ 接近成功！")
        return True
    else:
        print("\n✗ 验证失败")
        return False


# ============================================================
# 配置
# ============================================================

def get_config():
    parser = argparse.ArgumentParser(
        description="给定一句语言指令（如 Find the car），无人机在 AirSim 中搜索目标"
    )
    parser.add_argument("--config-file", default="configs/config_airsim.yaml")
    parser.add_argument("--goal", default="Find the car", type=str)
    parser.add_argument("--visualize", action='store_true', help="可视化检测结果")
    parser.add_argument("--map-file", default="maps/outdoor_map.pkl", help="预建地图文件")
    args = parser.parse_args()
    
    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    args = vars(args)
    args.update(config)
    args = SimpleNamespace(**args)

    # 配置里留空时自动回退到环境变量 OPENAI_API_KEY
    args.api_key = resolve_api_key(getattr(args, "api_key", None))

    # 计算地图参数
    args.map_size = args.map_size_cm // args.map_resolution
    args.global_width = args.map_size
    args.global_height = args.map_size
    args.local_width = int(args.global_width / args.global_downscaling)
    args.local_height = int(args.global_height / args.global_downscaling)
    args.device = torch.device("cuda:0" if args.cuda else "cpu")
    
    return args


def get_camera_params(width=640, height=480, fov_degrees=90):
    """计算相机内参"""
    fov_rad = math.radians(fov_degrees)
    fx = fy = (width / 2.0) / math.tan(fov_rad / 2.0)
    cx = width / 2.0
    cy = height / 2.0
    
    return {
        'fx': fx,
        'fy': fy,
        'cx': cx,
        'cy': cy,
        'rgb_width': width,
        'rgb_height': height
    }


# ============================================================
# 主程序
# ============================================================

def main():
    args = get_config()

    # 确保输出目录存在
    for d in ('maps', 'outputs', 'debug'):
        Path(d).mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("预建地图 + YOLO 混合导航")
    print("="*60)
    print(f"目标: {args.goal}")
    print(f"地图文件: {args.map_file}")
    print("="*60)
    
    # ========== 加载预建地图 ==========
    
    try:
        navigator = PrebuiltMapNavigator(args.map_file)
        USE_PREBUILT_MAP = True
        print("✓ 使用预建地图模式\n")
    except FileNotFoundError:
        print("⚠️  未找到预建地图，使用实时探索模式\n")
        USE_PREBUILT_MAP = False
        navigator = None
    
    # ========== 初始化 ==========
    
    env = AirSimEnv(args)
    graph = Graph(args)
    BEV_map = BEV_Map(args)
    yolo = FastYOLODetector('yolov8n.pt')
    
    print("✓ 组件初始化完成\n")
    
    # ========== 重置 ==========
    
    obs, rgbd, infos = env.reset()
    
    if rgbd is None:
        print("错误：reset返回None")
        exit()
    
    print(f"Reset后 rgbd shape: {rgbd.shape}")
    print(f"Reset后 depth sum: {np.sum(rgbd[0, 3]):.2f}\n")
    
    BEV_map.init_map_and_pose()
    graph.reset()
    graph.set_text_goal(args.goal)
    
    # 相机参数
    camera_params = get_camera_params(640, 480, 90)
    
    # ========== 目标解析 ==========
    
    target_keywords = args.goal.lower().split()
    main_target = target_keywords[-1]  # "Find the car" -> "car"
    
    print(f"开始搜索目标: {main_target}")
    print("="*60 + "\n")
    
    # ========== 主循环 ==========
    
    step = 0
    max_steps = 30 if USE_PREBUILT_MAP else 50
    found_goal = False
    
    while step < max_steps and not found_goal:
        print(f"\n[Step {step}]")
        
        # 1. 获取观测
        obs, rgbd, infos = env.get_observation()
        
        if obs is None or np.sum(rgbd[0, 3]) == 0:
            print("  [警告] 观测无效，跳过")
            step += 1
            continue
        
        # 2. YOLO检测
        detections = yolo.detect_with_depth(obs['rgb'], obs['depth'][:,:,0])
        
        print(f"  [YOLO] 检测到 {len(detections)} 个物体")
        if detections:
            labels = [d['label'] for d in detections]
            print(f"  [YOLO] 物体: {labels}")
        
        # 可视化
        if args.visualize and len(detections) > 0:
            Path('debug').mkdir(exist_ok=True)
            vis_img = yolo.visualize_detections(obs['rgb'], detections)
            cv2.imwrite(f'debug/step_{step:03d}.jpg', 
                       cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
        
        # 3. 检查是否找到目标
        target_det = next(
            (d for d in detections if main_target in d['label'].lower()),
            None
        )
        
        if target_det:
            print(f"  ✓ 发现目标: {target_det['label']}, 置信度: {target_det['confidence']:.2f}")
            
            # 接近目标
            success = approach_target_safe(
                env, yolo, target_det, main_target, 
                obs, camera_params, stop_distance=8.0
            )
            
            if success:
                found_goal = True
                break
            else:
                print("  [接近失败] 继续探索...")
        
        # ========== 4. 探索策略 ==========
        
        if USE_PREBUILT_MAP and navigator:
            # ✅ 方案A：使用预建地图
            print("\n  [地图查询] 搜索目标位置...")
            
            current_state = env.client.getMultirotorState()
            current_pos = [
                current_state.kinematics_estimated.position.x_val,
                current_state.kinematics_estimated.position.y_val
            ]
            
            target_obj = navigator.query_object(main_target, current_pos)
            
            if target_obj:
                target_pos = target_obj['position']
                print(f"  ✓ 地图中找到 {main_target} at ({target_pos[0]:.1f}, {target_pos[1]:.1f})")
                print(f"    置信度: {target_obj['confidence']:.2f}")
                
                # 飞向目标（保持安全距离）
                approach_distance = 10.0
                dx = target_pos[0] - current_pos[0]
                dy = target_pos[1] - current_pos[1]
                dist = np.hypot(dx, dy)
                
                if dist > approach_distance:
                    # 飞到目标10米外
                    ratio = (dist - approach_distance) / dist
                    waypoint_x = current_pos[0] + dx * ratio
                    waypoint_y = current_pos[1] + dy * ratio
                    
                    print(f"  → 飞向目标附近: ({waypoint_x:.1f}, {waypoint_y:.1f})")
                    
                    env.client.moveToPositionAsync(
                        waypoint_x, waypoint_y, -env.altitude, velocity=3
                    ).join()
                    time.sleep(1)
                else:
                    print("  ✓ 已在目标附近，环视搜索")
                    # 原地360°环视
                    for yaw_deg in range(0, 360, 45):
                        env.client.rotateToYawAsync(yaw_deg, 20).join()
                        time.sleep(0.5)
                        
                        obs_scan, _, _ = env.get_observation()
                        if obs_scan is None:
                            continue
                            
                        dets_scan = yolo.detect(obs_scan['rgb'])
                        
                        if any(main_target in d['label'].lower() for d in dets_scan):
                            print("  ✓ 环视中发现目标！")
                            break
            else:
                print("  ✗ 地图中未找到目标")
                print("  → 切换到实时探索模式")
                USE_PREBUILT_MAP = False
        
        else:
            # ✅ 方案B：实时探索（简化版）
            print("\n  [实时探索] 随机探索...")
            
            current_state = env.client.getMultirotorState()
            pos = current_state.kinematics_estimated.position
            orientation = current_state.kinematics_estimated.orientation
            yaw = airsim.to_eularian_angles(orientation)[2]
            
            # 随机选择方向
            explore_distance = 15.0
            explore_angle = yaw + np.random.uniform(-np.pi/3, np.pi/3)
            
            explore_x = pos.x_val + explore_distance * np.cos(explore_angle)
            explore_y = pos.y_val + explore_distance * np.sin(explore_angle)
            
            print(f"  → 探索: ({explore_x:.1f}, {explore_y:.1f})")
            
            env.client.moveToPositionAsync(
                explore_x, explore_y, -env.altitude, velocity=2
            ).join()
            time.sleep(1)
        
        step += 1
    
    # ========== 结束 ==========
    
    if found_goal:
        print("\n" + "="*60)
        print("✓✓✓ 任务完成！成功找到目标")
        print(f"总步数: {step}")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("✗ 探索结束，未找到目标")
        print(f"总步数: {step}")
        print("="*60)
    
    env.close()


if __name__ == "__main__":
    main()