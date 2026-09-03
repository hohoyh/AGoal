# build_offline_map.py - 离线语义建图
# 操控无人机按栅格巡航，用 GroundingDINO + SAM 抽取实例，聚合成带语义描述的俯视地图（.pkl）

import sys
sys.path.append('third_party/Grounded-Segment-Anything/')

import airsim
import json
import numpy as np
import time
import pickle
import torch
import base64
import cv2
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.cluster import DBSCAN

# GroundingDINO + SAM
from groundingdino.util.inference import load_model as load_groundingdino, predict as grounding_predict
from segment_anything import sam_model_registry, SamPredictor
import groundingdino.datasets.transforms as T

# 本项目模块
from src.semantic_graph.llm_enricher import LlmEnricher
from src.mapping.semantic_map_optimizer import SemanticMapOptimizer, EulerAngleHelper
from src.mapping.map_post_processor import MapPostProcessor


class OfflineSemanticMapperSAM:
    """..."""
    
    def __init__(self, client, config_path='configs/mapping_config.yaml', map_size=None, resolution=None):
        # ... (此函数保持不变, 它已经正确地从 config 加载了) ...
        self.client = client
        
        self.optimizer = SemanticMapOptimizer(config_path=config_path)
        self.post_processor = MapPostProcessor(config_path=config_path)
        self.enricher = LlmEnricher(config_path=config_path) # ✅ 2. 在此初始化
        self.config = self.optimizer.config
        
        self.map_size = map_size if map_size is not None else self.config['map']['size']
        self.resolution = resolution if resolution is not None else self.config['map']['resolution']
        
        self.depth_max = self.config['projection']['depth_max']
        self.assoc_thresholds = self.config['association_thresholds']
        
        print(f"[初始化] 地图尺寸: {self.map_size}x{self.map_size} @ {self.resolution}m/px")
        print(f"[初始化] 最大深度: {self.depth_max}m")
        print("[初始化] 加载 GroundingDINO + SAM 模型...")
        
        groundingdino_config = 'third_party/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'
        groundingdino_checkpoint = 'data/models/groundingdino_swint_ogc.pth'
        self.groundingdino = load_groundingdino(groundingdino_config, groundingdino_checkpoint)
        
        sam_checkpoint = 'data/models/sam_vit_b_01ec64.pth'
        sam = sam_model_registry['vit_b'](checkpoint=sam_checkpoint)
        sam.to('cuda' if torch.cuda.is_available() else 'cpu')
        self.sam_predictor = SamPredictor(sam)
        
        print("✓ 模型加载完成\n")
        
        self.semantic_layers = self.optimizer.create_layered_grids(self.map_size)
        self.occupancy_grid = np.zeros((self.map_size, self.map_size), dtype=np.int8)
        self.objects_db = []
        self.object_id_counter = 0
        
        self.class_mapping = {}
        self.text_prompt_categories = []
        class_id_counter = 1
        
        fixed_ids = {
            'car': 1, 
            # 'truck': 2, 'bus': 3,
            # 'person': 4, 'bicycle': 5, 'motorcycle': 6,
            'tree': 7, 'building': 8, 
            # 'bench': 9,'traffic light': 10, 'stop sign': 11, 
            'road': 12,'lake':13,'grass': 14
        }
        
        all_classes = []
        for layer in self.config['classes']:
            all_classes.extend(self.config['classes'][layer])
        
        for cls in sorted(list(set(all_classes))):
            if cls in fixed_ids:
                self.class_mapping[cls] = fixed_ids[cls]
            else:
                while class_id_counter in fixed_ids.values():
                    class_id_counter += 1
                self.class_mapping[cls] = class_id_counter
                class_id_counter += 1
            
            self.text_prompt_categories.append(cls)

        self.class_instance_counters = {}
        
        self.text_prompt = " . ".join(self.text_prompt_categories)
        
        print(f"[初始化] 动态类别映射: {self.class_mapping}")
        print(f"[初始化] 动态文本提示: {self.text_prompt}")
    def extract_building_polygons_from_overhead(self):
        """
        ✅ 新函数: 从 Pass 1 的语义图提取建筑物多边形
        """
        print("\n[建筑轮廓提取] 开始...")
        
        # 1. 获取 building 层
        building_layer = self.semantic_layers['background'][:, :, 1]
        building_mask = (building_layer == 8).astype(np.uint8)  # 8 = building
        
        # 2. 形态学处理 (填充孔洞)
        kernel = np.ones((5, 5), np.uint8)
        building_mask = cv2.morphologyEx(building_mask, cv2.MORPH_CLOSE, kernel)
        
        # 3. 连通域分析 (每个连通域 = 一栋建筑)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            building_mask, connectivity=8
        )
        
        building_polygons = []
        
        for i in range(1, num_labels):  # 跳过背景 (label=0)
            # 提取单个建筑物
            building_i = (labels == i).astype(np.uint8)
            
            # 面积过滤 (太小的是噪声)
            area_pixels = stats[i, cv2.CC_STAT_AREA]
            area_m2 = area_pixels * (self.resolution ** 2)
            
            if area_m2 < 50:  # 小于50平米的忽略
                continue
            
            # 找轮廓
            contours, _ = cv2.findContours(
                building_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            if not contours:
                continue
            
            # 取最大轮廓
            contour = max(contours, key=cv2.contourArea)
            
            # 简化轮廓 (Douglas-Peucker)
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx_poly = cv2.approxPolyDP(contour, epsilon, True)
            
            # 转换到世界坐标
            world_poly = []
            for point in approx_poly:
                mx, my = point[0]
                x, y = self.map_to_world(mx, my)
                world_poly.append([x, y])
            
            # 计算中心
            center_x, center_y = centroids[i]
            world_center_x, world_center_y = self.map_to_world(
                int(center_x), int(center_y)
            )
            
            building_polygons.append({
                'id': f'building_{i}',
                'polygon': np.array(world_poly),  # 世界坐标多边形
                'center': [world_center_x, world_center_y, 0],
                'area_m2': area_m2,
                'pixel_count': area_pixels,
                'attributes': {}  # ✅ 待侧视扫描填充
            })
        
        print(f"  ✓ 提取到 {len(building_polygons)} 栋建筑物")
        
        # ✅ 保存到类属性
        self.building_polygons = building_polygons
        
        return building_polygons
    def systematic_scan_with_points(self, scan_points, altitude):
        """(✅ 修复: 动态设置俯仰角)"""
        if len(scan_points) == 0:
            print("  ⚠️  没有安全扫描点,跳过此Pass")
            return
        
        print(f"✓ 执行 {len(scan_points)} 个安全扫描点")
        print(f"  预计耗时: {len(scan_points) * 10 / 60:.1f} 分钟\n")
        
        # ✅ 根据高度决定俯仰角
        if altitude > 50:
            pitch_deg = -70.0 # 高空: 大角度俯视
        elif altitude > 20:
            pitch_deg = -45.0 # 中空: 45度
        else:
            pitch_deg = -30.0 # 低空: 浅角度
        
        print(f"  [扫描] 使用高度 {altitude}m, 俯仰角 {pitch_deg}°")

        for i, (x, y) in enumerate(scan_points):
            print(f"[{i+1}/{len(scan_points)}] 安全点: ({x:.1f}, {y:.1f})")
            
            if not self.safe_move_to_scan_point(x, y, altitude):
                continue
            
            for yaw_deg in [0, 90, 180, 270]:
                try:
                    # 1. 设置偏航 (Yaw)
                    self.client.rotateToYawAsync(yaw_deg, timeout_sec=5.0).join()
                    
                    # ✅ 2. 修复: 强制设置俯仰 (Pitch)
                    pitch_rad = np.radians(pitch_deg)
                    
                    q = airsim.to_quaternion(0, pitch_rad, 0)
                    pos_vec = airsim.Vector3r(0, 0, 0)
                    pose = airsim.Pose(pos_vec, q)
                    
                    self.client.simSetCameraPose("0", pose)
                    
                    time.sleep(0.2) # 等待相机稳定

                except Exception as e:
                    print(f"  [警告] rotateToYawAsync 或 simSetCameraPose 失败: {e}")
                    break
                
                self._process_single_view(altitude)
            
            # ✅ 3. 重置相机为默认(或水平)，准备下一次移动
            q_reset = airsim.to_quaternion(0, 0, 0)
            pos_vec_reset = airsim.Vector3r(0, 0, 0)
            pose_reset = airsim.Pose(pos_vec_reset, q_reset)
            try:
                self.client.simSetCameraPose("0", pose_reset)
            except Exception as e:
                print(f"  [警告] 重置相机姿态失败: {e}")

            # if (i + 1) % 10 == 0:
            #     self._save_checkpoint(f"maps/checkpoint_safe_{i+1}.pkl")
        
        print("\n✓ 安全区域扫描完成!")

    def safe_move_to_scan_point(self, x, y, target_altitude):
        """(✅ 修复: 增加了超时和异常处理)"""
        SAFETY_MARGIN = 10.0
        
        safe_z_target = -(target_altitude + SAFETY_MARGIN)
        z_target = -target_altitude
        
        try:
            current_state = self.client.getMultirotorState()
            current_z = current_state.kinematics_estimated.position.z_val
            
            if current_z > safe_z_target:
                print(f"    ...爬升到安全高度 {abs(safe_z_target)}m")
                self.client.moveToZAsync(safe_z_target, 3, timeout_sec=10.0).join()
                time.sleep(0.5)
            
            print(f"    ...水平移动到 ({x:.1f}, {y:.1f})")
            self.client.moveToPositionAsync(x, y, safe_z_target, velocity=5, timeout_sec=15.0).join()
            time.sleep(0.5)
            
            print(f"    ...下降到扫描高度 {abs(z_target)}m")
            self.client.moveToZAsync(z_target, 2, timeout_sec=10.0).join()
            time.sleep(0.5)
            return True
        
        except Exception as e:
            print(f"  [警告] safe_move_to_scan_point 失败 (可能超时或碰撞): {e}")
            print("  [警告] 尝试重置 API 并跳过此点...")
            try:
                self.client.enableApiControl(True)
                self.client.armDisarm(True)
            except:
                pass
            return False

    def identify_safe_scan_points(self, scan_area, grid_size, min_clearance):
        # ... (此函数保持不变) ...
        print(f"\n[智能区域识别] 分析建筑分布,寻找安全低空扫描区域...")
        
        building_layer = self.semantic_layers['background'][:, :, 1]
        building_mask = (building_layer == 8).astype(np.float32) # 8 is building
        
        buffer_pixels = int(min_clearance / self.resolution)
        kernel = np.ones((buffer_pixels*2, buffer_pixels*2), np.uint8)
        building_buffer = cv2.dilate(building_mask, kernel, iterations=1)
        
        safe_mask = (building_buffer < 0.5).astype(np.uint8)
        
        kernel_clean = np.ones((5, 5), np.uint8)
        safe_mask = cv2.morphologyEx(safe_mask, cv2.MORPH_OPEN, kernel_clean)
        
        x_min, x_max = scan_area['x']
        y_min, y_max = scan_area['y']
        
        safe_points = []
        y_points = list(np.arange(y_min, y_max, grid_size))
        
        for i, y in enumerate(y_points):
            if i % 2 == 0:
                x_points = np.arange(x_min, x_max, grid_size)
            else:
                x_points = np.arange(x_max, x_min, -grid_size)
            
            for x in x_points:
                mx, my = self.world_to_map(x, y)
                
                if 0 <= mx < self.map_size and 0 <= my < self.map_size:
                    if safe_mask[my, mx] > 0:
                        safe_points.append([x, y])
        
        print(f"  ✓ 原始网格点: {len(y_points) * len(list(np.arange(x_min, x_max, grid_size)))}")
        print(f"  ✓ 安全扫描点: {len(safe_points)}")
        print(f"  ✓ 过滤比例: {(1 - len(safe_points) / max(1, len(y_points) * len(list(np.arange(x_min, x_max, grid_size))))) * 100:.1f}%\n")
        
        self._save_safe_area_visualization(safe_mask, 'maps/safe_area_mask.png')
        
        return safe_points

    def _save_safe_area_visualization(self, safe_mask, filename):
        # ... (此函数保持不变) ...
        vis = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        
        vis[safe_mask > 0] = [0, 255, 0]
        vis[safe_mask == 0] = [255, 0, 0]
        
        building_layer = self.semantic_layers['background'][:, :, 1]
        building_outline = (building_layer == 8).astype(np.uint8)
        building_outline = cv2.Canny(building_outline * 255, 50, 150)
        vis[building_outline > 0] = [255, 255, 0]
        
        cv2.imwrite(filename, vis)
        print(f"  [调试] 安全区域可视化: {filename}")
        
    def world_to_map(self, x, y):
        # ... (此函数保持不变) ...
        mx = int(self.map_size / 2 + x / self.resolution)
        my = int(self.map_size / 2 - y / self.resolution)
        return mx, my
    
    def map_to_world(self, mx, my):
        # ... (此函数保持不变) ...
        x = (mx - self.map_size / 2) * self.resolution
        y = (self.map_size / 2 - my) * self.resolution
        return x, y
    
    def systematic_scan(self, scan_area, altitude=15, grid_size=10, pitch_deg=None):
        """(✅ 修复: 动态设置俯仰角)"""
        x_min, x_max = scan_area['x']
        y_min, y_max = scan_area['y']
        
        scan_points = []
        y_points = list(np.arange(y_min, y_max, grid_size))
        
        for i, y in enumerate(y_points):
            if i % 2 == 0:
                x_points = np.arange(x_min, x_max, grid_size)
            else:
                x_points = np.arange(x_max, x_min, -grid_size)
            
            for x in x_points:
                scan_points.append([x, y])
        
        print(f"✓ 生成 {len(scan_points)} 个扫描点\n")
        
        # ✅ 根据高度决定俯仰角
        if pitch_deg is None:  # ✅ 关键: 仅在 main 未指定 pitch 时才动态计算
            if altitude > 50:
                pitch_deg = -70.0 # 高空: 大角度俯视
            elif altitude > 20:
                pitch_deg = -45.0 # 中空: 45度
            else:
                pitch_deg = -30.0 # 低空: 浅角度
        
        print(f"  [扫描] 使用高度 {altitude}m, 俯仰角 {pitch_deg}°")
        
        for i, (x, y) in enumerate(scan_points):
            print(f"[{i+1}/{len(scan_points)}] 扫描点: ({x:.1f}, {y:.1f})")
            
            if not self.safe_move_to_scan_point(x, y, altitude):
                continue
            
            for yaw_deg in [0, 90, 180, 270]:
                try:
                    # 1. 设置偏航 (Yaw)
                    self.client.rotateToYawAsync(yaw_deg, timeout_sec=5.0).join()
                    
                    # ✅ 2. 修复: 强制设置俯仰 (Pitch)
                    pitch_rad = np.radians(pitch_deg)
                    
                    # 姿态是相对于机身的 (Roll=0, Pitch=动态, Yaw=0)
                    q = airsim.to_quaternion(0, pitch_rad, 0)
                    pos_vec = airsim.Vector3r(0, 0, 0)
                    pose = airsim.Pose(pos_vec, q)
                    
                    self.client.simSetCameraPose("0", pose)
                    
                    time.sleep(0.2) # 等待相机稳定
                    
                except Exception as e:
                    print(f"  [警告] rotateToYawAsync 或 simSetCameraPose 失败: {e}")
                    break
                
                self._process_single_view(altitude)
            
            # ✅ 3. 重置相机为默认(或水平)，准备下一次移动
            q_reset = airsim.to_quaternion(0, 0, 0) # 重置为 0 度俯仰
            pos_vec_reset = airsim.Vector3r(0, 0, 0)
            pose_reset = airsim.Pose(pos_vec_reset, q_reset)
            try:
                self.client.simSetCameraPose("0", pose_reset)
            except Exception as e:
                print(f"  [警告] 重置相机姿态失败: {e}")

            
            # if (i + 1) % 10 == 0:
            #     self._save_checkpoint(f"maps/checkpoint_{i+1}.pkl")
        
        print("\n✓ 扫描完成!")
    
    def _process_single_view(self, altitude):
        """✅ 修复: 使用 simGetCameraInfo 进行精确投影"""
        
        # 1. 获取图像
        responses = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
            airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True, False)
        ])
        
        if not responses or len(responses) < 2:
            return
        
        # ... (RGB 和 Depth 的解析保持不变)
        rgb_response = responses[0]
        rgb_data = np.frombuffer(rgb_response.image_data_uint8, dtype=np.uint8)
        rgb = rgb_data.reshape(rgb_response.height, rgb_response.width, 3)
        
        depth_response = responses[1]
        depth = np.array(depth_response.image_data_float, dtype=np.float32)
        depth = depth.reshape(depth_response.height, depth_response.width)
        
        # ✅ 2. 获取相机真实姿态
        try:
            camera_info = self.client.simGetCameraInfo("0")
            camera_pose = camera_info.pose
        except Exception as e:
            print(f"  [警告] simGetCameraInfo 失败: {e}")
            return
        
        # 3. GroundingDINO 检测
        detections = self._groundingdino_detect(rgb)
        if len(detections) == 0:
            return
        print(f"  [GroundingDINO] 检测到 {len(detections)} 个物体: {[d['label'] for d in detections]}")
        
        # 4. SAM 分割
        masks = self._sam_segment(rgb, detections)
        if len(masks) == 0:
            return
        print(f"  [SAM] 成功分割 {len(masks)} 个物体")
        
        # 5. ✅ 投影 (传入真实的相机姿态)
        painted_pixels = self._project_masks_to_map_optimized(
            masks, depth, camera_pose, rgb.shape
        )
        print(f"  [投影] 写入 {painted_pixels} 个像素到语义地图\n")
        
        # 6. ✅ 更新数据库 (传入真实的相机姿态)
        self._update_object_database(detections, masks, depth, camera_pose, altitude, rgb.shape, rgb)
        
        # 7. ✅ 更新占据栅格 (传入真实的相机姿态)
        self._update_occupancy(depth, camera_pose)

    def _groundingdino_detect(self, image):
        # ... (此函数已更新, 会自动过滤掉 "building bench" 这样的脏标签) ...
        image_pil = Image.fromarray(image)
        
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        image_transformed, _ = transform(image_pil, None)
        
        with torch.no_grad():
            boxes, logits, phrases = grounding_predict(
                model=self.groundingdino,
                image=image_transformed,
                caption=self.text_prompt,
                box_threshold=0.25,
                text_threshold=0.2,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
        
        H, W = image.shape[:2]
        detections = []
        
        for box, logit, phrase in zip(boxes, logits, phrases):
            clean_phrase = phrase.lower().strip()
            if clean_phrase not in self.class_mapping:
                continue
                
            cx, cy, w, h = box.cpu().numpy()
            x1 = int((cx - w/2) * W)
            y1 = int((cy - h/2) * H)
            x2 = int((cx + w/2) * W)
            y2 = int((cy + h/2) * H)
            
            detections.append({
                'label': clean_phrase,
                'confidence': float(logit),
                'bbox': [x1, y1, x2, y2]
            })
        
        return detections
    
    def _sam_segment(self, image, detections):
        # ... (此函数保持不变) ...
        self.sam_predictor.set_image(image)
        
        masks_result = []
        
        for det in detections:
            bbox = det['bbox']
            
            masks, scores, _ = self.sam_predictor.predict(
                box=np.array(bbox),
                multimask_output=False
            )
            
            if len(masks) > 0:
                masks_result.append({
                    'label': det['label'],
                    'confidence': det['confidence'],
                    'mask': masks[0],
                    'bbox': bbox
                })
        
        return masks_result
    
    def _project_masks_to_map_optimized(self, masks, depth, camera_pose, img_shape):
        """✅ 修复: 使用 camera_pose 进行精确投影"""
        H, W = img_shape[:2]
        depth_h, depth_w = depth.shape
        
        # 从 config 读取相机内参
        fx, fy = self.config['camera']['fx'], self.config['camera']['fy']
        cx, cy = depth_w / 2.0, depth_h / 2.0
        
        # ✅ 1. 获取相机姿态
        cam_pos = camera_pose.position
        R_world = EulerAngleHelper.quaternion_to_rotation_matrix(camera_pose.orientation)
        T_world = np.array([cam_pos.x_val, cam_pos.y_val, cam_pos.z_val])
        
        painted = 0
        
        for mask_data in masks:
            mask = mask_data['mask']
            label = mask_data['label']
            conf = mask_data['confidence']
            
            class_id = self.class_mapping.get(label, 0)
            if class_id == 0: continue
            
            layer_name = self.optimizer.get_layer_for_class(label)
            
            if mask.shape != (depth_h, depth_w):
                mask_resized = cv2.resize(
                    mask.astype(np.uint8), 
                    (depth_w, depth_h), 
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)
            else:
                mask_resized = mask
            
            ys, xs = np.where(mask_resized)
            stride = int(self.config['projection'].get('stride_px', 2))  # 推荐 1~3
            xs = xs[::stride]
            ys = ys[::stride]
            
            for x, y in zip(xs, ys):
                d = depth[y, x]
                
                if d < 0.1 or d > self.depth_max:
                    continue
                
                # ✅ 2. 像素 -> 相机坐标系 (NED: +Fwd, +Right, +Down)
                z_cam = d                 # Forward
                x_cam = (x - cx) * d / fx # Right
                y_cam = (y - cy) * d / fy # Down
                
                point_in_camera_frame = np.array([z_cam, x_cam, y_cam])
                
                # ✅ 3. 相机坐标系 -> 世界坐标系
                point_in_world_frame = R_world @ point_in_camera_frame + T_world
                
                x_world = point_in_world_frame[0] # North
                y_world = point_in_world_frame[1] # East
                
                # 4. 世界 -> 地图坐标
                mx, my = self.world_to_map(x_world, y_world)
                
                if 0 <= mx < self.map_size and 0 <= my < self.map_size:
                    self.semantic_layers[layer_name] = self.optimizer.bayesian_update(
                        self.semantic_layers[layer_name],
                        class_id,
                        conf,
                        mx, my
                    )
                    painted += 1
        
        return painted
    
    def _update_object_database(self, detections, masks, depth, camera_pose, altitude, img_shape, rgb_image):
        """
        ✅ 修复: 阶段 1 - 只累积, 不关联。
        我们不再尝试 'find_nearby_object'。
        """
        
        H, W = img_shape[:2]
        camera_params = {
            'fx': self.config['camera']['fx'], 
            'fy': self.config['camera']['fy'],
            'cx': W / 2, 'cy': H / 2
        }
        
        cam_pos = camera_pose.position
        R_world = EulerAngleHelper.quaternion_to_rotation_matrix(camera_pose.orientation)
        T_world = np.array([cam_pos.x_val, cam_pos.y_val, cam_pos.z_val])
        
        for det, mask_data in zip(detections, masks):
            label = det['label'] # ✅ 获取标签

            # ✅ 新增的检查：
            if label == 'grass' or label == 'lake':
                continue  # 跳过, 不把它们添加到 objects_db
            mask = mask_data['mask']
            bbox = det['bbox']
            
            x1, y1, x2, y2 = bbox
            depth_h, depth_w = depth.shape
            
            # --- 稳定锚点计算 (这部分逻辑是正确的, 保持) ---
            scale_x = depth_w / W; scale_y = depth_h / H
            x1_d = int(max(0, min(x1 * scale_x, depth_w - 1))); y1_d = int(max(0, min(y1 * scale_y, depth_h - 1)))
            x2_d = int(max(0, min(x2 * scale_x, depth_w - 1))); y2_d = int(max(0, min(y2 * scale_y, depth_h - 1)))
            if x2_d <= x1_d or y2_d <= y1_d: continue
            depth_region = depth[y1_d:y2_d, x1_d:x2_d]
            valid_depths = depth_region[(depth_region > 0.1) & (depth_region < self.depth_max)]
            if len(valid_depths) == 0: continue
            median_depth = float(np.median(valid_depths))
            valid_pixels_y_in_region, valid_pixels_x_in_region = np.where(
                (depth_region > 0.1) & (depth_region < self.depth_max)
            )
            if valid_pixels_y_in_region.size == 0: continue
            depth_diffs = np.abs(depth_region[valid_pixels_y_in_region, valid_pixels_x_in_region] - median_depth)
            median_pixel_idx = np.argmin(depth_diffs)
            stable_pixel_x_depth = valid_pixels_x_in_region[median_pixel_idx] + x1_d
            stable_pixel_y_depth = valid_pixels_y_in_region[median_pixel_idx] + y1_d
            stable_pixel_x_rgb = stable_pixel_x_depth / scale_x
            stable_pixel_y_rgb = stable_pixel_y_depth / scale_y
            # --- 稳定锚点计算结束 ---

            z_cam = median_depth
            x_cam = (stable_pixel_x_rgb - camera_params['cx']) * median_depth / camera_params['fx']
            y_cam = (stable_pixel_y_rgb - camera_params['cy']) * median_depth / camera_params['fy']
            
            point_in_camera_frame = np.array([z_cam, x_cam, y_cam])
            point_in_world_frame = R_world @ point_in_camera_frame + T_world
            
            x_world = point_in_world_frame[0]
            y_world = point_in_world_frame[1]
            z_world = point_in_world_frame[2]
            
            if self.optimizer.get_layer_for_class(det['label']) == 'ground':
                z_world = 0.0
            
            # --- 提取颜色 (保持) ---
            x1_crop = max(0, int(x1)); y1_crop = max(0, int(y1))
            x2_crop = min(rgb_image.shape[1], int(x2)); y2_crop = min(rgb_image.shape[0], int(y2))
            crop_img = rgb_image[y1_crop:y2_crop, x1_crop:x2_crop]
            
            if crop_img.size > 0:
                crop_hsv = cv2.cvtColor(crop_img, cv2.COLOR_RGB2HSV)
                hist_h = cv2.calcHist([crop_hsv], [0], None, [30], [0, 180]); hist_h = cv2.normalize(hist_h, hist_h).flatten()
                hist_s = cv2.calcHist([crop_hsv], [1], None, [32], [0, 256]); hist_s = cv2.normalize(hist_s, hist_s).flatten()
                color_hist = np.concatenate([hist_h, hist_s])
                avg_color = crop_img.mean(axis=(0, 1)).tolist()
            else:
                color_hist = None; avg_color = [0, 0, 0]
            
            obj_width = x2 - x1; obj_height = y2 - y1
            
            # ❌ 移除 'find_nearby_object_strict' 和 'if existing:'
            
            # ✅ 修复: 总是创建 "新物体" (这只是一个原始检测点)
            label = det['label']
            if label not in self.class_instance_counters:
                self.class_instance_counters[label] = 0
            self.class_instance_counters[label] += 1
            instance_id = f"{label}_{self.class_instance_counters[label]}"
            self.object_id_counter += 1
            
            yaw_rad, _, _ = airsim.to_eularian_angles(camera_pose.orientation)
            
            obj_data = {
                'id': self.object_id_counter, 'instance_id': instance_id, 'label': label,
                'position': [x_world, y_world, z_world], 'confidence': det['confidence'],
                'count': 1, # 'count' 现在只代表这个检测点本身
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'size': [float(obj_width), float(obj_height)],
                'color_hist': color_hist, 'avg_color': avg_color, 
                'crop_image': cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR),
                'first_seen_altitude': altitude,
                'first_seen_yaw': float(np.degrees(yaw_rad))
            }
            self.objects_db.append(obj_data)
            
            # (这个日志现在会刷屏, 这是正常的)
            # print(f"    [新物体] {instance_id} @ ({x_world:.1f}, {y_world:.1f}, {z_world:.1f})")
    def _find_nearby_object_strict(self, position, label, size, color_hist, threshold):
        """✅ 修复: 正确使用配置的阈值"""
        
        # ✅ 使用传入的threshold,如果为None才从config读取
        if threshold is None:
            threshold = self.assoc_thresholds.get(
                label,
                self.assoc_thresholds.get('default', 2.0)
            )
        
        # 1. 先按距离筛选候选
        candidates = []
        for obj in self.objects_db:
            if obj['label'] != label:
                continue
            
            dist = np.linalg.norm(
                np.array(obj['position'][:2]) - np.array(position[:2])
            )
            
            if dist < threshold:
                candidates.append((obj, dist))
        
        if not candidates:
            return None
        
        # 2. 如果只有一个候选,直接返回
        if len(candidates) == 1:
            return candidates[0][0]
        
        # 3. 多个候选 → 用颜色+尺寸精细匹配
        if color_hist is not None:
            best_match = None
            best_score = -1
            
            for obj, dist in candidates:
                if obj['color_hist'] is None:
                    continue
                
                # 颜色相似度
                color_sim = cv2.compareHist(
                    color_hist.astype(np.float32),
                    obj['color_hist'].astype(np.float32),
                    cv2.HISTCMP_CORREL
                )
                
                # 尺寸相似度
                size_diff = abs(obj['size'][0] - size[0]) + abs(obj['size'][1] - size[1])
                size_sim = 1.0 / (1.0 + size_diff / 100.0)
                
                # 距离权重
                dist_weight = 1.0 / (1.0 + dist)
                
                # 综合分数
                score = color_sim * 0.4 + size_sim * 0.3 + dist_weight * 0.3
                
                if score > best_score:
                    best_score = score
                    best_match = obj
            
            # ✅ 如果最佳匹配分数够高,返回;否则返回距离最近的
            if best_score > 0.5:  # 可调参数
                return best_match
        
        # 4. 降级策略: 返回距离最近的
        return min(candidates, key=lambda x: x[1])[0]
        
    def _update_occupancy(self, depth, camera_pose):
        """✅ 修复: 使用 camera_pose 进行精确投影"""
        H, W = depth.shape
        fx, fy = self.config['camera']['fx'], self.config['camera']['fy']
        cx, cy = W / 2, H / 2
        
        # ✅ 1. 获取相机姿态
        cam_pos = camera_pose.position
        R_world = EulerAngleHelper.quaternion_to_rotation_matrix(camera_pose.orientation)
        T_world = np.array([cam_pos.x_val, cam_pos.y_val, cam_pos.z_val])
        
        step = 4
        
        for v in range(0, H, step):
            for u in range(0, W, step):
                d = depth[v, u]
                if d < 0.1 or d > self.depth_max:
                    continue
                
                # ✅ 2. 像素 -> 相机坐标系 (NED)
                z_cam = d
                x_cam = (u - cx) * d / fx
                y_cam = (v - cy) * d / fy
                
                point_in_camera_frame = np.array([z_cam, x_cam, y_cam])
                
                # ✅ 3. 相机坐标系 -> 世界坐标系
                point_in_world_frame = R_world @ point_in_camera_frame + T_world
                
                x_world = point_in_world_frame[0]
                y_world = point_in_world_frame[1]
                
                # 4. 世界 -> 地图
                mx, my = self.world_to_map(x_world, y_world)
                
                if 0 <= mx < self.map_size and 0 <= my < self.map_size:
                    if d < 5:
                        self.occupancy_grid[my, mx] = -1
                    else:
                        if self.occupancy_grid[my, mx] == 0:
                            self.occupancy_grid[my, mx] = 1
    
    def _save_checkpoint(self, filename):
        # ... (此函数保持不变) ...
        merged_grid = self.optimizer.merge_layers(self.semantic_layers, self.map_size)
        map_data = {
            'semantic_map': merged_grid, 'semantic_layers': self.semantic_layers,
            'occupancy_grid': self.occupancy_grid, 'objects': self.objects_db,
            'map_size': self.map_size, 'resolution': self.resolution,
            'metadata': {'timestamp': time.time(), 'num_objects': len(self.objects_db)}
        }
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        with open(filename, 'wb') as f:
            pickle.dump(map_data, f)
        print(f"    [检查点] 已保存: {filename}")

    # ✅ 2. 在 save_map 函数之前, 添加这个新的辅助函数
    def _cluster_dbscan(self, raw_objects):
            """
            ✅ 修复: 聚类时, 找出 '最佳特写' 并保存
            """
            
            print(f"  [聚类] 开始聚类 {len(raw_objects)} 个原始检测点...")
            
            clustered_objects = []
            
            # 按类别分组
            objects_by_class = {}
            for obj in raw_objects:
                label = obj['label']
                if label not in objects_by_class:
                    objects_by_class[label] = []
                objects_by_class[label].append(obj)
            
            # 对每个类别单独执行 DBSCAN
            for label, objects in objects_by_class.items():
                if len(objects) == 0:
                    continue
                
                # 从 config 获取该类别的聚类距离
                assoc_threshold = self.assoc_thresholds.get(label, self.assoc_thresholds.get('default', 2.0))
                
                # 提取 2D 坐标 (North, East)
                positions_2d = np.array([obj['position'][:2] for obj in objects])
                
                # 运行 DBSCAN
                # min_samples=2 意味着至少 2 个检测点才能形成一个物体
                db = DBSCAN(eps=assoc_threshold, min_samples=2).fit(positions_2d)
                labels = db.labels_
                
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                print(f"    - 类别 '{label}': {len(objects)} 个检测点 -> {n_clusters} 个聚类 (距离={assoc_threshold}m)")
                
                # 合并每个聚类
                for cluster_id in range(n_clusters):
                    cluster_mask = (labels == cluster_id)
                    cluster_detections = [obj for i, obj in enumerate(objects) if cluster_mask[i]]
                    
                    if not cluster_detections:
                        continue
                    
                    # --- ✅ 关键: 找出最佳代表图像 ---
                    # 策略: 寻找 BBox 面积最大的那个检测点
                    # (这依赖于 _update_object_database 保存了 size 和 crop_image)
                    best_detection = max(cluster_detections, key=lambda d: d['size'][0] * d['size'][1])
                    representative_image_bgr = best_detection['crop_image']
                    
                    # 将该图像编码为 JPEG，然后转为 base64 字符串
                    # VLM (如 gpt-4o-mini) 可以直接接收 base64
                    representative_image_b64 = None
                    if representative_image_bgr is not None and representative_image_bgr.size > 0:
                        try:
                            # 'crop_image' 应该是 BGR numpy 数组
                            _, jpeg_buffer = cv2.imencode('.jpg', representative_image_bgr)
                            representative_image_b64 = base64.b64encode(jpeg_buffer).decode('utf-8')
                        except Exception as e:
                            print(f"    [警告] 图像编码失败: {e}")
                    # --- 修复结束 ---
                    
                    # 合并信息
                    avg_pos = np.mean([d['position'] for d in cluster_detections], axis=0)
                    avg_conf = np.mean([d['confidence'] for d in cluster_detections])
                    
                    merged_obj = {
                        'id': best_detection['id'], # 使用最佳检测的 ID
                        'instance_id': f"{label}_cluster_{cluster_id}", # 新的实例 ID
                        'label': label,
                        'position': avg_pos.tolist(),
                        'confidence': avg_conf,
                        'count': len(cluster_detections), # Count 现在代表簇中的检测点数量
                        'size': np.mean([d['size'] for d in cluster_detections], axis=0).tolist(),
                        'avg_color': np.mean([d['avg_color'] for d in cluster_detections], axis=0).tolist(),
                        'first_seen_altitude': best_detection['first_seen_altitude'],
                        
                        'color_hist': None, # (不再需要, 节省空间)
                        'crop_image': representative_image_b64 # ✅ 保存最佳特写的 Base64 字符串
                    }
                    clustered_objects.append(merged_obj)
                    
            print(f"  [聚类] 完成, 总共合并为 {len(clustered_objects)} 个唯一物体。")
            return clustered_objects
    

    # offline_mapper_sam_visual_enhanced.py

    def save_map(self, filename, scan_area):
        """✅ 改进版：完整语义图输出 + LLM 富集 + 多类别可视化"""
        print("\n" + "="*60)
        print("开始最终地图处理...")
        print("="*60)

        # 1️⃣ 合并与增强
        self.semantic_layers = self.optimizer.filter_by_observations(self.semantic_layers)
        merged_grid = self.optimizer.merge_layers(self.semantic_layers, self.map_size)
        print("✓ 分层网格已合并")

        enhanced_grid = self.post_processor.enhance_map(merged_grid, self.occupancy_grid)
        print("✓ 地图增强完成")

        # 2️⃣ 提取建筑对象 + 其他语义聚类
        if hasattr(self, 'building_polygons'):
            building_objects = []
            original_buildings_db = [
                obj for obj in self.objects_db 
                if obj['label'] == 'building' and obj['crop_image'] is not None
            ]

            for poly in self.building_polygons:
                # --- 查找最近俯视图 ---
                best_overhead_image_b64 = None
                if original_buildings_db:
                    poly_center = np.array(poly['center'][:2])
                    best_det = min(
                        original_buildings_db, 
                        key=lambda det: np.linalg.norm(np.array(det['position'][:2]) - poly_center)
                    )
                    try:
                        _, jpeg_buffer = cv2.imencode('.jpg', best_det['crop_image'])
                        best_overhead_image_b64 = base64.b64encode(jpeg_buffer).decode('utf-8')
                    except Exception as e:
                        print(f"    [警告] 俯视图图像编码失败 (Building {poly['id']}): {e}")

                obj = {
                    'id': poly['id'],
                    'instance_id': poly['id'],
                    'label': 'building',
                    'position': poly['center'],
                    'confidence': 1.0,
                    'count': poly['pixel_count'],
                    'size': [np.sqrt(poly['area_m2']), np.sqrt(poly['area_m2'])],
                    'avg_color': [128, 128, 128],
                    'crop_image': best_overhead_image_b64,
                    'first_seen_altitude': 70.0,
                    'attributes': poly.get('attributes', {})
                }
                building_objects.append(obj)

            # 其他类对象（car/tree/lake/grass...）
            other_objects = [obj for obj in self.objects_db if obj['label'] != 'building']
            clustered_other = self._cluster_dbscan(other_objects)
            all_objects = building_objects + clustered_other
        else:
            all_objects = self._cluster_dbscan(self.objects_db)

        # 3️⃣ LLM 富集描述
        enriched_objects = self.enricher.generate_descriptions(all_objects)
        filtered_objects = enriched_objects

        # 4️⃣ 统计与覆盖率
        class_counts = {}
        for obj in filtered_objects:
            label = obj['label']
            class_counts[label] = class_counts.get(label, 0) + 1

        actual_coverage = self._calculate_actual_coverage(enhanced_grid, scan_area)

        map_data = {
            'semantic_map': enhanced_grid,
            'semantic_layers': self.semantic_layers,
            'occupancy_grid': self.occupancy_grid,
            'objects': filtered_objects,
            'map_size': self.map_size,
            'resolution': self.resolution,
            'metadata': {
                'scan_time': time.time(),
                'num_objects': len(filtered_objects),
                'class_counts': class_counts,
                'coverage': actual_coverage,
                'scan_area': scan_area
            }
        }

        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        with open(filename, 'wb') as f:
            pickle.dump(map_data, f)

        print(f"\n✓ 地图已保存: {filename}")
        print(f"  物体总数: {len(filtered_objects)}") 
        print(f"  类别分布:")
        for cls, cnt in sorted(class_counts.items()):
            print(f"    {cls}: {cnt} 个实例")
        print(f"  覆盖率: {actual_coverage:.1%}")

        # === ✅ 5️⃣ 可视化：绘制完整 2D 语义俯视图 ===
        vis_path = filename.replace('.pkl', '_visual_hd.png')

        # ✅ 使用全局统一调色板
        palette = self.post_processor.palette  # <-- 从 map_post_processor 读取

        cls_map = enhanced_grid[:, :, 1].astype(np.int32)
        canvas = np.zeros((cls_map.shape[0], cls_map.shape[1], 3), dtype=np.uint8)

        # ✅ 遍历调色板上色（与全局一致）
        for cid, color in palette.items():
            mask = (cls_map == cid)
            canvas[mask] = color

        # ✅ 保存2D语义地图
        vis_path = filename.replace('.pkl', '_visual_hd.png')
        cv2.imwrite(vis_path, canvas)
        print(f"✓ 已输出语义可视化图像（使用全局palette）: {vis_path}")


        # 保存俯视图（2D语义地图）
        cv2.imwrite(vis_path, canvas)
        print(f"✓ 已输出语义可视化图像: {vis_path}")

        # === 6️⃣ 保存对象清单 + 对比图 ===
        self._save_object_list(filename.replace('.pkl', '_objects.txt'), filtered_objects)
        self.post_processor.visualize_comparison(
            merged_grid, enhanced_grid, filename.replace('.pkl', '_comparison.png')
        )


    def _calculate_actual_coverage(self, semantic_grid, scan_area):
        # ... (此函数保持不变) ...
        x_min, x_max = scan_area['x']
        y_min, y_max = scan_area['y']
        mx_min, my_max = self.world_to_map(x_min, y_min)
        mx_max, my_min = self.world_to_map(x_max, y_max)
        mx_min = max(0, min(mx_min, self.map_size - 1)); mx_max = max(0, min(mx_max, self.map_size - 1))
        my_min = max(0, min(my_min, self.map_size - 1)); my_max = max(0, min(my_max, self.map_size - 1))
        if my_min > my_max: my_min, my_max = my_max, my_min
        scan_region = semantic_grid[my_min:my_max, mx_min:mx_max, 0]
        if scan_region.size == 0: return 0.0
        covered_pixels = np.sum(scan_region > 0); total_pixels = scan_region.size
        return float(covered_pixels) / float(total_pixels)
    
    # offline_mapper_sam_visual_enhanced.py

    def _save_object_list(self, filename, objects):
        """✅ 修复: 在 .txt 中显示 LLM 描述"""
        import json # 确保导入 json

        with open(filename, 'w') as f:
            f.write(f"{'ID':<6} {'Instance ID':<20} {'Label':<12} {'Position (x,y,z)':<25} {'Confidence':<10} {'Count':<6} {'LLM Description'}\n")
            f.write("=" * 150 + "\n")

            for obj in sorted(objects, key=lambda x: x['label']):
                pos_str = f"({obj['position'][0]:>6.1f}, {obj['position'][1]:>6.1f}, {obj['position'][2]:>6.1f})"

                # 获取 LLM 描述
                llm_desc = obj.get('llm_description', 'N/A')
                if isinstance(llm_desc, dict):
                    # 提取自然语言描述,如果存在
                    desc_text = llm_desc.get('natural_description', json.dumps(llm_desc))
                else:
                    desc_text = str(llm_desc)

                f.write(
                    f"{obj['id']:<6} "
                    f"{obj['instance_id']:<20} "
                    f"{obj['label']:<12} "
                    f"{pos_str:<25} "
                    f"{obj['confidence']:<10.2f} "
                    f"{obj['count']:<6} "
                    f"{desc_text}\n" # ✅ 显示描述
                )
        print(f"  物体列表: {filename}")

    def _save_visualization_professional(self, filename, semantic_grid, scan_area, filtered_objects):
        """✅ 修复: 接收 filtered_objects, 解决模糊问题"""
        print(f"  [可视化] 生成精美高清地图...")
        
        # ✅ 1. 从 post_processor 获取唯一的调色板
        palette = self.post_processor.palette
        
        # ✅ 2. 创建彩色地图 (原始尺寸)
        color_map = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        cls_layer = semantic_grid[:, :, 1].astype(np.int32)
        
        for cid, color in palette.items():
            mask = (cls_layer == cid)
            # 修正: BGR -> RGB (matplotlib 需要 RGB)
            color_rgb = (color[0], color[1], color[2]) # 假设 palette 是 RGB
            color_map[mask] = color_rgb
        
        # ✅ 3. 叠加占据网格
        obstacle_mask = self.occupancy_grid == -1
        obstacle_edges = cv2.Canny(obstacle_mask.astype(np.uint8) * 255, 50, 150)
        color_map[obstacle_edges > 0] = [180, 0, 0]  # 深红色边界 (RGB)
        
        # ✅ 4. [已删除] 高斯平滑
        # ✅ 5. [已删除] 超分辨率放大
        color_map_hd = color_map
        target_size = self.map_size # 画布尺寸 = 数据尺寸
        
        # ✅ 6. 使用matplotlib
        fig, ax = plt.subplots(figsize=(14, 12), dpi=150)
        
        # ✅ 修复: 使用 'nearest' 插值法, 显示清晰的像素
        ax.imshow(color_map_hd, interpolation='nearest') # 已是 RGB
        
        # ✅ 7. 比例尺 (修正逻辑)
        scale_length_m = 50
        scale_length_px = int(scale_length_m / self.resolution)
        scale_x_start = int(target_size * 0.1)
        scale_y = int(target_size * 0.95)
        
        ax.plot([scale_x_start, scale_x_start + scale_length_px], 
                [scale_y, scale_y], 'k-', linewidth=4)
        ax.plot([scale_x_start, scale_x_start], 
                [scale_y - (target_size*0.01), scale_y + (target_size*0.01)], 'k-', linewidth=2)
        ax.plot([scale_x_start + scale_length_px, scale_x_start + scale_length_px], 
                [scale_y - (target_size*0.01), scale_y + (target_size*0.01)], 'k-', linewidth=2)
        ax.text(scale_x_start + scale_length_px / 2, scale_y - (target_size*0.02), 
                f'{scale_length_m}m', ha='center', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # ✅ 8. 图例 (适配 DownTown)
        legend_elements = []
        legend_mapping = {
            'Car': 1, 'Building': 8, 'Road': 12, 'Tree':7,'Lake':13,'Grass': 14
            
        }
        
        for label, cid in legend_mapping.items():
            if cid in palette and cid in self.class_mapping.values():
                color_rgb = np.array(palette[cid]) / 255.0
                legend_elements.append(
                    Patch(facecolor=color_rgb, label=label)
                )
        
        ax.legend(handles=legend_elements, loc='upper right', fontsize=11,
                 framealpha=0.9, edgecolor='black')
        
        # ✅ 9. 标题
        x_range = scan_area['x'][1] - scan_area['x'][0]
        y_range = scan_area['y'][1] - scan_area['y'][0]
        
        ax.set_title(
            f'Semantic Map - Scan Area: {x_range:.0f}m × {y_range:.0f}m\n'
            f'Resolution: {self.resolution}m/pixel  |  Objects: {len(filtered_objects)}', # ✅ 修复
            fontsize=14, fontweight='bold', pad=20
        )
        
        # ✅ 10. 指北针
        compass_x = int(target_size * 0.1)
        compass_y = int(target_size * 0.2)
        arrow_length = int(target_size * 0.1)
        
        ax.arrow(compass_x, compass_y, 0, -arrow_length, 
                head_width=arrow_length*0.2, head_length=arrow_length*0.15, 
                fc='red', ec='black', linewidth=2)
        ax.text(compass_x, compass_y - arrow_length - (arrow_length*0.1), 'N', 
               ha='center', fontsize=16, fontweight='bold',
               bbox=dict(boxstyle='circle', facecolor='white', edgecolor='black'))
        
        ax.axis('off')
        plt.tight_layout()
        
        plt.savefig(filename, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"  ✓ 精美可视化: {filename} (300 DPI)")
        
        simple_filename = filename.replace('_visual_hd.png', '_visual_simple.png')
        cv2.imwrite(simple_filename, cv2.cvtColor(color_map_hd, cv2.COLOR_RGB2BGR),
                   [cv2.IMWRITE_PNG_COMPRESSION, 3])
        print(f"  ✓ 简单预览: {simple_filename}")
    def scan_buildings_side_view(self, building_polygons, altitude):
        """
        ✅ 新函数: 为每栋建筑进行侧视环绕扫描
        """
        print(f"\n[侧视扫描] 准备为 {len(building_polygons)} 栋建筑补充属性...")
        
        for i, building in enumerate(building_polygons):
            print(f"\n[{i+1}/{len(building_polygons)}] 扫描 {building['id']}")
            
            # 1. 计算环绕点位 (4个方向)
            center = building['center']
            radius = max(15, np.sqrt(building['area_m2']) / 2 + 10)  # 距离建筑10米
            
            scan_points = [
                [center[0] + radius, center[1], altitude],  # 东侧
                [center[0], center[1] + radius, altitude],  # 南侧
                [center[0] - radius, center[1], altitude],  # 西侧
                [center[0], center[1] - radius, altitude],  # 北侧
            ]
            
            yaw_angles = [90, 180, 270, 0]  # 对应朝向建筑物
            
            building_images = []
            
            for point, yaw in zip(scan_points, yaw_angles):
                # 2. 飞到观测点
                try:
                    self.client.moveToPositionAsync(
                        point[0], point[1], -point[2], 
                        velocity=5, timeout_sec=15.0
                    ).join()
                    time.sleep(0.5)
                    
                    # 3. 朝向建筑物
                    self.client.rotateToYawAsync(yaw, timeout_sec=5.0).join()
                    
                    # 4. 设置相机俯仰角 (轻微俯视)
                    pitch_rad = np.radians(-30)
                    q = airsim.to_quaternion(0, pitch_rad, 0)
                    pose = airsim.Pose(airsim.Vector3r(0, 0, 0), q)
                    self.client.simSetCameraPose("0", pose)
                    
                    time.sleep(0.3)
                    
                    # 5. 拍照 (只需要 RGB)
                    responses = self.client.simGetImages([
                        airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
                    ])
                    
                    if responses:
                        rgb_data = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
                        rgb = rgb_data.reshape(responses[0].height, responses[0].width, 3)
                        building_images.append(rgb)
                    
                except Exception as e:
                    print(f"    [警告] 观测点失败: {e}")
                    continue
            
            # 6. ✅ 用 LLM/VLM 分析这4张侧视图 → 提取颜色、高度
            if len(building_images) > 0:
                attributes = self._analyze_building_images(building_images, building)
                building['attributes'] = attributes
                
                print(f"    ✓ 属性提取完成: {attributes}")
    def _analyze_building_images(self, images, building):
        """
        ✅ 用 VLM 分析多张侧视图,提取建筑物属性
        """
        # 1. 选择最清晰的一张图 (简单策略: 选最大尺寸的BBox)
        # 或者: 让 VLM 同时看4张图
        
        # 简化版: 只用第一张
        representative_image = images[0]
        
        # 2. 编码为 Base64
        _, jpeg_buffer = cv2.imencode('.jpg', cv2.cvtColor(representative_image, cv2.COLOR_RGB2BGR))
        image_b64 = base64.b64encode(jpeg_buffer).decode('utf-8')
        
        # 3. 调用 VLM (通过 enricher)
        # ✅ 但这里不需要完整的 LLM 描述,只需要提取属性
        
        prompt = f"""
    你正在看一张建筑物的侧视图 (从旁边拍摄)。

    请判断:
    1. 建筑物外墙的主色调 (例如: 黑色、白色、灰色、棕色、红砖色)
    2. 建筑物的大致高度 (例如: 低矮、中等、高层、超高层)
    3. 建筑物的材质 (例如: 玻璃幕墙、混凝土、砖墙)

    严格按照以下JSON格式输出:
    {{
    "color": "主色调",
    "height": "高度估计",
    "material": "材质"
    }}
    """
        
        try:
            # 调用 LLM API (复用 enricher 的客户端)
            if hasattr(self, 'enricher') and self.enricher.client:
                json_string = self.enricher._call_llm_api(prompt, image_b64)
                attributes = json.loads(json_string)
            else:
                attributes = {
                    "color": "未知",
                    "height": "未知",
                    "material": "未知"
                }
        except Exception as e:
            print(f"    [VLM 错误] {e}")
            attributes = {"color": "未知", "height": "未知", "material": "未知"}
        
        return attributes
# ============================================================
# 主程序 (✅ 适配 DownTown)
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="操控无人机栅格巡航，构建带语义描述的离线俯视地图"
    )
    parser.add_argument('--scan-area', default='[-150,150],[-150,150]',
                        help='扫描区域，格式: [x_min,x_max],[y_min,y_max]（米）')
    parser.add_argument('--output', default='maps/downtown_map_hd.pkl')
    parser.add_argument('--config', default='configs/mapping_config.yaml')
    args = parser.parse_args()

    # 确保输出目录存在
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path('outputs').mkdir(parents=True, exist_ok=True)

    # 解析扫描区域
    parts = args.scan_area.replace('[', '').replace(']', '').split(',')
    x_min, x_max = float(parts[0]), float(parts[1])
    y_min, y_max = float(parts[2]), float(parts[3])
    
    scan_area = {
        'x': [x_min, x_max],
        'y': [y_min, y_max]
    }
    
    SCAN_PASSES = [
    {
        'name': 'Overhead Scan (Building Footprints)', 
        'altitude': 80.0,  # ✅ 更高,保证完全俯视
        'grid_size': 15.0,
        'mode': 'overhead',  # ✅ 新模式
        'pitch': -90  # ✅ 垂直向下
    },
    {
        'name': 'Side View Scan (Building Attributes)', 
        'altitude': 25.0,  # ✅ 环绕建筑物高度
        'mode': 'side_view',  # ✅ 新模式
        'pitch': -30  # ✅ 轻微俯视
    },
    # ✅ 可选: 低空补充 Car/Tree
]
    # 连接AirSim
    print("连接AirSim...")
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    
    # ✅ 创建mapper (从 config 读取 map_size)
    mapper = OfflineSemanticMapperSAM(client, config_path=args.config)
    
    # ✅ Pass 1: 俯视扫描
    pass1 = SCAN_PASSES[0]
    print(f"\n{'='*60}")
    print(f"✅ Pass 1: {pass1['name']}")
    print(f"{'='*60}")
    
    # 强制垂直向下
    mapper.systematic_scan(
        scan_area, 
        altitude=pass1['altitude'], 
        grid_size=pass1['grid_size'],
        pitch_deg=pass1['pitch']  # ✅ 新增：强制传入 -90 度
    )
    
    # ✅ 关键: Pass 1 结束后立即提取建筑物轮廓
    building_polygons = mapper.extract_building_polygons_from_overhead()
    
    # ✅ Pass 2: 侧视扫描 (为每栋建筑补充属性)
    if len(building_polygons) > 0:
        pass2 = SCAN_PASSES[1]
        print(f"\n{'='*60}")
        print(f"✅ Pass 2: {pass2['name']}")
        print(f"{'='*60}")
        
        mapper.scan_buildings_side_view(
            building_polygons, 
            altitude=pass2['altitude']
        )
    
    # ✅ 最后: 保存地图 (LLM 在这里介入)
    mapper.save_map(args.output, scan_area)
    
    # 返回起点
    print("\n返回起点...")
    try:
        client.moveToPositionAsync(0, 0, -20, 5, timeout_sec=15.0).join()
        client.landAsync(timeout_sec=10.0).join()
    except Exception as e:
        print(f"  [警告] 返回或降落失败: {e}")
    
    client.armDisarm(False)
    client.enableApiControl(False)
    
    print("\n✓ 全部完成!")
    print(f"\n📊 输出文件:")
    print(f"  - 地图数据: {args.output}")
    print(f"  - 精美可视化: {args.output.replace('.pkl', '_visual_hd.png')}")
    print(f"  - 简单预览: {args.output.replace('.pkl', '_visual_simple.png')}")
    print(f"  - 物体列表: {args.output.replace('.pkl', '_objects.txt')}")
    print(f"  - 前后对比: {args.output.replace('.pkl', '_comparison.png')}")


if __name__ == '__main__':
    main()