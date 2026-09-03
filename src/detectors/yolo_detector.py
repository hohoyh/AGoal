# src/detectors/yolo_detector.py

from ultralytics import YOLO
import numpy as np
import cv2
import airsim

class FastYOLODetector:
    """
    快速YOLO检测器，用于户外场景物体检测
    """
    def __init__(self, model_path='yolov8n.pt'):
        """
        Args:
            model_path: YOLO模型路径，默认使用轻量级yolov8n
        """
        print(f"[YOLO] 加载模型: {model_path}")
        self.model = YOLO(model_path)
        
        # 定义户外类别（COCO数据集中的相关类别）
        # 完整列表见: https://docs.ultralytics.com/datasets/detect/coco/
        self.outdoor_classes = {
            0: 'person',
            1: 'bicycle', 
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck',
            9: 'traffic light',
            10: 'fire hydrant',
            11: 'stop sign',
            13: 'bench',
            14: 'bird',
            # 注意：YOLO的COCO模型没有"building"类别
            # 需要用语义推理或微调来识别建筑
        }
        
        print(f"[YOLO] 支持检测 {len(self.outdoor_classes)} 个类别")

    def estimate_3d_position(self, detection, depth_map, current_state, camera_params, altitude):
        """估算目标的 3D 世界坐标"""
        bbox = detection['bbox']
        x1, y1, x2, y2 = bbox
        
        # ✅ 获取 RGB 尺寸（从 camera_params 传入）
        rgb_h = camera_params.get('rgb_height', 480)
        rgb_w = camera_params.get('rgb_width', 640)
        
        # ✅ 深度图尺寸
        depth_h, depth_w = depth_map.shape
        
        # ✅ 缩放 bbox 到深度图尺寸
        scale_x = depth_w / rgb_w
        scale_y = depth_h / rgb_h
        
        x1 = int(x1 * scale_x)
        x2 = int(x2 * scale_x)
        y1 = int(y1 * scale_y)
        y2 = int(y2 * scale_y)
        
        print(f"    [调试] 原始bbox: {bbox}")
        print(f"    [调试] 缩放后bbox: [{x1}, {y1}, {x2}, {y2}]")
        print(f"    [调试] RGB尺寸: {rgb_w}×{rgb_h}, Depth尺寸: {depth_w}×{depth_h}")
        
        # 边界检查
        x1, x2 = max(0, x1), min(depth_w, x2)
        y1, y2 = max(0, y1), min(depth_h, y2)
        
        if x1 >= x2 or y1 >= y2:
            print(f"    [错误] bbox无效: ({x1},{y1}) -> ({x2},{y2})")
            return None
        
       
        
        
        # 1. 提取深度
        depth_region = depth_map[y1:y2, x1:x2]
        print(f"    [调试] depth_region shape: {depth_region.shape}")
        print(f"    [调试] depth_region min/max: {depth_region.min():.3f}/{depth_region.max():.3f}")
        # 过滤无效深度
        valid_depths = depth_region[(depth_region > 0.1) & (depth_region < 100)]
        
        print(f"    [调试] 有效深度点数: {len(valid_depths)}")
        
        if len(valid_depths) == 0:
            print("    [警告] bbox区域深度全部无效")
            return None

         # 使用中位数
        median_depth = float(np.median(valid_depths))
        print(f"    ✓ 中位数深度: {median_depth:.2f}m")
            
        # 2. bbox 中心
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        
        # 3. 像素 → 相机坐标
        fx, fy = camera_params['fx'], camera_params['fy']
        cx_cam, cy_cam = camera_params['cx'], camera_params['cy']
        
        x_cam = (cx - cx_cam) * median_depth / fx
        y_cam = (cy - cy_cam) * median_depth / fy
        z_cam = median_depth
        
        # 4. 相机 → 世界坐标
        drone_pos = current_state.kinematics_estimated.position
        drone_orientation = current_state.kinematics_estimated.orientation
        
        yaw = airsim.to_eularian_angles(drone_orientation)[2]
        
        # 简化版旋转（只考虑 yaw）
        x_world = drone_pos.x_val + z_cam * np.cos(yaw) - x_cam * np.sin(yaw)
        y_world = drone_pos.y_val + z_cam * np.sin(yaw) + x_cam * np.cos(yaw)
        # z_world = -env.altitude  # 保持固定高度
        z_world = -altitude
        return [x_world, y_world, z_world]
    
    
    def detect(self, image, conf_threshold=0.3):
        """
        检测图像中的物体
        
        Args:
            image: np.array, RGB图像 (H, W, 3)
            conf_threshold: 置信度阈值
            
        Returns:
            detections: list of dict, 每个dict包含:
                - label: str, 类别名称
                - confidence: float, 置信度
                - bbox: [x1, y1, x2, y2], 边界框坐标
        """
        # YOLO推理
        results = self.model(image, conf=conf_threshold, verbose=False)
        
        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue
                
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                bbox = box.xyxy[0].cpu().numpy()
                
                # 获取类别名称
                label = r.names[cls_id]
                
                detections.append({
                    'label': label,
                    'confidence': conf,
                    'bbox': bbox,  # [x1, y1, x2, y2]
                    'class_id': cls_id
                })
        
        return detections
    
    def detect_with_depth(self, image, depth_map):
        """
        结合深度图的检测（用于估算物体距离）
        
        Args:
            image: RGB图像
            depth_map: 深度图 (H, W)
            
        Returns:
            detections: 包含距离信息的检测结果
        """
        detections = self.detect(image)
        
        for det in detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            # 从bbox区域提取深度
            depth_region = depth_map[y1:y2, x1:x2]
            
            if depth_region.size > 0:
                # 使用中位数深度（更robust）
                median_depth = np.median(depth_region[depth_region > 0])
                det['distance'] = float(median_depth)
            else:
                det['distance'] = None
        
        return detections
    
    def visualize_detections(self, image, detections):
        """
        可视化检测结果（用于调试）
        
        Args:
            image: 原始图像
            detections: 检测结果列表
            
        Returns:
            vis_image: 带标注的图像
        """
        vis_image = image.copy()
        
        for det in detections:
            bbox = det['bbox']
            label = det['label']
            conf = det['confidence']
            
            x1, y1, x2, y2 = map(int, bbox)
            
            # 绘制框
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 绘制标签
            text = f"{label}: {conf:.2f}"
            if 'distance' in det and det['distance']:
                text += f" ({det['distance']:.1f}m)"
            
            cv2.putText(vis_image, text, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return vis_image