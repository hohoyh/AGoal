# run_image_goal.py - 图像目标驱动的零样本无人机导航（AirSim）

import sys
sys.path.append('third_party/Grounded-Segment-Anything/')

import airsim
import numpy as np
import cv2
import torch
import time
import networkx as nx
from PIL import Image
from pathlib import Path
import math

# GroundingDINO + SAM（third_party/ 已加入 sys.path，见 scripts/download_models.sh）
from groundingdino.util.inference import load_model as load_groundingdino, predict as grounding_predict
from segment_anything import sam_model_registry, SamPredictor
import groundingdino.datasets.transforms as T

# 本项目模块
from src.mapping.stmr_builder import STMRBuilder
from src.visualization.navigation_visualizer import NavigationVisualizer

# CLIP (用于图匹配)
from transformers import CLIPProcessor, CLIPModel


class AGoalDroneNavigator:
    """
    图像目标无人机导航器：把目标图片解析成场景图，再与实时观测图匹配并导航接近
    """
    
    def __init__(self, goal_image_path):
        print("\n" + "="*60)
        print("AGoal 无人机导航系统")
        print("="*60)
        # 记录目标图像路径（供后续CLIP外观特征计算）
        self.goal_image_path = goal_image_path

        # 1. 连接 AirSim
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        print("\n[初始化] 使用 AirSim 内置 Segmentation 模式，无需加载 GroundingDINO/SAM")
        # # 2. 加载 GroundingDINO + SAM
        # print("\n[初始化] 加载 GroundingDINO + SAM...")
        
        groundingdino_config = 'third_party/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'
        groundingdino_checkpoint = 'data/models/groundingdino_swint_ogc.pth'
        self.groundingdino = load_groundingdino(groundingdino_config, groundingdino_checkpoint)
        
        sam_checkpoint = 'data/models/sam_vit_b_01ec64.pth'
        sam = sam_model_registry['vit_b'](checkpoint=sam_checkpoint)
        sam.to('cuda' if torch.cuda.is_available() else 'cpu')
        self.sam_predictor = SamPredictor(sam)
        
        # print("  ✓ 模型加载完成")
        
        # 3. 加载 CLIP (用于节点相似度)
        print("\n[初始化] 加载 CLIP...")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model.eval()
        print("  ✓ CLIP 加载完成")
        
        # 检测类别 (可以自定义)
        self.text_prompt = "building . tree . car . road . window . door"
        
        # 4. 构建目标图 G_g
        print(f"\n[目标图] 构建目标图: {goal_image_path}")
        self.goal_graph = self._construct_goal_graph(goal_image_path)
        print(f"  ✓ 目标图节点数: {self.goal_graph.number_of_nodes()}")
        print(f"  ✓ 目标图边数: {self.goal_graph.number_of_edges()}")
        
        # 打印目标图结构
        print("\n  目标图结构:")
        for node in self.goal_graph.nodes(data=True):
            print(f"    节点: {node[0]} (中心物体: {node[1].get('is_central', False)})")
        for edge in self.goal_graph.edges(data=True):
            print(f"    边: {edge[0]} -[{edge[2].get('relation', 'near')}]-> {edge[1]}")
         # ✅ 添加可视化器
        
        self.visualizer = NavigationVisualizer(output_dir="outputs/visualization")
        # 5. 初始化场景图 G_t
        self.scene_graph = nx.Graph()
        
        # 6. 导航参数
        self.scan_altitude = 5.0  # 扫描高度
        self.sigma_1 = 0.3  # 阶段1→2 阈值
        self.sigma_2 = 0.7  # 阶段2→3 阈值
        
        self.blacklist = set()  # 黑名单

        # ✅ 新增: STMR 构建器
        print("\n[初始化] 加载 STMR...")
        self.stmr = STMRBuilder(map_size=100, grid_size=5)
        print("  ✓ STMR 加载完成")

        
        print("\n" + "="*60)
        print("初始化完成! 准备导航...")
        print("="*60)
    
    # ============================================================
    # 核心功能: 构建目标图 G_g
    # ============================================================
    def _set_candidate_target_pixel(self, pixel_xy):
        """
        将候选目标在图像坐标系 (x,y) 转换为 AirSim 世界坐标，
        并设置为当前导航目标 (self.current_goal_pos)。
        """
        x_pix, y_pix = pixel_xy

        # === 1. 获取相机参数 ===
        camera_pose = self.client.simGetCameraInfo("0")
        drone_state = self.client.getMultirotorState()
        drone_pos = drone_state.kinematics_estimated.position
        altitude = abs(drone_pos.z_val)

        # === 2. 简化投影变换 (近似法) ===
        # 假设相机视野 90°，分辨率 640x480
        img_w, img_h = 640, 480
        fov = 90.0  # degrees
        focal_len = img_w / (2 * np.tan(np.deg2rad(fov / 2)))

        # 图像坐标中心化
        cx, cy = img_w / 2.0, img_h / 2.0
        dx = (x_pix - cx) / focal_len
        dy = (y_pix - cy) / focal_len

        # 转换到相机坐标系下的向量 (近似)
        dir_cam = np.array([dx, dy, 1.0])
        dir_cam /= np.linalg.norm(dir_cam)

        # === 3. 近似射线落地点估计 ===
        # 用无人机当前位置 + 视线方向 × 高度
        scale = altitude / dir_cam[2]
        world_offset = dir_cam * scale
        target_world = np.array([
            drone_pos.x_val + world_offset[0],
            drone_pos.y_val + world_offset[1],
            0.0  # 假设地面高度为0
        ])

        self.current_goal_pos = target_world.tolist()
        print(f"  🎯 新候选目标位置: {self.current_goal_pos}")

    def _construct_goal_graph(self, image_path):
        """
        ✅ 核心步骤: 从目标图片构建目标图 G_g
        """
        
        # 1. 加载图片
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 2. GroundingDINO 检测
        detections = self._detect_objects(image_rgb, self.text_prompt)
        
        if len(detections) == 0:
            print("  ⚠️ 目标图片中未检测到物体!")
            return nx.Graph()
        
        print(f"  检测到物体: {[d['label'] for d in detections]}")
        
        # 3. SAM 分割
        masks = self._sam_segment(image_rgb, detections)
        
        # 4. 识别中心物体 (面积最大的)
        largest_mask = max(masks, key=lambda m: m['mask'].sum())
        central_label = largest_mask['label']
        
        print(f"  中心物体: {central_label}")
        
        # 5. 构建图
        goal_graph = nx.Graph()
        
        # 添加节点
        for mask_data in masks:
            label = mask_data['label']
            
            # 提取 CLIP 特征 (用于后续匹配)
            bbox = mask_data['bbox']
            x1, y1, x2, y2 = bbox
            crop = image_rgb[y1:y2, x1:x2]
            
            with torch.no_grad():
                inputs = self.clip_processor(images=crop, return_tensors="pt")
                features = self.clip_model.get_image_features(**inputs)
                features = features / features.norm(dim=-1, keepdim=True)
            
            goal_graph.add_node(
                label,
                is_central=(label == central_label),
                bbox=bbox,
                clip_features=features.cpu().numpy()
            )
        
        # 6. ✅ 添加边 (简化版: 基于空间距离)
        for i, mask1 in enumerate(masks):
            for j, mask2 in enumerate(masks):
                if i >= j:
                    continue
                
                # 计算 BBox 中心距离
                bbox1 = mask1['bbox']
                bbox2 = mask2['bbox']
                
                center1 = [(bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2]
                center2 = [(bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2]
                
                dist = np.linalg.norm(np.array(center1) - np.array(center2))
                
                # 如果距离较近,添加边
                if dist < 200:  # 像素距离阈值
                    # 判断方向
                    dx = center2[0] - center1[0]
                    dy = center2[1] - center1[1]
                    
                    if abs(dx) > abs(dy):
                        relation = "right_of" if dx > 0 else "left_of"
                    else:
                        relation = "below" if dy > 0 else "above"
                    
                    goal_graph.add_edge(
                        mask1['label'],
                        mask2['label'],
                        relation=relation,
                        distance=dist
                    )
        # 在 __init__ 或构建目标图后加：
        self.goal_img_feat = None
        if self.goal_image_path is not None:
            img_bgr = cv2.imread(self.goal_image_path)
            if img_bgr is not None:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                with torch.no_grad():
                    clip_in = self.clip_processor(images=img_rgb, return_tensors="pt")
                    feat = self.clip_model.get_image_features(**clip_in)
                    self.goal_img_feat = (feat / feat.norm(dim=-1, keepdim=True)).cpu().numpy()

        return goal_graph
    
    def _rank_building_candidates(self, masks, rgb, graph_score: float, alpha: float = 0.6):
        """
        对场景中的 building 候选做外观匹配：总分 = alpha*graph_match + (1-alpha)*clip_sim
        - masks: 你从 SAM 得到的实例列表 [{'label','bbox','mask',...}]
        - rgb: 当前帧 RGB
        - graph_score: 本帧图匹配分（0~1）
        - alpha: 图匹配权重（关系可信时加大；很多楼时建议 0.5~0.7）
        """
        ranked = []
        if self.goal_img_feat is None:
            # 没有目标图像，就只用图匹配
            for m in masks:
                if m['label'] == 'building':
                    ranked.append((alpha*graph_score, m))
            return sorted(ranked, key=lambda x: x[0], reverse=True)

        for m in masks:
            if m['label'] != 'building': 
                continue
            x1,y1,x2,y2 = m['bbox']
            crop = rgb[y1:y2, x1:x2]
            if crop.size == 0: 
                continue
            with torch.no_grad():
                inp = self.clip_processor(images=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), return_tensors="pt")
                f = self.clip_model.get_image_features(**inp)
                f = (f / f.norm(dim=-1, keepdim=True)).cpu().numpy()
            sim = float((f @ self.goal_img_feat.T)[0,0])  # 余弦相似
            total = alpha*graph_score + (1-alpha)*sim
            ranked.append((total, m))
        ranked.sort(key=lambda x: x[0], reverse=True)
        return ranked


    def _find_semantic_frontier(self):
        """
        ✅ 找到语义前沿点（即“未知但可到达”的区域中心”）
        """
        state = self.client.getMultirotorState()
        drone_pos = [state.kinematics_estimated.position.x_val,
                    state.kinematics_estimated.position.y_val]
        
        local_map, center = self.stmr.get_local_map(drone_pos)
        frontier_mask = (local_map == 0)  # 未探索
        explored_mask = (local_map > 0)
        frontier_candidates = []

        for y in range(1, local_map.shape[0]-1):
            for x in range(1, local_map.shape[1]-1):
                if frontier_mask[y, x] and np.any(explored_mask[y-1:y+2, x-1:x+2]):
                    frontier_candidates.append((x, y))
        
        if not frontier_candidates:
            return None

        # 求前沿中心
        frontier = np.mean(frontier_candidates, axis=0)
        wx = (frontier[0] - center[0]) * self.stmr.grid_size + drone_pos[0]
        wy = (center[1] - frontier[1]) * self.stmr.grid_size + drone_pos[1]
        wz = -self.scan_altitude
        return [wx, wy, wz]

    
    def _detect_objects(self, image, text_prompt):
        """
        GroundingDINO 检测
        """
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
                caption=text_prompt,
                box_threshold=0.25,
                text_threshold=0.2,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
        
        H, W = image.shape[:2]
        detections = []
        
        for box, logit, phrase in zip(boxes, logits, phrases):
            cx, cy, w, h = box.cpu().numpy()
            x1 = int((cx - w/2) * W)
            y1 = int((cy - h/2) * H)
            x2 = int((cx + w/2) * W)
            y2 = int((cy + h/2) * H)
            
            detections.append({
                'label': phrase.lower().strip(),
                'confidence': float(logit),
                'bbox': [x1, y1, x2, y2]
            })
        
        return detections
    
    def _sam_segment(self, image, detections):
        """
        SAM 分割
        """
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
    
    # ============================================================
    # 核心功能: 实时构建场景图 G_t
    # ============================================================
    
    def _update_scene_graph(self):
        """
        ✅ 使用 AirSim 获取 RGB + Segmentation 图
        - Segmentation 图:用于 STMR 地图更新
        - RGB 图:用于 GroundingDINO + SAM 构建场景图
        """

        # === 1. 从 AirSim 获取图像 ===
        responses = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),        # RGB
            airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False)  # 分割
        ])
        if not responses or len(responses) < 2:
            print("⚠️ 无法从 AirSim 获取图像帧")
            return

        # RGB 图
        rgb = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
        rgb = rgb.reshape(responses[0].height, responses[0].width, 3)

        # 分割图(仅用于语义地图)
        seg = np.frombuffer(responses[1].image_data_uint8, dtype=np.uint8)
        seg = seg.reshape(responses[1].height, responses[1].width, 3)
        
        # ✅ 保存为类成员变量,供可视化使用
        self.last_rgb = rgb.copy()
        self.last_seg = seg.copy()
        
        print(f"\n[调试] 分割图形状: {seg.shape}")
        print(f"[调试] 分割图唯一颜色数: {len(np.unique(seg.reshape(-1, 3), axis=0))}")
        print(f"[调试] 分割图示例像素: {seg[240, 320]}")
        
        # 保存分割图检查
        if hasattr(self, '_debug_count'):
            self._debug_count += 1
        else:
            self._debug_count = 0
        
        if self._debug_count % 10 == 0:
            cv2.imwrite(f'debug/seg_{self._debug_count}.png', seg)
            print(f"  ✓ 已保存分割图: debug/seg_{self._debug_count}.png")

        # === 2. 获取无人机位置 ===
        state = self.client.getMultirotorState()
        drone_pos = [
            state.kinematics_estimated.position.x_val,
            state.kinematics_estimated.position.y_val,
            state.kinematics_estimated.position.z_val
        ]

        # === 3. 更新语义地图 (Segmentation 用)
        self.stmr.update_map(seg, drone_pos)

        # === 4. 利用 RGB 图构建场景图 (与目标图逻辑一致) ===
        detections = self._detect_objects(rgb, self.text_prompt)
        if len(detections) == 0:
            print("⚠️ 当前帧未检测到物体,场景图为空")
            return

        masks = self._sam_segment(rgb, detections)
        if len(masks) == 0:
            print("⚠️ SAM 未生成有效分割掩膜,跳过本帧")
            return

        scene_graph = nx.Graph()

        for mask_data in masks:
            label = mask_data['label']
            bbox = mask_data['bbox']
            x1, y1, x2, y2 = bbox
            crop = rgb[y1:y2, x1:x2]

            # CLIP特征
            with torch.no_grad():
                inputs = self.clip_processor(images=crop, return_tensors="pt")
                features = self.clip_model.get_image_features(**inputs)
                features = features / features.norm(dim=-1, keepdim=True)

            # 添加节点
            scene_graph.add_node(
                label,
                label=label,
                bbox=bbox,
                clip_features=features.cpu().numpy(),
                position=(drone_pos[0], drone_pos[1], drone_pos[2])
            )

        # === 5. 添加节点间空间关系(基于 2D 图像中心) ===
        for i, m1 in enumerate(masks):
            for j, m2 in enumerate(masks):
                if i >= j:
                    continue
                bbox1, bbox2 = m1["bbox"], m2["bbox"]
                c1 = [(bbox1[0]+bbox1[2])/2, (bbox1[1]+bbox1[3])/2]
                c2 = [(bbox2[0]+bbox2[2])/2, (bbox2[1]+bbox2[3])/2]
                dist = np.linalg.norm(np.array(c1) - np.array(c2))
                if dist < 200:
                    dx, dy = c2[0]-c1[0], c2[1]-c1[1]
                    if abs(dx) > abs(dy):
                        rel = "right_of" if dx > 0 else "left_of"
                    else:
                        rel = "below" if dy > 0 else "above"
                    scene_graph.add_edge(m1["label"], m2["label"], relation=rel, distance=dist)

        self.scene_graph = scene_graph
        print(f"  ✅ 场景图更新成功: {scene_graph.number_of_nodes()} 节点, {scene_graph.number_of_edges()} 边")
        
        # 候选楼排序
        if any(m['label']=='building' for m in masks):
            ranked = self._rank_building_candidates(
                masks, rgb, 
                graph_score=getattr(self, 'last_match_score', 0.0), 
                alpha=0.6
            )
            if ranked:
                best_total, best_m = ranked[0]
                print(f"  🧭 候选楼分数(融合): {best_total:.3f}  (top-1 label={best_m['label']})")
                cx = (best_m['bbox'][0] + best_m['bbox'][2]) / 2
                cy = (best_m['bbox'][1] + best_m['bbox'][3]) / 2
                self._set_candidate_target_pixel((cx, cy))

    def _approach_best_match(self, matched_pairs):
        """
        ✅ 接近当前匹配度最高的候选目标
        - 从 matched_pairs 中选择中心目标节点（或第一个匹配）
        - 获取对应场景节点的三维坐标
        - 调用避障导航函数移动过去
        """

        if not matched_pairs:
            print("⚠️ 没有匹配到任何候选目标，无法接近")
            return False

        # 1. 选取中心目标节点（优先 is_central=True）
        central_node_goal = None
        for node, data in self.goal_graph.nodes(data=True):
            if data.get("is_central", False):
                central_node_goal = node
                break

        # 如果目标图中没有中心节点，则取第一个匹配节点
        if not central_node_goal:
            central_node_goal = matched_pairs[0][0]

        # 2. 找到对应场景节点
        scene_target = None
        for goal_node, scene_node in matched_pairs:
            if goal_node == central_node_goal:
                scene_target = scene_node
                break
        if not scene_target:
            print("⚠️ 未找到中心目标的匹配节点，默认取首个匹配节点")
            scene_target = matched_pairs[0][1]

        # 3. 获取目标位置
        target_data = self.scene_graph.nodes[scene_target]
        target_pos = target_data.get("position", [0, 0, -self.scan_altitude])

        print(f"  🚀 接近匹配目标: {scene_target} @ {target_pos}")

        # 4. 调用避障移动
        self._lidar_avoidance_move(
            [target_pos[0], target_pos[1], -5.0],
            velocity=2.5
        )

        # 5. 确认到达附近后再次更新场景图 + 匹配验证
        time.sleep(1.0)
        self._update_scene_graph()
        score, pairs = self._match_graphs(self.scene_graph, self.goal_graph)
        print(f"  🔁 接近后匹配验证: score={score:.3f}")
        if score > 0.75:
            print("  ✅ 接近验证通过，进入阶段3确认")
            self._stage3_approach(pairs)
        else:
            print("  ⚠️ 接近后仍不稳定，继续探索")
            self._stage1_exploration()

    def _match_graphs(self, scene_graph, goal_graph):
        """
        计算场景图与目标图的匹配分数（健壮版）
        - 缺失 label/clip_features 时自动降级
        - 节点语义相似度 + 关系结构相似度 融合
        返回: (match_score, matched_pairs)
        """
        import itertools

        if scene_graph is None or goal_graph is None:
            print("⚠️ 无法匹配：scene/goal graph 为空")
            return 0.0, []

        def _node_label(G, n, d):
            # 优先用属性里的 label，其次 cls/category/name，最后用节点ID（若为字符串）
            return (
                d.get("label")
                or d.get("cls")
                or d.get("category")
                or d.get("name")
                or (n if isinstance(n, str) else "unknown")
            )

        def _node_feat(d):
            # 可能是 numpy/tensor，也可能不存在
            f = d.get("clip_features", None)
            if f is None:
                return None
            # 统一成 torch.FloatTensor [1,D]
            if isinstance(f, list):
                f = np.array(f)
            if isinstance(f, np.ndarray):
                f = torch.from_numpy(f)
            if f.dim() == 1:
                f = f.unsqueeze(0)
            return f.float()

        matched_pairs = []
        node_scores = []

        # ---------- 1) 节点级匹配 ----------
        for ng, gd in goal_graph.nodes(data=True):
            g_label = _node_label(goal_graph, ng, gd)
            g_feat  = _node_feat(gd)

            best_score = 0.0
            best_match = None

            for ns, sd in scene_graph.nodes(data=True):
                s_label = _node_label(scene_graph, ns, sd)
                s_feat  = _node_feat(sd)

                # (a) 语义相似度：优先用 CLIP；若缺失，则用“类别相等”作为弱相似
                sim_sem = 0.0
                if g_feat is not None and s_feat is not None:
                    sim_sem = torch.nn.functional.cosine_similarity(g_feat, s_feat).item()
                elif g_label != "unknown" and s_label != "unknown":
                    sim_sem = 1.0 if (g_label == s_label) else 0.0
                else:
                    sim_sem = 0.0

                # (b) 类别一致加一点偏置（可帮忙 disambiguation）
                if g_label == s_label and sim_sem < 0.9:
                    sim_sem = sim_sem * 0.7 + 0.3

                if sim_sem > best_score:
                    best_score = sim_sem
                    best_match = (ng, ns)

            if best_match is not None:
                matched_pairs.append(best_match)
                node_scores.append(best_score)

        if not node_scores:
            print("⚠️ 节点匹配为空")
            return 0.0, []

        node_score = float(np.mean(node_scores))

        # ---------- 2) 关系（边）相似 ----------
        # 统计目标图中每种 relation 的数量，并看 scene_graph 中能匹配多少
        def _edge_rel_counts(G):
            cnt = {}
            for u, v, d in G.edges(data=True):
                rel = d.get("relation", None)
                if rel is None:
                    continue
                cnt[rel] = cnt.get(rel, 0) + 1
            return cnt

        goal_rel = _edge_rel_counts(goal_graph)
        scene_rel = _edge_rel_counts(scene_graph)

        if sum(goal_rel.values()) == 0:
            edge_score = 1.0  # 目标图没有边，则默认关系项满分
        else:
            hit = 0
            total = 0
            for rel, c in goal_rel.items():
                total += c
                hit += min(c, scene_rel.get(rel, 0))
            edge_score = hit / max(total, 1)

        # ---------- 3) 综合得分 ----------
        match_score = 0.7 * node_score + 0.3 * edge_score
        print(f"  🎯 匹配计算: 节点={node_score:.3f}, 关系={edge_score:.3f}, 综合={match_score:.3f}")
        return match_score, matched_pairs

    
    # ============================================================
    # 核心功能: 图匹配
    # ============================================================
    
    def _graph_matching(self):
        """
        ✅ 核心步骤: 计算实时场景图 G_t 与目标图 G_g 的匹配分数
        
        返回: (匹配分数, 匹配的节点对)
        """
        
        if self.scene_graph.number_of_nodes() == 0:
            return 0.0, []
        
        # 1. 节点匹配 (基于 CLIP 相似度)
        matched_pairs = []
        similarity_scores = []
        
        for goal_node, goal_data in self.goal_graph.nodes(data=True):
            goal_features = goal_data['clip_features']
            
            best_match = None
            best_sim = 0.0
            
            for scene_node, scene_data in self.scene_graph.nodes(data=True):
                if scene_node in self.blacklist:
                    continue
                
                # 类别必须匹配
                if scene_data['label'] != goal_node:
                    continue
                
                scene_features = scene_data['clip_features']
                
                # 计算余弦相似度
                sim = float(np.dot(goal_features.flatten(), scene_features.flatten()))
                
                if sim > best_sim:
                    best_sim = sim
                    best_match = scene_node
            
            if best_match and best_sim > 0.7:  # 相似度阈值
                matched_pairs.append((goal_node, best_match))
                similarity_scores.append(best_sim)
        
        # 2. 计算匹配分数 (简化版)
        if len(matched_pairs) == 0:
            return 0.0, []
        
        # 匹配分数 = 匹配节点数 / 目标图节点数
        node_score = len(matched_pairs) / self.goal_graph.number_of_nodes()
        
        # 平均相似度
        avg_similarity = np.mean(similarity_scores)
        
        # 综合分数
        final_score = (node_score + avg_similarity) / 2
        
        return final_score, matched_pairs
    
    def _compute_confidence(self, similarity_scores, matched_pairs, goal_graph, target_pos):
            """
            ✅ 计算置信度 C (综合节点匹配+相似度+距离)
            """
            # 1. 节点匹配比例
            Rn = len(matched_pairs) / max(goal_graph.number_of_nodes(), 1)

            # 2. 平均相似度
            if len(similarity_scores) > 0:
                S_node = float(np.mean(similarity_scores))
            else:
                S_node = 0.0

            # 3. 边与拓扑相似度 (此简化版未实现, 可设为常量)
            S_edge = 0.3 * S_node
            S_topo = 0.2 * S_node

            # 4. 距离惩罚
            drone_pos = self.client.getMultirotorState().kinematics_estimated.position
            dx = drone_pos.x_val - target_pos[0]
            dy = drone_pos.y_val - target_pos[1]
            dz = drone_pos.z_val - target_pos[2]
            dist = (dx**2 + dy**2 + dz**2) ** 0.5
            D_penalty = np.exp(-dist / 10.0)

            # 5. 综合置信度
            S_total = 0.5 * S_node + 0.3 * S_edge + 0.2 * S_topo
            C = 0.6 * S_total + 0.3 * Rn + 0.1 * D_penalty

            print(f"  [验证] S_node={S_node:.2f}, 匹配比例={Rn:.2f}, 距离={dist:.1f}m → 置信度 C={C:.2f}")
            return C

    def _lidar_avoidance_move(self, goal_xyz, safe_dist=5.0, velocity=3.0):
            """
            ✅ 使用 Lidar 实现真实避障导航（改进版）
            - 通过雷达点云检测前方障碍, 自动横向绕行
            - 解决原版坐标系方向错误导致"飞进建筑"的问题
            """

            lidarData = self.client.getLidarData(lidar_name="Lidar1", vehicle_name="Drone1")

            if len(lidarData.point_cloud) < 3:
                print("⚠️ 无点云数据, 直接前往目标")
                self.client.moveToPositionAsync(*goal_xyz, velocity=velocity).join()
                return

            pts = np.array(lidarData.point_cloud, dtype=np.float32).reshape(-1, 3)

            # AirSim 坐标系:
            #   X: 前方 (+x)
            #   Y: 右方 (+y)
            #   Z: 向下 (+z)
            # 因此我们避障时，左右绕行要基于Y轴判断

            # 1. 过滤前方区域 ±60°
            forward_mask = (pts[:, 0] > 0) & (np.abs(pts[:, 1]) < 10)
            forward_pts = pts[forward_mask]

            if forward_pts.size == 0:
                print("✅ 前方无障碍, 直飞")
                self.client.moveToPositionAsync(*goal_xyz, velocity=velocity).join()
                return

            dists = np.linalg.norm(forward_pts, axis=1)
            min_dist = np.min(dists)

            # 2. 若障碍太近，则绕行
            if min_dist < safe_dist:
                nearest = forward_pts[np.argmin(dists)]
                # 计算避障方向：
                #  如果障碍在右侧(Y>0)，则往左飞；反之往右
                if nearest[1] > 0:
                    side_dir = -1  # 左
                else:
                    side_dir = 1   # 右

                step_side = safe_dist * 2.0
                step_forward = safe_dist * 1.5

                # 构造新目标点：先横向绕行一点，再前进一点
                new_goal = np.array([
                    goal_xyz[0] + step_forward,
                    goal_xyz[1] + side_dir * step_side,
                    goal_xyz[2]
                ])

                print(f"🚧 检测障碍({min_dist:.2f}m)，绕行方向: {'左' if side_dir==-1 else '右'} → 新目标 {new_goal}")
                self.client.moveToPositionAsync(new_goal[0], new_goal[1], new_goal[2],
                                                velocity=velocity).join()
            else:
                print("✅ 前方安全, 直飞")
                self.client.moveToPositionAsync(*goal_xyz, velocity=velocity).join()



    # ============================================================
    # 核心功能: 三阶段探索
    # ============================================================
    
    def navigate_to_goal(self):
        """
        主导航循环 - 修复版
        """
        
        # 1. 起飞
        print("\n[1] 起飞...")
        self.client.takeoffAsync().join()
        self.client.moveToZAsync(-self.scan_altitude, 3).join()
        time.sleep(1)
        
        print(f"  ✓ 已到达扫描高度: {self.scan_altitude}m\n")
        
        # 2. 导航循环
        max_steps = 50
        
        for step in range(max_steps):
            print(f"\n{'='*60}")
            print(f"步骤 {step + 1}/{max_steps}")
            print(f"{'='*60}")
            
            # ✅ 第一步: 获取无人机状态
            state = self.client.getMultirotorState()
            drone_pos = state.kinematics_estimated.position
            
            # 2.1 更新场景图 (会设置 self.last_rgb 和 self.last_seg)
            self._update_scene_graph()
            print(f"  场景图节点数: {self.scene_graph.number_of_nodes()}")
            
            # 2.2 图匹配
            match_score, matched_pairs = self._graph_matching()
            print(f"  匹配分数: {match_score:.3f}")
            print(f"  匹配节点对: {len(matched_pairs)}")
            # ✅ 计算置信度
            confidence = 0.0
            if match_score > self.sigma_2 and matched_pairs:
                # 提取相似度分数
                similarity_scores = []
                for g_node, s_node in matched_pairs:
                    if g_node in self.goal_graph.nodes and s_node in self.scene_graph.nodes:
                        g_feat = self.goal_graph.nodes[g_node].get('clip_features')
                        s_feat = self.scene_graph.nodes[s_node].get('clip_features')
                        if g_feat is not None and s_feat is not None:
                            sim = float(np.dot(g_feat.flatten(), s_feat.flatten()))
                            similarity_scores.append(sim)
                
                # 计算置信度
                if similarity_scores:
                    target_pos = [drone_pos.x_val, drone_pos.y_val, drone_pos.z_val]
                    confidence = self._compute_confidence(
                        similarity_scores, 
                        matched_pairs, 
                        self.goal_graph, 
                        target_pos
                    )

            
            # ✅ 准备可视化数据 (在所有数据都准备好后)
            viz_data = {
                'rgb_image': getattr(self, 'last_rgb', None),
                'seg_image': getattr(self, 'last_seg', None),
                'semantic_map': self.stmr.topdown_map,
                'drone_pos': [drone_pos.x_val, drone_pos.y_val, drone_pos.z_val],
                'goal_pos': getattr(self, 'current_goal_pos', None),
                'trajectory': self.stmr.trajectory,
                'stage': 3 if match_score > self.sigma_2 else (2 if match_score > self.sigma_1 else 1),
                'match_score': match_score,
                'confidence': confidence,  # ✅ 改这里!使用计算出的confidence
                'matched_nodes': len(matched_pairs),
                'info_text': f"Nodes: {self.scene_graph.number_of_nodes()}, Score: {match_score:.2f}"
            }
            
            # ✅ 生成可视化
            try:
                self.visualizer.visualize_step(step + 1, viz_data)
            except Exception as e:
                print(f"  ⚠️  可视化失败: {e}")
            
            # 2.3 根据匹配分数选择策略
            if match_score < self.sigma_1:
                print("\n  [阶段 1: Zero Matching] 扩大探索...")
                self._stage1_exploration()
            
            elif match_score < self.sigma_2:
                print("\n  [阶段 2: Partial Matching] 推断目标位置...")
                success = self._stage2_inference(matched_pairs)
                if success:
                    print("\n✅ 导航成功!")
                    break
            
            else:
                print("\n  [阶段 3: Perfect Matching] 接近目标...")
                success = self._stage3_approach(matched_pairs)
                if success:
                    print("\n✅ 导航成功!")
                    break
            
            # ✅ 每5步保存高质量地图
            if step % 5 == 0:
                state = self.client.getMultirotorState()
                drone_pos_2d = [
                    state.kinematics_estimated.position.x_val,
                    state.kinematics_estimated.position.y_val
                ]
                
                self.stmr.visualize_topdown_map(
                    f'maps/semantic_hq_step_{step}.png', 
                    drone_pos=drone_pos_2d,
                    high_quality=True
                )
            
            # ✅ 每10步保存STMR矩阵
            if step % 10 == 0:
                self.stmr.visualize_topdown_map(f'maps/stmr_step_{step}.png')
                
                state = self.client.getMultirotorState()
                drone_pos_2d = [
                    state.kinematics_estimated.position.x_val,
                    state.kinematics_estimated.position.y_val
                ]
                
                local_matrix, center = self.stmr.get_local_map(drone_pos_2d)
                self.stmr.visualize_matrix(local_matrix, f"maps/matrix_step_{step}.png")
                
                stmr_text = self.stmr.matrix_to_text(local_matrix, center)
                print(f"\n[STMR Matrix]\n{stmr_text}")
                
                self.stmr.update_trajectory(drone_pos_2d)
                self.stmr.visualize_topdown_map("maps/occupancy_debug.png")
            
            time.sleep(0.5)
        
        # ✅ 生成导航过程视频
        try:
            print("\n[视频生成] 正在合成导航过程视频...")
            self.visualizer.create_video(fps=2, output_name="navigation_full.mp4")
        except Exception as e:
            print(f"  ⚠️  视频生成失败: {e}")
        
        # 3. 降落
        print("\n[完成] 降落...")
        self.client.landAsync().join()
        self.client.armDisarm(False)
        self.client.enableApiControl(False)


    def _spin_and_match_once(self, yaw_deg: float):
        """将机体旋转到 yaw_deg（度），抓一帧，更新场景图并返回匹配分数与匹配对"""
        # 1) 旋转
        self.client.rotateToYawAsync(yaw_deg, 2.0).join()
        time.sleep(0.2)  # 给渲染/传感器一点缓冲

        # 2) 更新场景图（会调用 DINO+SAM 构建 nodes/edges）
        self._update_scene_graph()

        # 3) 做一次匹配（你的现有匹配函数名若不同，替换这里）
        if getattr(self, "scene_graph", None) and self.goal_graph is not None:
            score, pairs = self._match_graphs(self.scene_graph, self.goal_graph)
            return float(score), pairs
        return 0.0, []

    def spin_match_then_decide(self, sweep_num: int = 6, accept: float = 0.55):
        """
        原地做 360° 多视角扫掠与匹配；返回 (best_score, best_pairs, best_yaw)
        - sweep_num=6 → 每60°一帧；你也可改成8或12
        - accept：达到该阈值就直接认定“看到了”，不再进入随机探索
        """
        # 当前朝向
        quat = self.client.getMultirotorState().kinematics_estimated.orientation
        _, _, yaw0 = airsim.to_eularian_angles(quat)  # 弧度
        yaw0_deg = math.degrees(yaw0)

        best = (0.0, [], yaw0_deg)
        step = 360.0 / sweep_num

        for k in range(sweep_num):
            yaw_deg = yaw0_deg + k * step
            score, pairs = self._spin_and_match_once(yaw_deg)
            print(f"  🔎 spin {k+1}/{sweep_num}: yaw={yaw_deg:.1f}°, score={score:.3f}, pairs={len(pairs)}")
            if score > best[0]:
                best = (score, pairs, yaw_deg)
            if score >= accept:
                print("  ✅ 多视角匹配已达阈值，立即进入接近阶段")
                return best

        # 转回最佳朝向，便于接近
        self.client.rotateToYawAsync(best[2], 2.0).join()
        return best
    
    def _stage1_exploration(self):
        """
        阶段 1: 零匹配 - 随机探索
        """
        
        # 简化版: 螺旋扫描
        current_pos = self.client.getMultirotorState().kinematics_estimated.position
        
        # 随机选择一个方向
        angle = np.random.uniform(0, 360)
        radius = 10  # 前进 10 米
        
        new_x = current_pos.x_val + radius * np.cos(np.radians(angle))
        new_y = current_pos.y_val + radius * np.sin(np.radians(angle))
        
        print(f"  → 飞向 ({new_x:.1f}, {new_y:.1f})")
        
        self.client.moveToPositionAsync(
            new_x, new_y, -self.scan_altitude,
            velocity=5, timeout_sec=10
        ).join()
        
        # 旋转观察
        for yaw in [0, 90, 180, 270]:
            self.client.rotateToYawAsync(yaw, timeout_sec=2).join()
            time.sleep(0.3)
            self._update_scene_graph()
        
        next_target = self._find_semantic_frontier()
        if next_target:
            print(f"🧭 前往语义前沿点: {next_target}")
            self._lidar_avoidance_move(next_target)
            self.stmr.visualize_topdown_map("maps/frontier_debug.png", drone_pos=next_target[:2])
        else:
            # （新逻辑：先强制多视角匹配，再决定是否随机）
            print("⚠️ 未找到前沿，先原地多视角匹配再决定")
            best_score, best_pairs, best_yaw = self.spin_match_then_decide(sweep_num=8, accept=0.60)
            if best_score >= 0.60:
                # 直接进入“阶段3：接近目标”
                print(f"  [多视角后切换] score={best_score:.3f} → 接近目标")
                self._approach_best_match(best_pairs)   # 用你现有的接近函数名替换
            else:
                print("  ↪︎ 仍未可靠匹配，才执行随机探索")
                self._random_explore()


    
    def _stage2_inference(self, matched_pairs):
        """
        阶段 2: 部分匹配 - 推断目标位置
        """
        
        # 找到中心物体
        central_node_goal = None
        for node, data in self.goal_graph.nodes(data=True):
            if data.get('is_central', False):
                central_node_goal = node
                break
        
        if not central_node_goal:
            return False
        
        # 检查中心物体是否已匹配
        central_matched = False
        central_scene_node = None

        # 找到已匹配节点的场景位置
        matched_positions = []
        for goal_node, scene_node in matched_pairs:
            goal_feat = self.goal_graph.nodes[goal_node]
            scene_feat = self.scene_graph.nodes[scene_node]
            matched_positions.append(scene_feat['position'])

        # 估计未观测目标的大致方向（简单策略：中心点外推）
        if len(matched_positions) >= 2:
            matched_positions = np.array(matched_positions)
            mean_pos = np.mean(matched_positions[:, :2], axis=0)
            direction = matched_positions[-1][:2] - matched_positions[0][:2]
            direction = direction / np.linalg.norm(direction)
            pred_point = mean_pos + direction * 15.0
            next_goal = [pred_point[0], pred_point[1], -self.scan_altitude]
            print(f"🔮 推测目标区域: {next_goal}")
            self._lidar_avoidance_move(next_goal)
            return False
        else:
            print("  匹配太少，继续探索")
            self._stage1_exploration()
            return False

        
        # for goal_node, scene_node in matched_pairs:
        #     if goal_node == central_node_goal:
        #         central_matched = True
        #         central_scene_node = scene_node
        #         break
        
        # if not central_matched:
        #     print("  中心物体未匹配, 继续探索...")
        #     self._stage1_exploration()
        #     return False
        
        # 飞向中心物体
        scene_node_data = self.scene_graph.nodes[central_scene_node]
        target_pos = scene_node_data['position']
        
        print(f"  → 飞向中心物体: {central_scene_node} @ ({target_pos[0]:.1f}, {target_pos[1]:.1f})")
        
        # self.client.moveToPositionAsync(
        #     target_pos[0], target_pos[1], -self.scan_altitude,
        #     velocity=3, timeout_sec=10
        # ).join()
        self._lidar_avoidance_move([target_pos[0], target_pos[1], -self.scan_altitude])

        
        return True
    
    def _stage3_approach(self, matched_pairs):
        """
        ✅ 阶段 3: 完美匹配 - 接近并验证 (论文式置信度)
        """
        # 1. 找到中心目标节点
        central_node_goal = None
        for node, data in self.goal_graph.nodes(data=True):
            if data.get('is_central', False):
                central_node_goal = node
                break
        if not central_node_goal:
            return False

        # 2. 找到匹配到的场景节点
        central_scene_node = None
        for goal_node, scene_node in matched_pairs:
            if goal_node == central_node_goal:
                central_scene_node = scene_node
                break
        if not central_scene_node:
            print("  ⚠️ 未找到对应场景节点")
            return False

        # 3. 获取目标坐标
        scene_node_data = self.scene_graph.nodes[central_scene_node]
        target_pos = scene_node_data['position']

        print(f"  → 接近目标: {central_scene_node}")
        # self.client.moveToPositionAsync(
        #     target_pos[0], target_pos[1], -5, velocity=2, timeout_sec=10
        # ).join()
        self._lidar_avoidance_move([target_pos[0], target_pos[1], -5], velocity=2)

        print("  ✓ 已到达目标附近!")
        
        next_target = self._find_semantic_frontier()
        if next_target:
            print(f"🧭 前往语义前沿点: {next_target}")
            self._lidar_avoidance_move(next_target)
        else:
            print("⚠️ 未找到前沿，随机探索")
            self._stage1_exploration()

        # 4. === 语义验证 ===
        center_label_goal = central_node_goal
        scene_label = scene_node_data['label']
        if scene_label != center_label_goal:
            print(f"  ❌ 语义不符: {scene_label} ≠ {center_label_goal}")
            return False

        # 5. === 置信度计算 ===
        # 重新获取最近一次匹配的相似度列表
        similarity_scores = []
        for g_node, s_node in matched_pairs:
            g_feat = self.goal_graph.nodes[g_node]['clip_features']
            s_feat = self.scene_graph.nodes[s_node]['clip_features']
            sim = float(np.dot(g_feat.flatten(), s_feat.flatten()))
            similarity_scores.append(sim)

        C = self._compute_confidence(
            similarity_scores, matched_pairs, self.goal_graph, target_pos
        )

        # 6. === 成功判定 ===
        if C >= 0.85:
            print("✅ 语义 + 置信度验证通过! 导航成功!")
            return True
        else:
             print(f"⚠️ 置信度不足 (C={C:.2f}), 继续探索...")

        # === 退回阶段 2 的探索逻辑 ===
        next_target = self._find_new_frontier()
        if next_target is not None:
            print(f"↩️  回退探索: 前往新的前沿点 {next_target}")
            self.client.moveToPositionAsync(
                next_target[0], next_target[1], -self.scan_altitude,
                velocity=3, timeout_sec=15
            ).join()
        else:
            print("⚠️ 未找到新的前沿, 执行随机探索")
            self._stage1_exploration()

        return False

    def _find_new_frontier(self):
        """
        ✅ 选择一个新的探索目标点
        - 优先选择 road/grass/lake/open_space
        - 若全是原点或距离太近, 加随机扰动
        """
        if self.scene_graph.number_of_nodes() == 0:
            return None

        # 当前无人机位置
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        current_xy = np.array([pos.x_val, pos.y_val], dtype=float)

        candidates = []
        for node, data in self.scene_graph.nodes(data=True):
            label = data.get("label", "")
            if label in ["road", "grass", "lake", "open_space", "building"]:
                p = np.array(data.get("position", [0, 0, 0]), dtype=float)
                # 排除太近的点
                if np.linalg.norm(p[:2] - current_xy) > 5.0:
                    candidates.append(p)

        # 如果没有合格候选，随机给一个方向
        if not candidates:
            dx, dy = np.random.uniform(-30, 30, size=2)
            new_point = current_xy + np.array([dx, dy])
            return [new_point[0], new_point[1], -self.scan_altitude]

        # 否则取最远的那个并加上随机偏移
        farthest = max(
            candidates,
            key=lambda p: np.linalg.norm(p[:2] - current_xy)
        )
        farthest = np.array(farthest)
        farthest[:2] += np.random.uniform(-10, 10, size=2)  # 防止陷入同点
        farthest[2] = -self.scan_altitude
        return farthest.tolist()



# ============================================================
# 主程序
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="给一张目标图片，无人机在 AirSim 中自主搜索并飞抵该目标"
    )
    parser.add_argument('--goal-image', required=True, help='目标图片路径')
    args = parser.parse_args()

    # 确保输出目录存在（语义地图、调试帧、可视化面板都会写到这里）
    for d in ('maps', 'outputs', 'debug'):
        Path(d).mkdir(parents=True, exist_ok=True)

    # 创建导航器
    navigator = AGoalDroneNavigator(args.goal_image)
    
    # 开始导航
    navigator.navigate_to_goal()
    
    print("\n" + "="*60)
    print("导航完成!")
    print("="*60)


if __name__ == '__main__':
    main()