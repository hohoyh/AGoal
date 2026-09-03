import os
import sys
import cv2
import numpy as np
import torch
import math
import dataclasses
import omegaconf
from openai import OpenAI
import base64
from io import BytesIO
import supervision as sv
import random
from PIL import Image
from sklearn.cluster import DBSCAN  
from collections import Counter 
from omegaconf import DictConfig
from pathlib import PosixPath, Path
from supervision.draw.color import ColorPalette
import copy
import skimage

from .graphbuilder import GraphBuilder
from .goalgraphdecomposer import GoalGraphDecomposer
from .overlap import GraphMatcher
from .scenegraphcorrector import SceneGraphCorrector
from .utils.slam_classes import MapObjectList
from .utils.utils import filter_objects, gobs_to_detection_list
from .utils.mapping import compute_spatial_similarities, merge_detections_to_objects

from ..utils.fmm.fmm_planner import FMMPlanner
from ..utils.fmm import pose_utils as pu
from ..utils.camera import get_camera_matrix
from ..utils.map import remove_small_frontiers
from ..utils.llm import LLM, VLM
from ..utils.config import resolve_api_key

sys.path.append('third_party/Grounded-Segment-Anything/')
from grounded_sam_demo import load_model, get_grounding_output
import GroundingDINO.groundingdino.datasets.transforms as T
from segment_anything import sam_model_registry, SamPredictor
from lightglue import LightGlue, DISK
from lightglue.utils import match_pair , numpy_image_to_torch


ADDITIONAL_PSL_OPTIONS = {
    'log4j.threshold': 'INFO'
}

ADDITIONAL_CLI_OPTIONS = [
    # '--postgres'
]


class RoomNode():
    def __init__(self, caption):
        self.caption = caption
        self.exploration_level = 0
        self.nodes = set()
        self.group_nodes = []


class GroupNode():
    def __init__(self, caption=''):
        self.caption = caption
        self.exploration_level = 0
        self.corr_score = 0
        self.center = None
        self.center_node = None
        self.nodes = []
        self.edges = set()
    
    def __lt__(self, other):
        return self.corr_score < other.corr_score
    
    def get_graph(self):
        self.center = np.array([node.center for node in self.nodes]).mean(axis=0)
        min_distance = np.inf
        for node in self.nodes:
            distance = np.linalg.norm(np.array(node.center) - np.array(self.center))
            if distance < min_distance:
                min_distance = distance
                self.center_node = node
            self.edges.update(node.edges)
        self.caption = self.graph_to_text(self.nodes, self.edges)

    def graph_to_text(self, nodes, edges):
        nodes_text = ', '.join([node.caption for node in nodes])
        edges_text = ', '.join([f"{edge.node1.caption} {edge.relation} {edge.node2.caption}" for edge in edges])
        return f"Nodes: {nodes_text}. Edges: {edges_text}."

class ObjectNode():
    def __init__(self):
        self.is_new_node = True
        self.caption = None
        self.object = None
        self.reason = None
        self.center = None
        self.room_node = None
        self.exploration_level = 0
        self.distance = 2
        self.score = 0.5
        self.edges = set()

    def __lt__(self, other):
        return self.score < other.score

    def add_edge(self, edge):
        self.edges.add(edge)

    def remove_edge(self, edge):
        self.edges.discard(edge)
    
    def set_caption(self, new_caption):
        for edge in list(self.edges):
            edge.delete()
        self.is_new_node = True
        self.caption = new_caption
        self.reason = None
        self.distance = 2
        self.score = 0.5
        self.exploration_level = 0
        self.edges.clear()
    
    def set_object(self, object):
        self.object = object
        self.object['node'] = self
    
    def set_center(self, center):
        self.center = center


class Edge():
    def __init__(self, node1, node2):
        self.node1 = node1
        self.node2 = node2
        node1.add_edge(self)
        node2.add_edge(self)
        self.relation = None

    def set_relation(self, relation):
        self.relation = relation

    def delete(self):
        self.node1.remove_edge(self)
        self.node2.remove_edge(self)

    def text(self):
        text = '({}, {}, {})'.format(self.node1.caption, self.node2.caption, self.relation)
        return text


class SubGraph():
    def __init__(self, center_node):
        self.center_node = center_node
        self.edges = self.center_node.edges
        self.center = self.center_node.center
        self.nodes = set()
        for edge in self.edges:
            self.nodes.add(edge.node1)
            self.nodes.add(edge.node2)

    def get_subgraph_2_text(self):
        text = ''
        edges = set()
        for node in self.nodes:
            text = text + node.caption + '/'
            edges.update(node.edges)
        text = text[:-1] + '\n'
        for edge in edges:
            text = text + edge.relation + '/'
        text = text[:-1]
        return text


class Graph():
    def __init__(self, args, is_navigation=True) -> None:
        self.args = args
        self.blacklist = set()
        self.map_resolution = args.map_resolution
        self.map_size_cm = args.map_size_cm
        self.map_size = args.map_size
        self.camera_matrix = get_camera_matrix(args.env_frame_height, args.env_frame_width, args.hfov)
        full_width, full_height = self.map_size, self.map_size
        self.full_width = full_width
        self.full_height = full_height
        self.visited = torch.zeros(full_width, full_height).float().cpu().numpy()
        self.device = args.device
        self.classes = ['item']
        self.BG_CLASSES = ["sky", "ground", "road", "grass"]
        self.rooms = ['parking lot', 'street', 'garden', 'plaza', 'building entrance', 'park']
        self.objects = MapObjectList(device=self.device)
        self.objects_post = MapObjectList(device=self.device)
        self.nodes = []
        self.edge_list = []
        self.group_nodes = []
        self.init_room_nodes()
        self.is_navigation = is_navigation
        self.set_cfg()
        
        self.groundingdino_config_file = 'third_party/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'
        self.groundingdino_checkpoint = 'data/models/groundingdino_swint_ogc.pth'
        self.sam_version = 'vit_b'
        self.sam_checkpoint = 'data/models/sam_vit_b_01ec64.pth'
        self.segment2d_results = []
        self.max_detections_per_object = 10
        self.threshold_list = {'bathtub': 3, 'bed': 3, 'cabinet': 2, 'chair': 1, 'chest_of_drawers': 3, 'clothes': 2, 'counter': 1, 'cushion': 3, 'fireplace': 3, 'gym_equipment': 2, 'picture': 3, 'plant': 3, 'seating': 0, 'shower': 2, 'sink': 2, 'sofa': 2, 'stool': 2, 'table': 1, 'toilet': 3, 'towel': 2, 'tv_monitor': 0}
        self.found_goal_times_threshold = 1
        self.N_max = 10
        self.node_space = 'building. tree. vehicle. person. sign. bench. lamp'
        self.relations = ["next to", "near", "far from", "north of", "south of", "east of", "west of"]
        self.prompt_edge_proposal = '''
Provide the most possible single spatial relationship for each of the following object pairs. Answer with only one relationship per pair, and separate each answer with a newline character.
Examples:
Input:
Object pair(s):
(cabinet, chair)
Output:
next to
Input:
Object pair(s):
(table, lamp)
(bed, nightstand)
Output:
on
next to
Object pair(s):
        '''
        self.prompt_room_predict = 'Which room is the most likely to have the [{}] in: [{}]. Only answer the room.'
        self.prompt_graph_corr_0 = 'What is the probability of A and B appearing together. [A:{}], [B:{}]. Even if you do not have enough information, you have to answer with a value from 0 to 1 anyway. Answer only the value of probability and do not answer any other text.'
        self.prompt_graph_corr_1 = 'What else do you need to know to determine the probability of A and B appearing together? [A:{}], [B:{}]. Please output a short question (output only one sentence with no additional text).'
        self.prompt_graph_corr_2 = 'Here is the objects and relationships near A: [{}] You answer the following question with a short sentence based on this information. Question: {}'
        self.prompt_graph_corr_3 = 'The probability of A and B appearing together is about {}. Based on the dialog: [{}], re-determine the probability of A and B appearing together. A:[{}], B:[{}]. Even if you do not have enough information, you have to answer with a value from 0 to 1 anyway. Answer only the value of probability and do not answer any other text.'
        self.prompt_image2text = 'Describe the object at the center of the image and indicate the spatial relationship between other objects and it.'
        self.prompt_create_relation = """
Given the image, please analyze the spatial relationship between {obj1} and {obj2}.
If there is a clear spatial relationship, describe it using the following template:
"{obj1} and {obj2}: {obj1} is <relation type> {obj2}"
If no clear spatial relationship exists, state: "No clear spatial relationship between {obj1} and {obj2}"

Example output format for a relation:
"table and book: table is under book"

Example output format if no relation:
"No clear spatial relationship between table and book"

Please provide the relationship you can determine from the image.
        """
        self.grounded_sam = self.get_grounded_sam(self.device)
        # 配置文件里的 api_key 通常是占位值，真实密钥从环境变量 OPENAI_API_KEY 读取
        api_key = resolve_api_key(getattr(self.args, "api_key", None))
        self.llm = LLM(self.args.base_url, api_key, self.args.llm_model)
        self.vlm = VLM(self.args.base_url, api_key, self.args.vlm_model)
        self.graphbuilder = GraphBuilder(self.llm)
        self.goalgraphdecomposer = GoalGraphDecomposer(self.llm)
        self.extractor = DISK(max_num_keypoints=2048).eval().to(self.device)
        self.image_matcher = LightGlue(features='disk').eval().to(self.device)
        self.navigate_steps = 0  # 添加这行
    

    # 在 graph.py 的 Graph 类中添加以下方法

    def verify_goal(self, text_goal, env, yolo):
        """
        当检测到疑似目标时，进行多帧置信度验证。
        结合YOLO检测分数 + LLM确认。
        """
        confidence_history = []

        for i in range(5):  # 连续5帧
            img = env.get_image()  # 获取当前图像
            detections = yolo.predict(img)
            
            detection_conf = 0
            for det in detections:
                if text_goal.lower().split()[-1] in det['label'].lower():
                    detection_conf = det['confidence']
                    break
            
            llm_conf = 0
            if detection_conf > 0.4:
                ans = self.llm(f"图中是{text_goal}吗？", img)
                if "yes" in ans.lower():
                    llm_conf = 1
            
            total_conf = detection_conf + llm_conf
            confidence_history.append(total_conf)

            if len(confidence_history) > 5:
                confidence_history.pop(0)

            # 连续下降 → 误检
            if len(confidence_history) >= 3 and all(
                confidence_history[i] > confidence_history[i + 1]
                for i in range(len(confidence_history) - 1)
            ):
                print("[Verification] 连续置信度下降，可能误检。重新搜索。")
                return False

        avg_conf = sum(confidence_history) / len(confidence_history)
        print(f"[Verification] 平均置信度: {avg_conf:.2f}")
        return avg_conf >= 1.5


    def try_match_object(self, detected_obj, target):
        """
        检查检测到的物体是否可能是目标。
        如果已在黑名单，则直接跳过。
        """
        if detected_obj in self.blacklist:
            return False  # 跳过已失败的匹配

        # 简单匹配逻辑（可以扩展成更复杂的语义匹配）
        if target.lower() in detected_obj.lower():
            return True
        else:
            # 加入黑名单
            self.blacklist.add(detected_obj)
            return False


    def decompose_outdoor_goal(self, text_goal):
        """
        户外目标分解 —— 使用LLM将复杂指令拆成多个子目标
        """
        prompt = f"""
        你是一架无人机的导航助手。
        请将下面的户外目标语句拆分为2~3个地标搜索步骤（landmark），按顺序输出。

        示例：
        输入："Find red building near parking lot"
        输出：["parking lot", "red building"]

        现在输入目标："{text_goal}"
        请直接输出JSON格式的列表。
        """
        try:
            subgoals = self.llm(prompt)
            return eval(subgoals)  # 解析成 Python list
        except Exception as e:
            print(f"[Decomposition Error] {e}")
            return [text_goal]  # 若失败则不分解


    def llm_evaluate_frontiers(self, frontier_locations, text_goal, current_pose):
        """
        使用LLM评估frontier的语义相关性
        
        Args:
            frontier_locations: (N, 2) frontier坐标
            text_goal: 文本目标描述
            current_pose: 当前位置 [x, y, theta]
        
        Returns:
            scores: (N,) 每个frontier的LLM评分 (0-1)
        """
        print("[DEBUG llm_evaluate_frontiers]")

        # --- 修正 current_pose 维度问题 ---
        if isinstance(current_pose, np.ndarray) and current_pose.ndim > 1:
            current_pose = current_pose.squeeze(0)
        elif torch.is_tensor(current_pose) and len(current_pose.shape) > 1:
            current_pose = current_pose.squeeze(0).cpu().numpy()
        # ----------------------------------
        
        print("  frontier_locations:", frontier_locations.shape)
        print("  first frontier:", frontier_locations[0] if len(frontier_locations) > 0 else "None")
        print("  full_pose shape:", self.full_pose.shape)
        print("  current_pose value:", current_pose)
        print("  current_pose shape:", np.shape(current_pose))

        
        # 1. 将frontier转换为方向描述
        directions = self._frontiers_to_directions(frontier_locations, current_pose)
        
        # 2. 获取每个方向的场景信息
        direction_contexts = self._get_direction_contexts(directions)
        
        # 3. 使用LLM评分
        scores = []
        for i, (direction, context) in enumerate(zip(directions, direction_contexts)):
            score = self._llm_score_single_frontier(direction, context, text_goal)
            scores.append(score)
        
        # 归一化到0-1
        scores = np.array(scores)
        if scores.max() > 0:
            scores = scores / scores.max()
        
        return scores


    def _frontiers_to_directions(self, frontier_locations, current_pose):
        """将frontier坐标转换为方向描述（北/南/东/西）"""
        directions = []
        current_x, current_y = current_pose[0], current_pose[1]
        
        for loc in frontier_locations:
            dx = loc[1] - current_x  # 注意坐标系
            dy = loc[0] - current_y
            
            # 计算角度
            angle = np.arctan2(dy, dx) * 180 / np.pi
            
            # 转换为方向描述
            if -45 <= angle < 45:
                direction = "right"
            elif 45 <= angle < 135:
                direction = "front"
            elif -135 <= angle < -45:
                direction = "back"
            else:
                direction = "left"
            
            directions.append({
                'label': direction,
                'angle': angle,
                'distance': np.sqrt(dx**2 + dy**2)
            })
        
        return directions


    def _get_direction_contexts(self, directions):
        """获取每个方向的场景上下文（已看到的物体）"""
        contexts = []
        
        for direction in directions:
            # 找该方向已探索过的物体
            objects_in_direction = []
            for node in self.nodes:
                if node.center is None:
                    continue
                
                # 计算物体相对当前位置的角度
                node_angle = self._compute_angle_to_node(node)
                
                # 判断是否在该方向的角度范围内（±30度）
                if abs(node_angle - direction['angle']) < 30:
                    objects_in_direction.append(node.caption)
            
            # 构建上下文描述
            if objects_in_direction:
                context = f"In {direction['label']} direction, I see: {', '.join(set(objects_in_direction))}"
            else:
                context = f"In {direction['label']} direction, unexplored area"
            
            contexts.append(context)
        
        return contexts


    def _compute_angle_to_node(self, node):
        """计算当前位置到节点的角度"""
        if not hasattr(self, 'full_pose') or node.center is None:
            return 0
        
        # ---- 修复: 确保full_pose是一维 [x, y, θ] ----
        if torch.is_tensor(self.full_pose):
            pose = self.full_pose.squeeze().cpu().numpy()
        elif isinstance(self.full_pose, np.ndarray):
            pose = self.full_pose.squeeze()
        else:
            pose = np.array(self.full_pose).squeeze()

        if pose.ndim != 1 or len(pose) < 2:
            print("[WARN] Invalid full_pose:", pose)
            return 0
        
        current_x, current_y = self.full_pose[0], self.full_pose[1]
        dx = node.center[1] - current_x
        dy = node.center[0] - current_y
        
        return np.arctan2(dy, dx) * 180 / np.pi


    def _llm_score_single_frontier(self, direction, context, text_goal):
        """
        使用LLM为单个frontier评分
        
        返回: 0-10的分数
        """
        
        # 构建prompt
        prompt = f"""You are a navigation assistant helping a robot find objects in an indoor environment.

    Current situation:
    - Goal: Find "{text_goal}"
    - Direction being evaluated: {direction['label']} direction
    - What the robot has observed: {context}

    Question: How likely is it that "{text_goal}" is in the {direction['label']} direction?

    Consider:
    1. Common sense about where objects are typically located
    2. Spatial relationships between objects
    3. What has already been observed in that direction

    Rate from 0 to 10:
    - 0: Very unlikely to find the goal there
    - 5: Uncertain
    - 10: Very likely to find the goal there

    Answer with ONLY a number from 0-10, no explanation."""

        try:
            # 调用LLM
            response = self.llm(prompt)
            
            # 解析分数
            score = self._parse_llm_score(response)
            
            return score
        
        except Exception as e:
            print(f"LLM scoring error: {e}")
            return 5.0  # 默认中等分数


    def _parse_llm_score(self, response):
        """从LLM响应中提取分数"""
        import re
        
        # 尝试提取数字
        numbers = re.findall(r'\d+\.?\d*', response)
        
        if numbers:
            score = float(numbers[0])
            # 确保在0-10范围内
            score = max(0, min(10, score))
            return score
        else:
            return 5.0  # 默认分数

    def set_cfg(self):
        cfg = {'dataset_config': PosixPath('tools/replica.yaml'), 'scene_id': 'room0', 'start': 0, 'end': -1, 'stride': 5, 'image_height': 680, 'image_width': 1200, 'gsa_variant': 'none', 'detection_folder_name': 'gsa_detections_${gsa_variant}', 'det_vis_folder_name': 'gsa_vis_${gsa_variant}', 'color_file_name': 'gsa_classes_${gsa_variant}', 'device': 'cuda', 'use_iou': True, 'spatial_sim_type': 'overlap', 'phys_bias': 0.0, 'match_method': 'sim_sum', 'semantic_threshold': 0.5, 'physical_threshold': 0.5, 'sim_threshold': 1.2, 'use_contain_number': False, 'contain_area_thresh': 0.95, 'contain_mismatch_penalty': 0.5, 'mask_area_threshold': 25, 'mask_conf_threshold': 0.95,
               'max_bbox_area_ratio': 0.5, 'skip_bg': True, 'min_points_threshold': 16, 'downsample_voxel_size': 0.025, 'dbscan_remove_noise': True, 'dbscan_eps': 0.1, 'dbscan_min_points': 10, 'obj_min_points': 0, 'obj_min_detections': 3, 'merge_overlap_thresh': 0.7, 'merge_visual_sim_thresh': 0.8, 'merge_text_sim_thresh': 0.8, 'denoise_interval': 20, 'filter_interval': -1, 'merge_interval': 20, 'save_pcd': True, 'save_suffix': 'overlap_maskconf0.95_simsum1.2_dbscan.1_merge20_masksub', 'vis_render': False, 'debug_render': False, 'class_agnostic': True, 'save_objects_all_frames': True, 'render_camera_path': 'replica_room0.json', 'max_num_points': 512}
        cfg = DictConfig(cfg)
        if self.is_navigation:
            cfg.sim_threshold = 0.8
            cfg.sim_threshold_spatial = 0.01
        self.cfg = cfg

    def set_agent(self, agent):
        self.agent = agent

    def set_navigate_steps(self, navigate_steps):
        self.navigate_steps = navigate_steps

    def set_room_map(self, room_map):
        self.room_map = room_map

    def set_fbe_free_map(self, fbe_free_map):
        self.fbe_free_map = fbe_free_map
    
    def set_observations(self, observations):
        self.observations = observations
        self.image_rgb = observations['rgb'].copy()
        self.image_depth = observations['depth'].copy()
        self.pose_matrix = self.get_pose_matrix()

    def set_obj_goal(self, obj_goal):
        self.obj_goal = obj_goal

    def set_image_goal(self, image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        self.instance_imagegoal = image
        text_goal = self.vlm(self.prompt_image2text, self.instance_imagegoal)
        self.set_text_goal(text_goal)

    def set_text_goal(self, text_goal):
        if isinstance(text_goal, dict) and 'intrinsic_attributes' in text_goal and 'extrinsic_attributes' in text_goal:
            text_goal = text_goal['intrinsic_attributes'] + ' ' + text_goal['extrinsic_attributes']
        self.text_goal = text_goal
        self.node_space = text_goal
        self.goalgraph = self.graphbuilder.build_graph_from_text(text_goal)
        self.goalgraph_decomposed = self.goalgraphdecomposer.goal_decomposition(self.goalgraph)

    def set_frontier_map(self, frontier_map):
        self.frontier_map = frontier_map

    def set_full_map(self, full_map):
        self.full_map = full_map

    def set_full_pose(self, full_pose):
        self.full_pose = full_pose

    def get_scenegraph(self):
        nodes = self.nodes
        edges = self.get_edges()
        caption_count = {}
        
        new_nodes = []
        
        node_id_map = {}
        for node in nodes:
            caption = node.caption
            if caption not in caption_count:
                caption_count[caption] = 0
            unique_id = "{}_{}".format(caption, caption_count[caption])
            new_nodes.append({
                'id': unique_id,
                'position': node.center
            })
            node_id_map[node] = unique_id
            caption_count[caption] += 1

        new_edges = []
        for edge in edges:
            source_id = node_id_map[edge.node1]
            target_id = node_id_map[edge.node2]
            new_edges.append({
                'source': source_id,
                'target': target_id,
                'type': edge.relation
            })

        self.scenegraph = {
            'nodes': new_nodes,
            'edges': new_edges
        }

        return self.scenegraph

    def get_nodes(self):
        return self.nodes
    
    def get_edges(self):
        edges = set()
        for node in self.nodes:
            edges.update(node.edges)
        edges = list(edges)
        return edges

    def get_seg_xyxy(self):
        return self.seg_xyxy

    def get_seg_caption(self):
        return self.seg_caption

    def init_room_nodes(self):
        room_nodes = []
        for caption in self.rooms:
            room_node = RoomNode(caption)
            room_nodes.append(room_node)
        self.room_nodes = room_nodes

    def get_grounded_sam(self, device):
        model = load_model(self.groundingdino_config_file, self.groundingdino_checkpoint, device=device)
        predictor = SamPredictor(sam_model_registry[self.sam_version](checkpoint=self.sam_checkpoint).to(device))
        return model, predictor
    
    def get_segmentation(
        self, model, image: np.ndarray
    ) -> tuple:
        groundingdino = model[0]
        sam_predictor = model[1]
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_resized, _ = transform(Image.fromarray(image), None)  # 3, h, w
        boxes_filt, caption = get_grounding_output(groundingdino, image_resized, caption=self.node_space, box_threshold=0.3, text_threshold=0.25, with_logits=False, device=self.device)
        if len(caption) == 0:
            return None, None, None, None
        sam_predictor.set_image(image)

        # size = image_pil.size
        H, W = image.shape[0], image.shape[1]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(self.device)

        mask, conf, _ = sam_predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes.to(self.device),
            multimask_output = False,
        )
        mask, xyxy, conf = mask.squeeze(1).cpu().numpy(), boxes_filt.squeeze(1).numpy(), conf.squeeze(1).cpu().numpy()
        return mask, xyxy, conf, caption

    def get_pose_matrix(self):
        x = self.map_size_cm / 100.0 / 2.0 + self.observations['gps'][0]
        y = self.map_size_cm / 100.0 / 2.0 - self.observations['gps'][1]
        t = (self.observations['compass'] - np.pi / 2)[0] # input degrees and meters
        pose_matrix = np.array([
            [np.cos(t), -np.sin(t), 0, x],
            [np.sin(t), np.cos(t), 0, y],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        return pose_matrix

    def segment2d(self):
        print('    segement2d...')
        with torch.no_grad():
            print('        sam_segmentation...')
            mask, xyxy, masks_conf, caption = self.get_segmentation(self.grounded_sam, self.image_rgb)

            self.seg_xyxy = xyxy
            self.seg_caption = caption
            self.clear_line()

        if caption is not None:
            print(f"✓ 检测到: {caption}")
        else:
            print("✗ 未检测到任何物体")
                    
        if caption is None:
            self.clear_line()
            return
        detections = sv.Detections(
            xyxy=xyxy,
            confidence=masks_conf,
            class_id=np.zeros_like(masks_conf).astype(int),
            mask=mask,
        )
        image_appear_efficiency = [''] * len(mask)
        self.segment2d_results.append({
            "xyxy": detections.xyxy,
            "confidence": detections.confidence,
            "class_id": detections.class_id,
            "mask": detections.mask,
            "classes": self.classes,
            "image_appear_efficiency": image_appear_efficiency,
            "image_rgb": self.image_rgb,
            "caption": caption,
        })
        self.clear_line()


    def mapping3d(self):
        print('    mapping3d...')
        depth_array = self.image_depth
        depth_array = depth_array[..., 0]

        gobs = self.segment2d_results[-1]
        
        unt_pose = self.pose_matrix
        
        adjusted_pose = unt_pose
        cam_K = self.camera_matrix
            
        idx = len(self.segment2d_results) - 1

        fg_detection_list, bg_detection_list = gobs_to_detection_list(
            cfg = self.cfg,
            image = self.image_rgb,
            depth_array = depth_array,
            cam_K = cam_K,
            idx = idx,
            gobs = gobs,
            trans_pose = adjusted_pose,
            class_names = self.classes,
            BG_CLASSES = self.BG_CLASSES,
            is_navigation = self.is_navigation
        )
        
        if len(fg_detection_list) == 0:
            self.clear_line()
            return
            
        if len(self.objects) == 0:
            # Add all detections to the map
            for i in range(len(fg_detection_list)):
                self.objects.append(fg_detection_list[i])

            # Skip the similarity computation 
            self.objects_post = filter_objects(self.cfg, self.objects)
            self.clear_line()
            return
                
        print('        compute_spatial_similarities...')
        spatial_sim = compute_spatial_similarities(self.cfg, fg_detection_list, self.objects)
        self.clear_line()
        spatial_sim[spatial_sim < self.cfg.sim_threshold_spatial] = float('-inf')
        
        self.objects = merge_detections_to_objects(self.cfg, fg_detection_list, self.objects, spatial_sim)
        
        self.objects_post = filter_objects(self.cfg, self.objects)
        self.clear_line()
            
    def get_caption(self):
        print('    get_caption...')
        for idx, object in enumerate(self.objects_post):
            caption_list = []
            for idx_det in range(len(object["image_idx"])):
                caption = self.segment2d_results[object["image_idx"][idx_det]]['caption'][object["mask_idx"][idx_det]]
                caption_list.append(caption)
            caption = self.find_modes(caption_list)[0]
            object['captions'] = [caption]
        self.clear_line()

    def update_node(self):
        print('    update_node...')
        # update nodes
        for i, node in enumerate(self.nodes):
            caption_ori = node.caption
            # caption_new = self.find_modes(self.objects_post[i]['captions'])[0]
            caption_new = node.object['captions'][0]
            if caption_ori != caption_new:
                node.set_caption(caption_new)
        # add new nodes
        new_objects = list(filter(lambda object: 'node' not in object, self.objects_post))
        # for i in range(node_num_ori, node_num_new):
        for new_object in new_objects:
            new_node = ObjectNode()
            # caption = self.find_modes(self.objects_post[i]['captions'])[0]
            caption = new_object['captions'][0]
            new_node.set_caption(caption)
            new_node.set_object(new_object)
            # self.create_new_edge(new_node)
            self.nodes.append(new_node)
        # get node.center and node.room
        for node in self.nodes:
            points = np.asarray(node.object['pcd'].points)
            center = points.mean(axis=0)
            x = int(center[0] * 100 / self.map_resolution)
            y = int(center[1] * 100 / self.map_resolution)
            y = self.map_size - 1 - y
            node.set_center([x, y])
            if 0 <= x < self.map_size and 0 <= y < self.map_size and hasattr(self, 'room_map'):
                if sum(self.room_map[0, :, y, x]!=0).item() == 0:
                    room_label = 0
                else:
                    room_label = torch.where(self.room_map[0, :, y, x]!=0)[0][0].item()
            else:
                room_label = 0
            if node.room_node is not self.room_nodes[room_label]:
                if node.room_node is not None:
                    node.room_node.nodes.discard(node)
                node.room_node = self.room_nodes[room_label]
                node.room_node.nodes.add(node)
        self.clear_line()

    def create_new_edge(self, new_node):
        # new_edges = []
        for j, old_node in enumerate(self.nodes):
            image = self.get_joint_image(old_node, new_node)
            if image is not None:
                response = self.vlm(self.prompt_create_relation.format(obj1=old_node.caption, obj2=new_node.caption), image)
                if "No clear spatial relationship" not in response:
                    response = response.lower()
                    objects = [old_node.caption.lower(), new_node.caption.lower()]
                    relations = self.graphbuilder.get_relations(response, objects)
                    new_edge = Edge(old_node, new_node)
                    if len(relations) > 0:
                        new_edge.set_relation(relations[0]['type'])
                    else:
                        new_edge.set_relation('')
                    # new_node.edges.add(new_edge)
                    # old_node.edges.add(new_edge)
                    # new_edges.append(new_edge)

    def update_edge(self):
        print('    update_edge...')
        old_nodes = []
        new_nodes = []
        for i, node in enumerate(self.nodes):
            if node.is_new_node:
                new_nodes.append(node)
                node.is_new_node = False
            else:
                old_nodes.append(node)
        if len(new_nodes) == 0:
            self.clear_line()
            return
        # create the edge between new_node and old_node
        new_edges = []
        for i, new_node in enumerate(new_nodes):
            for j, old_node in enumerate(old_nodes):
                new_edge = Edge(new_node, old_node)
                # new_node.edges.add(new_edge)
                # old_node.edges.add(new_edge)
                new_edges.append(new_edge)
        # create the edge between new_node
        for i, new_node1 in enumerate(new_nodes):
            for j, new_node2 in enumerate(new_nodes[i + 1:]):
                new_edge = Edge(new_node1, new_node2)
                # new_node1.edges.add(new_edge)
                # new_node2.edges.add(new_edge)
                new_edges.append(new_edge)
        # get all new_edges
        new_edges = set()
        for i, node in enumerate(self.nodes):
            node_new_edges = set(filter(lambda edge: edge.relation is None, node.edges))
            new_edges = new_edges | node_new_edges
        new_edges = list(new_edges)
        # get all relation proposals
        if len(new_edges) > 0:
            print(f'        LLM get all relation proposals...')
            node_pairs = []
            for new_edge in new_edges:
                node_pairs.append(new_edge.node1.caption)
                node_pairs.append(new_edge.node2.caption)
            prompt = self.prompt_edge_proposal + '\n({}, {})' * len(new_edges)
            prompt = prompt.format(*node_pairs)
            relations = self.llm(prompt=prompt)
            
            # 防御性检查 —— 避免 LLM 返回 None 或 ""
            if not relations or not isinstance(relations, str):
                print("[update_edge] Warning: empty or invalid relations from LLM, skip this update.")
                relations = []
            else:
                relations = relations.split('\n')
            
            if len(relations) == len(new_edges):
                for i, relation in enumerate(relations):
                    new_edges[i].set_relation(relation)
            self.clear_line()
            # discriminate all relation proposals
            for i, new_edge in enumerate(new_edges):
                print(f'        discriminate_relation  {i}/{len(new_edges)}...')
                if new_edge.relation == None:
                    new_edge.delete()
                self.clear_line()
            # get edges set
            # self.edges = set()
            # for node in self.nodes:
            #     self.edges.update(node.edges)
        self.clear_line()

    def update_group(self):
        for room_node in self.room_nodes:
            if len(room_node.nodes) > 0:
                room_node.group_nodes = []
                object_nodes = list(room_node.nodes)
                centers = [object_node.center for object_node in object_nodes]
                centers = np.array(centers)
                dbscan = DBSCAN(eps=10, min_samples=1)  
                clusters = dbscan.fit_predict(centers)  
                for i in range(clusters.max() + 1):
                    group_node = GroupNode()
                    indices = np.where(clusters == i)[0]
                    for index in indices:
                        group_node.nodes.append(object_nodes[index])
                    group_node.get_graph()
                    room_node.group_nodes.append(group_node)
    
    def insert_goal(self, goal=None):
        if goal is None:
            goal = self.obj_goal
        self.update_group()
        room_node_text = ''
        for room_node in self.room_nodes:
            if len(room_node.group_nodes) > 0:
                room_node_text = room_node_text + room_node.caption + ','
        # room_node_text[-2] = '.'
        if room_node_text == '':
            return None
        prompt = self.prompt_room_predict.format(goal, room_node_text)
        response = self.llm(prompt=prompt)
        response = response.lower()
        predict_room_node = None
        for room_node in self.room_nodes:
            if len(room_node.group_nodes) > 0 and room_node.caption.lower() in response:
                predict_room_node = room_node
        if predict_room_node is None:
            return None
        for group_node in predict_room_node.group_nodes:
            corr_score = self.graph_corr(goal, group_node)
            group_node.corr_score = corr_score
        sorted_group_nodes = sorted(predict_room_node.group_nodes)
        self.mid_term_goal = sorted_group_nodes[-1].center
        return self.mid_term_goal
    
    def update_scenegraph(self):
        print(f'update_observation {self.navigate_steps}...')
        self.segment2d()
        if len(self.segment2d_results) == 0:
            self.clear_line()
            return
        self.mapping3d()
        self.get_caption()
        self.update_node()
        self.update_edge()
        self.get_scenegraph()
        self.clear_line()

    def explore(self):
        overlap = self.overlap()
        if 0.5 <= overlap < 0.9 and len(self.matcher.common_nodes) >= 2:
            goal = self.explore_remaining()
        elif overlap >= 0.9 and len(self.matcher.common_nodes) < 2:
            goal = self.reasonableness_correction()
        else:
            goal = self.explore_subgraph()
        
        goal = self.get_goal(goal)
        
        return goal

    def explore_subgraph(self, goal=None):
        if goal == None:
            self.subgraph = self.goalgraph_decomposed['subgraph_1']
        self.subgraph = self.goalgraphdecomposer.graph_to_text(self.subgraph)
        return self.insert_goal(self.subgraph)
    
    def overlap(self):
        graph1 = self.scenegraph
        graph2 = self.goalgraph
        self.matcher = GraphMatcher(graph1, graph2, self.llm)
        overlap_score = self.matcher.overlap()
        return overlap_score
    
    def explore_remaining(self):
        G1 = self.matcher.G1
        G2 = self.matcher.G2
        common_nodes = self.matcher.common_nodes

        # Assign positions to the first two common nodes in the subgraph
        for i, node_id in enumerate(common_nodes):
            if i < 2:
                G2.nodes[node_id]['position'] = G1.nodes[node_id]['position']
            else:
                break

        # Calculate relative positions within the subgraph
        positions = self.matcher.calculate_relative_positions(G2, common_nodes)

        # Predict positions of the remaining nodes
        position = self.matcher.predict_remaining_node_positions(common_nodes, positions, G1)
        return position

    def reasonableness_correction(self):
        corrector = SceneGraphCorrector(self.llm)
        self.scenegraph = corrector.correct_scene_graph(self.scenegraph, self.obj_goal)
        return None

    def clear_line(self, line_num=1):
        for i in range(line_num):  
            sys.stdout.write('\033[F')
            sys.stdout.write('\033[J')
            sys.stdout.flush()  
    
    def find_modes(self, lst):  
        if len(lst) == 0:
            return ['object']
        else:
            counts = Counter(lst)  
            max_count = max(counts.values())  
            modes = [item for item, count in counts.items() if count == max_count]  
            return modes  
        
    def get_joint_image(self, node1, node2):
        image_idx1 = node1.object["image_idx"]
        image_idx2 = node2.object["image_idx"]
        image_idx = set(image_idx1) & set(image_idx2)
        if len(image_idx) == 0:
            return None
        conf_max = -np.inf
        # get joint images of the two nodes
        for idx in image_idx:
            conf1 = node1.object["conf"][image_idx1.index(idx)]
            conf2 = node2.object["conf"][image_idx2.index(idx)]
            conf = conf1 + conf2
            if conf > conf_max:
                conf_max = conf
                idx_max = idx
        image = self.segment2d_results[idx_max]["image_rgb"]
        image = Image.fromarray(image)
        return image

    def get_goal(self, goal=None):
        fbe_map = torch.zeros_like(self.full_map[0,0])
        if self.full_map.shape[1] == 1:
            fbe_map[self.fbe_free_map[0,0]>0] = 1
        else:
            fbe_map[self.full_map[0,1]>0] = 1
         # ✅ 临时修复：如果explored为空，使用obstacle的反面作为free space
        # if fbe_map.sum() == 0:
        #     print("[DEBUG] explored通道为空，使用obstacle反面")
        #     obstacle_map = self.full_map[0,0]
        #     fbe_map[obstacle_map == 0] = 1  # 非障碍=自由空间
            
        #     # 标记当前位置周围为已探索
        #     if hasattr(self, 'full_pose'):
        #         pose = self.full_pose.cpu().numpy().squeeze()
        #         x = int(pose[0] * 100 / self.map_resolution)
        #         y = int(pose[1] * 100 / self.map_resolution)
        #         y = self.map_size - 1 - y
                
        #         # 标记50x50的区域为已探索
        #         x1, x2 = max(0, x-25), min(self.map_size, x+25)
        #         y1, y2 = max(0, y-25), min(self.map_size, y+25)
        #         fbe_map[y1:y2, x1:x2] = 1

        fbe_map[skimage.morphology.binary_dilation(self.full_map[0,0].cpu().numpy(), skimage.morphology.disk(4))] = 3 # then dialte obstacle

        fbe_cp = copy.deepcopy(fbe_map)
        fbe_cpp = copy.deepcopy(fbe_map)
        fbe_cp[fbe_cp==0] = 4 # don't know space is 4
        fbe_cp[fbe_cp<4] = 0 # free and obstacle
        selem = skimage.morphology.disk(1)
        fbe_cpp[skimage.morphology.binary_dilation(fbe_cp.cpu().numpy(), selem)] = 0 # don't know space is 0 dialate unknown space
        
        diff = fbe_map - fbe_cpp # intersection between unknown area and free area 
        frontier_map = diff == 1
        frontier_map = remove_small_frontiers(frontier_map, min_size=5)
        frontier_locations = torch.stack([torch.where(frontier_map)[0], torch.where(frontier_map)[1]]).T
        num_frontiers = len(torch.where(frontier_map)[0])
        print(f"[DEBUG] 探索区域大小: {(fbe_map==1).sum().item()}")
        print(f"[DEBUG] Frontier数量: {num_frontiers}")

        if num_frontiers == 0:
            print("[DEBUG] 未找到frontier，地图可能太小")
            return None
        
        input_pose = np.zeros(7)
        input_pose[:3] = self.full_pose.cpu().numpy()
        input_pose[1] = self.map_size_cm/100 - input_pose[1]
        input_pose[2] = -input_pose[2]
        input_pose[4] = self.full_map.shape[-2]
        input_pose[6] = self.full_map.shape[-1]
        traversible, start = self.get_traversible(self.full_map.cpu().numpy()[0, 0, ::-1], input_pose)
        planner = FMMPlanner(traversible)
        state = [start[0] + 1, start[1] + 1]

        # ✅ 添加边界检查
        state[0] = max(1, min(state[0], traversible.shape[0] - 2))
        state[1] = max(1, min(state[1], traversible.shape[1] - 2))

        planner.set_goal(state)
        fmm_dist = planner.fmm_dist[::-1]
        frontier_locations += 1
        frontier_locations = frontier_locations.cpu().numpy()
        distances = fmm_dist[frontier_locations[:,0],frontier_locations[:,1]] / 20
        
        distance_threshold = 1.2
        idx_16 = np.where(distances>=distance_threshold)
        distances_16 = distances[idx_16]
        distances_16_inverse = 10 - (np.clip(distances_16, 0, 10 + distance_threshold) - distance_threshold)
        frontier_locations_16 = frontier_locations[idx_16]
        self.frontier_locations = frontier_locations
        self.frontier_locations_16 = frontier_locations_16
        if len(distances_16) == 0:
            return None
        num_16_frontiers = len(idx_16[0])  # 175
        scores = np.zeros((num_16_frontiers))
        
        scores += distances_16_inverse
        if isinstance(goal, list) or isinstance(goal, np.ndarray):
            goal = list(goal)

            planner = FMMPlanner(traversible)
            state = [goal[0] + 1, goal[1] + 1]
            # ✅ 添加边界检查
            state[0] = max(1, min(state[0], traversible.shape[0] - 2))
            state[1] = max(1, min(state[1], traversible.shape[1] - 2))
            planner.set_goal(state)
            fmm_dist = planner.fmm_dist[::-1]
            distances = fmm_dist[frontier_locations[:,0],frontier_locations[:,1]] / 20
            
            distances_16 = distances[idx_16]
            distances_16_inverse = 1 - (np.clip(distances_16, 0, 10 + distance_threshold) - distance_threshold) / 10
            if len(distances_16) == 0:
                return None
            scores += distances_16_inverse

        # ==================== 新增部分：LLM评分 ====================
        # 在选择最优frontier之前添加LLM语义评分
        if hasattr(self, 'text_goal') and self.text_goal and len(self.nodes) > 0:
            print("    [LLM] Evaluating frontiers...")
            try:
                # llm_scores = self.llm_evaluate_frontiers(
                #     frontier_locations_16 - 1,  # 转回原始坐标
                #     self.text_goal,
                #     self.full_pose.cpu().numpy().squeeze(0)  # ✅ 降维
                # )
                pose_data = self.full_pose
                if isinstance(pose_data, torch.Tensor):
                    pose_data = pose_data.squeeze(0).cpu().numpy()  # ← 这里保证是 (3,)
                elif isinstance(pose_data, np.ndarray):
                    pose_data = pose_data.squeeze()

                llm_scores = self.llm_evaluate_frontiers(
                    frontier_locations_16 - 1,
                    self.text_goal,
                    pose_data
                )
                
                # 融合LLM分数
                llm_weight = 0.5  # LLM分数的权重（可调整）
                scores += llm_weight * llm_scores
                
                print(f"    [LLM] Scores - min:{llm_scores.min():.1f}, max:{llm_scores.max():.1f}, mean:{llm_scores.mean():.1f}")
            
            except Exception as e:
                print(f"    [LLM] Evaluation failed: {e}")
        # =======================================================
    

        idx_16_max = idx_16[0][np.argmax(scores)]
        goal = frontier_locations[idx_16_max] - 1
        self.scores = scores
        return goal


    def get_traversible(self, map_pred, pose_pred):
        if isinstance(map_pred, torch.Tensor):
            map_pred = map_pred.cpu().numpy()
        if len(map_pred.shape) == 4:
            map_pred = map_pred[0, 0]
        grid = np.rint(map_pred)

        # Get pose prediction and global policy planning window
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = pose_pred
        gx1, gx2, gy1, gy2  = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]

        # Get curr loc
        r, c = start_y, start_x
        start = [int(r*100/self.map_resolution - gy1),
                 int(c*100/self.map_resolution - gx1)]
        # start = [int(start_x), int(start_y)]
        start = pu.threshold_poses(start, grid.shape)
        self.visited[gy1:gy2, gx1:gx2][start[0]-2:start[0]+3,
                                       start[1]-2:start[1]+3] = 1
        #Get traversible
        def add_boundary(mat, value=1):
            h, w = mat.shape
            new_mat = np.zeros((h+2,w+2)) + value
            new_mat[1:h+1,1:w+1] = mat
            return new_mat
        
        def delete_boundary(mat):
            new_mat = copy.deepcopy(mat)
            return new_mat[1:-1,1:-1]
        
        [gx1, gx2, gy1, gy2] = planning_window

        x1, y1, = 0, 0
        x2, y2 = grid.shape

        selem = skimage.morphology.square(1)
        traversible = skimage.morphology.binary_dilation(
                    grid[y1:y2, x1:x2],
                    selem) != True

        if not(traversible[start[0], start[1]]):
            print("Not traversible, step is  ", self.navigate_steps)

        # obstacle dilation do not dilate collision
        traversible = 1 - traversible
        selem = skimage.morphology.disk(4)
        traversible = skimage.morphology.binary_dilation(
                        traversible, selem) != True
        
        traversible[int(start[0]-y1)-1:int(start[0]-y1)+2,
            int(start[1]-x1)-1:int(start[1]-x1)+2] = 1
        traversible = traversible * 1.
        
        traversible[self.visited[gy1:gy2, gx1:gx2][y1:y2, x1:x2] == 1] = 1
        traversible = add_boundary(traversible)
        return traversible, start
    
    def reset(self):
        full_width, full_height = self.map_size, self.map_size
        self.full_width = full_width
        self.full_height = full_height
        self.visited = torch.zeros(full_width, full_height).float().cpu().numpy()
        self.segment2d_results = []
        self.objects = MapObjectList(device=self.device)
        self.objects_post = MapObjectList(device=self.device)
        self.nodes = []
        self.group_nodes = []
        self.init_room_nodes()
        self.edge_list = []
        # 添加这两行
        self.scenegraph = {'nodes': [], 'edges': []}
        self.goalgraph = {'nodes': [], 'edges': []}

    def graph_corr(self, goal, graph):
        prompt = self.prompt_graph_corr_0.format(graph.center_node.caption, goal)
        response_0 = self.llm(prompt=prompt)
        prompt = self.prompt_graph_corr_1.format(graph.center_node.caption, goal)
        response_1 = self.llm(prompt=prompt)
        prompt = self.prompt_graph_corr_2.format(graph.caption, response_1)
        response_2 = self.llm(prompt=prompt)
        prompt = self.prompt_graph_corr_3.format(response_0, response_1 + response_2, graph.center_node.caption, goal)
        response_3 = self.llm(prompt=prompt)
        corr_score = self.text2value(response_3)
        return corr_score
    
    def text2value(self, text):
        try:
            value = float(text)
        except:
            value = 0
        return value
    