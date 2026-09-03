# airsim_env.py
import warnings, logging, sys

logging.getLogger("tornado").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")

class _StderrFilter:
    def __init__(self, stream):
        self.stream = stream
    def write(self, msg):
        if "IOLoop is already running" not in msg:
            self.stream.write(msg)
    def flush(self):
        self.stream.flush()

sys.stderr = _StderrFilter(sys.stderr)


import airsim
import numpy as np
import cv2
import time
from PIL import Image
from src.navigation.dstar_lite import DStarLite

class AirSimEnv:
    """AirSim环境接口，模拟Habitat的API"""
    
    def __init__(self, args):
        self.args = args
        from src.navigation.semantic_explorer import SemanticExplorer

        self.map_size = 1000
        self.resolution = 0.5
        self.occupancy_grid = np.zeros((self.map_size, self.map_size), dtype=np.int8)  # 0 unknown, 1 free, -1 obstacle
        self.explorer = SemanticExplorer(map_size=self.map_size, resolution=self.resolution)

        # 连接AirSim 客户端
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()

        # 确保 API 控制开启（可重复调用是安全的）
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        # 飞行参数
        self.altitude = abs(getattr(self.args, "init_altitude", 5))  # 默认5m
        self.speed = getattr(self.args, "speed", 2)

        # 位置/姿态缓存
        self.last_pos = np.array([0.0, 0.0, -self.altitude])
        self.last_yaw = 0.0
        self.current_pos = np.array([0.0, 0.0, -self.altitude])
        # # === 新增：语义占据图 ===
        # from src.envs.airsim_env import SemanticOccupancyGrid
        # self.semantic_map = SemanticOccupancyGrid(map_size=self.map_size, resolution=self.resolution)


        # 不在 __init__ 里立即 takeoff —— 由 reset() 或外部调用控制起飞
        print("AirSimEnv initialized. API connected and enabled.")

        # 启动一个 keepalive 心跳线程，防止 AirSim 认为长时间无API调用
        import threading
        self._keepalive_stop = False
        def _keepalive():
            while not self._keepalive_stop:
                # 原本是 self.client.getMultirotorState()
                # 改成安全的空循环
                time.sleep(1.0)
        t = threading.Thread(target=_keepalive, daemon=True)
        t.start()
      


    
    def get_depth_map(self):
        # 返回 numpy depth HxW 单通道（float meters）
        responses = self.client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False)])
        r = responses[0]
        depth = airsim.list_to_2d_float_array(r.image_data_float, r.width, r.height)
        return depth
    
    def get_drone_position(self):
        """返回当前无人机位置 (airsim.Vector3r)"""
        state = self.client.getMultirotorState()
        return state.kinematics_estimated.position

    def update_occupancy_from_depth(self, depth_image, obstacle_dist=5.0):
        """
        用前视深度更新占据网格：
        - d < obstacle_dist   → 障碍(-1)
        - obstacle_dist~50m   → 可行(1)
        """
        import numpy as np
        if depth_image is None or depth_image.size == 0:
            return

        depth_image = np.clip(depth_image, 0.1, 50.0)
        H, W = depth_image.shape
        step = max(1, H // 160)   # ✅ 更密采样（约每3像素）

        # 当前位姿
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        yaw = self._get_yaw(state.kinematics_estimated.orientation)

        fx = fy = 320.0
        cx = W / 2.0
        cy = H / 2.0

        for v in range(0, H, step):
            for u in range(0, W, step):
                d = float(depth_image[v, u])
                if d <= 0.1 or d > 50.0:
                    continue

                # 像素 → 相机坐标
                x_cam = (u - cx) * d / fx
                y_cam = (v - cy) * d / fy
                z_cam = d

                # 相机 → 世界（仅yaw）
                xw = pos.x_val + z_cam * np.cos(yaw) - x_cam * np.sin(yaw)
                yw = pos.y_val + z_cam * np.sin(yaw) + x_cam * np.cos(yaw)

                mx, my = self.explorer.world_to_map(xw, yw)
                if 0 <= mx < self.map_size and 0 <= my < self.map_size:
                    if d < obstacle_dist:
                        self.occupancy_grid[mx, my] = -1  # 障碍
                    else:
                        if self.occupancy_grid[mx, my] == 0:
                            self.occupancy_grid[mx, my] = 1

        # 自身位置始终标为可行
        mx, my = self.explorer.world_to_map(pos.x_val, pos.y_val)
        if 0 <= mx < self.map_size and 0 <= my < self.map_size:
            self.occupancy_grid[mx, my] = 1

        # ✅ 调试信息
        free_count = np.sum(self.occupancy_grid == 1)
        obs_count = np.sum(self.occupancy_grid == -1)
        print(f"[调试] 更新占据图: free={free_count}, obstacle={obs_count}, 占障比例={obs_count/(free_count+1e-5):.2%}")


    def goto(self, world_x, world_y, altitude=None, vel=3.0):
        if altitude is None:
            altitude = -self.altitude  # or maintain configured z
        # moveToPositionAsync takes x,y,z in world coords (AirSim: NED maybe)
        self.client.moveToPositionAsync(world_x, world_y, altitude, vel).join()

    
    def get_rgb_image(self):
        """返回当前RGB图像（np.array格式，用于YOLO）"""
        responses = self.client.simGetImages([
            airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False),
            airsim.ImageRequest("front_center", airsim.ImageType.DepthPerspective, True, False)
        ])
        
        if responses and len(responses) > 0:
            response = responses[0]
            img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            img_rgb = img1d.reshape(response.height, response.width, 3)
            return img_rgb
        else:
            return None
        
    def takeoff(self):
        """起飞到固定高度"""
        print("Taking off...")
        self.client.takeoffAsync().join()
        self.client.moveToZAsync(-self.altitude, 2).join()
        print(f"Hovering at {self.altitude}m altitude")
        time.sleep(3)
    
    def reset(self):
        

        # 确保连接与控制被启用
        try:
            self.client.confirmConnection()
            self.client.enableApiControl(True)
            self.client.armDisarm(True)
        except Exception as e:
            print(f"[reset] API enable failed: {e}")

        # 起飞到设定高度（如果还未在空中）
        try:
            # 判断是否已经接近目标高度，否则重新起飞/爬升
            state = self.client.getMultirotorState()
            z = state.kinematics_estimated.position.z_val
            if abs(z + self.altitude) > 0.5:
                print("Taking off (from reset)...")
                self.client.takeoffAsync().join()
                self.client.moveToZAsync(-self.altitude, 3).join()
                time.sleep(1.0)
        except Exception as e:
            print(f"[reset] takeoff error: {e}")

        # 获取观测（重试几次）
        obs, rgbd, infos = None, None, None
        for _ in range(3):
            obs, rgbd, infos = self.get_observation()
            if rgbd is not None:
                break
            print("reset: retrying get_observation...")
            time.sleep(1.0)

        # 重置内部缓存值
        self.last_pos = np.array([0.0, 0.0, -self.altitude])
        self.last_yaw = 0.0
        self.current_pos = np.array([0.0, 0.0, -self.altitude])

        return obs, rgbd, infos

    def get_observation(self):
        """获取观测（模拟Habitat格式），对 simGetImages 结果做严格检查"""
        try:
            responses = self.client.simGetImages([
                airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False),
                airsim.ImageRequest("front_center", airsim.ImageType.DepthPerspective, True, False)
            ])
        except Exception as e:
            print(f"[get_observation] simGetImages failed: {e}")
            return None, None, None

        if responses is None or len(responses) < 2:
            print("[get_observation] Warning: simGetImages returned None or incomplete responses.")
            return None, None, None
        # 2. 获取位置
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        orientation = state.kinematics_estimated.orientation

        # 计算相对移动
        current_pos = np.array([pos.x_val, pos.y_val, pos.z_val])
        current_yaw = self._get_yaw(orientation)
        # ✅ 计算sensor_pose（相对移动）
        dx = current_pos[0] - self.last_pos[0]
        dy = current_pos[1] - self.last_pos[1]
        dtheta = current_yaw - self.last_yaw
        
        # 归一化角度到[-pi, pi]
        while dtheta > np.pi:
            dtheta -= 2*np.pi
        while dtheta < -np.pi:
            dtheta += 2*np.pi
        
        sensor_pose = np.array([dx, dy, dtheta])
        
        # 更新last值
        self.last_pos = current_pos.copy()
        self.last_yaw = current_yaw




        # RGB
        response_rgb = responses[0]
        if len(response_rgb.image_data_uint8) == 0:
            print("[get_observation] Warning: RGB data empty.")
            return None, None, None
        img1d = np.frombuffer(response_rgb.image_data_uint8, dtype=np.uint8)
        try:
            img_rgb = img1d.reshape(response_rgb.height, response_rgb.width, 3)
        except Exception as e:
            print(f"[get_observation] reshape RGB failed: {e}")
            return None, None, None

        # 深度
        response_depth = responses[1]
        depth_data = np.array(response_depth.image_data_float, dtype=np.float32)
        if depth_data.size == 0:
            print("[get_observation] Warning: depth data empty.")
            return None, None, None
        
        try:
            depth_img_raw = depth_data.reshape(response_depth.height, response_depth.width)
        except Exception as e:
            print(f"[get_observation] reshape depth failed: {e}")
            return None, None, None

        # ✅ 添加调试信息
        print(f"Depth response height: {response_depth.height}, width: {response_depth.width}")
        print(f"Depth data length: {len(depth_data)}")
        print(f"Depth min/max: {depth_img_raw.min()}/{depth_img_raw.max()}")

        # 归一化深度（用于BEV mapping）
        depth_img_normalized = np.clip(depth_img_raw, 0, self.args.max_depth)
        depth_img_normalized = (depth_img_normalized / self.args.max_depth * 255).astype(np.uint8)

        # ... 获取位置代码 ...

        # 转换为Habitat格式
        obs = {
            'rgb': img_rgb,
            'depth': depth_img_raw.reshape(response_depth.height, response_depth.width, 1),  # ✅ 原始深度（米）
            'gps': np.array([pos.x_val, pos.y_val]),
            'compass': np.array([current_yaw])
        }

        # 构建RGBD
        num_sem = self.args.num_sem_categories
        rgbd = np.zeros((self.args.frame_height, self.args.frame_width, 4 + num_sem), dtype=np.uint8)

        # RGB
        img_pil = Image.fromarray(img_rgb)
        img_resized = img_pil.resize((self.args.frame_width, self.args.frame_height))

        # Depth（用归一化的）
        depth_pil = Image.fromarray(depth_img_normalized)
        depth_resized = depth_pil.resize((self.args.frame_width, self.args.frame_height))

        rgbd[:, :, :3] = np.array(img_resized)
        rgbd[:, :, 3] = np.array(depth_resized)
        
        # 3. 语义通道

        # 转换为CHW格式
        rgbd = rgbd.transpose(2, 0, 1)
        rgbd = np.expand_dims(rgbd, axis=0)
        rgbd = rgbd.astype(np.float32) / 255.0
        
        # Infos
        infos = {
            'sensor_pose': sensor_pose,  # ✅ 使用真实的相对移动
            'agent_height': self.altitude,
            'episode_no': 0,
            'goal_name': self.args.goal if hasattr(self.args, 'goal') else 'unknown'
        }
        
        return obs, rgbd, infos
                
           
    
    def step(self, action):
        """执行动作"""
        if 'target_position' in action:
            target = action['target_position']
            print(f"[AirSim] 目标位置: {target}")
            self.move_to(target[0], target[1])

        # 获取观测
        obs, rgbd, infos = self.get_observation()
        if obs is None:
            print("[step] get_observation returned None")
            return None, None, False, {}

        # 从真实传感器计算 sensor_pose（已在 get_observation 更新 last_pos）
        # 不要覆盖 infos['sensor_pose'] 为一个常数
        # 但如需要调试，可打印位置
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        print(f"[AirSim] Current pos: ({pos.x_val:.2f}, {pos.y_val:.2f}, {pos.z_val:.2f})")

        # 检查深度
        print(f"Step获取观测完成，depth sum: {np.sum(rgbd[0, 3])}")

        done = False
        return obs, rgbd, done, infos
    
    def move_to(self, x, y, wait=True):
        """移动到指定位置（保持固定高度），带前置确认与超时处理"""
        try:
            # 再次确保 API 激活
            self.client.confirmConnection()
            self.client.enableApiControl(True)
        except Exception as e:
            print(f"[move_to] warning enableApiControl: {e}")

        print(f"Moving to ({x:.2f}, {y:.2f}, {-self.altitude:.2f})")
        try:
            fut = self.client.moveToPositionAsync(x, y, -abs(self.altitude), velocity=self.speed)
            if wait:
                fut.join(timeout_sec=10)  # join 增加超时参数（如果 SDK 版本支持）
        except TypeError:
            # 有些 airsim bindings 的 join 不支持 timeout_sec
            try:
                self.client.moveToPositionAsync(x, y, -abs(self.altitude), velocity=self.speed).join()
            except Exception as e:
                print(f"[move_to] move command failed: {e}")
        except Exception as e:
            print(f"[move_to] move command exception: {e}")

        # 更新内部位置缓存（粗略）
        self.current_pos = np.array([x, y, -self.altitude])
        time.sleep(0.5)

    
    def _get_yaw(self, quaternion):
        """从四元数提取yaw角"""
        import math
        q = quaternion
        siny_cosp = 2 * (q.w_val * q.z_val + q.x_val * q.y_val)
        cosy_cosp = 1 - 2 * (q.y_val * q.y_val + q.z_val * q.z_val)
        return math.atan2(siny_cosp, cosy_cosp)
    
    def close(self):
        """关闭环境"""
        print("Closing AirSimEnv...")
        try:
            self._keepalive_stop = True
        except:
            pass
        try:
            self.client.armDisarm(False)
            self.client.enableApiControl(False)
        except Exception as e:
            print(f"[close] error: {e}")

    def get_rgb_image(self):
        """返回当前RGB图像（np.array格式）"""
        response = self.client.simGetImage("0", airsim.ImageType.Scene)
        img = np.frombuffer(response, np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        return img

# class SemanticOccupancyGrid:
#     """基于 YOLO 检测的语义占据图"""

#     def __init__(self, map_size=1000, resolution=0.5):
#         self.map_size = map_size
#         self.resolution = resolution
#         # grid: (H,W,3): occupied prob / class id / confidence
#         self.grid = np.zeros((map_size, map_size, 3), dtype=np.float32)
#         # 类别颜色表
#         self.palette = {
#             "car": (0, 255, 0),
#             "person": (255, 0, 0),
#             "tree": (0, 128, 0),
#             "truck": (255, 128, 0),
#             "bus": (255, 255, 0),
#             "building": (128, 128, 128),
#             "road": (128, 64, 128)
#         }

#     def label_to_id(self, label: str) -> int:
#         label = label.lower()
#         mapping = {
#             "person": 1, "bicycle": 2, "car": 3, "motorcycle": 4,
#             "bus": 5, "truck": 6, "tree": 7, "building": 8, "road": 9
#         }
#         return mapping.get(label, 0)

#     def update_from_yolo(self, detections, depth, camera_pose, explorer):
#         """
#         用 YOLO 检测结果更新语义占据图
        
#         Args:
#             detections: [{'label':str, 'bbox':[x1,y1,x2,y2], 'confidence':float}]
#             depth: (H,W) float32 meters
#             camera_pose: (px, py, yaw)
#             explorer: 提供 world_to_map(x,y)->(ix,iy)
#         """
#         import numpy as np, math

#         px, py, yaw = camera_pose
#         H, W = depth.shape
        
#         # ✅ 相机内参（根据实际深度图尺寸）
#         fx = fy = max(W, H) * 0.5
#         cx, cy = W / 2.0, H / 2.0

#         painted = 0
#         skipped_boundary = 0
#         skipped_class = 0

#         for det in detections:
#             label = det.get("label", "").lower()
#             conf = float(det.get("confidence", det.get("score", 0.0)))
            
#             if conf < 0.25:
#                 continue

#             cls_id = self.label_to_id(label)
#             if cls_id == 0:
#                 skipped_class += 1
#                 continue

#             # ✅ 获取原始 bbox（在 RGB 图像上）
#             bbox_orig = det["bbox"]
            
#             # ✅ 缩放到深度图尺寸（重要！）
#             # 假设 RGB 是 640×480，depth 是 256×144
#             rgb_w, rgb_h = 640, 480  # 你可以从 det 中获取，或者作为参数传入
#             scale_x = W / rgb_w
#             scale_y = H / rgb_h
            
#             x1 = int(bbox_orig[0] * scale_x)
#             y1 = int(bbox_orig[1] * scale_y)
#             x2 = int(bbox_orig[2] * scale_x)
#             y2 = int(bbox_orig[3] * scale_y)
            
#             # 边界检查
#             x1, y1 = max(0, x1), max(0, y1)
#             x2, y2 = min(W - 1, x2), min(H - 1, y2)
            
#             if x2 <= x1 or y2 <= y1:
#                 continue

#             # 提取深度
#             region = depth[y1:y2 + 1, x1:x2 + 1]
#             valid = region[(region > 0.1) & (region < 80.0)]
            
#             if valid.size == 0:
#                 continue
            
#             d = float(np.median(valid))
            
#             # ✅ 限制深度范围（避免投影过远）
#             if d > 50.0:  # 超过50米的忽略
#                 continue

#             # bbox中心像素 → 世界坐标
#             cx_img = 0.5 * (x1 + x2)
#             cy_img = 0.5 * (y1 + y2)
            
#             # 像素 → 相机坐标
#             x_cam = (cx_img - cx) * d / fx
#             z_cam = d
            
#             # 相机 → 世界坐标（只考虑 yaw）
#             Xw = px + z_cam * math.cos(yaw) - x_cam * math.sin(yaw)
#             Yw = py + z_cam * math.sin(yaw) + x_cam * math.cos(yaw)

#             # ✅ 世界坐标 → 地图坐标
#             try:
#                 ix, iy = explorer.world_to_map(Xw, Yw)
#             except Exception as e:
#                 print(f"    [警告] world_to_map 失败: {e}")
#                 continue
            
#             # ✅ 边界检查（防止越界）
#             if not (0 <= ix < self.map_size and 0 <= iy < self.map_size):
#                 skipped_boundary += 1
#                 continue

#             # ✅ 写入地图（3×3 邻域）
#             for dy in (-1, 0, 1):
#                 for dx in (-1, 0, 1):
#                     col = ix + dx
#                     row = iy + dy
                    
#                     if 0 <= col < self.map_size and 0 <= row < self.map_size:
#                         if conf >= self.grid[row, col, 2]:
#                             self.grid[row, col, 0] = 1.0
#                             self.grid[row, col, 1] = cls_id
#                             self.grid[row, col, 2] = conf
#                             painted += 1

#         # ✅ 调试信息
#         if painted == 0:
#             print(f"[语义图] 本帧未写入: 越界={skipped_boundary}, 未映射类别={skipped_class}")
#         else:
#             print(f"[语义图] 成功写入 {painted} 个像素 (跳过: 越界={skipped_boundary}, 类别={skipped_class})")


#     def get_color_map(self):
#         import numpy as np
#         color_map = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        
#         # 给每个类别染色
#         palette = {
#             1: (255, 0, 0),     # person
#             2: (255, 128, 0),   # bicycle
#             3: (0, 255, 0),     # car
#             4: (0, 255, 255),   # moto
#             5: (255, 255, 0),   # bus
#             6: (255, 0, 255),   # truck
#             7: (0, 128, 0),     # tree
#             8: (160, 160, 160), # building
#             9: (128, 64, 128),  # road
#         }
        
#         cls_layer = self.grid[:, :, 1].astype(np.int32)
        
#         for cid, color in palette.items():
#             mask = (cls_layer == cid)
#             color_map[mask] = color
        
#         return color_map

#     def save_visualization(self, path="debug/semantic_map.png"):
#         import cv2
#         from pathlib import Path
        
#         Path(path).parent.mkdir(parents=True, exist_ok=True)
        
#         cm = self.get_color_map()
#         cv2.imwrite(path, cv2.cvtColor(cm, cv2.COLOR_RGB2BGR))