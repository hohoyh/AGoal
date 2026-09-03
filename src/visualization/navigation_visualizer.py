# src/visualization/navigation_visualizer.py - 导航过程可视化工具
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import os

class NavigationVisualizer:
    """
    为无人机导航过程生成汇报用的可视化图像（RGB、分割、语义地图、匹配状态）
    每一步生成一个完整的状态面板
    """
    
    def __init__(self, output_dir="visualization_output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 定义阶段颜色
        self.stage_colors = {
            1: "#FF6B6B",  # Stage 1: 红色
            2: "#4ECDC4",  # Stage 2: 青色
            3: "#95E1D3",  # Stage 3: 绿色
        }
        
        # 定义语义颜色(与STMR一致)
        self.semantic_colors = {
            0: [240, 240, 240],   # 未探索
            1: [169, 169, 169],   # 建筑
            5: [34, 139, 34],     # 树木
            4: [139, 119, 101],   # 道路
            8: [255, 215, 0],     # 车辆
        }
        
        print(f"✅ 可视化器初始化完成, 输出目录: {output_dir}")
    
    def visualize_step(self, step_num, data):
        """
        生成单步可视化
        
        Args:
            step_num: 步骤编号
            data: 包含所有可视化数据的字典
                {
                    'rgb_image': numpy array,      # RGB观测
                    'seg_image': numpy array,      # 分割图
                    'semantic_map': numpy array,   # STMR语义地图
                    'drone_pos': [x, y, z],        # 无人机位置
                    'goal_pos': [x, y, z],         # 目标位置(可选)
                    'stage': 1/2/3,                # 当前阶段
                    'match_score': float,          # 匹配分数
                    'confidence': float,           # 置信度(可选)
                    'matched_nodes': int,          # 匹配节点数
                    'trajectory': [[x,y],...],     # 轨迹点
                    'info_text': str,              # 额外信息文本
                }
        """
        
        # 创建大画布 (2行3列布局)
        fig = plt.figure(figsize=(18, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # ==================== 第一行 ====================
        
        # [1,1] RGB观测
        ax1 = fig.add_subplot(gs[0, 0])
        if 'rgb_image' in data and data['rgb_image'] is not None:
            ax1.imshow(cv2.cvtColor(data['rgb_image'], cv2.COLOR_BGR2RGB))
            ax1.set_title(f"RGB Observation (Step {step_num})", fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # [1,2] 分割图
        ax2 = fig.add_subplot(gs[0, 1])
        if 'seg_image' in data and data['seg_image'] is not None:
            ax2.imshow(data['seg_image'])
            ax2.set_title("Semantic Segmentation", fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        # [1,3] 语义地图 + 轨迹
        ax3 = fig.add_subplot(gs[0, 2])
        if 'semantic_map' in data and data['semantic_map'] is not None:
            # 绘制语义地图
            map_vis = self._render_semantic_map(
                data['semantic_map'], 
                data.get('drone_pos'),
                data.get('goal_pos'),
                data.get('trajectory', [])
            )
            ax3.imshow(map_vis)
            ax3.set_title("Semantic Map + Trajectory", fontsize=12, fontweight='bold')
        ax3.axis('off')
        
        # ==================== 第二行 ====================
        
        # [2,1] 状态信息面板
        ax4 = fig.add_subplot(gs[1, 0])
        self._draw_status_panel(ax4, step_num, data)
        
        # [2,2] 匹配得分可视化
        ax5 = fig.add_subplot(gs[1, 1])
        self._draw_score_panel(ax5, data)
        
        # [2,3] 图匹配示意图
        ax6 = fig.add_subplot(gs[1, 2])
        self._draw_graph_matching(ax6, data)
        
        # ==================== 保存 ====================
        
        # 添加大标题
        stage = data.get('stage', 1)
        stage_names = {1: "Zero Matching", 2: "Partial Matching", 3: "Perfect Matching"}
        fig.suptitle(
            f"Step {step_num} - Stage {stage}: {stage_names.get(stage, 'Unknown')}", 
            fontsize=16, 
            fontweight='bold',
            color=self.stage_colors.get(stage, '#000000')
        )
        
        # 保存
        save_path = os.path.join(self.output_dir, f"step_{step_num:03d}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"  ✅ 步骤 {step_num} 可视化已保存: {save_path}")
    
    # ============================================================
    # 辅助绘图函数
    # ============================================================
    
    def _render_semantic_map(self, semantic_map, drone_pos, goal_pos, trajectory):
        """渲染语义地图(带轨迹和位置标记)"""
        h, w = semantic_map.shape
        map_vis = np.zeros((h, w, 3), dtype=np.uint8)
        
        # 填充语义颜色
        for sid, color in self.semantic_colors.items():
            mask = (semantic_map == sid)
            map_vis[mask] = color
        
        # 放大显示
        scale = 4
        map_vis = cv2.resize(map_vis, (w*scale, h*scale), interpolation=cv2.INTER_NEAREST)
        
        # ✅ 绘制轨迹 (调整粗细)
        if len(trajectory) > 1:
            traj_pixels = []
            for pos in trajectory:
                gx = int((pos[0] + 50) / 5 * scale)
                gy = int((50 - pos[1]) / 5 * scale)
                traj_pixels.append([gx, gy])
            
            points = np.array(traj_pixels, dtype=np.int32)
            cv2.polylines(map_vis, [points], False, (255, 69, 0), 
                        thickness=1)  # ✅ 从3改为1
        
        # ✅ 绘制当前位置 (调整大小)
        if drone_pos is not None:
            gx = int((drone_pos[0] + 50) / 5 * scale)
            gy = int((50 - drone_pos[1]) / 5 * scale)
            cv2.circle(map_vis, (gx, gy), 4, (255, 255, 255), -1)  # ✅ 从12改为4
            cv2.circle(map_vis, (gx, gy), 3, (30, 144, 255), -1)   # ✅ 从8改为3
        
        # ✅ 绘制目标位置 (调整大小)
        if goal_pos is not None:
            gx = int((goal_pos[0] + 50) / 5 * scale)
            gy = int((goal_pos[1] + 50) / 5 * scale)
            cv2.drawMarker(map_vis, (gx, gy), (255, 0, 0), 
                        markerType=cv2.MARKER_STAR, 
                        markerSize=8, thickness=2)  # ✅ 从20/3改为8/2
        
        return map_vis
        
    def _draw_status_panel(self, ax, step_num, data):
        """绘制状态信息面板"""
        ax.axis('off')
        
        stage = data.get('stage', 1)
        match_score = data.get('match_score', 0.0)
        confidence = data.get('confidence', 0.0)
        matched_nodes = data.get('matched_nodes', 0)
        drone_pos = data.get('drone_pos', [0, 0, 0])
        
        # 文本内容
        info_lines = [
            f"📍 Step: {step_num}",
            f"🎯 Stage: {stage}",
            f"",
            f"📊 Match Score: {match_score:.3f}",
            f"✅ Matched Nodes: {matched_nodes}",
            f"",
            f"🚁 Drone Position:",
            f"   X: {drone_pos[0]:.2f} m",
            f"   Y: {drone_pos[1]:.2f} m",
            f"   Z: {drone_pos[2]:.2f} m",
        ]
        
        if confidence > 0:
            info_lines.insert(5, f"🔍 Confidence: {confidence:.2f}")
        
        if 'info_text' in data:
            info_lines.append("")
            info_lines.append(f"💬 {data['info_text']}")
        
        # 绘制文本
        y_pos = 0.95
        for line in info_lines:
            ax.text(0.05, y_pos, line, 
                   transform=ax.transAxes,
                   fontsize=11,
                   verticalalignment='top',
                   family='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3) if line.startswith('📊') else None)
            y_pos -= 0.08
        
        ax.set_title("Status Information", fontsize=12, fontweight='bold')
    
    def _draw_score_panel(self, ax, data):
        """绘制匹配得分柱状图"""
        match_score = data.get('match_score', 0.0)
        confidence = data.get('confidence', 0.0)
        stage = data.get('stage', 1)
        
        # 数据
        metrics = ['Match\nScore', 'Confidence', 'Stage\nProgress']
        values = [match_score, confidence, stage / 3.0]
        colors = ['#4ECDC4', '#95E1D3', self.stage_colors.get(stage, '#666666')]
        
        # 绘制柱状图
        bars = ax.barh(metrics, values, color=colors, alpha=0.7)
        
        # 添加阈值线
        ax.axvline(x=0.3, color='red', linestyle='--', linewidth=1, alpha=0.5, label='σ₁ (0.3)')
        ax.axvline(x=0.7, color='green', linestyle='--', linewidth=1, alpha=0.5, label='σ₂ (0.7)')
        
        # 数值标注
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax.text(val + 0.02, i, f'{val:.2f}', 
                   va='center', fontsize=10, fontweight='bold')
        
        ax.set_xlim(0, 1.1)
        ax.set_xlabel('Score', fontsize=11)
        ax.set_title('Matching Metrics', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(axis='x', alpha=0.3)
    
    def _draw_graph_matching(self, ax, data):
        """绘制图匹配示意图"""
        ax.axis('off')
        
        matched_nodes = data.get('matched_nodes', 0)
        total_nodes = 5  # 假设目标图有5个节点
        
        # 绘制场景图和目标图
        scene_graph_pos = np.array([[0.2, 0.8], [0.2, 0.5], [0.2, 0.2]])
        goal_graph_pos = np.array([[0.8, 0.8], [0.8, 0.5], [0.8, 0.2]])
        
        # 场景图节点
        for i, pos in enumerate(scene_graph_pos):
            color = 'lightblue' if i < matched_nodes else 'lightgray'
            circle = plt.Circle(pos, 0.08, color=color, ec='black', linewidth=2)
            ax.add_patch(circle)
            ax.text(pos[0], pos[1], f'S{i+1}', ha='center', va='center', 
                   fontsize=10, fontweight='bold')
        
        # 目标图节点
        for i, pos in enumerate(goal_graph_pos):
            color = 'lightgreen' if i < matched_nodes else 'lightgray'
            circle = plt.Circle(pos, 0.08, color=color, ec='black', linewidth=2)
            ax.add_patch(circle)
            ax.text(pos[0], pos[1], f'G{i+1}', ha='center', va='center',
                   fontsize=10, fontweight='bold')
        
        # 匹配连线
        for i in range(min(matched_nodes, len(scene_graph_pos), len(goal_graph_pos))):
            ax.plot([scene_graph_pos[i][0], goal_graph_pos[i][0]], 
                   [scene_graph_pos[i][1], goal_graph_pos[i][1]], 
                   'g--', linewidth=2, alpha=0.7)
        
        # 标签
        ax.text(0.2, 0.95, 'Scene Graph', ha='center', fontsize=11, fontweight='bold')
        ax.text(0.8, 0.95, 'Goal Graph', ha='center', fontsize=11, fontweight='bold')
        ax.text(0.5, 0.05, f'Matched: {matched_nodes}/{total_nodes}', 
               ha='center', fontsize=10, 
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title('Graph Matching', fontsize=12, fontweight='bold')
    
    # ============================================================
    # 批量处理和视频生成
    # ============================================================
    
    def create_video(self, fps=2, output_name="navigation_process.mp4"):
        """将所有步骤图片合成视频"""
        import glob
        
        image_files = sorted(glob.glob(os.path.join(self.output_dir, "step_*.png")))
        
        if not image_files:
            print("⚠️  未找到可视化图片!")
            return
        
        # 读取第一张图片获取尺寸
        first_img = cv2.imread(image_files[0])
        h, w = first_img.shape[:2]
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_path = os.path.join(self.output_dir, output_name)
        video = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
        
        print(f"\n🎬 生成视频: {video_path}")
        
        for img_file in image_files:
            img = cv2.imread(img_file)
            video.write(img)
            print(f"  ✅ 添加帧: {os.path.basename(img_file)}")
        
        video.release()
        print(f"\n✅ 视频生成完成! 共 {len(image_files)} 帧")
        print(f"   输出路径: {video_path}")


# ============================================================
# 示例用法
# ============================================================

if __name__ == '__main__':
    """
    示例: 如何在导航代码中使用可视化器
    """
    
    # 初始化可视化器
    viz = NavigationVisualizer(output_dir="ppt_visualization")
    
    # 模拟导航步骤数据
    for step in range(1, 6):
        
        # 准备数据 (实际使用时从导航器获取)
        data = {
            'rgb_image': np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            'seg_image': np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            'semantic_map': np.random.randint(0, 6, (20, 20), dtype=np.int32),
            'drone_pos': [step * 2.0, step * 1.5, -15.0],
            'goal_pos': [10.0, 8.0, 0.0],
            'trajectory': [[i*2.0, i*1.5] for i in range(step+1)],
            'stage': min((step // 2) + 1, 3),
            'match_score': 0.2 + step * 0.15,
            'confidence': 0.1 + step * 0.12,
            'matched_nodes': min(step, 5),
            'info_text': f"Exploring region {step}"
        }
        
        # 生成可视化
        viz.visualize_step(step, data)
    
    # 生成视频
    viz.create_video(fps=1, output_name="demo_navigation.mp4")
    
    print("\n" + "="*60)
    print("✅ 演示完成! 请查看 ppt_visualization/ 目录")
    print("="*60)