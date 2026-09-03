import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

class DepthEstimator:
    def __init__(self, model_type='DPT_Large'):
        """
        初始化深度估计器
        model_type: 'DPT_Large', 'DPT_Hybrid', 'MiDaS_small' 等
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 加载MiDaS模型
        self.model = torch.hub.load('intel-isl/MiDaS', model_type)
        self.model.to(self.device)
        self.model.eval()
        
        # 加载对应的transforms
        midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
        
        if model_type == 'DPT_Large' or model_type == 'DPT_Hybrid':
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform
    
    def estimate_depth(self, image_path, output_path=None, show=True):
        """
        估计单张图像的深度
        
        参数:
            image_path: 输入图像路径
            output_path: 输出深度图路径（可选）
            show: 是否显示结果
        
        返回:
            depth_map: numpy数组格式的深度图
        """
        # 读取图像
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 预处理
        input_batch = self.transform(img).to(self.device)
        
        # 推理
        with torch.no_grad():
            prediction = self.model(input_batch)
            
            # 调整大小到原始图像尺寸
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode='bicubic',
                align_corners=False
            ).squeeze()
        
        # 转换为numpy数组
        depth_map = prediction.cpu().numpy()
        
        # 归一化到0-255
        depth_normalized = cv2.normalize(depth_map, None, 0, 255, 
                                        cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # 应用颜色映射
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_MAGMA)
        
        # 保存结果
        if output_path:
            cv2.imwrite(output_path, depth_colored)
            print(f"深度图已保存到: {output_path}")
        
        # 显示结果
        if show:
            self._visualize_results(img, depth_map, depth_colored)
        
        return depth_map
    
    def _visualize_results(self, original_img, depth_map, depth_colored):
        """可视化原图和深度图"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 原始图像
        axes[0].imshow(original_img)
        axes[0].set_title('原始图像', fontsize=14)
        axes[0].axis('off')
        
        # 深度图（灰度）
        axes[1].imshow(depth_map, cmap='gray')
        axes[1].set_title('深度图（灰度）', fontsize=14)
        axes[1].axis('off')
        
        # 深度图（彩色）
        axes[2].imshow(cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB))
        axes[2].set_title('深度图（彩色）', fontsize=14)
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def batch_process(self, image_list, output_dir='./depth_outputs'):
        """
        批量处理多张图像
        
        参数:
            image_list: 图像路径列表
            output_dir: 输出目录
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        for idx, img_path in enumerate(image_list):
            print(f"\n处理图像 {idx+1}/{len(image_list)}: {img_path}")
            
            filename = os.path.basename(img_path)
            name, ext = os.path.splitext(filename)
            output_path = os.path.join(output_dir, f"{name}_depth{ext}")
            
            depth_map = self.estimate_depth(img_path, output_path, show=False)
            results.append(depth_map)
        
        print(f"\n批量处理完成！结果保存在: {output_dir}")
        return results


# ============== 使用示例 ==============

def main():
    # 初始化深度估计器
    # 可选模型: 'DPT_Large'(最准确), 'DPT_Hybrid'(平衡), 'MiDaS_small'(最快)
    estimator = DepthEstimator(model_type='MiDaS_small')
    
    # 方式1: 处理单张图像
    image_path = '图片1.png'  # 替换为你的图像路径
    depth_map = estimator.estimate_depth(
        image_path=image_path,
        output_path='output_depth.png',
        show=True
    )
    
    # 方式2: 批量处理
    # image_list = ['image1.jpg', 'image2.jpg', 'image3.jpg']
    # results = estimator.batch_process(image_list, output_dir='./depth_results')
    
    # 方式3: 获取深度值进行进一步处理
    print(f"深度图形状: {depth_map.shape}")
    print(f"深度值范围: {depth_map.min():.2f} ~ {depth_map.max():.2f}")
    
    # 可以进一步分析深度信息
    # 例如：找到最近和最远的点
    min_depth_pos = np.unravel_index(depth_map.argmin(), depth_map.shape)
    max_depth_pos = np.unravel_index(depth_map.argmax(), depth_map.shape)
    print(f"最近点位置: {min_depth_pos}")
    print(f"最远点位置: {max_depth_pos}")


if __name__ == "__main__":
    main()