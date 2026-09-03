# scripts/test_semantic_nav.py
# 在预建语义地图上做语言指令导航（LLM 解析指令 -> 语义匹配 -> 飞行）

import os
import sys
import time
import argparse

# 把仓库根目录加入 sys.path，保证 `src.xxx` 可以被导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.envs.airsim_env import AirSimEnv
from src.semantic_graph.semantic_navigator import SemanticNavigator


def main():
    parser = argparse.ArgumentParser(
        description="基于 LLM 语义地图的语言指令导航测试"
    )

    # 由 build_offline_map.py 生成的、包含 LLM 描述的语义地图
    parser.add_argument('--map-file', default='maps/downtown_map_hd.pkl')

    # LLM 接口配置（真实密钥用环境变量 OPENAI_API_KEY 注入）
    parser.add_argument('--config-file', default='src/semantic_graph/semantic_config.yaml')

    # 语言指令
    parser.add_argument('--goal', default='Find the building', type=str)
    
    args = parser.parse_args()

    # === 初始化环境 ===
    print("连接 AirSim...")
    env = AirSimEnv(None) # navigator 会填充 args
    
    # (YOLO 是可选的, 暂不加载)
    # yolo = FastYOLODetector('yolov8n.pt') 
    
    print("加载导航器...")
    nav = SemanticNavigator(env, yolo_detector=None,
                        map_file=args.map_file,
                        config_file=args.config_file)

    # === 起飞 ===
    print("起飞...")
    try:
        env.client.takeoffAsync().join()
        env.client.moveToZAsync(nav.cruise_z, 2).join() # 飞到巡航高度
        time.sleep(1.0)
    except Exception as e:
        print(f"✗ 起飞失败: {e}")
        env.close()
        return

    # === 执行语义导航 ===
    text_goal = args.goal
    success = nav.run_semantic_navigation(text_goal)

    print("\n" + "="*30)
    print("  任务结果:", "✅ 成功" if success else "❌ 失败")
    print("="*30)
    
    # 安全降落
    print("...任务结束, 正在降落。")
    env.client.landAsync().join()
    env.close()


if __name__ == "__main__":
    main()