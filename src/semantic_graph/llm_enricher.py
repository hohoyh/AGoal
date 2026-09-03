# src/semantic_graph/llm_enricher.py
# (✅ 最终优化版：采纳了 Claude 的详细 Prompt 策略)

import numpy as np
import yaml
import json
import base64

# ⚠️ 确保您已经安装了 openai 库: pip install openai
try:
    from openai import OpenAI
except ImportError:
    print("错误: OpenAI 库未安装。请运行: pip install openai")
    exit()

from ..utils.config import resolve_api_key

class LlmEnricher:
    """
    负责在建图最后一步,为物体数据库(objects_db)丰富LLM生成的描述。
    ✅ 使用针对俯拍优化的、按类别定制的详细 Prompt。
    """
    def __init__(self, config_path='configs/mapping_config.yaml'):
        print("[LLM Enricher] 正在加载...")
        
        # 1. 加载配置文件
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        if 'llm' not in self.config:
            print(f"  [错误] '{config_path}' 中未找到 'llm:' 配置块。")
            self.client = None
            return

        llm_config = self.config['llm']
        
        # 2. ✅ 读取您的 API 设置
        self.api_key = resolve_api_key(llm_config.get('api_key'))
        self.base_url = llm_config.get('base_url')
        self.model = llm_config.get('llm_model', 'gpt-4o-mini')
        
        # ✅ 3. 读取您要丰富的类别列表
        self.enrich_categories = llm_config.get('enrich_categories', ['building'])
        print(f"  [LLM] 将只为以下类别生成描述: {self.enrich_categories}")
        
        # ✅ (从 config 加载关联阈值)
        self.assoc_thresholds = self.config.get('association_thresholds', {'default': 20.0})
        
        if not self.api_key:
            print(f"  [警告] 'llm.api_key' 未在 {config_path} 中设置，"
                  f"且环境变量 OPENAI_API_KEY 也为空，将跳过 LLM 描述生成。")
            self.client = None
            return

        # 4. ✅ 初始化 OpenAI 客户端
        try:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            print(f"  ✓ LLM 客户端已初始化 (模型: {self.model})")
        except Exception as e:
            print(f"  [错误] 初始化 OpenAI 客户端失败: {e}")
            self.client = None

    def _call_llm_api(self, text_prompt, base64_image_data):
        """
        (✅ 真实 API 调用: 接收并发送图像)
        """
        if not self.client:
            print("    [LLM 跳过] 客户端未初始化。")
            return json.dumps({"error": "LLM 客户端未初始化。"})

        # 准备消息体
        messages_payload = [
            {
                "role": "system", 
                "content": "You are a helpful assistant outputting structured JSON."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": text_prompt # 我们的文本提示
                    }
                ]
            }
        ]
        
        # ✅ 如果有图像数据, 则添加到 payload
        if base64_image_data:
            messages_payload[1]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image_data}",
                    "detail": "low" # 节省 token
                }
            })
        else:
            print("    [警告] 正在进行无图像的纯文本描述。")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                response_format={ "type": "json_object" },
                messages=messages_payload
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"    [LLM 错误] API 调用失败: {e}")
            return json.dumps({"error": str(e)})

    # ========== ✅ 采纳 Claude 的新函数 (来自您的 Prompt) ==========
    
    def _compute_spatial_relations_8dir(self, obj, all_objects):
        """
        计算周围物体的8方向关系 (东/西/南/北/东南/东北/西南/西北)
        """
        relations = []
        
        for other_obj in all_objects:
            if obj['id'] == other_obj['id']:
                continue
            
            # 计算相对位置
            dx = other_obj['position'][0] - obj['position'][0]  # North方向 (+为北, -为南)
            dy = other_obj['position'][1] - obj['position'][1]  # East方向 (+为东, -为西)
            dist = np.sqrt(dx**2 + dy**2)
            
            # ✅ 建筑物看更远的范围
            threshold = self.assoc_thresholds.get(obj['label'], self.assoc_thresholds.get('default', 20.0))
            
            if dist > threshold:
                continue
            
            # ✅ 计算8方向 (关键: 用户会说"东侧有树")
            angle = np.degrees(np.arctan2(dy, dx))  # -180 ~ 180度
            
            if -22.5 <= angle < 22.5:
                direction = "北侧"
            elif 22.5 <= angle < 67.5:
                direction = "东北侧"
            elif 67.5 <= angle < 112.5:
                direction = "东侧"       # ✅ 用户会说"东侧有树"
            elif 112.5 <= angle < 157.5:
                direction = "东南侧"
            elif angle >= 157.5 or angle < -157.5:
                direction = "南侧"
            elif -157.5 <= angle < -112.5:
                direction = "西南侧"
            elif -112.5 <= angle < -67.5:
                direction = "西侧"       # ✅ 用户会说"西侧有车"
            else:  # -67.5 <= angle < -22.5
                direction = "西北侧"
            
            # ✅ 按类别聚合 (例如: "东侧有3棵树" 而不是列出3次)
            relations.append({
                'type': other_obj['label'],
                'direction': direction,
                'distance': dist,
                'id': other_obj['instance_id']
            })
        
        # ✅ 聚合同方向同类型的物体
        aggregated = {}
        for rel in relations:
            key = (rel['type'], rel['direction'])
            if key not in aggregated:
                aggregated[key] = {'count': 0, 'min_dist': 999, 'ids': []}
            aggregated[key]['count'] += 1
            aggregated[key]['min_dist'] = min(aggregated[key]['min_dist'], rel['distance'])
            aggregated[key]['ids'].append(rel['id'])
        
        # ✅ 格式化输出
        formatted = []
        for (obj_type, direction), info in aggregated.items():
            if info['count'] == 1:
                formatted.append(f"- {direction}有 {obj_type} (距离 {info['min_dist']:.1f}米)")
            else:
                formatted.append(f"- {direction}有 {info['count']} 个 {obj_type} (最近 {info['min_dist']:.1f}米)")
        
        return formatted, aggregated # ✅ 返回聚合后的字典和格式化列表

    # ========== ✅ Building 精简 Prompt ==========
    def _building_prompt_simple(self, obj, context, relations_text, relations_dict):
        """
        建筑物专用Prompt - 精简版
        只关注: 颜色、形状、大小、周围物体的方位
        """
        # ✅ 新增: 从 obj['attributes'] 获取侧视扫描的属性
        side_view_attrs = obj.get('attributes', {})
        color_from_side = side_view_attrs.get('color', '未知')
        height_from_side = side_view_attrs.get('height', '未知')
        material_from_side = side_view_attrs.get('material', '未知')

        # ✅ 将relations转换成更易读的格式
        relations_text_list = "\n".join(relations_text) if relations_text else "周围50米内无其他物体"
        
        # ✅ 动态构建 spatial_relations JSON
        spatial_json = {"east": "无", "west": "无", "south": "无", "north": "无"}
        summary_parts = []
        
        for (obj_type, direction), info in relations_dict.items():
            text = f"{info['count']} 个 {obj_type}" if info['count'] > 1 else f"{obj_type}"
            if direction == "东侧":
                spatial_json["east"] = text
                summary_parts.append(f"东侧有 {text}")
            elif direction == "西侧":
                spatial_json["west"] = text
                summary_parts.append(f"西侧有 {text}")
            elif direction == "南侧":
                spatial_json["south"] = text
                summary_parts.append(f"南侧有 {text}")
            elif direction == "北侧":
                spatial_json["north"] = text
                summary_parts.append(f"北侧有 {text}")
            # (省略了东南、西北等, 以保持JSON简洁)

        summary_text = ", ".join(summary_parts) if summary_parts else "周围环境开阔"

        return f"""你是一个建筑物识别AI。请综合**俯视图**和**侧视扫描**的信息,生成描述。

        **上下文信息**:
        {context}

        **侧视扫描获得的属性** (来自环绕拍摄):
        - 外墙颜色: {color_from_side}
        - 建筑高度: {height_from_side}
        - 外墙材质: {material_from_side}

        **已知的周围物体** (来自俯视图):
        {relations_text_list}

        **任务**:
        1. 你看到的俯视图展示了建筑物的**屋顶**
        2. 侧视扫描已经告诉你外墙颜色和高度
        3. 请综合这两部分信息,生成完整描述

        请严格按照以下JSON格式输出:
        {{
        "id": "{obj['instance_id']}",
        "category": "building",
        "attributes": {{
            "roof_color": "屋顶颜色 (从俯视图判断)",
            "wall_color": "{color_from_side}",  # 来自侧视
            "building_height": "{height_from_side}",  # 来自侧视
            "building_shape": "俯视形状 (从俯视图判断)",
            "material": "{material_from_side}",  # 来自侧视
            "description": "综合描述"
        }},
        "spatial_relations": {{
            "east": "{spatial_json['east']}",
            "west": "{spatial_json['west']}",
            "south": "{spatial_json['south']}",
            "north": "{spatial_json['north']}",
            "summary": "{summary_text}"
        }},
        "natural_description": "用一句话完整描述: 这是一座{color_from_side}外墙、{height_from_side}的建筑,从上方看是XX形状,东西南北各有什么物体。"
        }}
        """

    def _build_prompt(self, obj, all_objects):
        """
        ✅ 精简版: 专注建筑物颜色和周围物体的方位关系
        """
        label = obj['label']
        
        # 1. 收集文本上下文
        context = f"物体类型: {label}, ID: {obj['instance_id']}\n"
        context += f"中心坐标 (North, East, Down): ({obj['position'][0]:.1f}, {obj['position'][1]:.1f}, {obj['position'][2]:.1f})\n"
        context += f"聚类观测次数: {obj['count']}\n"
        
        # ✅ 2. 计算带8方向的空间关系 (关键!)
        relations_text_list, relations_dict = self._compute_spatial_relations_8dir(obj, all_objects)
        context += "附近物体:\n" + ("\n".join(relations_text_list) if relations_text_list else "无")
        
        # ✅ 3. 只为 building 生成详细描述
        if label == 'building':
            prompt = self._building_prompt_simple(obj, context, relations_text_list, relations_dict)
        else:
            # 其他类别跳过或使用简单描述
            obj['llm_description'] = "N/A (Not building)"
            return None, None # ✅ 返回 None, 表示跳过
        
        return prompt, obj.get('crop_image') # ✅ 返回 Prompt 和 图像

    def generate_descriptions(self, clustered_objects):
        """
        (✅ 核心函数 - 已更新为调用新的 _build_prompt)
        """
        if not self.client:
            print("  [LLM 错误] LLM 客户端未初始化, 跳过描述生成。")
            return clustered_objects
            
        print(f"  [LLM] 开始为 {len(clustered_objects)} 个唯一物体生成描述...")
        
        for obj in clustered_objects:
            
            # 1. 检查该类别是否在您的配置列表中
            if obj['label'] not in self.enrich_categories:
                obj['llm_description'] = "N/A (Skipped by config)" # 跳过
                continue

            # 2. ✅ 调用新的、更智能的 Prompt 构建器
            prompt_text, base64_image_data = self._build_prompt(obj, clustered_objects)
            
            # ✅ 如果 Prompt 构建器返回 None (例如因为不是 building), 则跳过
            if prompt_text is None:
                continue

            try:
                # 3. 调用 LLM API
                json_string_output = self._call_llm_api(prompt_text, base64_image_data)
                
                obj['llm_description'] = json.loads(json_string_output) 
                print(f"    ✓ 成功描述: {obj['instance_id']}")
                
            except Exception as e:
                print(f"    ✗ 描述失败: {obj['instance_id']} - {e}")
                obj['llm_description'] = {"error": str(e)}
       
        print("  [LLM] 全部描述生成完毕。")
        return clustered_objects