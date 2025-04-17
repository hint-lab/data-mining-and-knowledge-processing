import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class MedicalEntityExtractor:
    def __init__(self, model_name="Qwen/Qwen2.5-3B", device=""):
        """
        初始化医学实体提取器
        
        参数:
            model_name: Qwen模型名称 (使用较小的3B版本)
            device: 计算设备 (默认使用CPU避免CUDA错误)
        """
        self.model_name = model_name
        
        # 设置设备
        if device == "":
            # 检查CUDA是否可用
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                print("使用CUDA设备")
            else:
                self.device = torch.device('cpu')
                print("CUDA不可用，使用CPU设备")
        else:
            self.device = torch.device(device)
            print(f"使用设备: {self.device}")
        
        # 加载tokenizer和模型
        print(f"加载模型: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # 设置token - 确保在模型加载前设置
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # 使用保守的模型配置
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto", 
            "low_cpu_mem_usage": True,  # 减少内存使用
        }
        
        if self.device.type == 'cpu':
            print("使用CPU运行模型，将使用全精度")
            model_kwargs["torch_dtype"] = torch.float32
        else:
            print("使用CUDA运行模型，将使用半精度以提高性能")
            model_kwargs["torch_dtype"] = torch.float16
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                **model_kwargs
            )
            self.model = self.model.to(self.device)  # 确保模型在正确的设备上
        except Exception as e:
            print(f"模型加载异常: {str(e)}")
            print("尝试使用较小的batch_size加载模型...")
            
            # 使用更保守的配置重试
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float32  # 使用全精度
            )
            self.model = self.model.to(self.device)
        
        print("模型加载完成")
    
    def extract_entities(self, text):
        """
        从文本中提取医学实体（症状、疾病、基因和药物）
        
        参数:
            text: 输入文本
                
        返回:
            dict: 包含提取的医学实体
        """
        # 检查文本长度，截断过长文本以避免处理错误
        max_text_length = 1024  # 更小的最大长度以提高稳定性
        if len(text) > max_text_length:
            print(f"警告: 文本过长 ({len(text)} 字符)，将被截断至 {max_text_length} 字符")
            text = text[:max_text_length]
        
        # 修改提示模板 - 提取症状、疾病、基因和药物
        prompt = f"""请从以下医学病例描述中提取医学实体，如果找到任何包括症状、疾病、检查类型和药物，请将它们列在相应的数组中。如果没有找到任何实体，请返回空数组。请确保JSON格式正确，数组项之间和属性之间使用逗号分隔。只返回json文件内容，不要返回其他内容。请以以下JSON格式返回：
{{
  "symptoms": [],
  "diseases": [],
  "checks": [],
  "drugs": []
}}

#医学病例描述：
{text}

#JSON输出：
"""
        
        try:
            # 生成结果 - 使用try/except捕获错误
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # 确保attention_mask正确设置
            attention_mask = torch.ones_like(inputs["input_ids"])
            
            # 调整生成参数
            generation_config = {
                "max_new_tokens": 512,  # 减小生成长度，因为只需要提取实体
                "do_sample": True,      # 使用采样以获得更多样化的结果
                "num_beams": 1,         # 使用贪婪解码
                "temperature": 0.1,     # 保持低温度以获得更确定性的输出
                "repetition_penalty": 1.1,
                "pad_token_id": self.tokenizer.eos_token_id
            }
            
            with torch.no_grad():
                try:
                    outputs = self.model.generate(
                        inputs["input_ids"],
                        attention_mask=attention_mask,
                        **generation_config
                    )
                except Exception as e:
                    print(f"生成过程中出错: {str(e)}")
                    print("尝试使用更保守的生成参数...")
                    
                    # 更保守的生成参数
                    outputs = self.model.generate(
                        inputs["input_ids"],
                        max_new_tokens=256,
                        num_beams=1,
                        early_stopping=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 打印原始响应，用于调试
            print(f"模型原始响应: {response}")
            
        except Exception as e:
            print(f"生成过程中出错: {str(e)}")
            return {"symptoms": [], "diseases": [], "checks": [], "drugs": []}
        
        # 改进的JSON提取和错误处理
        try:
            # 尝试找到JSON部分
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            
            if json_start == -1 or json_end <= json_start:
                print("无法在响应中找到有效的JSON结构")
                print(f"响应内容: {response}")
                
                # 尝试使用更直接的方法提取实体
                result = self.extract_entities_manually(response)
                if result is not None:
                    return result
                
                return {"symptoms": [], "diseases": [], "checks": [], "drugs": []}
            
            # 提取JSON字符串
            json_str = response[json_start:json_end]
            
            # 尝试清理JSON字符串
            json_str = self.clean_json_string(json_str)
            
            # 解析JSON
            try:
                result = json.loads(json_str)
                print(f"#"*30)
                print(f"提取到的实体: {result}")
                print(f"#"*30)
                # 检查提取到的非空实体类型数量
                entity_types_found = 0
                non_empty_types = []
                
                for entity_type in ["symptoms", "diseases", "checks", "drugs"]:
                    entities = result.get(entity_type, [])
                    if entities and len(entities) > 0:  # 确保列表非空
                        entity_types_found += 1
                        non_empty_types.append(entity_type)
                
                # 打印提取到的实体
                print("\n提取到的实体:")
                for entity_type in ["symptoms", "diseases", "checks", "drugs"]:
                    entities = result.get(entity_type, [])
                    if entities:
                        print(f"- {entity_type}: {', '.join(entities)}")
                
                # 如果提取到的非空实体类型少于2种，则跳过
                if entity_types_found < 2:
                    print(f"只提取到 {entity_types_found} 种非空实体类型 ({', '.join(non_empty_types)}), 需要至少2种，将跳过当前文章")
                    return None
                else:
                    print(f"成功提取到 {entity_types_found} 种实体类型 ({', '.join(non_empty_types)})")
                    return result
                
            except json.JSONDecodeError as e:
                print(f"JSON解析错误: {e}")
                print(f"问题的JSON字符串: {json_str}")
                print(f"原始响应: {response}")
                
                # 如果出错，尝试一个更直接的方法提取实体列表
                result = self.extract_entities_manually(response)
                return result
        except Exception as e:
            print(f"处理JSON时出错: {str(e)}")
            print(f"响应内容: {response}")
            
            # 尝试使用更直接的方法提取实体
            result = self.extract_entities_manually(response)
            if result is not None:
                return result
                
            return {"symptoms": [], "diseases": [], "checks": [], "drugs": []}

    def clean_json_string(self, json_str):
        """清理JSON字符串，修复常见问题"""
        # 替换单引号为双引号
        if '"' not in json_str and "'" in json_str:
            json_str = json_str.replace("'", '"')
        
        # 处理尾部逗号问题
        json_str = json_str.replace(",]", "]").replace(",}", "}")
        
        # 处理省略号问题
        json_str = json_str.replace("...", "")
        
        # 确保键名使用双引号
        json_str = re.sub(r'([{,])\s*([a-zA-Z0-9_]+):', r'\1"\2":', json_str)
        
        # 修复常见的数组和对象格式问题 - 确保在键值对之间有逗号
        json_str = re.sub(r'(\]\s*)\n\s*("|\w)', r'\1,\n\2', json_str)
        
        # 修复各种实体数组之间的格式问题
        for entity_type in ["symptoms", "diseases", "checks", "drugs"]:
            json_str = re.sub(f'("{entity_type}"\\s*:\\s*\\[\\s*\\])\\s+(")', r'\1,\n  \2', json_str)
        
        # 处理模型重复生成JSON的情况
        # 查找所有完整的JSON对象
        json_pattern = r'\{[^{}]*\}'
        matches = re.findall(json_pattern, json_str)
        
        if matches:
            # 尝试解析每个JSON对象，选择第一个非空的对象
            for json_obj in matches:
                try:
                    parsed = json.loads(json_obj)
                    # 检查是否有非空实体
                    has_entities = False
                    for entity_type in ["symptoms", "diseases", "checks", "drugs"]:
                        if parsed.get(entity_type, []) and len(parsed[entity_type]) > 0:
                            has_entities = True
                            break
                    
                    if has_entities:
                        json_str = json_obj
                        break
                except json.JSONDecodeError:
                    continue
            
            # 如果没有找到非空对象，使用第一个对象
            if not json_str.startswith('{'):
                json_str = matches[0]
        
        # 移除JSON后的解释性文本
        json_str = re.sub(r'\}\s*[\n\r].*$', '}', json_str, flags=re.DOTALL)
        
        # 移除JSON前的解释性文本
        json_str = re.sub(r'^.*?\{', '{', json_str, flags=re.DOTALL)
        
        return json_str

    def extract_entities_manually(self, text):
        """手动从响应中提取实体列表，适用于JSON解析失败的情况"""
        result = {
            "symptoms": [],
            "diseases": [],
            "checks": [],
            "drugs": []
        }
        
        try:
            # 尝试提取各种实体列表
            for entity_type in ["symptoms", "diseases", "checks", "drugs"]:
                # 尝试多种模式匹配实体
                patterns = [
                    f'"{entity_type}"\\s*:\\s*\\[(.*?)\\]',  # 标准JSON格式
                    f'"{entity_type}"\\s*:\\s*\\[(.*?)\\]',  # 带引号的键
                    f'{entity_type}\\s*:\\s*\\[(.*?)\\]',    # 不带引号的键
                    f'{entity_type}\\s*=\\s*\\[(.*?)\\]',    # 使用等号
                    f'{entity_type}\\s*:\\s*\\[(.*?)\\]',    # 冒号后可能有空格
                ]
                
                for pattern in patterns:
                    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
                    if matches:
                        for content in matches:
                            if content.strip():
                                # 尝试多种方式提取引号内的内容
                                items = []
                                
                                # 方法1: 标准JSON格式
                                items.extend(re.findall(r'"([^"]*)"', content))
                                
                                # 方法2: 单引号
                                items.extend(re.findall(r"'([^']*)'", content))
                                
                                # 方法3: 没有引号，用逗号分隔
                                if not items:
                                    items = [item.strip() for item in content.split(',') if item.strip()]
                                
                                # 添加到结果中
                                result[entity_type].extend([item.strip() for item in items if item.strip()])
            
            # 检查提取到的非空实体类型数量
            entity_types_found = 0
            if len(result["symptoms"]) > 0:
                entity_types_found += 1
            if len(result["diseases"]) > 0:
                entity_types_found += 1
            if len(result["checks"]) > 0:
                entity_types_found += 1
            if len(result["drugs"]) > 0:
                entity_types_found += 1
            
            # 打印提取到的实体
            print("\n手动提取到的实体:")
            for entity_type in ["symptoms", "diseases", "checks", "drugs"]:
                entities = result[entity_type]
                if entities:
                    print(f"- {entity_type}: {', '.join(entities)}")
            
            # 如果提取到的非空实体类型少于2种，则跳过
            if entity_types_found < 2:
                print(f"手动提取只得到 {entity_types_found} 种非空实体类型，需要至少2种，将跳过当前文章")
                return None
            else:
                print(f"手动提取成功得到 {entity_types_found} 种实体类型")
                return result
                
        except Exception as e:
            print(f"手动提取实体失败: {str(e)}")
            print(f"原始响应: {text}")
            return None 