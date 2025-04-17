import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def translate_to_chinese(text, model, tokenizer, device):
    """
    将文本翻译成中文
    
    参数:
        text: 要翻译的文本
        model: 翻译模型
        tokenizer: 分词器
        device: 计算设备
        
    返回:
        str: 翻译后的中文文本
    """
    # 检查输入是否为空
    if text is None or not isinstance(text, str) or not text.strip():
        print("警告: 输入文本为空或无效")
        return ""
        
    try:
        # 清理输入文本
        text = text.strip()
        
        # 构建翻译提示
        prompt = f"""请将以下英文文本翻译成中文，只需要输出翻译结果，不要解释，不要提问，不要回答： 英文原文： {text} 中文翻译："""
        
        # 生成翻译
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # 设置生成参数 - 调整参数以获得更精确的输出
        generation_config = {
            "max_new_tokens": 1024,
            "do_sample": False,     # 使用确定性生成
            "num_beams": 1,         # 使用贪婪解码
            # "temperature": 0.1,     # 降低温度使输出更确定
            "repetition_penalty": 1.1,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,  # 添加结束标记
            # "early_stopping": True   # 添加早停以避免生成额外内容
        }
        
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **generation_config
            )
        
        # 解码翻译结果
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取翻译部分（去除提示）
        translated_text = translated_text.replace(prompt, "").strip()
        
        # 只保留第一段文本，去除可能的额外内容
        translated_text = translated_text.split('\n')[0].strip()
        
        if not translated_text:
            print("警告: 翻译结果为空")
            return text  # 如果翻译为空，返回原文
            
        print(f"翻译完成，原文长度: {len(text)}, 译文长度: {len(translated_text)}")
        return translated_text
        
    except Exception as e:
        print(f"翻译过程中出错: {str(e)}")
        # 如果翻译失败，返回原文
        return text

def init_translation_model(model_name="Qwen/Qwen2.5-3B", device=None):
    """
    初始化翻译模型
    
    参数:
        model_name: 模型名称
        device: 计算设备
        
    返回:
        tuple: (model, tokenizer)
    """
    print(f"初始化翻译模型: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    if device is not None:
        model = model.to(device)
    
    print("翻译模型初始化完成")
    return model, tokenizer 