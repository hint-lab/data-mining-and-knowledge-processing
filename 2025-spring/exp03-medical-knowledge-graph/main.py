import os
import torch
from models.entity_extractor import MedicalEntityExtractor
from utils.data_processor import DataProcessor
from utils.translation import translate_to_chinese

def init_models(model_name="Qwen/Qwen2.5-3B"):
    """
    初始化所有需要的模型
    
    参数:
        model_name: 模型名称
        
    返回:
        tuple: (extractor, model, tokenizer)
    """
    print(f"初始化模型: {model_name}")
    
    # 设置设备
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("使用CUDA设备")
    else:
        device = torch.device('cpu')
        print("CUDA不可用，使用CPU设备")
    
    # 初始化实体提取器
    print("初始化实体提取器...")
    extractor = MedicalEntityExtractor(model_name=model_name, device=device)
    
    # 使用实体提取器的模型和tokenizer进行翻译
    model = extractor.model
    tokenizer = extractor.tokenizer
    
    print("模型初始化完成")
    return extractor, model, tokenizer

def merge_entities(entities1, entities2):
    """
    合并两个实体字典，去除重复项
    """
    if not entities1:
        return entities2
    if not entities2:
        return entities1
    
    result = {
        "symptoms": list(set(entities1.get("symptoms", []) + entities2.get("symptoms", []))),
        "diseases": list(set(entities1.get("diseases", []) + entities2.get("diseases", []))),
        "checks": list(set(entities1.get("checks", []) + entities2.get("checks", []))),
        "drugs": list(set(entities1.get("drugs", []) + entities2.get("drugs", [])))
    }
    return result

def main():
    # 初始化所有模型
    extractor, model, tokenizer = init_models()
    
    # 设置输入输出路径
    input_file = "data/raw/Open-Patients.jsonl"  # 原始文章数据
    output_dir = "data/processed"          # 处理后的数据目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载原始数据
    print("加载原始数据...")
    articles = DataProcessor.load_json_data(input_file)
    if not articles:
        print("没有找到原始数据，请确保数据文件存在")
        return
    
    # 处理每篇文章
    processed_articles = []
    for i, article in enumerate(articles):
        # 测试
        if i >10:
            break
        print(f"\n{'='*50}")
        print(f"处理第 {i+1}/{len(articles)} 篇文章")
        print(f"{'='*50}")
        
        # 打印原文标题
        print("\n原文:")
        print(article["description"])
        
        # 翻译标题
        print("\n翻译...")
        translated = translate_to_chinese(
            article["description"],
            model,
            tokenizer,
            extractor.device
        )
        print(f"翻译结果: {translated}")
        
         
        # 从中提取实体
        print("\n从翻译中提取医学实体...")
        entities = extractor.extract_entities(translated)
        
        if entities:
            for key, values in entities.items():
                if values:
                    print(f"- {key}: {', '.join(values)}")
            
            # 合并结果
            processed_article = {
                "id": article["_id"],
                "translated": translated,
                **entities
            }
            processed_articles.append(processed_article)
            print(f"\n成功处理文章: {translated}")
            print(f"提取的实体数量: 症状({len(entities['symptoms'])}), 疾病({len(entities['diseases'])}), 药物({len(entities['drugs'])}), 检查({len(entities['checks'])})")
        else:
            print(f"\n跳过文章: {translated} (未提取到实体)")
    
    # 保存处理后的数据
    print("\n保存处理后的数据...")
    json_output = os.path.join(output_dir, "processed_articles.json")
    DataProcessor.save_json_data(processed_articles, json_output)
    
    # 保存为Neo4j格式
    print("保存为Neo4j格式...")
    neo4j_output = os.path.join(output_dir, "neo4j")
    DataProcessor.save_to_neo4j_format(processed_articles, neo4j_output)
    
    print("\n处理完成！")
    print(f"- 处理文章数: {len(processed_articles)}")
    print(f"- JSON输出: {json_output}")
    print(f"- Neo4j输出: {neo4j_output}")

if __name__ == "__main__":
    main() 