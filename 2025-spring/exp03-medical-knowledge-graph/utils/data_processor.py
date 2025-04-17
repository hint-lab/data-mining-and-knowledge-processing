import json
import os
import csv
from typing import Dict, List, Any

class DataProcessor:
    @staticmethod
    def load_json_data(file_path: str) -> List[Dict[str, Any]]:
        """
        从JSONL文件加载数据（每行一个JSON对象）
        
        参数:
            file_path: JSONL文件路径
            
        返回:
            List[Dict]: 加载的数据列表
        """
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:  # 跳过空行
                        continue
                        
                    try:
                        item = json.loads(line)
                        data.append(item)
                    except json.JSONDecodeError as e:
                        print(f"第 {line_num} 行JSON解析失败: {str(e)}")
                        print(f"问题行内容: {line[:100]}...")
                        continue
                        
            if not data:
                print("没有找到有效的JSON数据")
                return []
            
            print(f"成功加载 {len(data)} 条数据")
            return data
            
        except Exception as e:
            print(f"加载JSONL文件失败: {str(e)}")
            return []
    
    @staticmethod
    def save_json_data(data: List[Dict[str, Any]], file_path: str) -> bool:
        """
        保存数据到JSON文件
        
        参数:
            data: 要保存的数据
            file_path: 保存路径
            
        返回:
            bool: 是否保存成功
        """
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"保存JSON文件失败: {str(e)}")
            return False
    
    @staticmethod
    def save_to_neo4j_format(data: List[Dict[str, Any]], output_dir: str) -> bool:
        """
        将数据保存为Neo4j兼容的CSV格式
        
        参数:
            data: 要保存的数据
            output_dir: 输出目录
            
        返回:
            bool: 是否保存成功
        """
        try:
            # 确保输出目录存在
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存节点
            nodes_file = os.path.join(output_dir, 'nodes.csv')
            with open(nodes_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['id', 'type', 'name'])
                
                # 用于去重的集合
                seen_nodes = set()
                
                for article in data:
                    # 添加症状节点
                    for symptom in article.get('symptoms', []):
                        node_id = f"symptom_{symptom}"
                        if node_id not in seen_nodes:
                            writer.writerow([node_id, 'Symptom', symptom])
                            seen_nodes.add(node_id)
                    
                    # 添加疾病节点
                    for disease in article.get('diseases', []):
                        node_id = f"disease_{disease}"
                        if node_id not in seen_nodes:
                            writer.writerow([node_id, 'Disease', disease])
                            seen_nodes.add(node_id)
                    
                    # 添加检查节点
                    for check in article.get('checks', []):
                        node_id = f"check_{check}"
                        if node_id not in seen_nodes:
                            writer.writerow([node_id, 'Check', check])
                            seen_nodes.add(node_id)
                    
                    # 添加药物节点
                    for drug in article.get('drugs', []):
                        node_id = f"drug_{drug}"
                        if node_id not in seen_nodes:
                            writer.writerow([node_id, 'Drug', drug])
                            seen_nodes.add(node_id)
            
            # 保存关系
            relationships_file = os.path.join(output_dir, 'relationships.csv')
            with open(relationships_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['start_id', 'end_id', 'type'])
                
                # 用于去重的集合
                seen_relationships = set()
                
                for article in data:
                    # 症状-疾病关系
                    for symptom in article.get('symptoms', []):
                        for disease in article.get('diseases', []):
                            rel_id = f"symptom_{symptom}-disease_{disease}"
                            if rel_id not in seen_relationships:
                                writer.writerow([
                                    f"symptom_{symptom}",
                                    f"disease_{disease}",
                                    'RELATED_TO'
                                ])
                                seen_relationships.add(rel_id)
                    
                    # 症状-检查关系
                    for symptom in article.get('symptoms', []):
                        for check in article.get('checks', []):
                            rel_id = f"symptom_{symptom}-check_{check}"
                            if rel_id not in seen_relationships:
                                writer.writerow([
                                    f"symptom_{symptom}",
                                    f"check_{check}",
                                    'RELATED_TO'
                                ])
                                seen_relationships.add(rel_id)
                    
                    # 疾病-检查关系
                    for disease in article.get('diseases', []):
                        for check in article.get('checks', []):
                            rel_id = f"disease_{disease}-check_{check}"
                            if rel_id not in seen_relationships:
                                writer.writerow([
                                    f"disease_{disease}",
                                    f"check_{check}",
                                    'RELATED_TO'
                                ])
                                seen_relationships.add(rel_id)
                    
                    # 疾病-药物关系
                    for disease in article.get('diseases', []):
                        for drug in article.get('drugs', []):
                            rel_id = f"disease_{disease}-drug_{drug}"
                            if rel_id not in seen_relationships:
                                writer.writerow([
                                    f"disease_{disease}",
                                    f"drug_{drug}",
                                    'RELATED_TO'
                                ])
                                seen_relationships.add(rel_id)
            
            return True
        except Exception as e:
            print(f"保存Neo4j格式文件失败: {str(e)}")
            return False 