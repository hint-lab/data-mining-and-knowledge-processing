import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Union
import json
import os

class DataLoader:
    """
    数据加载器类，用于加载和预处理数据
    """
    def __init__(self, config):
        """
        初始化数据加载器
        
        参数:
            config: 配置对象，包含数据路径等参数
        """
        self.config = config
        
    def load_csv(self, file_path: str) -> Tuple[List[str], List[int]]:
        """
        加载CSV格式的数据文件
        
        参数:
            file_path (str): CSV文件路径
            
        返回:
            Tuple[List[str], List[int]]: 文本列表和标签列表
        """
        try:
            # 读取CSV文件，不使用列名
            df = pd.read_csv(file_path, header=None, names=['label', 'title', 'text'])
            
            # 将标签从 1,2 转换为 0,1
            df['label'] = df['label'].map({1: 0, 2: 1})
            
            # 合并标题和文本
            texts = [f"{title} {text}" for title, text in zip(df['title'], df['text'])]
            labels = df['label'].tolist()
            
            return texts, labels
        except Exception as e:
            print(f"加载CSV文件时出错: {str(e)}")
            raise
    
    def load_json(self, file_path: str) -> Tuple[List[str], List[int]]:
        """
        加载JSON格式的数据文件
        预期JSON格式为：[{"text": "文本", "label": 0}, ...]
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            texts = [item['text'] for item in data]
            labels = [item['label'] for item in data]
            
            return texts, labels
        except Exception as e:
            print(f"加载JSON文件时出错: {str(e)}")
            raise
    
    def load_txt(self, text_file: str, label_file: str) -> Tuple[List[str], List[int]]:
        """
        加载文本文件
        text_file: 每行一个文本
        label_file: 每行一个标签
        """
        try:
            with open(text_file, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f]
            
            with open(label_file, 'r', encoding='utf-8') as f:
                labels = [int(line.strip()) for line in f]
            
            if len(texts) != len(labels):
                raise ValueError("文本数量与标签数量不匹配")
            
            return texts, labels
        except Exception as e:
            print(f"加载文本文件时出错: {str(e)}")
            raise
    
    def preprocess_text(self, text: str) -> str:
        """
        文本预处理
        
        参数:
            text (str): 输入文本
            
        返回:
            str: 预处理后的文本
        """
        # 去除多余的空白字符
        text = ' '.join(text.split())
        # 去除特殊字符（保留基本标点）
        text = text.replace('\n', ' ').replace('\t', ' ').replace('""', '"')
        return text
    
    def load_and_split_data(self, 
                           file_path: str, 
                           file_type: str = 'csv',
                           test_size: float = 0.1,
                           val_size: float = 0.1,
                           random_state: int = 42) -> Tuple[List[str], List[str], List[str], List[int], List[int], List[int]]:
        """
        加载并分割数据为训练集、验证集和测试集
        
        参数:
            file_path: 数据文件路径
            file_type: 文件类型 ('csv', 'json', 'txt')
            test_size: 测试集比例
            val_size: 验证集比例
            random_state: 随机种子
        
        返回:
            train_texts, val_texts, test_texts, train_labels, val_labels, test_labels
        """
        # 根据文件类型加载数据
        if file_type == 'csv':
            texts, labels = self.load_csv(file_path)
        elif file_type == 'json':
            texts, labels = self.load_json(file_path)
        elif file_type == 'txt':
            # 对于txt文件，需要同时提供文本和标签文件
            text_file = file_path
            label_file = file_path.replace('texts', 'labels')
            texts, labels = self.load_txt(text_file, label_file)
        else:
            raise ValueError(f"不支持的文件类型: {file_type}")
        
        # 文本预处理
        texts = [self.preprocess_text(text) for text in texts]
        
        # 首先分割出测试集
        train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
            texts, labels, 
            test_size=test_size, 
            random_state=random_state
        )
        
        # 从剩余数据中分割出验证集
        val_size_adjusted = val_size / (1 - test_size)  # 调整验证集比例
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_val_texts, train_val_labels,
            test_size=val_size_adjusted,
            random_state=random_state
        )
        
        print(f"数据集大小:")
        print(f"训练集: {len(train_texts)}")
        print(f"验证集: {len(val_texts)}")
        print(f"测试集: {len(test_texts)}")
        
        return train_texts, val_texts, test_texts, train_labels, val_labels, test_labels

def load_data_from_dir(data_dir: str, file_type: str = 'csv') -> Tuple[List[str], List[int]]:
    """
    从目录中加载所有数据文件
    """
    texts, labels = [], []
    for file_name in os.listdir(data_dir):
        if file_name.endswith(file_type):
            file_path = os.path.join(data_dir, file_name)
            loader = DataLoader(None)  # 如果不需要配置可以传入None
            file_texts, file_labels = loader.load_csv(file_path) if file_type == 'csv' else loader.load_json(file_path)
            texts.extend(file_texts)
            labels.extend(file_labels)
    return texts, labels

# 使用示例
if __name__ == "__main__":
    # 配置示例
    class Config:
        data_path = "dataset/test.csv"  # 更新为正确的路径
        file_type = "csv"
        test_size = 0.1
        val_size = 0.1
        random_state = 42
    
    config = Config()
    loader = DataLoader(config)
    
    # 加载并分割数据
    train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = loader.load_and_split_data(
        config.data_path,
        file_type=config.file_type,
        test_size=config.test_size,
        val_size=config.val_size,
        random_state=config.random_state
    )
    
    print("数据加载完成！")
    print(f"示例标签: {train_labels[0]}")
    print(f"示例文本: {train_texts[0][:100]}...")  # 只打印前100个字符
