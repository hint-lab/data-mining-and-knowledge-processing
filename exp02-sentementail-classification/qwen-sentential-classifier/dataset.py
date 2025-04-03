import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class SentimentDataset(Dataset):
    """
    情感分类数据集类
    
    参数:
        texts (List[str]): 文本列表
        labels (List[int]): 标签列表
        tokenizer: 分词器
        max_len (int): 最大序列长度
    """
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # 确保tokenizer有padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.texts)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        
        参数:
            idx (int): 样本索引
            
        返回:
            dict: 包含input_ids、attention_mask和labels的字典
        """
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # 使用tokenizer处理文本
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,  # 添加特殊标记
            max_length=self.max_len,  # 最大长度
            padding='max_length',  # 填充到最大长度
            truncation=True,  # 截断过长的文本
            return_attention_mask=True,  # 返回注意力掩码
            return_tensors='pt'  # 返回PyTorch张量
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }