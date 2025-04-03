import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from config import Config
from dataset import SentimentDataset
from load_data import DataLoader as DataLoaderClass
from model import SentimentClassifier
import torch.nn as nn
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def set_hf_mirrors():
    """
    设置Hugging Face镜像，加速模型下载
    """
    # 设置环境变量
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    # 可选的其他镜像
    # os.environ['HF_ENDPOINT'] = 'https://huggingface.tuna.tsinghua.edu.cn'
    # os.environ['HF_ENDPOINT'] = 'https://mirror.sjtu.edu.cn/hugging-face'
    
    # 设置模型缓存目录（可选）
    os.environ['HF_HOME'] = './hf_cache'
    
# 设置镜像
set_hf_mirrors()

def evaluate(model, eval_loader, device):
    """
    评估模型性能
    
    参数:
        model: 模型对象
        eval_loader: 评估数据加载器
        device: 计算设备（CPU/GPU）
        
    返回:
        Tuple[float, float]: 平均损失和准确率
    """
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            
            _, predictions = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(predictions == labels)
            total_predictions += len(labels)
            total_loss += loss.item()
            
    return total_loss / len(eval_loader), correct_predictions.double() / total_predictions

def train(train_texts, train_labels, val_texts=None, val_labels=None):
    """
    训练模型
    
    参数:
        train_texts (List[str]): 训练文本列表
        train_labels (List[int]): 训练标签列表
        val_texts (List[str], optional): 验证文本列表
        val_labels (List[int], optional): 验证标签列表
        
    返回:
        model: 训练好的模型
    """
    # 清理 GPU 缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 加载配置
    config = Config()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 初始化tokenizer和模型
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # 确保tokenizer有padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = SentimentClassifier(config.model_name, config.num_classes)
    model.to(device)
    
    # 准备训练数据
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, config.max_seq_length)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    
    # 准备验证数据
    if val_texts is not None and val_labels is not None:
        val_dataset = SentimentDataset(val_texts, val_labels, tokenizer, config.max_seq_length)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    
    # 计算总训练步数
    total_steps = len(train_loader) * config.num_epochs
    
    # 优化器
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    
    # 添加学习率调度器
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # 训练循环
    best_accuracy = 0
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            
            loss.backward()
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{config.num_epochs}')
        print(f'Average training loss: {avg_train_loss:.4f}')
        
        if val_texts is not None and val_labels is not None:
            val_loss, val_accuracy = evaluate(model, val_loader, device)
            print(f'Validation Loss: {val_loss:.4f}')
            print(f'Validation Accuracy: {val_accuracy:.4f}')
            
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                model.save_model(config.model_save_path)
                print(f"保存新的最佳模型，准确率: {val_accuracy:.4f}")
    
    return model

def predict(text, model_path=None):
    """
    使用训练好的模型进行预测
    
    参数:
        text (str): 待预测的文本
        model_path (str, optional): 模型路径
        
    返回:
        int: 预测的标签（0或1）
    """
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 初始化并加载模型
    model = SentimentClassifier(config.model_name, config.num_classes)
    if model_path:
        model.load_model(model_path)
    model.to(device)
    model.eval()
    
    # 预处理文本
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=config.max_seq_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        _, predictions = torch.max(outputs, dim=1)
    
    return predictions.item()

if __name__ == "__main__":
    # 设置Hugging Face镜像
    set_hf_mirrors()
    
    # 加载配置
    config = Config()
    
    # 加载数据
    data_loader = DataLoaderClass(config)
    
    # 分别加载训练集、验证集和测试集
    print("加载训练集...")
    train_texts, train_labels = data_loader.load_csv("dataset/train.csv")
    print("加载验证集...")
    val_texts, val_labels = data_loader.load_csv("dataset/dev.csv")
    print("加载测试集...")
    test_texts, test_labels = data_loader.load_csv("dataset/test.csv")
    
    # 打印数据集大小
    print(f"训练集: {len(train_texts)} 样本")
    print(f"验证集: {len(val_texts)} 样本")
    print(f"测试集: {len(test_texts)} 样本")
    
    # 训练模型
    print("开始训练模型...")
    model = train(train_texts, train_labels, val_texts, val_labels)
    
    # 预测示例
    example_text = "这个产品质量非常好，我很满意！"
    prediction = predict(example_text, config.model_save_path)
    sentiment = "正面" if prediction == 1 else "负面"
    print(f"示例文本: '{example_text}'")
    print(f"情感预测: {sentiment} (类别 {prediction})")
