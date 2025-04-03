class Config:
    """
    模型配置类，包含所有可配置参数
    """
    # 模型参数
    model_name = "bert-base-chinese"  # 改为使用BERT模型，更兼容旧版本PyTorch
    max_seq_length = 128  # 最大序列长度，超过此长度的文本将被截断
    num_classes = 2  # 分类类别数量，二分类为2
    
    # 训练参数
    batch_size = 16  # 批次大小，可根据GPU内存调整
    learning_rate = 2e-5  # 学习率
    num_epochs = 5  # 训练轮数
    
    # 路径配置
    train_path = "dataset/train.csv"  # 训练集路径
    dev_path = "dataset/dev.csv"  # 验证集路径
    test_path = "dataset/test.csv"  # 测试集路径
    model_save_path = "saved_models/sentiment_model.pth"  # 模型保存路径