# 实验 2　基于 TextCNN 的情感分类

使用相同的 TextCNN 和 Amazon 评论，对比 A（随机初始化词嵌入）与 B（用实验 1 的 Word2Vec 向量初始化）。复用实验 1 的词表、分词规则和训练样本。

## 运行

建议用 Python 3.12。普通 CPU 可运行，无需 GPU 或模型服务。

1. 从实验仓库根目录安装依赖。实验 1 的数据准备及词向量训练见 [实验 1](../exp01-word2vec/README.md)。
2. 使用实验 1 同一次训练生成的训练 CSV 和 `window_5` 目录。该目录包含 `embedding.npy`、`vocab.json`、`preprocessing.json`，父目录的 `run_manifest.json` 用来核验数据来源和文件摘要。
3. 以下路径对应课堂本地准备的样本。若你在实验 1 使用 `outputs/formal`，把命令中的 `outputs/skipgram` 换成该目录。

~~~bash
python -m pip install -r 2026-autumn/exp02-textcnn/requirements.txt

python 2026-autumn/exp02-textcnn/run_lab.py train \
  --train-csv 2026-autumn/exp01-word2vec/data/train_sample.csv \
  --dev-csv 2025-spring/exp02-sentiment-classificationn/dataset/dev.csv \
  --test-csv 2025-spring/exp02-sentiment-classificationn/dataset/test.csv \
  --handoff 2026-autumn/exp01-word2vec/outputs/skipgram/window_5 \
  --output-dir 2026-autumn/exp02-textcnn/outputs/my-run \
  --epochs 5

python 2026-autumn/exp02-textcnn/run_lab.py predict \
  --output-dir 2026-autumn/exp02-textcnn/outputs/my-run \
  "The screen is very clear, but the battery life is terrible."
~~~

训练输出目录须是新目录。Amazon CSV 原标签为 1（负面）和 2（正面），训练时映射为 0 和 1。开发集和测试集若有同文评论，程序在开发集中剔除，与训练集重叠则报错。课堂数据固定后，每组应使用同一份训练、开发和测试数据。

## 比较

输入统一为 100 个词位，卷积窗口为 3、4、5，每个窗口用 100 个滤波器。池化前屏蔽超出有效词长的窗口，少于窗口宽度的短句使用一次补齐后的卷积。两个模型都继续训练词嵌入。初始化时两者的 PAD 行均为零，UNK 行共享同一随机初值。

两组共享训练顺序、随机种子和训练预算。各自按开发集 Macro-F1 选出最佳轮次，再在测试集上评估。混淆矩阵的行是真实类别，列是预测类别，顺序为负面、正面。

| 输出 | 内容 |
|---|---|
| `metrics.json` | 两组测试 Accuracy、Macro-F1、混淆矩阵和数据校验信息 |
| `history.csv`、`curves.png` | 每轮训练损失和开发集变化 |
| `errors.csv` | 误判或两组判断不同的评论及未登录词数量 |
| `A_random.pt`、`B_word2vec.pt` | 可用于课堂推理的两组模型权重 |

从 `errors.csv` 选 5 条评论说明错误可能来自哪里。转折、反讽和词表覆盖可以作为分析线索，判断要引用具体文本。用同一条新评论现场比较两个模型的输出。
