# 实验1　词向量表示与相似度分析

本目录是2026秋季实验1的完整参考实现。用相同训练数据比较 Word2Vec 的两个窗口（2和5），计算词对相似度、查看近邻、画二维投影，最后将词向量交给实验2的TextCNN。

建议4学时：前两节简讲和动手做，后两节抽两组展示结果、讨论并修改。没有得到预期相似词也可以完成实验，关键是用语料和运行结果解释原因。

## 1. 安装与快速跑通

建议 Python **3.12**，普通CPU即可运行。不需要GPU、API密钥或NLTK数据下载。

```bash
cd 2026-autumn/exp01-word2vec
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_lab.py --train-csv examples/demo_train.csv --output-dir outputs/demo
```

Windows PowerShell将激活命令换成 `.venv\Scripts\Activate.ps1`，后续均使用 `python`。

`examples/demo_train.csv` 是本课程人工编写的24条英文**合成评论**，标签1为负面、2为正面，仅用于检查程序能否运行和文件能否交接。不要将它的词向量效果当作正式实验结论。

## 2. 准备正式训练数据

### 方式A　下载一个课堂训练样本

```bash
python prepare_data.py --download --size 5000 --seed 42 \
  --exclude-csv ../../2025-spring/exp02-sentiment-classificationn/dataset/dev.csv \
                ../../2025-spring/exp02-sentiment-classificationn/dataset/test.csv
```

脚本从 [Amazon Polarity镜像](https://huggingface.co/datasets/fancyzhx/amazon_polarity) 的 **train** 部分随机选取位置，按批次获取数据，去重并排除开发、测试集中出现的文本；不下载test部分。镜像的0/1标签会转换成与原Amazon CSV一致的1/2。外部服务需要网络，失败会明确退出，可以改用本地CSV方式。

这种小样本取法按随机位置分批获取，并不保证类别均衡或覆盖所有产品。正式开课由教师准备一次，查看类别分布和语料覆盖，固定文件及SHA256后发给全班；各组不要各自下载不同样本进行横向排名。镜像可能更新，同一随机种子不保证未来下载字节完全相同，以已固定CSV及清单为准。

成功下载的批次缓存在 `data/download_cache`。遇到503或网络中断时，重新执行相同命令会复用缓存；最终CSV只在取得足够样本后写出，不把半份数据当成完整训练集。一次数据准备应使用同一缓存目录，重新准备另一版本时使用新的 `--cache-dir`。

### 方式B　从已有的Amazon训练CSV抽样

从[原讲义的Kaggle数据源](https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews)取得 `train.csv`，运行：

```bash
python prepare_data.py --train-csv /你的路径/train.csv --size 5000 --seed 42 \
  --output data/train_sample.csv \
  --exclude-csv ../../2025-spring/exp02-sentiment-classificationn/dataset/dev.csv \
                ../../2025-spring/exp02-sentiment-classificationn/dataset/test.csv
```

本地方式遍历训练CSV，去重和排除重叠后做蓄水池抽样；读取大文件可能需要数分钟，并需要保存去重指纹的内存。支持无表头的三列CSV和 `label,title,text` 表头；不会把第一条评论误当列名。缺列、错误标签或没有有效英文内容会报错。

两种方式均输出 `data/train_sample.csv` 和 `data/train_sample.manifest.json`，记录来源、抽样方式、随机种子、数量、类别分布和文件摘要。已有文件不会被覆盖。训练数据不随代码提交；下载和使用须遵守数据来源的要求。

## 3. 完成两个窗口的对照训练

```bash
python run_lab.py --train-csv data/train_sample.csv --output-dir outputs/formal \
  --exclude-csv ../../2025-spring/exp02-sentiment-classificationn/dataset/dev.csv \
                ../../2025-spring/exp02-sentiment-classificationn/dataset/test.csv
```

默认设置：CBOW、128维、10轮、min_count=2、随机种子42、单线程，分别训练window=2和window=5。窗口表示左右最大距离，本实验关闭随机缩小窗口，使两种设定容易解释。其余训练参数相同。程序自动在固定 `PYTHONHASHSEED` 的子进程中运行；相同环境、输入和参数下可复现，同一结果不代表跨平台逐位一致。

保留否定词（not/no/never）和英文缩写，不默认删除停用词；统一预处理函数见 `lab_common.tokenize`。训练前检查重复文本和显式提供的开发/测试文件重叠。**不使用dev/test训练词向量，也不因为它们没有标签就把它们加入训练。** 没有传入排除文件时，程序无法判断你在别处保存的划分是否重叠。

可选参数：

```bash
# 拓展：固定其他条件，改为Skip-gram；保存到新的目录
python run_lab.py --train-csv data/train_sample.csv --algorithm skipgram --output-dir outputs/skipgram
# 可选的t-SNE投影
python run_lab.py --train-csv data/train_sample.csv --projection tsne --output-dir outputs/tsne
# 每行一个英文查询词，便于补充自己关心的词
python run_lab.py --train-csv data/train_sample.csv --query-file my_words.txt --output-dir outputs/custom
```

正式运行上述拓展时也应加入相同的 `--exclude-csv` 参数。目录非空会退出，避免覆盖已有实验记录。

## 4. 看结果与现场查询

| 文件 | 内容 |
|---|---|
| `comparison.csv` | 两个窗口的词表规模、训练耗时与训练损失 |
| `word_pairs.csv` | 5组词对的实测相似度；缺词标为oov |
| `neighbors.csv` | 默认10个查询词各自的Top-5近邻 |
| `training.log` | Gensim的训练过程记录 |
| `run_manifest.json` | 输入摘要、参数、版本、划分核对及完成状态 |
| `window_2/`、`window_5/` | 各组模型、词向量、词表、投影坐标及图片 |

两个模型的投影使用同一批最多200个高频词，图中仅标注前30个，全部坐标在 `projection.csv`。每张图单独拟合，坐标轴并未对齐，不能按两图中某个点的绝对位置判定词向量变化。

```bash
python query_model.py --vectors outputs/formal/window_5/vectors.kv good bad battery new_unknown_word
```

词表外的词会明确显示OOV，不能现场要求普通Word2Vec为任意新词生成向量。缺词记录不伪造相似度。训练损失与预测样本数量有关，不能用两个窗口损失的大小直接判断语义质量；词对相似度也不预设达到某个数值。

## 5. 交给实验2的文件

统一使用 `window_5`：

- `embedding.npy`：float32矩阵，行号就是词ID。
- `vocab.json`：双向词表、向量维度、PAD=0和UNK=1。
- `preprocessing.json`：预处理约定；实验2复用同版 `lab_common.tokenize`。
- `word2vec.model`：可继续训练的模型；`vectors.kv`用于只读向量查询。

PAD行为全零，UNK为按种子生成的固定随机向量，其余行逐一对应已训练词向量。只加载自己或教师提供的可信Gensim模型文件。

```python
import json
import numpy as np
from lab_common import tokenize

folder = "outputs/formal/window_5/"
matrix = np.load(folder + "embedding.npy", allow_pickle=False)
vocab = json.load(open(folder + "vocab.json", encoding="utf-8"))
ids = [vocab["token_to_id"].get(w, vocab["unk_id"]) for w in tokenize("not a good phone")]
assert matrix.shape == (len(vocab["id_to_token"]), vocab["vector_size"])
```

## 6. 学生要解释什么

1. 窗口2和5的近邻有什么差异？举具体词例和对应语境。
2. 找出至少2个不符合直觉的结果，说明可能与共现、词频或多义性有什么关系。
3. 哪些结论来自真实运行，哪些只是解释假设？二维图有什么局限？
4. 为什么近邻看着合理仍不能保证情感分类准确？交给实验2验证。

提交代码、参数与结果对照表、图、案例分析和模型交接文件。汇报直接复用输出，不需要制作复杂界面。该目录提供完整参考实现，学生仍须独立解释方法与结果。

## 验证

```bash
python -m unittest discover -s tests -v
```

检查CSV首行/多行读取、否定词保留、划分重叠拒绝、抽样一致性、下载标签映射、重复训练一致性，以及导出矩阵与词表逐行对应。GitHub Actions使用演示数据执行，不依赖外部数据下载。

技术依据：[Gensim Word2Vec实现及参数说明](https://github.com/piskvorky/gensim/blob/develop/gensim/models/word2vec.py)。
