1.根据课程要求“去标点符号以及基础停用词”，对代码lab_common.py做出以下修改：

1.1 新加入的frozenset中包含了所有的英文停用词
BASIC_STOPWORDS = frozenset({
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "did", "do", "does", "doing", "down", "during",
    "each", "for", "from", "further", "had", "has", "have", "he", "her", "here",
    "hers", "him", "his", "how", "i", "if", "in", "into", "is", "it", "its", "me",
    "my", "of", "on", "or", "our", "ours", "out", "over", "she", "so", "than",
    "that", "the", "their", "theirs", "them", "then", "there", "these", "they",
    "this", "those", "through", "to", "under", "until", "up", "was", "we", "were",
    "what", "when", "where", "which", "while", "who", "why", "will", "with", "you",
    "your", "yours",
})

1.2 保留表示否定含义的词
PRESERVED_NEGATIONS = ("no", "not", "never")

1.3 修改并新增了预处理阶段的代码
修改后的PREPROCESSING代码：
PREPROCESSING = {
    "version": 2, "language": "English", "unicode": "NFKC",
    "lowercase": True, "html_unescape": True, "remove_html_tags": True,
    "token_pattern": r"[a-z]+(?:'[a-z]+)?|[0-9]+", "remove_stopwords": True,
    "stopword_list": "basic_english_function_words_v1",
    "stopwords": sorted(BASIC_STOPWORDS),
    "preserved_negations": list(PRESERVED_NEGATIONS),
    "combine": "title + space + text",
}
1.3.1 version由原先的1改为2，功能无实质改变，只是便于区分

1.3.2 remove_stopwords由False改为True,表明激活删除停用词操作

1.3.3 新增以下三行代码：
    "stopword_list": "basic_english_function_words_v1",
    "stopwords": sorted(BASIC_STOPWORDS),  //挑选出停用词并删除
    "preserved_negations": list(PRESERVED_NEGATIONS),  //保留带有否定含义的词



1.4 在tokenize()函数中添加筛选停用词并删除的功能
tokens = re.findall(PREPROCESSING["token_pattern"], re.sub(r"<[^>]*>", " ", text).lower())
    if PREPROCESSING["remove_stopwords"]:  //如果识别到停用词
        tokens = [token for token in tokens if token not in BASIC_STOPWORDS]
    return tokens


2.同时，在test_lab1.py中，也进行了简单的修改

2.1 删除了第32行的it和i，因为i和it都在上面frozenset里的英文停用词中
def test_tokenizer_keeps_negation_and_apostrophes(self):
        self.assertEqual(tokenize("<p>NOT good &amp; I don’t recommend it.</p>"),
                         ['not', 'good', "don't", 'recommend'])

2.2 修改后对运行项目进行自检操作
在终端中输入“python -m unittest discover -s tests -v”，最终显示：
Ran 8 tests in 14.632s

OK

表明8项测试全部通过

3.核心代码run_lab.py
该段代码仅改变了窗口的类型，同时控制了语料库等其余参数相同，体现了实验1的核心原理：对照训练窗口2和窗口5
for window in (2, 5):  //窗口2和窗口5
        folder = a.output_dir / f"window_{window}"
        folder.mkdir()
        start = time.perf_counter()
        model = Word2Vec(
            vector_size=a.vector_size, 
            window=window, 
            min_count=a.min_count,
            sg=int(a.algorithm == "skipgram"), 
            seed=a.seed, 
            **FIXED_PARAMETERS
        )
        model.build_vocab(corpus) //两窗口使用同一预料库corpus
        model.train(
            corpus, 
            total_examples=model.corpus_count, 
            epochs=a.epochs, 
            compute_loss=True  //激活计算损失功能
        )
        seconds = time.perf_counter() - start
        model.save(str(folder / "word2vec.model"))
        model.wv.save(str(folder / "vectors.kv"))






