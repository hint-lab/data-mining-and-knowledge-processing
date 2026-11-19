                     //Part 1: 修改后的lab_common.py代码

"""Shared CSV and preprocessing contract for experiments 1 and 2 (stdlib only)."""
import csv
import hashlib
import html
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path


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
PRESERVED_NEGATIONS = ("no", "not", "never")

PREPROCESSING = {
    "version": 2, "language": "English", "unicode": "NFKC",
    "lowercase": True, "html_unescape": True, "remove_html_tags": True,
    "token_pattern": r"[a-z]+(?:'[a-z]+)?|[0-9]+", "remove_stopwords": True,
    "stopword_list": "basic_english_function_words_v1",
    "stopwords": sorted(BASIC_STOPWORDS),
    "preserved_negations": list(PRESERVED_NEGATIONS),
    "combine": "title + space + text",
}


def tokenize(text):
    text = unicodedata.normalize("NFKC", html.unescape(text)).replace("’", "'")
    tokens = re.findall(PREPROCESSING["token_pattern"], re.sub(r"<[^>]*>", " ", text).lower())
    if PREPROCESSING["remove_stopwords"]:
        tokens = [token for token in tokens if token not in BASIC_STOPWORDS]
    return tokens


@dataclass(frozen=True)
class Review:
    label: int  # Amazon CSV: 1=negative, 2=positive. Lab 2 maps these to 0/1.
    title: str
    text: str

    def tokens(self):
        return tokenize(self.title + " " + self.text)

    def key(self):
        # Compare model-visible text, irrespective of sentiment label.
        return hashlib.sha256(" ".join(self.tokens()).encode()).digest()


def read_reviews(path):
    """Read headerless Amazon CSV or label/polarity,title,text/content header."""
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        first = True
        for row in reader:
            if not row or not any(s.strip() for s in row):
                continue
            fields = [s.strip().lower() for s in row]
            if first and fields in (["label", "title", "text"],
                                    ["polarity", "title", "text"],
                                    ["label", "title", "content"]):
                first = False
                continue
            first = False
            if len(row) != 3 or row[0].strip() not in {"1", "2"}:
                raise ValueError(f"{path}: CSV line {reader.line_num}: expected label 1/2,title,text")
            review = Review(int(row[0]), row[1].strip(), row[2].strip())
            if not review.tokens():
                raise ValueError(f"{path}: CSV line {reader.line_num}: empty English text")
            yield review


def write_reviews(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "title", "text"])
        writer.writerows((r.label, r.title, r.text) for r in rows)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def exclusion_keys(paths):
    return {row.key() for path in paths for row in read_reviews(path)}


def require_training_file(path):
    if Path(path).stem.lower() in {"dev", "test", "valid", "validation"}:
        raise ValueError("Do not use a development/test file for training. Supply a training CSV.")


def write_table(path, fields, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


                           //Part 2 run_lab.py核心代码

for window in (2, 5):  
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
        model.build_vocab(corpus)
        model.train(
            corpus, 
            total_examples=model.corpus_count, 
            epochs=a.epochs, 
            compute_loss=True
        )
        seconds = time.perf_counter() - start
        model.save(str(folder / "word2vec.model"))
        model.wv.save(str(folder / "vectors.kv"))









