"""Shared CSV and preprocessing contract for experiments 1 and 2 (stdlib only)."""
import csv
import hashlib
import html
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

PREPROCESSING = {
    "version": 1, "language": "English", "unicode": "NFKC",
    "lowercase": True, "html_unescape": True, "remove_html_tags": True,
    "token_pattern": r"[a-z]+(?:'[a-z]+)?|[0-9]+", "remove_stopwords": False,
    "combine": "title + space + text",
}


def tokenize(text):
    text = unicodedata.normalize("NFKC", html.unescape(text)).replace("’", "'")
    return re.findall(PREPROCESSING["token_pattern"], re.sub(r"<[^>]*>", " ", text).lower())


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
