"""Prepare training-only data; never downloads or trains on the test split."""
import argparse
import json
import random
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from lab_common import Review, read_reviews, write_reviews, write_json, sha256, exclusion_keys, require_training_file

DATASET = "fancyzhx/amazon_polarity"
API = "https://datasets-server.huggingface.co/rows"


def get_batch(offset, length=100):
    query = urllib.parse.urlencode({"dataset": DATASET, "config": "amazon_polarity",
                                   "split": "train", "offset": offset, "length": length})
    for attempt in range(3):
        try:
            req = urllib.request.Request(API + "?" + query, headers={"User-Agent": "course-lab1/1.0"})
            with urllib.request.urlopen(req, timeout=30) as response:
                return json.load(response)
        except (OSError, ValueError):
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def cached_batch(offset, length, cache_dir):
    if cache_dir is None:
        return get_batch(offset, length)
    path = Path(cache_dir) / f"train-{offset}-{length}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    result = get_batch(offset, length)
    if "num_rows_total" not in result or "rows" not in result:
        raise ValueError("Unexpected dataset server response")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    write_json(temporary, result)
    temporary.replace(path)
    return result


def download_sample(size, seed, excluded, cache_dir=None):
    # Random contiguous batches avoid downloading the entire 3.6M-row training set.
    # This is a classroom sample, not a claim of stratified or independent sampling.
    first = cached_batch(0, 1, cache_dir)
    total = int(first["num_rows_total"])
    if total < size:
        raise ValueError("Requested more rows than the training split contains")
    rng = random.Random(seed)
    selected, keys, offsets = [], set(), []
    for _ in range(max(20, size // 100 * 5 + 10)):
        offset = rng.randrange(max(1, total - 99))
        result = cached_batch(offset, 100, cache_dir)
        if result.get("num_rows_total") != total:
            raise ValueError("Dataset changed during download; retry with a new output path")
        offsets.append(offset)
        for item in result["rows"]:
            if item.get("truncated_cells"):
                raise ValueError("Dataset viewer returned truncated text; use the local CSV route")
            row = item["row"]
            if row["label"] not in (0, 1):
                raise ValueError("Unexpected Hugging Face label encoding")
            review = Review(row["label"] + 1, row["title"] or "", row["content"] or "")
            key = review.key()
            if not review.tokens() or key in excluded or key in keys:
                continue
            keys.add(key)
            selected.append(review)
            if len(selected) == size:
                return selected, {"dataset": DATASET, "split": "train", "offsets": offsets,
                                  "sampling": "random training batches; deduplicated", "source_rows": total}
        print(f"Downloaded {len(selected)}/{size} training rows", flush=True)
    raise ValueError("Not enough unique eligible rows; use a local training CSV")


def local_sample(path, size, seed, excluded):
    require_training_file(path)
    rng, selected, seen = random.Random(seed), [], set()
    eligible = 0
    for row in read_reviews(path):
        key = row.key()
        if key in excluded or key in seen:
            continue
        seen.add(key)
        eligible += 1
        if len(selected) < size:
            selected.append(row)
        else:
            j = rng.randrange(eligible)
            if j < size:
                selected[j] = row
    if len(selected) < size:
        raise ValueError(f"Only {len(selected)} eligible unique training rows; requested {size}")
    return selected, {"source_name": Path(path).name, "source_sha256": sha256(path),
                      "eligible_rows": eligible, "sampling": "reservoir after deduplication/exclusion"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--train-csv", type=Path)
    source.add_argument("--download", action="store_true", help="Fetch only the public Amazon polarity TRAIN split")
    p.add_argument("--output", type=Path, default=Path("data/train_sample.csv"))
    p.add_argument("--size", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cache-dir", type=Path, default=Path("data/download_cache"))
    p.add_argument("--exclude-csv", type=Path, nargs="*", default=[])
    a = p.parse_args()
    if a.size < 1:
        p.error("--size must be positive")
    if a.output.exists() or a.output.with_suffix(".manifest.json").exists():
        p.error("Output already exists; use a new path to preserve the fixed dataset")
    try:
        excluded = exclusion_keys(a.exclude_csv)
        rows, info = download_sample(a.size, a.seed, excluded, a.cache_dir) if a.download else local_sample(a.train_csv, a.size, a.seed, excluded)
        a.output.parent.mkdir(parents=True, exist_ok=True)
        write_reviews(a.output, rows)
        info.update({"seed": a.seed, "rows": len(rows), "labels": dict(Counter(r.label for r in rows)),
                     "csv_sha256": sha256(a.output), "label_encoding": {"1": "negative", "2": "positive"},
                     "excluded": [{"name": f.name, "sha256": sha256(f)} for f in a.exclude_csv]})
        write_json(a.output.with_suffix(".manifest.json"), info)
        print(f"Saved {len(rows)} rows to {a.output}; label counts: {info['labels']}")
    except (ValueError, OSError, KeyError) as error:
        p.exit(1, f"Data preparation failed: {error}\n")


if __name__ == "__main__":
    main()
