"""Run a controlled window=2 versus window=5 Word2Vec experiment."""
import argparse
from collections import Counter
import json
import importlib.metadata
import logging
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from lab_common import PREPROCESSING, read_reviews, exclusion_keys, require_training_file, sha256, tokenize, write_json, write_table

QUERIES = ["good", "bad", "great", "phone", "battery", "screen", "price", "quality", "easy", "poor"]
PAIRS = [("good", "great"), ("good", "bad"), ("phone", "battery"), ("expensive", "cheap"), ("screen", "battery")]
FIXED_PARAMETERS = {"workers": 1, "shrink_windows": False, "alpha": 0.025,
                    "min_alpha": 0.0001, "negative": 5, "hs": 0, "sample": 1e-3}


def export_handoff(model, folder, seed):
    import numpy as np
    words = ["<PAD>", "<UNK>"] + model.wv.index_to_key
    matrix = np.zeros((len(words), model.vector_size), dtype=np.float32)
    matrix[1] = np.random.default_rng(seed).normal(0, 0.05, model.vector_size)
    matrix[2:] = model.wv.vectors
    np.save(folder / "embedding.npy", matrix, allow_pickle=False)
    write_json(folder / "vocab.json", {"pad_id": 0, "unk_id": 1, "id_to_token": words,
               "token_to_id": {word: i for i, word in enumerate(words)}, "vector_size": model.vector_size})
    write_json(folder / "preprocessing.json", PREPROCESSING)


def project_vectors(model, words, method, seed, folder):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = model.wv[words]
    if method == "pca":
        from sklearn.decomposition import PCA
        xy = PCA(n_components=2, svd_solver="full").fit_transform(x)
    else:
        from sklearn.manifold import TSNE
        xy = TSNE(n_components=2, perplexity=min(30, max(1, (len(words)-1)/3)),
                  random_state=seed, init="pca", learning_rate="auto").fit_transform(x)
    write_table(folder / "projection.csv", ["word", "x", "y"],
                [{"word": w, "x": float(v[0]), "y": float(v[1])} for w, v in zip(words, xy)])
    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    ax.scatter(xy[:, 0], xy[:, 1], s=20, alpha=0.65, color="#285887")
    for word, (xx, yy) in zip(words[:30], xy[:30]):
        ax.annotate(word, (xx, yy), xytext=(3, 3), textcoords="offset points", fontsize=8)
    ax.set_title(f"{method.upper()} | window={model.window} | {len(words)} words (30 labelled)")
    ax.set_xlabel("Projection axis 1"); ax.set_ylabel("Projection axis 2")
    ax.text(0, -0.1, "Exploratory projection; axes of separately fitted models are not aligned.",
            transform=ax.transAxes, fontsize=9)
    fig.savefig(folder / f"{method}.png", dpi=160)
    plt.close(fig)


def experiment(a):
    import gensim
    import numpy as np
    from gensim.models import Word2Vec
    require_training_file(a.train_csv)
    rows = list(read_reviews(a.train_csv))
    if not rows:
        raise ValueError("Training CSV is empty")
    keys = [r.key() for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate training texts found; run prepare_data.py first")
    excluded = exclusion_keys(a.exclude_csv)
    overlap = len(set(keys) & excluded)
    if overlap:
        raise ValueError(f"{overlap} training texts also occur in development/test files")
    corpus = [r.tokens() for r in rows]
    counts = Counter(w for sentence in corpus for w in sentence)
    words = sorted((w for w in counts if counts[w] >= a.min_count), key=lambda w: (-counts[w], w))[:a.plot_words]
    if len(words) < 3:
        raise ValueError("Need at least 3 retained words. Increase training data or reduce --min-count")
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        raise ValueError("Output directory is not empty; choose a new run directory")
    a.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=a.output_dir / "training.log", level=logging.INFO,
                        format="%(asctime)s %(message)s", force=True)
    queries = QUERIES if a.query_file is None else [w.strip().lower() for w in a.query_file.read_text().splitlines() if w.strip()]
    write_json(a.output_dir / "run_manifest.json", {
        "status": "running", "training_name": a.train_csv.name, "training_sha256": sha256(a.train_csv),
        "rows": len(rows), "labels": dict(Counter(r.label for r in rows)),
        "token_count": sum(map(len, corpus)), "preprocessing": PREPROCESSING,
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items() if k != "exclude_csv"},
        "exclusion_files": [{"name": p.name, "sha256": sha256(p)} for p in a.exclude_csv],
        "overlap_count": overlap, "python": platform.python_version(), "gensim": gensim.__version__,
        "numpy": np.__version__, "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "packages": {p: importlib.metadata.version(p) for p in ['gensim','numpy','scipy','matplotlib','scikit-learn']},
        "fixed_parameters": FIXED_PARAMETERS,
        "label_encoding": {"1": "negative", "2": "positive"},
        "handoff": "window_5/embedding.npy + vocab.json + preprocessing.json",
    })
    results, pairs, neighbors = [], [], []
    for window in (2, 5):
        folder = a.output_dir / f"window_{window}"
        folder.mkdir()
        start = time.perf_counter()
        model = Word2Vec(vector_size=a.vector_size, window=window, min_count=a.min_count,
                         sg=int(a.algorithm == "skipgram"), seed=a.seed, **FIXED_PARAMETERS)
        model.build_vocab(corpus)
        model.train(corpus, total_examples=model.corpus_count, epochs=a.epochs, compute_loss=True)
        seconds = time.perf_counter() - start
        model.save(str(folder / "word2vec.model"))
        model.wv.save(str(folder / "vectors.kv"))
        export_handoff(model, folder, a.seed)
        project_vectors(model, words, a.projection, a.seed, folder)
        results.append({"window": window, "algorithm": a.algorithm, "vector_size": a.vector_size,
                        "epochs": a.epochs, "vocab_size": len(model.wv), "rows": len(rows),
                        "train_seconds": round(seconds, 4), "training_loss": model.get_latest_training_loss()})
        for left, right in PAIRS:
            ok = left in model.wv and right in model.wv
            pairs.append({"window": window, "word1": left, "word2": right,
                          "similarity": float(model.wv.similarity(left, right)) if ok else "",
                          "status": "ok" if ok else "oov"})
        for word in queries:
            if word not in model.wv:
                neighbors.append({"window": window, "query": word, "rank": "", "neighbor": "", "similarity": "", "status": "oov"})
                continue
            for rank, (near, score) in enumerate(model.wv.most_similar(word, topn=5), 1):
                neighbors.append({"window": window, "query": word, "rank": rank, "neighbor": near,
                                  "similarity": float(score), "status": "ok"})
        print(f"window={window}: {len(model.wv)} words, training {seconds:.2f}s", flush=True)
    write_table(a.output_dir / "comparison.csv", list(results[0]), results)
    write_table(a.output_dir / "word_pairs.csv", list(pairs[0]), pairs)
    write_table(a.output_dir / "neighbors.csv", ["window", "query", "rank", "neighbor", "similarity", "status"], neighbors)
    manifest_path = a.output_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = "complete"
    write_json(manifest_path, manifest)
    print(f"Complete. Results and the lab-2 handoff are in {a.output_dir}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-csv", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/run1"))
    p.add_argument("--algorithm", choices=["cbow", "skipgram"], default="cbow")
    p.add_argument("--vector-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--min-count", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--plot-words", type=int, default=200)
    p.add_argument("--projection", choices=["pca", "tsne"], default="pca")
    p.add_argument("--query-file", type=Path)
    p.add_argument("--exclude-csv", type=Path, nargs="*", default=[])
    a = p.parse_args()
    if a.vector_size < 2 or a.epochs < 1 or a.min_count < 1 or a.plot_words < 3 or not 0 <= a.seed <= 2**32-1:
        p.error("Invalid dimension, epochs, min-count, plot-words, or seed")
    # Gensim initialization also depends on Python's process-level string hash.
    if os.environ.get("PYTHONHASHSEED") != str(a.seed):
        env = dict(os.environ, PYTHONHASHSEED=str(a.seed))
        return subprocess.call([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], env=env)
    try:
        experiment(a)
    except (ValueError, OSError) as error:
        p.exit(1, f"Experiment failed: {error}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
