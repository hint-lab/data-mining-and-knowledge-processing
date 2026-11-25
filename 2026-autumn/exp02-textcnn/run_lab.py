"""TextCNN sentiment classification with two embedding initializations."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

LAB1 = Path(__file__).resolve().parents[1] / "exp01-word2vec"
sys.path.insert(0, str(LAB1))
from lab_common import PREPROCESSING, read_reviews, require_training_file, sha256, tokenize, write_json, write_table

MAX_LENGTH = 100
BATCH_SIZE = 64
KERNELS = (3, 4, 5)
LABELS = ("negative", "positive")


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.convolutions = nn.ModuleList(
            nn.Conv2d(1, 100, (width, embedding_size)) for width in KERNELS
        )
        self.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(300, 2))

    def forward(self, token_ids):
        embedded = self.embedding(token_ids).unsqueeze(1)
        # Mask windows that extend into padding before 1-max pooling.
        lengths = token_ids.ne(0).sum(dim=1)
        features = []
        for width, conv in zip(KERNELS, self.convolutions):
            activated = conv(embedded).squeeze(-1).relu()
            valid_starts = (lengths - width + 1).clamp_min(1)
            valid = torch.arange(activated.size(-1), device=token_ids.device) < valid_starts[:, None]
            pooled = activated.masked_fill(~valid[:, None, :], 0).amax(dim=-1)
            features.append(pooled)
        return self.classifier(torch.cat(features, dim=1))


def load_handoff(folder, train_file):
    vocab = json.loads((folder / "vocab.json").read_text(encoding="utf-8"))
    preprocessing = json.loads((folder / "preprocessing.json").read_text(encoding="utf-8"))
    manifest = json.loads((folder.parent / "run_manifest.json").read_text(encoding="utf-8"))
    if preprocessing != PREPROCESSING or manifest["training_sha256"] != sha256(train_file):
        raise ValueError("The vocabulary and vectors must come from Lab 1 trained on this exact training CSV.")
    ids = vocab["token_to_id"]
    tokens = vocab["id_to_token"]
    if (vocab["pad_id"], vocab["unk_id"]) != (0, 1) or ids != {t: i for i, t in enumerate(tokens)}:
        raise ValueError("Unexpected Lab 1 vocabulary mapping.")
    matrix = np.load(folder / "embedding.npy", allow_pickle=False)
    if (matrix.shape != (len(tokens), vocab["vector_size"])
            or not np.isfinite(matrix).all() or matrix[0].any()):
        raise ValueError("Invalid Lab 1 embedding matrix.")
    return vocab, torch.tensor(matrix, dtype=torch.float32)


def encode(tokens, vocab):
    ids = [vocab["token_to_id"].get(token, 1) for token in tokens[:MAX_LENGTH]]
    return ids + [0] * (MAX_LENGTH - len(ids))


def load_splits(train_file, dev_file, test_file):
    require_training_file(train_file)
    train, dev, test = (list(read_reviews(p)) for p in (train_file, dev_file, test_file))
    keys = lambda rows: {row.key() for row in rows}
    train_keys, test_keys = keys(train), keys(test)
    if train_keys & (keys(dev) | test_keys):
        raise ValueError("Training overlaps an evaluation split. Exclude both evaluation CSVs in Lab 1.")
    # The original dev/test files can share a review. Keep the test set intact.
    clean_dev = [row for row in dev if row.key() not in test_keys]
    for name, rows in (("train", train), ("dev", clean_dev), ("test", test)):
        if {row.label for row in rows} != {1, 2}:
            raise ValueError(f"{name} must contain both Amazon labels 1 and 2.")
    return train, clean_dev, test, len(dev) - len(clean_dev)


def dataset(rows, vocab):
    ids = torch.tensor([encode(row.tokens(), vocab) for row in rows], dtype=torch.long)
    labels = torch.tensor([row.label - 1 for row in rows], dtype=torch.long)
    return TensorDataset(ids, labels)


def metrics(target, prediction):
    matrix = torch.bincount(target * 2 + prediction, minlength=4).reshape(2, 2)
    true_positive = matrix.diag()
    totals = matrix.sum(0) + matrix.sum(1)
    f1 = 2 * true_positive / totals.clamp_min(1)
    return {"accuracy": float(true_positive.sum() / matrix.sum()),
            "macro_f1": float(f1.mean()), "confusion_matrix": matrix.tolist()}


@torch.inference_mode()
def evaluate(model, loader, loss_fn):
    model.eval()
    total = 0.0
    predictions, targets = [], []
    for x, y in loader:
        logits = model(x)
        total += loss_fn(logits, y).item() * y.size(0)
        predictions.append(logits.argmax(dim=1))
        targets.append(y)
    target, prediction = torch.cat(targets), torch.cat(predictions)
    return {**metrics(target, prediction), "loss": total / len(target)}, prediction.tolist()


def train_model(name, data, initial, vocab, epochs, seed):
    train, dev, test = data
    # Reset model, dropout and minibatch ordering for each arm.
    torch.manual_seed(seed)
    model = TextCNN(len(vocab["id_to_token"]), vocab["vector_size"])
    if initial is not None:
        with torch.no_grad():
            # Share the random UNK row and zero PAD row between A and B.
            model.embedding.weight[2:].copy_(initial[2:])
    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True,
                              generator=torch.Generator().manual_seed(seed))
    dev_loader = DataLoader(dev, batch_size=BATCH_SIZE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_score = -1.0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        for x, y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * y.size(0)
        score, _ = evaluate(model, dev_loader, loss_fn)
        history.append({"model": name, "epoch": epoch, "train_loss": loss_sum / len(train),
                        "dev_loss": score["loss"], "dev_macro_f1": score["macro_f1"]})
        print(f"{name} epoch {epoch}: dev macro-F1 {score['macro_f1']:.4f}", flush=True)
        if score["macro_f1"] > best_score:
            best_score, best_epoch = score["macro_f1"], epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    score, predictions = evaluate(model, DataLoader(test, batch_size=BATCH_SIZE), loss_fn)
    return model, {"best_dev_epoch": best_epoch, **score}, predictions, history


def plot_history(history, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for name in ("A_random", "B_word2vec"):
        rows = [row for row in history if row["model"] == name]
        for ax, key, ylabel in zip(axes, ("train_loss", "dev_macro_f1"), ("Train loss", "Dev Macro-F1")):
            ax.plot([row["epoch"] for row in rows], [row[key] for row in rows], marker="o", label=name)
            ax.set(xlabel="Epoch", ylabel=ylabel)
            ax.legend()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def run(args):
    train, dev, test, removed = load_splits(args.train_csv, args.dev_csv, args.test_csv)
    vocab, embedding = load_handoff(args.handoff, args.train_csv)
    data = tuple(dataset(rows, vocab) for rows in (train, dev, test))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {"counts": {"train": len(train), "dev": len(dev), "test": len(test),
                         "dev_removed_test_overlap": removed},
              "label_order": list(LABELS), "seed": args.seed, "epochs": args.epochs,
              "input_sha256": {n: sha256(f) for n, f in (("train", args.train_csv),
                                ("dev", args.dev_csv), ("test", args.test_csv),
                                ("embedding", args.handoff / "embedding.npy"))}, "models": {}}
    history, predictions = [], {}
    for name, initial in (("A_random", None), ("B_word2vec", embedding)):
        model, report["models"][name], predictions[name], series = train_model(
            name, data, initial, vocab, args.epochs, args.seed
        )
        torch.save(model.state_dict(), args.output_dir / f"{name}.pt")
        history.extend(series)
    errors = []
    for review, a, b in zip(test, predictions["A_random"], predictions["B_word2vec"]):
        gold = review.label - 1
        if a == b == gold:
            continue
        tokens = review.tokens()[:MAX_LENGTH]
        errors.append({"gold": LABELS[gold], "A": LABELS[a], "B": LABELS[b], "title": review.title,
                       "text": review.text, "oov_count": sum(t not in vocab["token_to_id"] for t in tokens)})
    write_json(args.output_dir / "metrics.json", report)
    write_json(args.output_dir / "vocab.json", vocab)
    write_table(args.output_dir / "history.csv", ["model", "epoch", "train_loss", "dev_loss", "dev_macro_f1"], history)
    write_table(args.output_dir / "errors.csv", ["gold", "A", "B", "title", "text", "oov_count"], errors)
    plot_history(history, args.output_dir / "curves.png")
    print(json.dumps(report, ensure_ascii=False, indent=2))


@torch.inference_mode()
def predict(args):
    vocab = json.loads((args.output_dir / "vocab.json").read_text(encoding="utf-8"))
    tokens = tokenize(args.text)
    if not tokens:
        raise ValueError("Enter an English review with at least one token.")
    x = torch.tensor([encode(tokens, vocab)], dtype=torch.long)
    for name in ("A_random", "B_word2vec"):
        model = TextCNN(len(vocab["id_to_token"]), vocab["vector_size"])
        model.load_state_dict(torch.load(args.output_dir / f"{name}.pt", map_location="cpu", weights_only=True))
        probabilities = model.eval()(x).softmax(dim=-1)[0]
        print(name, LABELS[int(probabilities.argmax())],
              {label: round(float(value), 4) for label, value in zip(LABELS, probabilities)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    for name in ("train-csv", "dev-csv", "test-csv", "handoff", "output-dir"):
        train.add_argument(f"--{name}", required=True, type=Path)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--epochs", type=int, default=5)
    inference = sub.add_parser("predict")
    inference.add_argument("--output-dir", type=Path, required=True)
    inference.add_argument("text")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.command == "train":
        if args.epochs < 1:
            parser.error("--epochs must be at least 1")
        run(args)
    else:
        predict(args)


if __name__ == "__main__":
    main()
