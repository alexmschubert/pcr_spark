"""Train the published three-head SPARK architecture on user-provided curves."""

import argparse
import csv
import json
import random
import re
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .model import SPARK
from .preprocess import GENE_ORDER, prepare_curve, render_curve


def load_rows(path: Path):
    """Read labeled curves; preserve all Fn cycles for training-only statistics."""
    with path.open(newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError("input CSV has no header")
        required = {"curve_id", "gene", "split", "groundtruth_target", "expert_call"}
        if not required.issubset(reader.fieldnames):
            raise ValueError(f"input CSV needs columns: {', '.join(sorted(required))}")
        fn_columns = {int(match.group(1)): field for field in reader.fieldnames
                      if (match := re.fullmatch(r"fn_(\d+)", field))}
        if sorted(fn_columns) != list(range(1, max(fn_columns, default=0) + 1)) or len(fn_columns) < 40:
            raise ValueError("input CSV needs contiguous fn_1 ... fn_N columns, N >= 40")
        fn_fields = [fn_columns[i] for i in sorted(fn_columns)]
        rows = []
        ids = set()
        for line, row in enumerate(reader, start=2):
            try:
                curve_id = row["curve_id"]
                if not curve_id or curve_id in ids:
                    raise ValueError("curve_id must be nonempty and unique")
                if "/" in curve_id or "\\" in curve_id or curve_id in (".", ".."):
                    raise ValueError("curve_id cannot contain path separators")
                ids.add(curve_id)
                if row["gene"] not in GENE_ORDER:
                    raise ValueError(f"unknown gene {row['gene']!r}")
                if row["split"] not in ("train", "val"):
                    raise ValueError("split must be 'train' or 'val'")
                y, expert = int(row["groundtruth_target"]), int(row["expert_call"])
                if y not in (0, 1) or expert not in (0, 1):
                    raise ValueError("groundtruth_target and expert_call must be 0 or 1")
                values = np.asarray([float(row[field]) for field in fn_fields], dtype=np.float64)
                if not np.isfinite(values).all():
                    raise ValueError("all Fn values must be finite")
                rows.append({"id": curve_id, "gene": row["gene"], "split": row["split"],
                             "values": values, "labels": (y, int(expert > y), int(expert < y)),
                             "sample_id": row.get("sample_id"), "plate_id": row.get("plate_id")})
            except (TypeError, ValueError) as error:
                raise ValueError(f"CSV line {line}: {error}") from error
    train = [row for row in rows if row["split"] == "train"]
    val = [row for row in rows if row["split"] == "val"]
    if not train or not val:
        raise ValueError("the CSV must include both train and val curves")
    for group in ("sample_id", "plate_id"):
        train_groups = {row[group] for row in train if row[group]}
        val_groups = {row[group] for row in val if row[group]}
        if train_groups & val_groups:
            raise ValueError(f"{group} overlaps between train and val")
    return train, val


def training_normalization(train):
    """Original rule: average each training curve's mean and population std."""
    mean = float(np.mean([row["values"].mean() for row in train]))
    std = float(np.mean([row["values"].std() for row in train]))
    if not np.isfinite(std) or std <= 0:
        raise ValueError("training curves have zero or invalid fluorescence variation")
    return mean, std


class CurveDataset(Dataset):
    def __init__(self, rows, image_dir: Path, mean: float, std: float):
        self.rows, self.image_dir, self.mean, self.std = rows, image_dir, mean, std

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = self.image_dir / f"curve_{row['id']}.png"
        image, sequence, gene = prepare_curve(row["values"], row["gene"], path,
                                               mean=self.mean, std=self.std)
        labels = torch.tensor(row["labels"], dtype=torch.float32)
        return image, sequence, gene, labels


def prepare_images(rows, image_dir: Path, existing: bool):
    """Render each image once unless an existing image directory was supplied."""
    if not existing:
        image_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        path = image_dir / f"curve_{row['id']}.png"
        if path.exists():
            continue
        if existing:
            raise FileNotFoundError(f"missing input image: {path}")
        render_curve(row["values"][:40]).save(path)


def run_epoch(model, loader, criterion, optimizer, device):
    training = optimizer is not None
    model.train(training)
    loss_total = 0.0
    correct = 0
    for image, sequence, gene, labels in loader:
        image, sequence, gene, labels = (item.to(device) for item in
                                          (image, sequence, gene, labels))
        if training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            outputs = model(image, sequence, gene)
            loss = sum(criterion(outputs[:, head], labels[:, head]) for head in range(3))
            if training:
                loss.backward()
                optimizer.step()
        loss_total += loss.item() * len(labels)
        correct += int(((outputs[:, 0] > 0.5) == labels[:, 0]).sum().item())
    return loss_total / len(loader.dataset), correct / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path,
                        help="CSV with labels, explicit train/val split, and Fn columns")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--image-dir", type=Path,
                        help="optional existing curve_<curve_id>.png directory")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--random-vit-init", action="store_true",
                        help="use random ViT weights instead of the published ImageNet initialization")
    args = parser.parse_args()
    if args.batch_size < 1 or args.epochs < 1 or args.learning_rate <= 0:
        parser.error("batch size, epochs, and learning rate must be positive")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = ("cuda" if torch.cuda.is_available() else
              "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else
              "cpu") if args.device == "auto" else args.device

    train, val = load_rows(args.input)
    mean, std = training_normalization(train)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = args.image_dir or args.output_dir / "rendered_images"
    prepare_images(train + val, image_dir, existing=args.image_dir is not None)
    (args.output_dir / "normalization.json").write_text(
        json.dumps({"mean": mean, "std": std, "gene_order": GENE_ORDER}, indent=2) + "\n")
    print(f"train={len(train)}, val={len(val)}, mean={mean:.8f}, std={std:.8f}, device={device}")

    train_loader = DataLoader(CurveDataset(train, image_dir, mean, std),
                              batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(CurveDataset(val, image_dir, mean, std),
                            batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = SPARK(pretrained_vit=not args.random_vit_init).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0)

    best_loss = float("inf")
    with (args.output_dir / "history.csv").open("w", newline="") as history_file:
        writer = csv.DictWriter(history_file,
                                fieldnames=("epoch", "train_loss", "val_loss", "val_accuracy"))
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            train_loss, _ = run_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_accuracy = run_epoch(model, val_loader, criterion, None, device)
            writer.writerow({"epoch": epoch, "train_loss": train_loss,
                             "val_loss": val_loss, "val_accuracy": val_accuracy})
            history_file.flush()
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), args.output_dir / "best_model.pth")
            print(f"epoch {epoch:02d}: train_loss={train_loss:.5f}, "
                  f"val_loss={val_loss:.5f}, val_accuracy={val_accuracy:.4f}", flush=True)
    torch.save(model.state_dict(), args.output_dir / "last_model.pth")


if __name__ == "__main__":
    main()
