"""Score a user-supplied CSV of fluorescence curves with a local checkpoint."""

import argparse
import csv
from pathlib import Path

import torch

from .model import load_checkpoint
from .preprocess import TRAIN_MEAN, TRAIN_STD, prepare_curve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path,
                        help="CSV with curve_id, gene, fn_1 ... fn_40")
    parser.add_argument("--output", required=True, type=Path,
                        help="output CSV with predicted positive probabilities")
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="trusted local SPARK state dict; never uploaded")
    parser.add_argument("--trust-legacy-checkpoint", action="store_true",
                        help="allow pickle loading with old PyTorch; only for a trusted local file")
    parser.add_argument("--image-dir", type=Path,
                        help="optional archived PNGs named curve_<curve_id>.png")
    parser.add_argument("--mean", type=float, default=TRAIN_MEAN)
    parser.add_argument("--std", type=float, default=TRAIN_STD)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threshold", type=float,
                        help="optional prespecified threshold for binary calls")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.threshold is not None and not 0 < args.threshold < 1:
        parser.error("--threshold must be strictly between 0 and 1")
    device = ("cuda" if torch.cuda.is_available() else
              "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else
              "cpu") if args.device == "auto" else args.device
    model = load_checkpoint(args.checkpoint, device=device,
                            trust_legacy_pickle=args.trust_legacy_checkpoint)
    fields = [f"fn_{i}" for i in range(1, 41)]

    with args.input.open(newline="") as source, args.output.open("w", newline="") as target:
        reader = csv.DictReader(source)
        required = {"curve_id", "gene", *fields}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"input CSV needs columns: {', '.join(['curve_id', 'gene', *fields])}")
        output_columns = ["curve_id", "gene", "p_positive"]
        if args.threshold is not None:
            output_columns.append("positive_call")
        writer = csv.DictWriter(target, fieldnames=output_columns)
        writer.writeheader()

        batch = []

        def score_batch():
            if not batch:
                return
            images = torch.stack([item[2][0] for item in batch]).to(device)
            sequences = torch.stack([item[2][1] for item in batch]).to(device)
            genes = torch.stack([item[2][2] for item in batch]).to(device)
            with torch.inference_mode():
                probabilities = model(images, sequences, genes)[:, 0].cpu().tolist()
            for (curve_id, gene, _), probability in zip(batch, probabilities):
                result = {"curve_id": curve_id, "gene": gene,
                          "p_positive": f"{probability:.8f}"}
                if args.threshold is not None:
                    result["positive_call"] = int(probability >= args.threshold)
                writer.writerow(result)
            batch.clear()

        for row_number, row in enumerate(reader, start=2):
            curve_id = row["curve_id"]
            if not curve_id:
                raise ValueError(f"row {row_number}: curve_id is empty")
            if "/" in curve_id or "\\" in curve_id or curve_id in (".", ".."):
                raise ValueError(f"row {row_number}: curve_id cannot contain path separators")
            image_path = (args.image_dir / f"curve_{curve_id}.png"
                          if args.image_dir is not None else None)
            try:
                values = [float(row[field]) for field in fields]
                inputs = prepare_curve(values, row["gene"], image_path,
                                       mean=args.mean, std=args.std)
            except (ValueError, OSError) as error:
                raise ValueError(f"row {row_number} ({curve_id}): {error}") from error
            batch.append((curve_id, row["gene"], inputs))
            if len(batch) >= args.batch_size:
                score_batch()
        score_batch()


if __name__ == "__main__":
    main()
