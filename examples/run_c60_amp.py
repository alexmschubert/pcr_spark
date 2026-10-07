"""Download public C60.amp qPCR curves and score them with a local SPARK model.

Requires Rscript (base R only) to read the .rda file in chipPCR's source archive.
The source archive, converted CSV, and predictions stay in --output-dir.
No model weights or study data are downloaded by this script.
"""

import argparse
import csv
import hashlib
import json
import subprocess
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import torch

from spark.model import load_checkpoint
from spark.preprocess import prepare_curve


SOURCE_URL = "https://cran.r-project.org/src/contrib/chipPCR_1.0-2.tar.gz"
SOURCE_SHA256 = "3d6071a894958bd6447043fee502756d791191595ef4dcefc2fbc6b52db36060"
DATA_MEMBER = "chipPCR/data/C60.amp.rda"
GENE_ENCODING = "E gene"  # Historical external-validation convention, not the biological gene.


def source_archive(output_dir: Path, local_archive: Path | None) -> Path:
    archive = local_archive or output_dir / "chipPCR_1.0-2.tar.gz"
    if not archive.exists() and local_archive is None:
        print(f"Downloading {SOURCE_URL}", flush=True)
        with urllib.request.urlopen(SOURCE_URL, timeout=60) as response, archive.open("wb") as target:
            while chunk := response.read(1024 * 1024):
                target.write(chunk)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError(f"unexpected chipPCR archive SHA-256: {digest}")
    return archive


def convert_source(archive: Path, output_dir: Path, rscript: str) -> Path:
    rda = output_dir / "C60.amp.rda"
    with tarfile.open(archive, "r:gz") as source:
        member = source.extractfile(DATA_MEMBER)
        if member is None:
            raise ValueError(f"{DATA_MEMBER} missing from the pinned archive")
        with member, rda.open("wb") as target:
            target.write(member.read())
    csv_path = output_dir / "C60.amp.csv"
    expression = (
        "args <- commandArgs(TRUE); "
        "loaded <- load(args[1]); "
        "stopifnot(identical(loaded, 'C60.amp')); "
        "write.csv(C60.amp, args[2], row.names=FALSE)"
    )
    subprocess.run([rscript, "--vanilla", "-e", expression, str(rda), str(csv_path)], check=True)
    return csv_path


def read_curves(path: Path):
    """Return the 32 raw, 45-cycle traces with their public dilution labels."""
    with path.open(newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or reader.fieldnames[0] != "Index":
            raise ValueError("unexpected C60.amp columns")
        names = reader.fieldnames[1:]
        rows = list(reader)
    if len(rows) != 45 or len(names) != 32 or [int(row["Index"]) for row in rows] != list(range(45)):
        raise ValueError("expected 32 curves, each with cycles indexed 0-44")
    curves = []
    for name in names:
        parts = name.split(".")
        if len(parts) != 3 or parts[0] not in ("Vim", "MLC2v") or not parts[1].isdigit():
            raise ValueError(f"unexpected reaction name: {name}")
        values = np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite fluorescence in {name}")
        curves.append({"id": name, "target": "Vimentin" if parts[0] == "Vim" else "MLC-2v",
                       "dilution": int(parts[1]), "known_positive": int(parts[1] != "0"),
                       "values": values})
    if sum(curve["target"] == "Vimentin" for curve in curves) != 16:
        raise ValueError("expected 16 Vimentin and 16 MLC-2v reactions")
    return curves


def score(curves, checkpoint: Path, device: str, batch_size: int, trust_legacy: bool):
    """Use all 32 unlabeled curves for the original external-set normalization."""
    mean = float(np.mean([curve["values"].mean() for curve in curves]))
    std = float(np.mean([curve["values"].std() for curve in curves]))
    model = load_checkpoint(checkpoint, device=device, trust_legacy_pickle=trust_legacy)
    probabilities = []
    for start in range(0, len(curves), batch_size):
        batch = curves[start:start + batch_size]
        inputs = [prepare_curve(curve["values"], GENE_ENCODING, mean=mean, std=std)
                  for curve in batch]
        images = torch.stack([item[0] for item in inputs]).to(device)
        sequences = torch.stack([item[1] for item in inputs]).to(device)
        genes = torch.stack([item[2] for item in inputs]).to(device)
        with torch.inference_mode():
            probabilities.extend(model(images, sequences, genes)[:, 0].cpu().tolist())
    return probabilities, mean, std


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="local SPARK state dict")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/c60_amp"))
    parser.add_argument("--archive", type=Path, help="optional already downloaded chipPCR archive")
    parser.add_argument("--rscript", default="Rscript", help="path to Rscript executable")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threshold", type=float, help="optional prespecified classification threshold")
    parser.add_argument("--trust-legacy-checkpoint", action="store_true",
                        help="allow pickle loading on old PyTorch only for a trusted local checkpoint")
    args = parser.parse_args()
    if args.batch_size < 1 or (args.threshold is not None and not 0 < args.threshold < 1):
        parser.error("batch size must be positive and threshold must be in (0, 1)")
    device = ("cuda" if torch.cuda.is_available() else
              "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else
              "cpu") if args.device == "auto" else args.device
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = source_archive(args.output_dir, args.archive)
    curves = read_curves(convert_source(archive, args.output_dir, args.rscript))
    probabilities, mean, std = score(curves, args.checkpoint, device, args.batch_size,
                                     args.trust_legacy_checkpoint)

    output = args.output_dir / "predictions.csv"
    columns = ["curve_id", "assay_target", "dilution", "known_positive", "p_positive"]
    if args.threshold is not None:
        columns.append("positive_call")
    with output.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=columns)
        writer.writeheader()
        for curve, probability in zip(curves, probabilities):
            row = {"curve_id": curve["id"], "assay_target": curve["target"],
                   "dilution": curve["dilution"], "known_positive": curve["known_positive"],
                   "p_positive": f"{probability:.8f}"}
            if args.threshold is not None:
                row["positive_call"] = int(probability >= args.threshold)
            writer.writerow(row)
    (args.output_dir / "normalization.json").write_text(
        json.dumps({"mean": mean, "std": std, "gene_encoding": GENE_ENCODING}, indent=2) + "\n")
    print(f"Scored {len(curves)} curves (16 Vimentin, 16 MLC-2v). Results: {output}")
    print(f"External-set normalization: mean={mean:.8f}, std={std:.8f}")


if __name__ == "__main__":
    main()
