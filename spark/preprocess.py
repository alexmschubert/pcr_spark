"""Input transformations used by the released SPARK checkpoint."""

from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torchvision import transforms


GENE_ORDER = ("E gene", "MS2", "N gene", "ORF1ab", "RnaseP", "S gene")
# Mean of the per-curve means and mean of the per-curve standard deviations,
# computed on the original training split. They are not patient-level data.
TRAIN_MEAN = 155626.8370536778
TRAIN_STD = 94477.0057018847

_IMAGE_TRANSFORM = transforms.Compose([
    transforms.Lambda(lambda image: image.convert("RGB")),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])


def render_curve(curve: np.ndarray) -> Image.Image:
    """Render the original 640x480, blue-line, axes-free Matplotlib image.

    Matplotlib's default figure size (6.4 x 4.8 inches), 100 dpi, line colour,
    autoscaling, and margins are intentional. These settings exactly reproduced
    an archived training image in a pixel comparison using Matplotlib 3.10.9.
    Different Matplotlib versions may make small rasterization differences.
    """
    figure, axis = plt.subplots(figsize=(6.4, 4.8), dpi=100)
    axis.plot(curve, linewidth=6)
    axis.axis("off")
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=100)
    plt.close(figure)
    buffer.seek(0)
    with Image.open(buffer) as image:
        return image.convert("RGB")


def encode_gene(gene: str) -> torch.Tensor:
    """Create the six-element one-hot vector; unknown genes are rejected."""
    if gene not in GENE_ORDER:
        raise ValueError(f"unknown gene {gene!r}; expected one of {GENE_ORDER}")
    encoded = torch.zeros(len(GENE_ORDER), dtype=torch.float32)
    encoded[GENE_ORDER.index(gene)] = 1.0
    return encoded


def prepare_curve(values, gene: str, image_path: str | Path | None = None,
                  mean: float = TRAIN_MEAN, std: float = TRAIN_STD):
    """Convert raw Fn values into the three model inputs (without a batch axis).

    Only cycles 1-40 enter the model and image. Pass an existing PNG to reproduce
    an archived evaluation exactly; otherwise it is rendered from the raw curve.
    For a newly trained model, supply normalization values fitted on its own
    training split. Never estimate them from the evaluation set.
    """
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size < 40 or not np.isfinite(values[:40]).all():
        raise ValueError("a curve needs at least 40 finite Fn values")
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        raise ValueError("mean must be finite and std must be positive")
    first_40 = values[:40]
    sequence = (torch.tensor(first_40, dtype=torch.float32) -
                torch.tensor(mean, dtype=torch.float32)) / torch.tensor(std, dtype=torch.float32)
    if image_path is None:
        image = render_curve(first_40)
    else:
        with Image.open(image_path) as source:
            image = source.copy()
    return _IMAGE_TRANSFORM(image), sequence.unsqueeze(-1), encode_gene(gene)
