"""Released SPARK architecture: ViT-B/32 + LSTM + gene + fluorescence range.

The three heads predict ground truth, laboratory false-positive error, and
laboratory false-negative error, respectively. Only head 0 is used for the
paper's positive/negative calls. The auxiliary heads are not required at
inference and do not feed back into head 0.
"""

from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn
from torchvision.models import vit_b_32


class SPARK(nn.Module):
    """The exact dimensions and module names of the released three-head model."""

    def __init__(self, pretrained_vit: bool = False):
        super().__init__()
        # The released checkpoint contains the fine-tuned ViT weights.
        # ImageNet initialization is an explicit choice for new training runs.
        self.vit = vit_b_32(weights="IMAGENET1K_V1" if pretrained_vit else None)
        self.vit_classifier = nn.Linear(1000, 512)
        self.lstm = nn.LSTM(1, 512, num_layers=3, batch_first=True)
        self.lstm_fc = nn.Linear(512, 512)
        # 512 image + 512 sequence + 6 gene + 64 repeated fluorescence range.
        self.fc = nn.Sequential(
            nn.Linear(1094, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.heads = nn.ModuleList([nn.Linear(64, 1) for _ in range(3)])

    def forward(self, image: torch.Tensor, sequence: torch.Tensor,
                gene: torch.Tensor) -> torch.Tensor:
        """Return probabilities [batch, 3], with ground-truth risk in column 0.

        image: [batch, 3, 224, 224], sequence: [batch, 40, 1],
        gene: [batch, 6] in the order documented in preprocess.py.
        """
        if image.ndim != 4 or image.shape[1:] != (3, 224, 224):
            raise ValueError("image must have shape [batch, 3, 224, 224]")
        if sequence.ndim != 3 or sequence.shape[1:] != (40, 1):
            raise ValueError("sequence must have shape [batch, 40, 1]")
        if gene.ndim != 2 or gene.shape[1] != 6:
            raise ValueError("gene must have shape [batch, 6]")
        if image.shape[0] != sequence.shape[0] or image.shape[0] != gene.shape[0]:
            raise ValueError("image, sequence, and gene batch sizes must agree")

        image_features = self.vit_classifier(self.vit(image))
        sequence_features = self.lstm_fc(self.lstm(sequence)[0][:, -1, :])
        fluorescence_range = (sequence.max(dim=1).values -
                              sequence.min(dim=1).values).expand(-1, 64)
        fused = self.fc(torch.cat((image_features, sequence_features,
                                   gene, fluorescence_range), dim=1))
        return torch.cat([torch.sigmoid(head(fused)) for head in self.heads], dim=1)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu",
                    trust_legacy_pickle: bool = False) -> SPARK:
    """Load a local state dict, with strict architecture verification.

    Modern PyTorch uses its restricted weights-only loader. PyTorch 1.12
    requires an explicit opt-in because its loader can execute pickle code.
    """
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch 1.12, used by the original environment
        if not trust_legacy_pickle:
            raise RuntimeError(
                "This PyTorch version cannot load checkpoints with weights_only=True. "
                "Upgrade PyTorch, or explicitly trust this local checkpoint by "
                "setting trust_legacy_pickle=True. Never trust a downloaded or "
                "unverified pickle checkpoint."
            ) from None
        state = torch.load(path, map_location="cpu")
    if isinstance(state, Mapping) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint must contain a PyTorch state dict")
    model = SPARK()
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()
