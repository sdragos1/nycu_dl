import numpy as np
import torch
from torch import Tensor


def dice_score(logits: Tensor,
               targets: Tensor,
               threshold: float = 0.5,
               eps: float = 1e-6) -> Tensor:
    preds = (torch.sigmoid(logits) > threshold).float()

    targets = targets.float()

    preds_flat = preds.view(preds.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)

    intersection = (preds_flat * targets_flat).sum(dim=1)
    union = preds_flat.sum(dim=1) + targets_flat.sum(dim=1)

    dice = (2.0 * intersection + eps) / (union + eps)
    return dice.mean()


def rle_encode(targets: np.ndarray) -> str:
    targets = targets.flatten(order="F").astype(np.uint8)
    pixels = np.concatenate([[0], targets, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)
