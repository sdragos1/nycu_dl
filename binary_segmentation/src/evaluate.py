from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader

from utils import dice_score


def evaluate(model: nn.Module, val_loader: DataLoader, criterion: Callable, device: str) -> tuple[float, float]:
    model.eval()

    running_vloss = 0
    running_dice_score = 0
    with torch.no_grad():
        for i, vdata in enumerate(val_loader):
            images, masks = vdata
            images, masks = images.to(device), masks.to(device)
            pred_mask = model(images)
            loss = criterion(pred_mask, masks)
            running_vloss += loss.item()
            running_dice_score += dice_score(pred_mask, masks)
    n = len(val_loader)
    avg_vloss = running_vloss / n
    avg_dice_score = running_dice_score / n

    print("Validation: ", f"{avg_vloss:.4f}", " - ", f"{avg_dice_score:.4f}")
    return running_vloss / n, running_dice_score / n
