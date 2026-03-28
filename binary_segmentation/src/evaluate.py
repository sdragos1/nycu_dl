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
    print(f"Val loss: {running_vloss} | Dice score: {running_dice_score}")
    return running_vloss, running_dice_score
