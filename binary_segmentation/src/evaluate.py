from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader


def evaluate(model: nn.Module, val_loader: DataLoader, criterion: Callable, device: str) -> float:
    model.eval()

    running_vloss = 0
    with torch.no_grad():
        for i, vdata in enumerate(val_loader):
            images, masks = vdata
            images, masks = images.to(device), masks.to(device)
            pred_mask = model(images)
            loss = criterion(pred_mask, masks)
            running_vloss += loss.item()
    return running_vloss
