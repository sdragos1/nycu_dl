from typing import Any

import numpy as np
import torch
from numpy import floating
from torch import nn
from torch.utils.data import DataLoader

from src.evaluator import Evaluation
from src.sample import sample_ddim
from src.noise_scheduler import LinearNoiseScheduler


def evaluate(model: nn.Module, scheduler: LinearNoiseScheduler, evaluator: Evaluation, val_loader: DataLoader,
             device: torch.device) -> floating[Any]:
    model.eval()

    accuracies = []
    with torch.no_grad():
        for batch_idx, (images, labels_t) in enumerate(val_loader):
            labels_t = labels_t.to(device)
            samples = sample_ddim(model, labels_t, scheduler, device)
            samples.clamp(min=-1, max=1)
            acc = evaluator.eval(samples, labels_t)
            accuracies.append(acc)

    return np.mean(accuracies)
