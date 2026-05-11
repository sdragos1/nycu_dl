import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.evaluator import Evaluation
from src.sample import sample
from src.noise_scheduler import LinearNoiseScheduler


def evaluate(model: nn.Module, scheduler: LinearNoiseScheduler, val_loader: DataLoader, device: torch.device) -> float:
    model.eval()

    accuracies = []
    evaluator = Evaluation()
    with torch.no_grad():
        for batch_idx, (images, labels_t) in enumerate(val_loader):
            labels_t = labels_t.to(device)
            samples = sample(model, labels_t, scheduler, device)
            acc = evaluator.eval(samples, labels_t)
            accuracies.append(acc)

    return np.sum(accuracies) / len(accuracies)
