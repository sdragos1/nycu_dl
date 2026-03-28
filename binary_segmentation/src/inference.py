import argparse
import os

import numpy as np
import torch

from oxford_pet import get_test_dataloader
from constants import DATASET_DIR
from utils import dice_score


def inference(model_path: str, device: str, batch_size: int):
    if not os.path.exists(model_path):
        raise FileNotFoundError

    test_loader = get_test_dataloader(DATASET_DIR, batch_size)
    model = torch.load(model_path)
    model.to(device)

    scores = []
    inf_row: tuple[str, str]

    for i, data in enumerate(test_loader):
        images, masks = data
        images = images.to(device)
        masks = masks.to(device)

        preds = model(images)
        score = dice_score(preds, masks)
        scores.append(score)
    print("Scores: ", np.mean(scores))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', required=True)
    parser.add_argument('-d', '--device', required=True)
    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    args = parse_args()
    inference(args.model_path, args.device, 32)
