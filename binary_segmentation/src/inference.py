import argparse
import os

import numpy as np
import torch

from constants import DATASET_DIR, INFERENCE_DIR
from oxford_pet import get_test_dataloader
from utils import dice_score, rle_encode


def inference(model_path: str, device: str, batch_size: int):
    if not os.path.exists(model_path):
        raise FileNotFoundError

    test_loader = get_test_dataloader(DATASET_DIR, batch_size)
    model = torch.load(model_path)
    model.to(device)

    scores = []
    inf = list[tuple[str, str]]

    for i, data in enumerate(test_loader):
        images, masks, image_ids = data
        images = images.to(device)
        masks = masks.to(device)
        inf_row = (image_ids, rle_encode(masks))
        inf.append(inf_row)

        preds = model(images)
        score = dice_score(preds, masks)
        scores.append(score)
    print("Scores: ", np.mean(scores))

    model_name = model_path.split('/')[-1]
    with open(INFERENCE_DIR / f"inference_{model_name}.csv", 'r') as f:
        f.write('image_id,encoded_mask')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', required=True)
    parser.add_argument('-d', '--device', required=True)
    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    args = parse_args()
    inference(args.model_path, args.device, 32)
