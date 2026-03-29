import argparse
import csv
import os

import numpy as np
import torch
from torch import nn

from constants import DATASET_DIR, INFERENCE_DIR
from models import get_model
from oxford_pet import get_test_dataloader
from utils import dice_score, rle_encode

DEFAULT_TEST_SPLIT_FILE = DATASET_DIR / "annotations" / "test.txt"


def _load_model(model_name: str, model_path: str, device: str) -> nn.Module:
    ckpt = torch.load(model_path, map_location=device)

    if isinstance(ckpt, nn.Module):
        model = ckpt
    else:
        model = get_model(model_name, out_channels=1)
        if model is None:
            raise ValueError("Failed to initialize model")

        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt

        if not isinstance(state_dict, dict):
            raise ValueError("Unsupported checkpoint fmt")

        cleaned_state_dict = {
            key.replace("_orig_mod.", "", 1): value
            for key, value in state_dict.items()
        }
        model.load_state_dict(cleaned_state_dict)

    model.to(device)
    model.eval()
    return model


def inference(dataset: str, model_name: str, model_path: str, device: str, batch_size: int, split: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError

    test_loader = get_test_dataloader(dataset, split, batch_size)
    model = _load_model(model_name, model_path, device)

    scores = []
    rows: list[tuple[str, str]] = []

    with torch.no_grad():
        for _, data in enumerate(test_loader):
            images, masks, image_ids = data
            images = images.to(device)
            masks = masks.to(device)

            preds = model(images)
            score = dice_score(preds, masks)
            scores.append(score.item())

            for idx, image_id in enumerate(image_ids):
                orig_image_path = DATASET_DIR / "images" / f"{image_id}.jpg"
                from PIL import Image
                with Image.open(orig_image_path) as orig_img:
                    orig_w, orig_h = orig_img.size

                pred_tensor = preds[idx: idx + 1]
                pred_resized = torch.nn.functional.interpolate(
                    pred_tensor,
                    size=(orig_h, orig_w),
                    mode='bilinear',
                    align_corners=False
                )

                pred_mask = (torch.sigmoid(pred_resized) > 0.5).to(torch.uint8).cpu().numpy()[0, 0]

                encoded_mask = rle_encode(pred_mask)
                rows.append((image_id, encoded_mask))

    print("Scores: ", np.mean(scores))

    INFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    output_path = INFERENCE_DIR / f"inference_{model_name}.csv"

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "encoded_mask"])
        writer.writerows(rows)

    print(f"Saved submission file to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run binary segmentation inference and export submission CSV."
    )
    parser.add_argument(
        "-m",
        "--model",
        required=True,
        default="unet",
        type=str,
        help="The model to run inference on.",
        choices=["unet", "resnet"]
    )
    parser.add_argument(
        "-p",
        "--model-path",
        dest="model_path",
        required=True,
        help="Path to trained model checkpoint (.pth).",
    )
    parser.add_argument(
        "-d",
        "--device",
        default="cuda",
        help="Device to run inference on (e.g. cuda, cpu).",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--split-file-path",
        "--split",
        dest="split_file_path",
        default=str(DEFAULT_TEST_SPLIT_FILE),
        help="Path to split txt file listing test image ids.",
    )
    parser.add_argument('--dataset', type=str, default=DATASET_DIR, help="Path to dataset.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    inference(args.dataset, args.model, args.model_path, args.device, args.batch_size, args.split_file_path)
