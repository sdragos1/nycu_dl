import argparse
import os

import torch
from torch import nn
from torchvision.utils import save_image
from tqdm import tqdm

from src.ddpm.unet import UNet
from src.iclevr_dataset import ICLEVRDataset
from src.noise_scheduler import LinearNoiseScheduler

from src.utils import get_device


@torch.no_grad()
def sample(model: nn.Module, labels_t: torch.Tensor,
           noise_scheduler: LinearNoiseScheduler, device: torch.device) -> torch.Tensor:
    x = torch.randn((labels_t.size(0), 3, 64, 64)).to(device)

    for t in reversed(range(noise_scheduler.total_steps)):
        timestamps_t = torch.full((labels_t.size(0),), t, device=device, dtype=torch.long)

        pred = model(x, timestamps_t, labels_t)
        alpha = noise_scheduler.alpha[t]

        sigma = torch.sqrt(noise_scheduler.beta[t])
        minus_sqrt_alpha_bar = noise_scheduler.minus_sqrt_alpha_bar[t]
        mean = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / minus_sqrt_alpha_bar) * pred)

        if t > 0:
            noise = torch.randn_like(x)
            x = mean + sigma * noise
        else:
            x = mean
    return x


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Sample images from trained UNet')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the trained model checkpoint (.pt)')
    parser.add_argument('--save-dir', type=str, default='results/samples', help='Directory to save generated images')
    parser.add_argument('--labels', nargs='+', type=str, required=True,
                        help='Custom labels to generate (e.g. --labels "gray cube" "red sphere")')
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()

    noise_scheduler = LinearNoiseScheduler(beta_start=1e-4, beta_end=2e-2, total_steps=1000, device=device)

    print(f"Generating image for custom labels: {args.labels}")
    conditions = [args.labels]
    objects = ICLEVRDataset().objects
    num_classes = len(objects)

    os.makedirs(args.save_dir, exist_ok=True)

    model = UNet(24).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    for i, labels in enumerate(tqdm(conditions)):
        label_t = torch.zeros(num_classes, dtype=torch.float32)
        for label in labels:
            if label not in objects:
                print(f"\nWarning: Label '{label}' not found in objects.json. Skipping this label.")
                continue
            label_t[objects[label]] = 1.0

        label_t = label_t.unsqueeze(0).to(device)

        gen_img = sample(model, label_t, noise_scheduler, device)
        gen_img = (gen_img + 1) / 2

        if args.labels:
            save_name = f"custom_{'_'.join(args.labels).replace(' ', '_')}.png"
            save_path = os.path.join(args.save_dir, save_name)
        else:
            save_path = os.path.join(args.save_dir, f'sample_{i}.png')

        save_image(gen_img, save_path)

    print(f"Finished! Generated images are saved in {args.save_dir}")


if __name__ == '__main__':
    main()
