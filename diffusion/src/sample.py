import os

import hydra
import torch
from omegaconf import DictConfig
from torch import nn
from torchvision.utils import save_image

from src.ddpm.unet import UNet
from src.iclevr_dataset import ICLEVRDataset
from src.noise_scheduler import LinearNoiseScheduler

from src.utils import get_device


@torch.no_grad()
def sample_ddim(model: nn.Module, labels_t: torch.Tensor,
                noise_scheduler: LinearNoiseScheduler, device: torch.device,
                ddim_steps: int = 50, eta: float = 0.0) -> torch.Tensor:
    x_t = torch.randn((labels_t.size(0), 3, 64, 64)).to(device)

    steps_t = torch.linspace(noise_scheduler.total_steps - 1, 0, ddim_steps, device=device, dtype=torch.long)

    for i in range(len(steps_t) - 1):
        t = steps_t[i].item()
        t_next = steps_t[i + 1].item()
        timestamps_t = torch.full((labels_t.size(0),), t, device=device, dtype=torch.long)

        alpha_bar = noise_scheduler.alpha_bar[t]
        alpha_bar_next = noise_scheduler.alpha_bar[t_next]
        sqrt_alpha_bar = noise_scheduler.sqrt_alpha_bar[t]
        sqrt_alpha_bar_next = noise_scheduler.sqrt_alpha_bar[t_next]
        minus_sqrt_alpha_bar = noise_scheduler.minus_sqrt_alpha_bar[t]
        z = torch.randn_like(x_t)

        pred = model(x_t, timestamps_t, labels_t)

        sigma = eta * torch.sqrt((1 - alpha_bar_next) / (1 - alpha_bar) * (1 - alpha_bar / alpha_bar_next))
        direction = torch.sqrt(1 - alpha_bar_next - (sigma ** 2)) * pred
        x_0 = 1 / sqrt_alpha_bar * (x_t - minus_sqrt_alpha_bar * pred)

        x_t = sqrt_alpha_bar_next * x_0 + direction + sigma * z

    return x_t


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


@hydra.main(config_path="../conf", config_name="sample", version_base=None)
def main(cfg: DictConfig) -> None:
    if not cfg.labels:
        raise ValueError("No labels provided. Override with: labels=[\"gray cube\",\"red sphere\"]")

    device = get_device()
    os.makedirs(cfg.save_dir, exist_ok=True)

    noise_scheduler = LinearNoiseScheduler(
        beta_start=cfg.beta_start,
        beta_end=cfg.beta_end,
        total_steps=cfg.num_timesteps,
        device=device
    )

    objects = ICLEVRDataset(root=cfg.data_root or None).objects
    num_classes = len(objects)

    model = UNet(cfg.model).to(device)
    model.load_state_dict(torch.load(cfg.model_path, map_location=device))
    model.eval()

    print(f"Generating image for labels: {list(cfg.labels)}")

    label_t = torch.zeros(num_classes, dtype=torch.float32)
    for label in cfg.labels:
        if label not in objects:
            print(f"Warning: '{label}' not found in objects.json. Skipping.")
            continue
        label_t[objects[label]] = 1.0

    label_t = label_t.unsqueeze(0).to(device)

    gen_img = sample_ddim(model, label_t, noise_scheduler, device,
                          ddim_steps=cfg.ddim_steps, eta=cfg.eta)
    gen_img = (gen_img + 1) / 2

    save_name = f"{'_'.join(cfg.labels).replace(' ', '_')}.png"
    save_path = os.path.join(cfg.save_dir, save_name)
    save_image(gen_img, save_path)

    print(f"Saved to {save_path}")


if __name__ == '__main__':
    main()
