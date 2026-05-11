import argparse

import torch
import wandb
from torch.nn import MSELoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.ddpm.unet import UNet
from src.iclevr_dataset import train_data_loader
from src.noise_scheduler import LinearNoiseScheduler
from src.utils import seed_all, get_device


def sample_timesteps(batch_size: int, num_timesteps: int, device: torch.device) -> torch.Tensor:
    return torch.randint(low=1, high=num_timesteps, size=(batch_size,), device=device).long()


def train(batch_size: int, beta_start: float, beta_end: float, num_timesteps: int, lr: float, weight_decay: float,
          epochs: int, save_dir: str):
    device = get_device()

    train_loader = train_data_loader(batch_size)
    noise_scheduler = LinearNoiseScheduler(beta_start, beta_end, num_timesteps, device)
    criterion = MSELoss()
    model = UNet(24).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=5,
    )

    for epoch in range(epochs):
        print('EPOCH {}:'.format(epoch + 1))
        for i, data in enumerate(train_loader):
            images_t, labels_t = data
            images_t = images_t.to(device)
            labels_t = labels_t.to(device)
            timesteps_t = sample_timesteps(batch_size, num_timesteps, device)
            images_t, noises_t = noise_scheduler.noise(images_t, timesteps_t)

            optimizer.zero_grad()
            pred = model(images_t, timesteps_t, labels_t)
            loss = criterion(pred, noises_t)
            loss.backward()
            optimizer.step()
            scheduler.step(loss)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train UNet')
    parser.add_argument("--disable-wandb", action="store_true", default=False, help="Disable wandb")
    return parser.parse_args()


if __name__ == "__main__":
    seed = 42
    run_name = f"unet-{str(seed)}"

    args = parse_args()
    wandb_mode = 'online'
    if args.disable_wandb:
        wandb_mode = 'disabled'
    wandb.init(project="DL-DDPM", name=f"unet-{str(seed)}", save_code=True, mode=wandb_mode)
    seed_all(seed)
    train(
        batch_size=8,
        beta_start=1e-4,
        beta_end=2e-2,
        num_timesteps=1000,
        lr=1e-3,
        weight_decay=1e-5,
        epochs=10,
        save_dir=f'results/{run_name}',
    )
