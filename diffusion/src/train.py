import argparse
import os

import torch
import wandb
from torch.nn import MSELoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from src.ddpm.unet import UNet
from src.evaluate import evaluate
from src.iclevr_dataset import train_val_data_loaders
from src.noise_scheduler import LinearNoiseScheduler
from src.utils import seed_all, get_device


def sample_timesteps(batch_size: int, num_timesteps: int, device: torch.device) -> torch.Tensor:
    return torch.randint(low=1, high=num_timesteps, size=(batch_size,), device=device).long()


def train(batch_size: int, beta_start: float, beta_end: float, num_timesteps: int, lr: float, weight_decay: float,
          epochs: int, save_dir: str):
    device = get_device()
    os.makedirs(save_dir, exist_ok=True)

    train_loader, val_loader = train_val_data_loaders(batch_size)
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

    step = 0
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        best_val_acc = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for i, data in enumerate(pbar):
            images_t, labels_t = data

            curr_batch_size = images_t.shape[0]

            images_t = images_t.to(device)
            labels_t = labels_t.to(device)
            timesteps_t = sample_timesteps(curr_batch_size, num_timesteps, device)
            images_t, noises_t = noise_scheduler.noise(images_t, timesteps_t)

            optimizer.zero_grad()
            pred = model(images_t, timesteps_t, labels_t)
            loss = criterion(pred, noises_t)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            step += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.6f}"})
            wandb.log({
                "train/loss": loss.item(),
                "train/lr": optimizer.param_groups[0]['lr'],
                "epoch": epoch + 1,
                "step": step
            })
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch + 1} finished. Average Loss: {avg_epoch_loss:.4f}")
        wandb.log({"train/epoch_loss": avg_epoch_loss, "epoch": epoch + 1})
        scheduler.step(avg_epoch_loss)

        val_acc = evaluate(model, noise_scheduler, val_loader, device)
        wandb.log({"val/val_acc": val_acc, "epoch": epoch + 1, "step": step})
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = os.path.join(save_dir, f"model_epoch_{epoch + 1}.pt")
            torch.save(model.state_dict(), checkpoint_path)


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
        batch_size=16,
        beta_start=1e-4,
        beta_end=2e-2,
        num_timesteps=1000,
        lr=1e-3,
        weight_decay=1e-5,
        epochs=10,
        save_dir=f'results/{run_name}',
    )
