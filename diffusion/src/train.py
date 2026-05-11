import os

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.nn import MSELoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from src.ddpm.unet import UNet
from src.evaluate import evaluate
from src.evaluator import Evaluation
from src.iclevr_dataset import train_val_data_loaders
from src.noise_scheduler import LinearNoiseScheduler
from src.utils import seed_all, get_device, model_parameters_count


def sample_timesteps(batch_size: int, num_timesteps: int, device: torch.device) -> torch.Tensor:
    return torch.randint(low=1, high=num_timesteps, size=(batch_size,), device=device).long()


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    t = cfg.train
    m = cfg.model

    device = get_device()
    os.makedirs(t.save_dir, exist_ok=True)

    seed_all(t.seed)

    wandb_mode = 'disabled' if t.disable_wandb else 'online'
    wandb.init(project=t.wandb_project, name=t.run_name, save_code=True, mode=wandb_mode,
               config=OmegaConf.to_container(cfg, resolve=True))

    train_loader, val_loader = train_val_data_loaders(t.batch_size, root=t.data_root or None)
    noise_scheduler = LinearNoiseScheduler(t.beta_start, t.beta_end, t.num_timesteps, device)
    criterion = MSELoss()
    model = UNet(m).to(device)
    optimizer = AdamW(model.parameters(), lr=t.lr, weight_decay=t.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=t.lr_factor, patience=t.lr_patience)
    scaler = torch.cuda.amp.GradScaler()

    print(f"Model parameters: {model_parameters_count(model):,}")

    step = 0
    best_val_acc = 0.0
    evaluator = Evaluation(device)

    for epoch in range(t.epochs):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{t.epochs}")
        for data in pbar:
            images_t, labels_t = data
            curr_batch_size = images_t.shape[0]

            images_t = images_t.to(device)
            labels_t = labels_t.to(device)
            timesteps_t = sample_timesteps(curr_batch_size, t.num_timesteps, device)
            images_t, noises_t = noise_scheduler.noise(images_t, timesteps_t)

            optimizer.zero_grad()
            with torch.amp.autocast(str(device)):
                pred = model(images_t, timesteps_t, labels_t)
                loss = criterion(pred, noises_t)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

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

        val_acc = evaluate(model, noise_scheduler, evaluator, val_loader, device)
        print(f"Epoch {epoch + 1} Validation Accuracy: {val_acc:.4f}")
        wandb.log({"val/val_acc": val_acc, "epoch": epoch + 1, "step": step})

        if val_acc > best_val_acc:
            print(f"Validation improved {best_val_acc:.4f} → {val_acc:.4f}. Saving model...")
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(t.save_dir, f"model_epoch_{epoch + 1}.pt"))


if __name__ == "__main__":
    main()
