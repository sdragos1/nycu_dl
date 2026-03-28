from datetime import datetime
from typing import Callable

import torch
from torch import nn, optim
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import tv_tensors
from torchvision.transforms import v2 as trans

from constants import DATASET_DIR, SAVED_MODELS_DIR
from evaluate import evaluate
from models import get_model
from oxford_pet import get_train_val_dataloaders


def save_model(model: nn.Module, timestamp: str, epoch_idx: int) -> None:
    model_path = SAVED_MODELS_DIR / f"bin_segm_{timestamp}_{epoch_idx}.pth"
    torch.save(model.state_dict(), model_path)


def get_transforms() -> tuple[trans.Compose, trans.Compose]:
    train_transform = trans.Compose([
        trans.ToImage(),
        trans.ToDtype({tv_tensors.Image: torch.float32}, scale=True),  # scale image pixels to [0, 1]
        trans.ToDtype({tv_tensors.Mask: torch.float32}, scale=False),  # cast mask {0,1} without scaling
        trans.RandomHorizontalFlip(p=0.5),
        trans.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.0)),
    ])

    val_transform = trans.Compose([
        trans.ToImage(),
        trans.ToDtype({tv_tensors.Image: torch.float32}, scale=True),  # scale image pixels to [0, 1]
        trans.ToDtype({tv_tensors.Mask: torch.float32}, scale=False),  # cast mask {0,1} without scaling
        trans.Resize((256, 256)),
    ])

    return train_transform, val_transform


def train_epoch(tb_writer: SummaryWriter, epoch_index: int, model: nn.Module, optimizer: Optimizer,
                criterion: Callable,
                train_loader: DataLoader,
                device: str):
    running_loss = 0.
    last_loss = 0.

    model.train()
    for i, data in enumerate(train_loader):
        images, masks = data
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        pred_masks = model(images)
        loss = criterion(pred_masks, masks)
        loss.backward()

        optimizer.step()
        running_loss += loss.item()
        if i % 10 == 0:
            last_loss = running_loss / 10
            print('  batch {} loss: {}'.format(i, last_loss))
            tb_x = epoch_index * len(train_loader) + i
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.
    return last_loss


def train(epochs: int, batch_size: int, device: str, lr: float) -> None:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/binary_segmentation_{}'.format(timestamp))

    train_transform, val_transform = get_transforms()

    train_loader, val_loader = get_train_val_dataloaders(
        root=DATASET_DIR,
        batch_size=batch_size,
        train_transform=train_transform,
        val_transform=val_transform,
    )

    model = get_model("unet", out_channels=1)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    best_val_loss: float = 1_000_000.
    best_dice_score: float = 0.
    for epoch in range(epochs):
        print('EPOCH {}:'.format(epoch + 1))
        avg_loss = train_epoch(writer, epoch, model, optimizer, criterion, train_loader, device)
        avg_vloss, avg_dice = evaluate(model, val_loader, criterion, device)

        if avg_vloss < best_val_loss and avg_dice > best_dice_score:
            best_val_loss = avg_vloss
            save_model(model, timestamp, epoch)
        writer.add_scalars('Training vs. Validation Loss',
                           {'Training': avg_loss, 'Validation': avg_vloss},
                           epoch + 1)
        writer.flush()


if __name__ == "__main__":
    train(5, 16, 'cuda', 0.003)
