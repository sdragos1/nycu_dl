import argparse
from datetime import datetime
from typing import Callable

import torch
from torch import nn, optim
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from constants import DATASET_DIR, SAVED_MODELS_DIR
from evaluate import evaluate
from models import get_model
from oxford_pet import get_train_val_dataloaders
from utils import dice_loss_criterion


def save_model(model: nn.Module, timestamp: str, epoch_idx: int) -> None:
    model_path = SAVED_MODELS_DIR / f"bin_segm_{timestamp}_{epoch_idx}.pth"
    torch.save(model.state_dict(), model_path)


def train_epoch(tb_writer: SummaryWriter, epoch_index: int, model: nn.Module, optimizer: Optimizer,
                criterion: Callable,
                train_loader: DataLoader,
                device: str,
                scheduler=None):
    running_loss = 0.
    last_loss = 0.
    epoch_loss = 0.0

    model.train()
    for i, data in enumerate(train_loader):
        images, masks, _ = data
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        pred_masks = model(images)
        bce = criterion(pred_masks, masks)
        dice_loss = dice_loss_criterion(pred_masks, masks)

        loss = bce + dice_loss
        loss.backward()

        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()
        running_loss += loss.item()
        if i % 10 == 0:
            last_loss = running_loss / 10
            print('  batch {} loss: {}'.format(i, last_loss))
            tb_x = epoch_index * len(train_loader) + i
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.
    return epoch_loss / len(train_loader)


def train(epochs: int, batch_size: int, device: str, lr: float, ts: str, vs: str) -> None:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/binary_segmentation_{}'.format(timestamp))

    train_loader, val_loader = get_train_val_dataloaders(
        root=DATASET_DIR,
        batch_size=batch_size,
        train_split=ts,
        val_split=vs
    )

    model = get_model("unet", out_channels=1)
    model = torch.compile(model)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        steps_per_epoch=len(train_loader),
        epochs=epochs
    )
    criterion = nn.BCEWithLogitsLoss()
    best_dice_score: float = 0.
    for epoch in range(epochs):
        print('EPOCH {}:'.format(epoch + 1))
        avg_loss = train_epoch(writer, epoch, model, optimizer, criterion, train_loader, device, scheduler)
        avg_vloss, avg_dice = evaluate(model, val_loader, criterion, device)

        if avg_dice > best_dice_score:
            best_dice_score = avg_dice
            save_model(model, timestamp, epoch)

        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning Rate', current_lr, epoch + 1)

        writer.add_scalars('Training vs. Validation Loss',
                           {'Training': avg_loss, 'Validation': avg_vloss},
                           epoch + 1)
        writer.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train binary segmentation model.")
    parser.add_argument("-e", "--epochs", type=int, default=50, help="Number of epochs.")
    parser.add_argument("-b", "--batch-size", type=int, default=16, help="Training batch size.")
    parser.add_argument("-d", "--device", default="cuda", help="Device for training (e.g. cuda, cpu).")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--ts", type=str, default="./dataset/oxford-iiit-pet/annotations/train.txt",
                        help="Path to train split file")
    parser.add_argument("--vs", type=str, default="./dataset/oxford-iiit-pet/annotations/val.txt",
                        help="Path to val split file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.epochs, args.batch_size, args.device, args.lr, args.ts, args.vs)
