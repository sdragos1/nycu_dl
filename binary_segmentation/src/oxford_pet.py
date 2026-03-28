import os.path
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision import tv_tensors
from torchvision.transforms import v2 as trans


class OxfordPetDataset(Dataset):
    _TRIMAP_FOREGROUND = 1
    _TRIMAP_BACKGROUND = 2
    _TRIMAP_UNCLASSIFIED = 3

    _BINARY_FOREGROUND = 1
    _BINARY_BACKGROUND = 0

    def __init__(
            self,
            root: Path | str,
            split: str | Path = "train",
            transform: Optional[Callable] = None
    ):
        self.root = Path(root)
        self.split_file = split
        self.transform = transform

        self.images_dir = self.root / "images"
        self.masks_dir = self.root / "annotations" / "trimaps"

        self.filenames = self._load_split_filenames()

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, str]:
        image_id = self.filenames[idx]
        image_path = self.images_dir / f"{image_id}.jpg"
        mask_path = self.masks_dir / f"{image_id}.png"

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)

        mask = np.array(mask)
        mask = np.where(
            mask == self._TRIMAP_FOREGROUND,
            self._BINARY_FOREGROUND,
            self._BINARY_BACKGROUND,
        ).astype(np.uint8)
        mask = tv_tensors.Mask(mask)
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        mask = mask.unsqueeze(0)
        return image, mask, image_id

    def _load_split_filenames(self) -> list[str]:
        if not os.path.exists(self.split_file):
            raise FileNotFoundError(f"File not found: {self.split_file}")

        with open(self.split_file) as f:
            lines = f.readlines()

        filenames = [line.split()[0] for line in lines]
        return filenames


default_train_transform = trans.Compose([
    trans.ToImage(),
    trans.ToDtype({tv_tensors.Image: torch.float32, "others": None}, scale=True),
    trans.ToDtype({tv_tensors.Mask: torch.float32, "others": None}, scale=False),
    trans.RandomHorizontalFlip(p=0.5),
    trans.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.0)),
    trans.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    trans.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

default_val_transform = trans.Compose([
    trans.ToImage(),
    trans.ToDtype({tv_tensors.Image: torch.float32, "others": None}, scale=True),
    trans.ToDtype({tv_tensors.Mask: torch.float32, "others": None}, scale=False),
    trans.Resize((256, 256)),
    trans.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


def get_train_val_dataloaders(
        root: Path | str = "./dataset/oxford-iiit-pet",
        train_split: Path | str = "./dataset/oxford-iiit-pet/annotations/train.txt",
        val_split: Path | str = "./dataset/oxford-iiit-pet/annotations/val.txt",
        batch_size: int = 32,
        num_workers: int = 8,
        train_transform: Optional[Callable] = default_train_transform,
        val_transform: Optional[Callable] = default_val_transform,
) -> tuple[DataLoader, DataLoader]:
    train_dataset = OxfordPetDataset(
        root, split=train_split, transform=train_transform
    )
    val_dataset = OxfordPetDataset(
        root, split=val_split, transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
        persistent_workers=True
    )

    return train_loader, val_loader


def get_test_dataloader(
        root: Path | str = "./dataset/oxford-iiit-pet",
        split: Path | str = "./dataset/oxford-iiit-pet/annotations/test.txt",
        batch_size: int = 1,
        num_workers: int = 2,
        transform: Optional[Callable] = default_val_transform,
) -> DataLoader:
    test_dataset = OxfordPetDataset(
        root, split=split, transform=transform
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
        persistent_workers=True
    )
    return test_loader
