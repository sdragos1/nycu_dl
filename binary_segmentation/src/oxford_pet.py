from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision import tv_tensors


class OxfordPetDataset(Dataset):
    _TRIMAP_FOREGROUND = 1
    _TRIMAP_BACKGROUND = 2
    _TRIMAP_UNCLASSIFIED = 3

    _BINARY_FOREGROUND = 1
    _BINARY_BACKGROUND = 0

    def __init__(
            self,
            root: Path | str,
            split: str = "train",
            transform: Optional[Callable] = None
    ):
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val', or 'test', got {split}")

        self.root = Path(root)
        self.split = split
        self.transform = transform

        self.images_dir = self.root / "images"
        self.masks_dir = self.root / "annotations" / "trimaps"
        self.split_file = self.root / "annotations" / f"{split}.txt"

        self.filenames = self._load_split_filenames()

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        filename = self.filenames[idx]
        image_path = self.images_dir / f"{filename}.jpg"
        mask_path = self.masks_dir / f"{filename}.png"

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
        return image, mask

    def _load_split_filenames(self) -> list[str]:
        if not self.split_file.exists():
            raise FileNotFoundError(f"Split file not found: {self.split_file}")

        with open(self.split_file) as f:
            lines = f.readlines()

        filenames = [line.split()[0] for line in lines]
        return filenames


def get_train_val_dataloaders(
        root: Path | str = "./dataset/oxford-iiit-pet",
        batch_size: int = 32,
        num_workers: int = 4,
        train_transform: Optional[Callable] = None,
        val_transform: Optional[Callable] = None,
) -> tuple[DataLoader, DataLoader]:
    train_dataset = OxfordPetDataset(
        root, split="train", transform=train_transform
    )
    val_dataset = OxfordPetDataset(
        root, split="val", transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader


def get_test_dataloader(
        root: Path | str = "./dataset/oxford-iiit-pet",
        batch_size: int = 1,
        num_workers: int = 2,
        transform: Optional[Callable] = None,
) -> DataLoader:
    test_dataset = OxfordPetDataset(
        root, split="test", transform=transform
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return test_loader
