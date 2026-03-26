from enum import Enum
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2 as trans


class TrimapClass(Enum):
    FOREGROUND = 1,
    BACKGROUND = 2,
    UNCLASSIFIED = 3,


class BinaryMask(Enum):
    FOREGROUND = 1
    BACKGROUND = 0


class OxfordPetDataset(Dataset):
    def __init__(self, data_dir: Path, mode='train'):
        super().__init__()
        self._mode = mode
        self._data_dir = data_dir
        self._ann_dir = data_dir / "annotations"
        self._images_dir = data_dir / "images"
        self._mask_dir = self._ann_dir / "trimaps"
        self._train_ann_file = self._ann_dir / "trainval.txt"
        self._test_ann_file = self._ann_dir / "test.txt"
        self._trans = self._get_transform()

        self._split_filenames = self._read_split_filenames()
        print(self._split_filenames)

    def __len__(self):
        return len(self._split_filenames)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        image_ph = self._get_image_path(self._split_filenames[idx])
        mask_ph = self._get_mask_path(self._split_filenames[idx])
        img = Image.open(image_ph).convert("RGB")
        mask = Image.open(mask_ph)
        mask = np.array(mask)
        mask[mask != TrimapClass.FOREGROUND.value] = BinaryMask.BACKGROUND.value
        img, mask = self._trans(img, mask)
        mask = mask.squeeze(0).long()
        return img, mask

    def _read_split_filenames(self) -> list[str]:
        split_ann_file = self._train_ann_file if self._mode == "train" else self._test_ann_file
        with open(split_ann_file, "r") as f:
            lines = f.readlines()
        filenames = [f"{line.split(" ")[0]}" for line in lines]
        return filenames

    def _get_image_path(self, filename):
        return self._images_dir / f"{filename}.jpg"

    def _get_mask_path(self, filename):
        return self._mask_dir / f"{filename}.png"

    @staticmethod
    def _training_transform() -> trans.Compose:
        return trans.Compose([
            trans.ToImage(),
            trans.ToDtype(torch.float32, scale=True),
            trans.RandomHorizontalFlip(p=0.5),
            trans.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.0)),
        ])

    @staticmethod
    def _test_transform() -> trans.Compose:
        return trans.Compose([
            trans.ToImage(),
            trans.ToDtype(torch.float32, scale=True),
        ])

    def _get_transform(self) -> trans.Compose:
        if self._mode == "train":
            return self._training_transform()
        return self._test_transform()
