import json
from pathlib import Path
from typing import Literal

import torch
from PIL import Image
from torchvision.transforms import v2 as T

from torch.utils.data import Dataset, random_split, DataLoader

ROOT = Path(__file__).parent.parent / "data"

DatasetMode = Literal["train", "test", "new_test"]


class ICLEVRDataset(Dataset):
    def __init__(self, mode: DatasetMode = "train"):
        self.mode = mode
        self.objects = self._load_objects()
        self.num_classes = len(self.objects)
        if self.mode == "train":
            self.train_data = self._load_train_data()
            self.train_keys = list(self.train_data.keys())
        if self.mode in ["test", "new_test"]:
            self.test_data = self._load_test_data()
        self.transform = T.Compose([
            T.ToImage(),
            T.Resize((64, 64)),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __getitem__(self, idx: int):
        if self.mode == "train":
            fname = self.train_keys[idx]
            labels = self.train_data[fname]
            img_path = ROOT / 'iclevr' / fname
            image = Image.open(img_path).convert('RGB')
            image_t = self.transform(image)

            indices = []
            for label in labels:
                indices.append(self.objects[label])

            label_t = torch.zeros(self.num_classes, dtype=torch.float32)

            for indice in indices:
                label_t[indice] = 1.0
            return image_t, label_t
        if self.mode in ["test", "new_test"]:
            return self.test_data[idx]
        return None

    def __len__(self):
        if self.mode == "train":
            return len(self.train_data)
        else:
            return len(self.test_data)

    @staticmethod
    def _load_objects() -> dict[str, int]:
        label_file = ROOT / 'objects.json'
        with open(label_file, encoding='utf-8', mode="r") as f:
            labels = json.load(f)
        return labels

    @staticmethod
    def _load_train_data() -> dict[str, list[str]]:
        train_file = ROOT / "train.json"
        with open(train_file, encoding='utf-8', mode="r") as f:
            train_dict = json.load(f)
        return train_dict

    def _load_test_data(self) -> list[list[str]]:
        test_file = ROOT / f"{self.mode}.json"
        with open(test_file, encoding='utf-8', mode="r") as f:
            test_list = json.load(f)
        return test_list


def train_val_data_loaders(batch_size: int = 32) -> tuple[DataLoader, DataLoader]:
    full_dataset = ICLEVRDataset("train")
    train_size = int(len(full_dataset) - batch_size)
    val_size = batch_size

    train_sub, val_sub = random_split(full_dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_sub, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_sub, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
