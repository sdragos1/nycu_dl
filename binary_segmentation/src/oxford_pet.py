from pathlib import Path

import torch

from constants import TRAIN_ANN_FILE


class OxfordPetDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.split_filenames = self._read_split_filenames(TRAIN_ANN_FILE)

    def _read_split_filenames(self, split_ann_file: Path):
        with open(split_ann_file, "r") as f:
            lines = f.readlines()
            print(lines)
        return None

dataset = OxfordPetDataset()