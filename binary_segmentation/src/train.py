from oxford_pet import OxfordPetDataset
from src import DATASET_DIR

dataset = OxfordPetDataset(DATASET_DIR, "train")

dataset.__getitem__(0)