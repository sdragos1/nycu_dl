from pathlib import Path

ROOT = Path(__file__).parent.parent

DATASET_DIR = ROOT / "dataset" / "oxford-iiit-pet"
ANNOTATIONS_DIR = DATASET_DIR / "annotations"
IMAGES_DIR = DATASET_DIR / "images"

TRAIN_ANN_FILE = ANNOTATIONS_DIR / "trainval.txt"
TEST_ANN_FILE = ANNOTATIONS_DIR / "test.txt"

SAVED_MODELS_DIR = DATASET_DIR / "saved_models"
