from dataclasses import dataclass, field
from typing import List


@dataclass
class UNetConfig:
    image_channels: int = 3
    num_classes: int = 24
    down_channels: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    mid_channels: List[int] = field(default_factory=lambda: [512, 512, 512])
    down_samples: List[bool] = field(default_factory=lambda: [True, True, False])
    emb_dim: int = 256
    num_layers: int = 2
    num_heads: int = 8
    groups: int = 8
    out_channels: int = 16


@dataclass
class TrainConfig:
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    num_timesteps: int = 1000
    batch_size: int = 32
    lr: float = 2e-4
    weight_decay: float = 1e-4
    lr_patience: int = 8
    lr_factor: float = 0.5
    epochs: int = 100
    seed: int = 42
    save_dir: str = "results/unet-42"
    run_name: str = "unet-42"
    wandb_project: str = "DL-DDPM"
    disable_wandb: bool = False


@dataclass
class Config:
    model: UNetConfig = field(default_factory=UNetConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
