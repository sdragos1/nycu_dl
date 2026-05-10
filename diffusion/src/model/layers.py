import math

import torch
from torch import nn


class ContextEmbeddingBlock(nn.Module):
    def __init__(self, emb_dim: int, num_classes: int):
        super(ContextEmbeddingBlock, self).__init__()
        self.emb_dim = emb_dim
        self.num_classes = num_classes
        self.cwm = nn.Embedding(self.num_classes, self.emb_dim)
        self.fusion = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim),
            nn.SiLU(),
            nn.Linear(self.emb_dim, self.emb_dim),
            nn.LayerNorm(self.emb_dim),
        )
        self.layer_norm = nn.LayerNorm(self.emb_dim)

    def forward(self, timesteps_t: torch.Tensor, classes_t: torch.Tensor) -> torch.Tensor:
        embedded_timesteps_t = self._time_sinusoidal_encoding(timesteps_t)
        embedded_classes_t = classes_t @ self.cwm.weight
        return self.fusion(embedded_classes_t + embedded_timesteps_t)

    def _time_sinusoidal_encoding(self, timesteps_t: torch.Tensor) -> torch.Tensor:
        half_dim = self.emb_dim // 2
        freqs = torch.exp(
            torch.arange(start=0, end=half_dim, device=timesteps_t.device)
            * (-math.log(10000.0) / (half_dim - 1))
        )
        angles = timesteps_t[:, None].float() * freqs[None, :]
        emb = torch.cat([angles.sin(), angles.cos()], dim=-1)
        return emb


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groups=8):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU()
        )

    def forward(self, x):
        return self.conv_block(x)
