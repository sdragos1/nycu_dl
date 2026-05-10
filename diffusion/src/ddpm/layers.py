import math

import torch
from torch import nn


class ContextEmbeddingLayer(nn.Module):
    def __init__(self, emb_dim: int, num_classes: int):
        super(ContextEmbeddingLayer, self).__init__()
        self.emb_dim = emb_dim
        self.num_classes = num_classes
        self.cwm = nn.Embedding(self.num_classes, self.emb_dim)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim),
            nn.SiLU(),
            nn.Linear(self.emb_dim, self.emb_dim),
        )
        self.layer_norm = nn.LayerNorm(self.emb_dim)

    def forward(self, timesteps_t: torch.Tensor, classes_t: torch.Tensor) -> torch.Tensor:
        embedded_timesteps_t = self._time_sinusoidal_encoding(timesteps_t)
        embedded_classes_t = self.layer_norm(classes_t @ self.cwm.weight)
        return self.fusion_mlp(embedded_classes_t + embedded_timesteps_t)

    def _time_sinusoidal_encoding(self, timesteps_t: torch.Tensor) -> torch.Tensor:
        half_dim = self.emb_dim // 2
        freqs = torch.exp(
            torch.arange(start=0, end=half_dim, device=timesteps_t.device)
            * (-math.log(10000.0) / (half_dim - 1))
        )
        angles = timesteps_t[:, None].float() * freqs[None, :]
        emb = torch.cat([angles.sin(), angles.cos()], dim=-1)
        return emb


class ConvLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groups: int = 8):
        super(ConvLayer, self).__init__()
        self.residual = nn.Sequential(
            nn.GroupNorm(groups, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
        )

    def forward(self, x):
        return self.residual(x)


class EmbeddingFusionLayer(nn.Module):
    def __init__(self, emb_dim: int, out_channels: int):
        super(EmbeddingFusionLayer, self).__init__()
        self.fusion = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fusion(x)


class AttentionLayer(nn.Module):
    def __init__(self, num_heads: int, out_channels, groups: int = 8):
        super(AttentionLayer, self).__init__()
        self.norm = nn.GroupNorm(groups, out_channels)
        self.mha = nn.MultiheadAttention(out_channels, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        att = x.reshape(batch_size, channels, height * width)
        att = self.norm(att)
        att = att.transpose(1, 2)
        att, _ = self.mha(att, att, att)
        att = att.transpose(1, 2).reshape(batch_size, channels, height, width)
        return att + x
