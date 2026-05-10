import torch
from torch import nn

from src.ddpm.layers import ConvLayer, EmbeddingFusionLayer, AttentionLayer, ContextEmbeddingLayer


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, emb_dim: int, groups: int = 8,
                 num_layers: int = 2, down_sample: bool = True, num_heads: int = 8):
        super(DownBlock, self).__init__()
        self.num_layers = num_layers
        self.down_sample = down_sample

        self.first_resnet_layers = nn.ModuleList([
            ConvLayer(in_channels if i == 0 else out_channels, out_channels, groups)
            for i in range(num_layers)
        ])
        self.fusion_layers = nn.ModuleList([
            EmbeddingFusionLayer(emb_dim, out_channels)
            for _ in range(num_layers)
        ])
        self.second_resnet_layers = nn.ModuleList([
            ConvLayer(out_channels, out_channels, groups)
            for i in range(num_layers)
        ])
        self.att_layers = nn.ModuleList([
            AttentionLayer(num_heads, out_channels, groups)
            for _ in range(num_layers)
        ])
        self.residual_layers = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers + 1)
            ]
        )
        self.down_sample_layer = nn.Conv2d(out_channels, out_channels,
                                           4, 2, 1) if self.down_sample else nn.Identity()

    def forward(self, x: torch.Tensor, emb_t: torch.Tensor) -> torch.Tensor:
        out = x

        for i in range(self.num_layers):
            resnet_input = out
            out = self.first_resnet_layers[i](out)
            out += self.fusion_layers[i](emb_t)[:, :, None, None]
            out = self.second_resnet_layers[i](out)
            out += self.residual_layers[i](resnet_input)
            out = self.att_layers[i](out)
        out = self.down_sample_layer(out)
        return out


class MidBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, emb_dim: int, num_layers: int, groups: int = 8,
                 num_heads: int = 8):
        super(MidBlock, self).__init__()
        self.num_layers = num_layers
        self.first_resnet_layers = nn.ModuleList([
            ConvLayer(in_channels if i == 0 else out_channels, out_channels, groups)
            for i in range(num_layers + 1)
        ])
        self.fusion_layers = nn.ModuleList([
            EmbeddingFusionLayer(emb_dim, out_channels)
            for _ in range(num_layers + 1)
        ])
        self.second_resnet_layers = nn.ModuleList([
            ConvLayer(out_channels, out_channels, groups)
            for _ in range(num_layers + 1)
        ])
        self.att_layers = nn.ModuleList([
            AttentionLayer(num_heads, out_channels, groups)
            for _ in range(num_layers)
        ])
        self.residual_layers = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers + 1)
            ]
        )

    def forward(self, x: torch.Tensor, emb_t: torch.Tensor) -> torch.Tensor:
        out = x
        resnet_input = out
        out = self.first_resnet_layers[0](out)
        out += self.fusion_layers[0](emb_t)[:, :, None, None]
        out = self.second_resnet_layers[0](out)
        out += self.residual_layers[0](resnet_input)

        for i in range(self.num_layers):
            out = self.att_layers[i](out)
            resnet_input = out
            out = self.first_resnet_layers[i + 1](out)
            out += self.fusion_layers[i + 1](emb_t)[:, :, None, None]
            out = self.second_resnet_layers[i + 1](out)
            out += self.residual_layers[i + 1](resnet_input)

        return out


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, emb_dim: int, num_layers: int = 2, groups: int = 8,
                 up_sample: bool = True, num_heads: int = 8):
        super(UpBlock, self).__init__()
        self.num_layers = num_layers
        self.up_sample = up_sample

        self.first_resnet_layers = nn.ModuleList([
            ConvLayer(in_channels if i == 0 else out_channels, out_channels, groups)
            for i in range(num_layers)
        ])
        self.fusion_layers = nn.ModuleList([
            EmbeddingFusionLayer(emb_dim, out_channels)
            for _ in range(num_layers)
        ])
        self.second_resnet_layers = nn.ModuleList([
            ConvLayer(out_channels, out_channels, groups)
            for _ in range(num_layers)
        ])
        self.att_layers = nn.ModuleList([
            AttentionLayer(num_heads, out_channels, groups)
            for _ in range(num_layers)
        ])
        self.residual_layers = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers + 1)
            ]
        )
        self.up_sample_conv = nn.ConvTranspose2d(in_channels // 2, in_channels // 2,
                                                 4, 2, 1) \
            if self.up_sample else nn.Identity()

    def forward(self, x: torch.Tensor, residuals: torch.Tensor, emb_t: torch.Tensor) -> torch.Tensor:
        x = self.up_sample_conv(x)
        x = torch.cat([x, residuals], dim=1)

        for i in range(self.num_layers):
            resnet_input = x
            x = self.first_resnet_layers[i](x)
            x += self.fusion_layers[i](emb_t)[:, :, None, None]
            x = self.second_resnet_layers[i](x)
            x += self.residual_layers[i](resnet_input)
            x = self.att_layers[i](x)
        return x


class UNet(nn.Module):
    def __init__(self, num_classes: int):
        super(UNet, self).__init__()
        self.image_channels = 1
        self.down_channels = [64, 128, 256, 512]
        self.mid_channels = [512, 512, 256]
        self.down_samples = [True, True, False]
        self.up_samples = list(reversed(self.down_samples))
        self.emb_dim = 256
        self.num_layers = 2
        self.num_heads = 4
        self.num_classes = num_classes

        self.ctx_embedding_layer = ContextEmbeddingLayer(self.emb_dim, self.num_classes)
        self.in_conv = nn.Conv2d(self.image_channels, self.down_channels[0], kernel_size=3, padding=(1, 1))

        self.down_blocks: nn.ModuleList = nn.ModuleList([
            DownBlock(self.down_channels[i],
                      self.down_channels[i + 1],
                      self.emb_dim,
                      down_sample=self.down_samples[i],
                      num_layers=self.num_layers,
                      num_heads=self.num_heads)
            for i in range(len(self.down_channels) - 1)
        ])
        self.mid_blocks: nn.ModuleList = nn.ModuleList([
            MidBlock(self.mid_channels[i],
                     self.mid_channels[i + 1],
                     self.emb_dim,
                     num_layers=self.num_layers,
                     num_heads=self.num_heads)
            for i in range(len(self.mid_channels) - 1)
        ])
        self.up_blocks: nn.ModuleList = nn.ModuleList([
            UpBlock(self.down_channels[i] * 2,
                    self.down_channels[i - 1] if i != 0 else 16,
                    self.emb_dim,
                    up_sample=self.up_samples[i],
                    num_layers=self.num_layers,
                    num_heads=self.num_heads)
            for i in range(len(self.down_channels) - 1)
        ])
        self.norm = nn.GroupNorm(num_groups=self.num_heads, num_channels=self.emb_dim)
        self.out_conv = nn.Conv2d(16, self.image_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, timesteps_t: torch.Tensor, classes_t: torch.Tensor) -> torch.Tensor:
        out = self.in_conv(x)
        emb_t = self.ctx_embedding_layer(timesteps_t, classes_t)

        residuals = [out]
        for idx, down in enumerate(self.down_blocks):
            out = down(out, emb_t)
            residuals.append(out)

        for mid in self.mid_blocks:
            out = mid(out, emb_t)

        for up in self.ups:
            residual = residuals.pop()
            out = up(out, residual, emb_t)
        out = self.norm(out)
        out = nn.SiLU()(out)
        out = self.out_conv(out)
        return out
