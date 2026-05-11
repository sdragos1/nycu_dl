import torch
from omegaconf import DictConfig
from torch import nn

from src.ddpm.layers import ConvLayer, EmbeddingFusionLayer, AttentionLayer, ContextEmbeddingLayer


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, emb_dim: int, groups: int = 8,
                 num_layers: int = 2, down_sample: bool = True, num_heads: int = 8,
                 apply_attention: bool = True):
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
            AttentionLayer(num_heads, out_channels, groups) if apply_attention else nn.Identity()
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
    def __init__(self, in_channels: int, out_channels: int, upsample_channels: int, emb_dim: int,
                 num_layers: int = 2, groups: int = 8, up_sample: bool = True, num_heads: int = 8,
                 apply_attention: bool = True):
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
            AttentionLayer(num_heads, out_channels, groups) if apply_attention else nn.Identity()
            for _ in range(num_layers)
        ])
        self.residual_layers = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers + 1)
            ]
        )
        self.up_sample_conv = nn.ConvTranspose2d(upsample_channels, upsample_channels,
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
    def __init__(self, cfg: DictConfig):
        super(UNet, self).__init__()
        self.cfg = cfg

        self.ctx_embedding_layer = ContextEmbeddingLayer(cfg.emb_dim, cfg.num_classes)
        self.in_conv = nn.Conv2d(cfg.image_channels, cfg.down_channels[0], kernel_size=3, padding=(1, 1))

        self.down_blocks: nn.ModuleList = nn.ModuleList([
            DownBlock(cfg.down_channels[i],
                      cfg.down_channels[i + 1],
                      cfg.emb_dim,
                      groups=cfg.groups,
                      down_sample=cfg.down_samples[i],
                      num_layers=cfg.num_layers,
                      num_heads=cfg.num_heads,
                      apply_attention=(i > 0))
            for i in range(len(cfg.down_channels) - 1)
        ])
        self.mid_blocks: nn.ModuleList = nn.ModuleList([
            MidBlock(cfg.mid_channels[i],
                     cfg.mid_channels[i + 1],
                     cfg.emb_dim,
                     groups=cfg.groups,
                     num_layers=cfg.num_layers,
                     num_heads=cfg.num_heads)
            for i in range(len(cfg.mid_channels) - 1)
        ])


        up_block_list = []
        decoder_in = list(cfg.mid_channels)[-1]
        for i in reversed(range(len(cfg.down_channels) - 1)):
            skip_ch = cfg.down_channels[i]
            in_ch = decoder_in + skip_ch
            out_ch = cfg.down_channels[i - 1] if i > 0 else cfg.out_channels
            up_block_list.append(
                UpBlock(in_channels=in_ch,
                        out_channels=out_ch,
                        upsample_channels=decoder_in,
                        emb_dim=cfg.emb_dim,
                        groups=cfg.groups,
                        up_sample=cfg.down_samples[i],
                        num_layers=cfg.num_layers,
                        num_heads=cfg.num_heads,
                        apply_attention=(i > 0))
            )
            decoder_in = out_ch
        self.up_blocks: nn.ModuleList = nn.ModuleList(up_block_list)
        self.norm = nn.GroupNorm(num_groups=cfg.groups, num_channels=cfg.out_channels)
        self.out_conv = nn.Conv2d(cfg.out_channels, cfg.image_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, timesteps_t: torch.Tensor, classes_t: torch.Tensor) -> torch.Tensor:
        out = self.in_conv(x)
        emb_t = self.ctx_embedding_layer(timesteps_t, classes_t)

        residuals = []
        for down in self.down_blocks:
            residuals.append(out)
            out = down(out, emb_t)

        for mid in self.mid_blocks:
            out = mid(out, emb_t)

        for up in self.up_blocks:
            residual = residuals.pop()
            out = up(out, residual, emb_t)

        out = self.norm(out)
        out = torch.nn.functional.silu(out)
        out = self.out_conv(out)
        return out
