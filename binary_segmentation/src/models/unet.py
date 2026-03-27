import torch
from torch import nn, Tensor


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class UNetEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetEncoderBlock, self).__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.conv(x)
        skip = x
        x = self.pool(x)
        return x, skip


class UNetDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetDecoderBlock, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, 2, 2)
        self.conv = DoubleConv(in_channels, out_channels, kernel_size=3)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.upsample(x)
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        features = [64, 128, 256, 512]

        self._encoder = nn.ModuleList()
        for f in features:
            self._encoder.append(UNetEncoderBlock(in_channels, f))
            in_channels = f

        self._bottleneck: nn.Module = DoubleConv(features[-1], features[-1] * 2)

        self._decoder = nn.ModuleList()
        for f in reversed(features):
            self._decoder.append(UNetDecoderBlock(f * 2, f))

        self._output = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        skip_features: list[Tensor] = []
        for encoder in self._encoder:
            x, skip = encoder(x)
            skip_features.append(skip)
        x = self._bottleneck(x)
        for decoder in self._decoder:
            skip = skip_features.pop()
            x = decoder(x, skip)
        x = self._output(x)
        return x
