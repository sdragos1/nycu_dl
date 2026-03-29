import torch
import torch.nn as nn
import torch.nn.functional as func


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample: nn.Module | None = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNetLayer(nn.Module):
    def __init__(self, in_channels, out_channels, blocks, stride=1):
        super(ResNetLayer, self).__init__()
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = [ResidualBlock(in_channels, out_channels, stride, downsample)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels // 2 + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            if x.size(2) != skip.size(2) or x.size(3) != skip.size(3):
                x = func.interpolate(x, size=(skip.size(2), skip.size(3)), mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ResNet34UNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(ResNet34UNet, self).__init__()
        layers = [3, 4, 6, 3]

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = ResNetLayer(64, 64, layers[0], stride=1)
        self.layer2 = ResNetLayer(64, 128, layers[1], stride=2)
        self.layer3 = ResNetLayer(128, 256, layers[2], stride=2)
        self.layer4 = ResNetLayer(256, 512, layers[3], stride=2)

        self.dec1 = DecoderBlock(in_channels=512, skip_channels=256, out_channels=256)
        self.dec2 = DecoderBlock(in_channels=256, skip_channels=128, out_channels=128)
        self.dec3 = DecoderBlock(in_channels=128, skip_channels=64, out_channels=64)
        self.dec4 = DecoderBlock(in_channels=64, skip_channels=64, out_channels=64)

        self.upsample = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=1)
        )

    def forward(self, x):
        enc0 = self.conv(x)
        pool = self.max_pool(enc0)

        enc1 = self.layer1(pool)
        enc2 = self.layer2(enc1)
        enc3 = self.layer3(enc2)
        enc4 = self.layer4(enc3)

        d1 = self.dec1(enc4, enc3)
        d2 = self.dec2(d1, enc2)
        d3 = self.dec3(d2, enc1)
        d4 = self.dec4(d3, enc0)

        out = self.upsample(d4)
        out = self.segmentation_head(out)

        return out
