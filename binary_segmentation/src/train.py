import torch

from models.unet import UNet

model = UNet(in_channels=3, out_channels=2)
x = torch.randn(1, 3, 256, 256)
y = model(x)
print(y.shape)
