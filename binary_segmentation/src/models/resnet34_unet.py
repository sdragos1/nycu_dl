from torch import nn


class ResNet34UNet(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet34UNet, self).__init__()
