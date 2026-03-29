from models import UNet, ResNet34UNet


def get_model(model: str, in_channels=3, out_channels=2):
    if model == 'unet':
        return UNet(in_channels=in_channels, out_channels=out_channels)
    if model == 'resnet':
        return ResNet34UNet(in_channels=in_channels, out_channels=out_channels)
    return None
