from unet import UNet
import torch


net = UNet(n_channels=3, n_classes=2, bilinear=True)
torch.save(net, 'unet.pth')
