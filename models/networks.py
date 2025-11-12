import torch
import torch.nn as nn
from models.unet import UNetorg
from models.vnet import VNetorg


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()
        self.net = UNetorg(in_channels=in_channels, num_classes=out_channels)

    def forward(self, x, perturbation=False):
        x = self.net(x, perturbation)
        return x

    def detach_model(self):
        for param in self.parameters():
            param.detach_()

    def ema_update(self, student, ema_decay, cur_step=None):
        if cur_step is not None:
            ema_decay = min(1 - 1 / (cur_step + 1), ema_decay)
        for t_param, s_param in zip(self.parameters(), student.parameters()):
            t_param.data.mul_(ema_decay).add_(1 - ema_decay, s_param.data)


class VNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, normalization='batchnorm'):
        super(VNet, self).__init__()
        self.net = VNetorg(in_channels=in_channels, out_channels=out_channels, normalization=normalization)

    def forward(self, x, perturbation=False):
        x = self.net(x, perturbation)
        return x

    def detach_model(self):
        for param in self.parameters():
            param.detach_()

    def ema_update(self, student, ema_decay, cur_step=None):
        if cur_step is not None:
            ema_decay = min(1 - 1 / (cur_step + 1), ema_decay)
        for t_param, s_param in zip(self.parameters(), student.parameters()):
            t_param.data.mul_(ema_decay).add_(1 - ema_decay, s_param.data)