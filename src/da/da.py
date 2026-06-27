import torch.nn.functional as F
from .grl import WarmStartGradientReverseLayer
from typing import Optional
import torch.nn as nn
import torch
from src.zoo.rtdetr.hybrid_encoder import CSPRepLayer, ConvNormLayer


# DA_Pixel: takes backbone features x[0]=512ch, x[1]=1024ch, x[2]=2048ch
# Each adapt reduces to 64 channels -> concat = 192ch -> fusion CSPRepLayer(192,192)
class DA_Pixel(nn.Module):
    def __init__(self, in_channel=None):
        super(DA_Pixel, self).__init__()

        # x[0]: 512 -> 64
        self.adapt1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1, bias=False),
        )

        # x[1]: 1024 -> 64
        self.adapt2 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.SiLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, bias=False),
        )

        # x[2]: 2048 -> 64
        self.adapt3 = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1024),
            nn.SiLU(inplace=True),
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.SiLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
            nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0, bias=False),
        )

        # concat of adapt1+2+3 = 64+64+64 = 192 channels
        self.fusion1 = CSPRepLayer(192, 192)
        self.ffn = nn.Linear(192, 1)
        self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=2., max_iters=1000, auto_step=True)

    def forward(self, x):
        x1 = self.grl(x[0])
        x2 = self.grl(x[1])
        x3 = self.grl(x[2])

        x1 = self.adapt1(x1)
        x2 = self.adapt2(x2)
        x3 = self.adapt3(x3)

        # Resize x2, x3 to match x1 spatial size
        x2 = F.interpolate(x2, size=x1.shape[2:], mode='nearest')
        x3 = F.interpolate(x3, size=x1.shape[2:], mode='nearest')

        x = torch.cat([x1, x2, x3], dim=1)  # 192 channels
        HFA = x
        x = self.fusion1(x)
        fusion = x
        x = nn.AdaptiveAvgPool2d((1, 1))(x)
        img_feature = x
        da_pixel_x = x.view(x.size(0), -1)
        da_pixel_x = self.ffn(da_pixel_x)
        da_pixel_x = torch.sigmoid(da_pixel_x)
        return da_pixel_x, fusion, HFA


class DomainDiscriminator1(nn.Module):
    def __init__(self, in_channels: int, hidden_size=1024):
        super(DomainDiscriminator1, self).__init__()
        self.pool_layer = nn.Sequential(
            nn.Conv2d(in_channels, hidden_size, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(32, hidden_size),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1)
        )
        self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True)
        self.bottleneck = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x[2]
        x = self.grl(x)
        x = self.pool_layer(x)
        fea = x
        x = self.bottleneck(x)
        return x, fea


class DomainDiscriminator2(nn.Module):
    def __init__(self, in_channels: int, hidden_size=1024):
        super(DomainDiscriminator2, self).__init__()
        self.pool_layer = nn.Sequential(
            nn.Conv2d(in_channels, hidden_size, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(32, hidden_size),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1)
        )
        self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True)
        self.bottleneck = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x[2]
        x = self.grl(x)
        x = self.pool_layer(x)
        fea = x
        x = self.bottleneck(x)
        return x, fea


class Mix_Attention(nn.Module):
    def __init__(self, in_channel=None):
        super(Mix_Attention, self).__init__()
        self.local_attention = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.Softmax(dim=1)
        )
        self.global_attention = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.Softmax(dim=1)
        )
        self.adapt = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x_source, x_target):
        x = torch.cat((x_source, x_target), dim=0)
        return self.adapt(x)


# DA_Instance: takes encoder features x[0]=384ch, x[1]=384ch, x[2]=384ch
# Each adapt reduces: adapt1->64ch, adapt2->64ch, adapt3->128ch -> concat=256ch
class DA_Instance(nn.Module):
    def __init__(self, in_channel=None):
        super(DA_Instance, self).__init__()

        # x[0]: 384 -> 64
        self.adapt1 = nn.Sequential(
            nn.Conv2d(384, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1, bias=False),
        )

        # x[1]: 384 -> 64
        self.adapt2 = nn.Sequential(
            nn.Conv2d(384, 128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1, bias=False),
        )

        # x[2]: 384 -> 64
        self.adapt3 = nn.Sequential(
            nn.Conv2d(384, 128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1, bias=False),
        )

        self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=2., max_iters=1000, auto_step=True)

        # concat of adapt1+2+3 = 64+64+64 = 192 channels
        self.fusion = CSPRepLayer(192, 192)

        self.adapt = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.ffn = nn.Linear(128, 1)

    def forward(self, x):
        x1 = self.grl(x[0])
        x2 = self.grl(x[1])
        x3 = self.grl(x[2])

        x1 = self.adapt1(x1)
        x2 = self.adapt2(x2)
        x3 = self.adapt3(x3)

        # Resize to match spatial dims
        x2 = F.interpolate(x2, size=x1.shape[2:], mode='nearest')
        x3 = F.interpolate(x3, size=x1.shape[2:], mode='nearest')

        ins_adapt_x = torch.cat([x1, x2, x3], dim=1)  # 192 channels
        HFA = ins_adapt_x
        ins_adapt_x = self.fusion(ins_adapt_x)
        fusion = ins_adapt_x

        da_instance = self.adapt(ins_adapt_x)
        ins_feature = da_instance
        da_instance = da_instance.view(da_instance.size(0), -1)
        da_instance = self.ffn(da_instance)
        da_instance = torch.sigmoid(da_instance)

        return da_instance, fusion, HFA, ins_feature
