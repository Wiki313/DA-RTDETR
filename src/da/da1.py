import torch.nn.functional as F
from .grl import WarmStartGradientReverseLayer
from typing import Optional
import torch.nn as nn
import torch
from src.zoo.rtdetr.hybrid_encoder import CSPRepLayer,ConvNormLayer

# modifiy from H2xxx
class DA_Pixel(nn.Module):
    """
    Local domain classifier for image-level class-agnostic feature alignment
    """

    def __init__(self, in_channel=None):
        super(DA_Pixel, self).__init__()
        self.adapt1 = nn.Sequential(

            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),

            nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),

            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1, bias=False),

        )

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

            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False),
        )

        self.adapt4 = nn.Sequential(

            nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),

            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),

            # nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1, bias=False),
            # nn.BatchNorm2d(128),
            # nn.SiLU(inplace=True),

            # nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1, bias=False),
        )

        self.fusion1 = CSPRepLayer(256,256)
        #self.fusion2 = CSPRepLayer(512,256)
        #self.fusion3 = CSPRepLayer(512,256)
        # self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.ffn = nn.Linear(256, 1)

        self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=2., max_iters=1000, auto_step=True)

    def forward(self, x):
        x1 = self.grl(x[0])
        x2 = self.grl(x[1])
        x3 = self.grl(x[2])

        x1 = self.adapt1(x1)
        x2 = self.adapt2(x2)
        x3 = self.adapt3(x3)

        x = torch.concat([x1, x2, x3], dim=1)

        #y1 = self.grl(y[0])
        #y2 = self.grl(y[1])
        #y3 = self.grl(y[2])

        #x1 = torch.cat([x1, y1], dim=1)
        #x2 = torch.cat([x2, y2], dim=1)
        #x3 = torch.cat([x3, y3], dim=1)

        #x1 = self.conv1(x1)
        #x2 = self.conv2(x2)
        #x3 = self.conv3(x3)
        #x1 = self.fusion1(x1)
        #x2 = self.fusion2(x2)
        #x3 = self.fusion3(x3)

        #adapt_x = torch.cat([x1, x2, x3], dim=1)
        x = self.fusion1(x)
        x = nn.AdaptiveAvgPool2d((1, 1))(x)
        #x1 = nn.AdaptiveAvgPool2d((1, 1))(x1)
        #x2 = nn.AdaptiveAvgPool2d((1, 1))(x2)
        #x3 = nn.AdaptiveAvgPool2d((1, 1))(x3)

        # for module in self.adapt1:
        #     x1 = module(x1)
        #
        # for module in self.adapt2:
        #     x2 = module(x2)
        #
        # for module in self.adapt3:
        #     x3 = module(x3)

        #da_pixel_x = torch.cat([x1, x2, x3], dim=1)
        img_feature = x
        da_pixel_x = x.view(x.size(0), -1)
        da_pixel_x = self.ffn(da_pixel_x)
        da_pixel_x = torch.sigmoid(da_pixel_x)
        # x = torch.sigmoid(self.da_conv3(x)) # 为什么当初在最后一层是sigmod
        return da_pixel_x


class Mix_Attention(nn.Module):
    def __init__(self, in_channel=None):
        super(Mix_Attention, self).__init__()
        # 域内注意力
        self.local_attention = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.Softmax(dim=1)
        )

        # 域间注意力
        self.global_attention = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.Softmax(dim=1)
        )

        self.adapt = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.AdaptiveAvgPool2d((1, 1))
        )
    def forward(self, x_source, x_target):
        # 域内注意力
        # local_weights = self.local_attention(x_source)
        # x1_adjusted = x_source * local_weights
        #
        # # 域间注意力
        # global_weights = self.global_attention(x1_adjusted)
        # x2_adjusted = x_target * global_weights
        # x = torch.cat((x1_adjusted, x2_adjusted), dim=0 )

        x = torch.cat((x_source, x_target), dim=0 )

        return self.adapt(x)

class DA_Instance(nn.Module):
    """
    Local domain classifier for image-level class-agnostic feature alignment
    先写死
    """

    def __init__(self, in_channel=None):
        super(DA_Instance, self).__init__()
        self.adapt1 = nn.Sequential(

            nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),

            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1, bias=False),

        )

        self.adapt2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),

            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1, bias=False),
        )

        self.adapt3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
        )
        self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=2., max_iters=1000, auto_step=True)
        self.fusion = CSPRepLayer(256,256)
        # self.mix_att =Mix_Attention()
        self.adapt = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1, bias=False),
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
        ins_adapt_x = torch.cat([x1, x2, x3], dim=1)
        ins_adapt_x = self.fusion(ins_adapt_x)

        # p_source_scale, p_target_scale = torch.chunk(p_adapt_x, 2, dim=0)
        # i_source_scale, i_target_scale = torch.chunk(ins_adapt_x, 2, dim=0)
        # source_scale = torch.cat((p_source_scale, i_source_scale), dim=0)
        # target_scale = torch.cat((p_target_scale, i_target_scale), dim=0)
        # da_instance = self.mix_att(source_scale, target_scale)
        # da_instance = da_instance.view(da_instance.size(0), -1)
        # da_instance = self.ffn(da_instance)
        # da_instance = torch.sigmoid(da_instance)

        da_instance = self.adapt(ins_adapt_x)
        ins_feature = da_instance
        da_instance = da_instance.view(da_instance.size(0), -1)

        da_instance = self.ffn(da_instance)
        da_instance = torch.sigmoid(da_instance)

        return da_instance


