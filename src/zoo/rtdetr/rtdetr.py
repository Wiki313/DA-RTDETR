"""by lyuwenyu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import numpy as np

from src.core import register
from src.da import DA_Pixel, DA_Instance

__all__ = ['RTDETR', ]


@register
class RTDETR(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, backbone: nn.Module, encoder, decoder, multi_scale=None, ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder

        # add domain adaptation
        self.da_pixel_block = DA_Pixel()
        self.da_instance_block = DA_Instance()
        #self.single_domain_a = DomainDiscriminator1(2048,1024)
        #self.single_domain_b = DomainDiscriminator2(256,128)
        # self.is_source = is_source
        # self.is_pixel_da = is_pixel_da
        # self.is_instance_da = is_instance_da
        self.multi_scale = multi_scale

    def forward(self, x_source, x_target=None, targets=None, use_pixel_da=False, use_instance_da=False, use_query_da=False):
        

        if self.multi_scale and self.training:
            sz = np.random.choice(self.multi_scale)
            x_source = F.interpolate(x_source, size=[sz, sz])
            if x_target is not None:
                x_target = F.interpolate(x_target, size=[sz, sz])

        if x_target is not None:
            x = torch.cat((x_source, x_target), dim=0)
        else:
            x = x_source

        x = self.backbone(x)  # [8,512,76,76] [8,1024,38,38] [8,2048,19,19]

        # add domain adaptation
        if self.training and (use_pixel_da or use_instance_da):
           da_pixel_x = self.da_pixel_block(x)
           #da_pixel_x = self.single_domain_a(x)

        x = self.encoder(x)  # x的输出尺寸[8,256,76,76] [8,256,38,38] [8,256,19,19]，da_query_x是对抗模块的输出

        if self.training and use_instance_da:
           da_instance_x = self.da_instance_block(x)

        list_source = []
        list_target = []
        if x_target is not None:
            for it in x:
                it_s, it_t = torch.chunk(it, 2, dim=0)
                list_source.append(it_s)
                list_target.append(it_t)
        else:
            list_source = x

        x_source = self.decoder(list_source, targets)


        if x_target is not None:
            x_target = self.decoder(list_target, None)
        
        
        if self.training:
        
          source_logits = list(x_source.values())[0]
          target_logits = list(x_target.values())[0]


          source_logits = source_logits.reshape((source_logits.size(0)*source_logits.size(1)),source_logits.size(2))

          target_logits = target_logits.reshape((target_logits.size(0) * target_logits.size(1)), target_logits.size(2))

          x_target['coral'] = [source_logits, target_logits]
        

        if self.training and use_pixel_da:
            x_target['da_pixel'] = da_pixel_x


        #if self.training and use_hinge_da:
            #x_target['da_hinge'] = da_query_x

        if self.training and use_instance_da:
            x_target['da_instance'] = da_instance_x

       # if self.training:
           # x_target['consistency'] = [img_fea, ins_fea, da_instance_x]

        #if self.training and use_query_da:
            #x_target['da_query'] = da_query_x
        if self.training:
          return [x_source, x_target]
        else:
          return x_source

    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self
