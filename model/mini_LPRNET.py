#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 13:14:16 2019

@author: xingyu
"""

import torch.nn as nn
import torch
import torch.nn.functional as F


class small_basic_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(small_basic_block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out, kernel_size=1),
        )
    def forward(self, x):
        return self.block(x)

class LPRNet(nn.Module):
    def __init__(self, class_num, dropout_rate ,r):
        super(LPRNet, self).__init__()
        self.r = r
        self.class_num = class_num
        self.outh=1
        self.outw=18
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=int(64/self.r), kernel_size=3, stride=1), # 0
            nn.BatchNorm2d(num_features=int(64/self.r)),
            nn.ReLU(),  # 2
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1)),
            small_basic_block(ch_in=int(64/self.r), ch_out=int(128/self.r)),    # *** 4 ***
            nn.BatchNorm2d(num_features=int(128/self.r)),
            nn.ReLU(),  # 6
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(2, 1, 2)),
            small_basic_block(ch_in=int((int(128/self.r)-1)/2)+1, ch_out=int(256/self.r)),   # 8
            nn.BatchNorm2d(num_features=int(256/self.r)),
            nn.ReLU(),  # 10
            small_basic_block(ch_in=int(256/self.r), ch_out=int(256/self.r)),   # *** 11 ***
            nn.BatchNorm2d(num_features=int(256/self.r)),   # 12
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(4, 1, 2)),  # 14
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=int((int(256/self.r)-1)/4)+1, out_channels=int(256/self.r), kernel_size=(1, 4), stride=1),  # 16
            nn.BatchNorm2d(num_features=int(256/self.r)),
            nn.ReLU(),  # 18
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=int(256/self.r), out_channels=class_num, kernel_size=(1, 1), stride=1), # 20
            nn.BatchNorm2d(num_features=class_num),
            nn.ReLU(),  # *** 22 ***
        )
        self.container = nn.Sequential(
            nn.Conv2d(in_channels=int(256/self.r)+class_num+int(128/self.r)+int(64/self.r), out_channels=self.class_num, kernel_size=(1,1), stride=(1,1)),
            # nn.BatchNorm2d(num_features=self.class_num),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=self.class_num, out_channels=self.lpr_max_len+1, kernel_size=3, stride=2),
            # nn.ReLU(),
        )

    def forward(self, x):
        keep_features = list()
        for i, layer in enumerate(self.backbone.children()):
            x = layer(x)
            if i in [2, 6, 13, 22]: # [2, 4, 8, 11, 22]
                keep_features.append(x)

        global_context = list()
        for i, f in enumerate(keep_features):
            f = F.interpolate(f,size=(self.outh,self.outw),mode='bilinear')
            f_pow = torch.pow(f, 2)
            f_mean = torch.mean(f_pow)
            f = torch.div(f, f_mean)
            global_context.append(f)

        x = torch.cat(global_context, 1)
        x = self.container(x)
        #x = self.backbone(x)
        logits = torch.mean(x, dim=2)

        return logits
        

CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
     '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
     '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
     '新',
     '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
     'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
     'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
     'W', 'X', 'Y', 'Z', 'I', 'O', '-'
     ]
"""
CHARS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
     'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
     'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
     'W', 'X', 'Y', 'Z', 'I', 'O', '-']
"""
if __name__ == "__main__":
    
    from torchsummary import summary  
    
    lprnet = LPRNet(class_num=len(CHARS), dropout_rate=0)
    print(lprnet)
    
    summary(lprnet, (3,24,94), device="cpu")