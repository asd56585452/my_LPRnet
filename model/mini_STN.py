#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 13:18:36 2019

@author: xingyu
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
from math import floor

class STNet(nn.Module):
    
    def __init__(self,r,resw,resh):
        super(STNet, self).__init__()
        # Spatial transformer localization-network
        self.w=int(((resw-3.0)/1.0)+1)
        self.w=int(((self.w-2.0)/2.0)+1)
        self.w=int(((self.w-5.0)/1.0)+1)
        self.w=int(((self.w-3.0)/3.0)+1)
        self.h=int(((resh-3.0)/1.0)+1)
        self.h=int(((self.h-2.0)/2.0)+1)
        self.h=int(((self.h-5.0)/1.0)+1)
        self.h=int(((self.h-3.0)/3.0)+1)
        if self.w<1:
            self.w=1
        if self.h<1:
            self.h=1
        self.r32=int(32/r)
        print(resw,resh,self.w,self.h,self.r32)
        
        self.localization = nn.Sequential(
                nn.Conv2d(3, self.r32, kernel_size=3),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True),
                nn.Conv2d(self.r32, self.r32, kernel_size=5),
                nn.MaxPool2d(3, stride=3),
                nn.ReLU(True)
                )
        # Regressor for the 3x2 affine matrix
        self.fc_loc = nn.Sequential(
                nn.Linear( self.r32*self.h*self.w , self.r32),
                nn.ReLU(True),
                nn.Linear(self.r32, 3*2)
                )
        # Initialize the weights/bias with identity transformation 
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1,0,0,0,1,0], dtype=torch.float))
        
    def forward(self, x):
        # xx = F.interpolate(x,size=(24,94),mode='bilinear')
        xs = self.localization(x)
        xs = xs.view(-1, self.r32*self.h*self.w )
        theta = self.fc_loc(xs)
        theta = theta.view(-1,2,3)
        
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        
        return x
        
    
if __name__ == "__main__":
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = STNet().to(device)
    
    input = torch.Tensor(2, 3, 24, 94).to(device)
    output = model(input)
    print('output shape is', output.shape)