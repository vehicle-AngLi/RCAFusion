# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Self_Attention(nn.Module):
    def __init__(self, channel):
        super(Self_Attention, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel, kernel_size=1)
        self.sig = nn.Sigmoid()
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=1)
        self.conv3 = nn.Conv2d(channel, channel, kernel_size=1)

    def forward(self,x):
        x1 = self.conv1(x)
        x1 = self.sig(x1)
        x1 = torch.transpose(x1, dim0=3, dim1=2)
        x2 = self.conv2(x)
        x2 = self.sig(x2)
        # print('\n', x1.size(), x2.size())
        x3 = torch.matmul(x1,x2)
        x4 = torch.matmul(x,x3)
        out = self.conv3(x4)
        return self.sig(out)


class Rubik_Cube_Attention(nn.Module):
    def __init__(self, channel, ratio=16, kernel_size=7):
        super(Rubik_Cube_Attention, self).__init__()
        self.channel_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.line1 = nn.Conv2d(channel, channel // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.line2 = nn.Conv2d(channel // ratio, channel, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.space = nn.Conv2d(1, 1, kernel_size, padding = 3, bias = False)
        self.satt = Self_Attention(channel)
        self.conv = nn.Conv2d(channel, channel, 1)

    def forward(self,x):
        ##训练通道权重
        ##channel attention
        x_ca = self.channel_avg_pool(x)
        x_ca = self.relu(self.line1(x_ca))
        x_ca = self.relu(self.line2(x_ca))
        x_ca = self.sigmoid(x_ca)
        ##同时训练空间权重
        ##spatial attention
        x_sa = torch.mean(x, dim=1, keepdim=True)
        x_sa = self.space(x_sa)
        x_sa = self.sigmoid(x_sa)
        ##直接相乘
        ##multiple
        x_attention = x*x_sa*x_ca
        ##训练自注意力机制
        ##self-attention
        x_self = self.satt(x)
        ##相加
        ##addition
        out = 0.1*x_self+x_attention
        out = self.sigmoid(self.conv(out))
        return out

