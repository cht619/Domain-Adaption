#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/8/17 19:12
# @Author  : CHT
# @Blog    : https://www.zhihu.com/people/xia-gan-yi-dan-chen-hao-tian
# @Site    : 
# @File    : networks.py
# @Function: 
# @Software: PyCharm


import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd.variable import *


class Classifier(nn.Module):
    def __init__(self, in_dim, out_dim, bottle_neck_dim=500):
        super(Classifier, self).__init__()
        if bottle_neck_dim:
            self.bottleneck = nn.Linear(in_dim, bottle_neck_dim)
            self.fc = nn.Linear(bottle_neck_dim, out_dim)
            self.main = nn.Sequential(
                self.bottleneck,
                nn.Sequential(
                    nn.BatchNorm1d(bottle_neck_dim),
                    nn.LeakyReLU(0.2, inplace=True),
                    self.fc
                ),
                nn.Softmax(dim=-1)
            )
        else:
            self.fc = nn.Linear(in_dim, out_dim)
            self.main = nn.Sequential(
                self.fc,
                nn.Softmax(dim=-1)
            )

        self.net = nn.Sequential(
            nn.Linear(in_dim, 2048),
            nn.BatchNorm1d(2048, 0.8),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024, 0.8),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 10),
            nn.Sigmoid(),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        for module in self.main.children():
            # children 就是 nn.Sequential里面的网络，这里是三层
            # print(module)
            x = module(x)
        # x = self.net(x)
        return x


class AdversarialNetwork(nn.Module):
    def __init__(self):
        super(AdversarialNetwork, self).__init__()
        self.main = nn.Sequential()

    def forward(self, x):
        for module in self.main:  # 第一代结构
            x = module(x)
        return x


class LargeDiscriminator(AdversarialNetwork):
    def __init__(self, in_feature):
        super(LargeDiscriminator, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, 1024)
        self.ad_layer2 = nn.Linear(1024, 1024)
        self.ad_layer3 = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()
        self.main = nn.Sequential(
            self.ad_layer1,
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            self.ad_layer2,
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            self.ad_layer3,
            self.sigmoid
        )

    def forward(self, x):
        for module in self.main:
            x = module(x)
        return x


