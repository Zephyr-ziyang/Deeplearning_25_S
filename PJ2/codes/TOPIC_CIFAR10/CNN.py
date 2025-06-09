import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import seaborn as sns
import warnings
import os
import time
import json
import copy
from tqdm import tqdm
import math

# 定义高性能CNN模型（支持多种配置）
class HighPerfCIFAR10Model(nn.Module):
    def __init__(self, num_classes=10, channels=[64, 128, 256], activation=nn.LeakyReLU(0.1)):
        super(HighPerfCIFAR10Model, self).__init__()
        self.activation = activation
        
        # 初始卷积块
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[0]),
            activation,
            nn.Conv2d(channels[0], channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[0]),
            activation,
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2)
        )
        
        # 残差块1
        self.res_block1 = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[1]),
            activation,
            nn.Conv2d(channels[1], channels[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[1]),
            activation,
        )
        
        self.shortcut1 = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], kernel_size=1, stride=1),
            nn.BatchNorm2d(channels[1])
        )

        # 卷积块2
        self.conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3)
        )
        
        # 残差块2
        self.res_block2 = nn.Sequential(
            nn.Conv2d(channels[1], channels[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[2]),
            activation,
            nn.Conv2d(channels[2], channels[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[2]),
            activation,
        )
        
        self.shortcut2 = nn.Sequential(
            nn.Conv2d(channels[1], channels[2], kernel_size=1, stride=1),
            nn.BatchNorm2d(channels[2])
        )
        
        # 最终分类层
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(channels[2], 512),
            nn.BatchNorm1d(512),
            activation,
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        
        # 第一个残差连接
        identity = self.shortcut1(x)
        x = self.res_block1(x)
        x = x + identity
        x = self.activation(x)
        
        x = self.conv2(x)
        
        # 第二个残差连接
        identity = self.shortcut2(x)
        x = self.res_block2(x)
        x = x + identity
        x = self.activation(x)
        
        x = self.fc(x)
        return x