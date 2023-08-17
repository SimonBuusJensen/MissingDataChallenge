import torch
from torch import nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import torch.optim as optim
from pytorch_msssim import ssim
import numpy as np
import torchvision.models as models
import cv2
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import InstanceNorm2d, BatchNorm2d

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Using a pretrained resnet34 model as the encoder
        self.resnet = models.resnet34(pretrained=True)

        self.intial_upsample = nn.Upsample(scale_factor=512/360.0, mode='bilinear', align_corners=True)

        self.upsample_to_original = nn.Upsample(size=(360,360), mode='bilinear', align_corners=True)

        original_first_layer = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Copy the pre-trained weights over
        with torch.no_grad():
            self.resnet.conv1.weight[:, :3] = original_first_layer.weight
            # Initialize the weights for the 4th channel
            self.resnet.conv1.weight[:, 3].uniform_(-0.1, 0.1)

        self.attention1 = SelfAttention(256)
        self.attention2 = SelfAttention(512)
        self.attention3 = SelfAttention(64)

        # Decoder
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            InstanceNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            InstanceNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            InstanceNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            InstanceNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            InstanceNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            InstanceNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.up4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            InstanceNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            InstanceNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )


        self.final = nn.Conv2d(64, 3, kernel_size=1) # Assuming the input image is RGB

        self.final_act = nn.Sigmoid()

    def forward(self, x, mask):
        # Combine mask and image as input
        x = torch.cat([x, mask], dim=1)  # Assuming mask has the same number of channels as x

        # Encoder path
        up_input = self.intial_upsample(x)

        x1 = self.resnet.conv1(up_input)
        x1 = self.resnet.bn1(x1)
        x1 = self.resnet.relu(x1)
        #print(x1.shape)
        x2 = self.resnet.layer1(self.resnet.maxpool(x1))
        #print(x2.shape)
        x3 = self.resnet.layer2(x2)
        #print(x3.shape)
        x4 = self.resnet.layer3(x3)
        #print(x4.shape)
        x5 = self.resnet.layer4(x4)
        x5 = self.attention2(x5)
        #print(x5.shape)

        # Decoder path
        x = self.up1(x5)
        x = self.dec1(torch.cat([x, x4], dim=1))
        #print(x.shape)
        x = self.attention1(x)
        x = self.up2(x)
        x = self.dec2(torch.cat([x, x3], dim=1))
        x = self.up3(x)
        x = self.dec3(torch.cat([x, x2], dim=1))
        #x = self.attention3(x)
        x = self.up4(x)
        x = self.dec4(torch.cat([x, x1], dim=1))

        #print(x.shape)

        x = self.upsample_to_original(x)
        # Final layer
        x = self.final(x)

        x = self.final_act(x)

        return x
# Instantiate the model

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels//8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels//8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, x):
        B, C, W, H = x.size()
        Q = self.query(x).view(B, -1, W*H).permute(0, 2, 1)
        K = self.key(x).view(B, -1, W*H)
        V = self.value(x).view(B, -1, W*H)

        attention = self.softmax(torch.bmm(Q, K) / (C**0.5))
        out = torch.bmm(V, attention.permute(0, 2, 1))
        return out.view(B, C, W, H)
