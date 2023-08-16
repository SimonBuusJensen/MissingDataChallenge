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
from torchviz import make_dot
from torch.nn import InstanceNorm2d, BatchNorm2d


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.intial_upsample = nn.Upsample(size=(512,512), mode='bilinear', align_corners=True)

        self.upsample_to_original = nn.Upsample(size=(360,360), mode='bilinear', align_corners=True)


        #self.attention0 = SelfAttention(512)
        #self.attention1 = SelfAttention(256)
        #self.attention2 = SelfAttention(128)
        #self.attention3 = SelfAttention(64)
        #self.attention4 = SelfAttention(64)

        self.enc1 = self._block(4, 64)  # input has 4 channels because of 3 for image and 1 for mask
        self.enc2 = self._block(64, 128)
        self.enc3 = self._block(128, 256)
        self.enc4 = self._block(256, 512)

        self.b = self.bottleneck(512,512)

        # Decoder
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding = 1)
        self.dec1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2,padding = 1)
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2,padding = 1)
        self.dec3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.up4 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2,padding = 1)
        self.dec4 = nn.Sequential(
            nn.Conv2d(68, 64, kernel_size=3, padding=1),
            BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.up5 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2,padding = 1)
        self.dec5 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.final = nn.Conv2d(64, 3, kernel_size=1) # Assuming the input image is RGB

        self.final_act = nn.Sigmoid()

    def _block(self, in_channels, out_channels, kernel_size=3):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=1, stride=2), # Adding stride to replace max-pooling
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def bottleneck(self, in_channels, out_channels, kernel_size=3):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=1), # Adding stride to replace max-pooling
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, mask):
        # Combine mask and image as input
        x0 = torch.cat([x, mask], dim=1)  # Assuming mask has the same number of channels as x
        # Encoder path
        x_up_init = self.intial_upsample(x0)

        x1 = self.enc1(x_up_init)

        #print("x1",x1.shape)
        x2 = self.enc2(x1)
        #print("x2",x2.shape)
        x3 = self.enc3(x2)
        #print("x3",x3.shape)
        x4 = self.enc4(x3)
        #print("x4",x4.shape)
        #x5 = self.resnet.layer4(x4)
        #print("x5",x5.shape)
        #x5 = self.attention2(x5)
        x = self.b(x4)
        #print("bottleneck", x.shape)

        # Decoder path
        x = self.up1(x)
        #print("up1",x.shape)
        x = self.dec1(torch.cat([x, x3], dim=1))
        #x = self.attention1(x)
        #print("x1",x.shape)

        x = self.up2(x)
        #print("up2", x.shape)
        x = self.dec2(torch.cat([x, x2], dim=1))
        #self.attention2(x)
        #print("x2",x.shape)

        x = self.up3(x)
        #print("up3", x.shape)
        x = self.dec3(torch.cat([x, x1], dim=1))
        #x = self.attention3(x)
        #print("x3",x.shape)

        x = self.up4(x)
        #print("up4", x.shape, x_up_init.shape)
        x = self.dec4(torch.cat([x, x_up_init], dim=1))
        #x = self.attention4(x)
        #print("x4",x.shape)

        # Final layer
        x = self.final(x)

        x = self.final_act(x)

        x = self.upsample_to_original(x)
        return x


"""
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
# Instantiate the model

def verify_model_shape(model):
    # Random tensor for the image and mask
    img = torch.rand(2, 3, 360, 360)
    mask = torch.rand(2, 1, 360, 360)

    # Pass the tensor through the model

    output = model(img, mask)
    print("Output Shape",output.shape)
    #dot = make_dot(output)
    #dot.view()
    # Check the output shape

    print("The model is set up correctly.")




model = UNet()
verify_model_shape(model)
"""
