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
import random
import torchvision.models as models
from torch.nn import InstanceNorm2d, BatchNorm2d
import cv2
from unet_test_2 import UNet

class InpaintingDataset(Dataset):
    def __init__(self, img_dir, masked_img_dir, mask_dir, txt_file, transform=None):
        self.img_dir = img_dir
        self.masked_img_dir = masked_img_dir
        self.mask_dir = mask_dir
        self.transform = transform

        with open(txt_file, 'r') as f:
            self.file_list = f.read().splitlines()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        base_name = self.file_list[idx]

        img_path = os.path.join(self.img_dir, base_name + '.jpg')
        masked_img_path = os.path.join(self.masked_img_dir, base_name + '_stroke_masked.png')
        mask_path = os.path.join(self.mask_dir, base_name + '_stroke_mask.png')

        img = Image.open(img_path).convert("RGB")
        masked_img = Image.open(masked_img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")

        if self.transform:
            img = self.transform(img)
            masked_img = self.transform(masked_img)
            # Convert the 3-channel mask image to a 1-channel binary mask
            mask = self.transform(mask)
            mask = torch.max(mask, dim=0)[0].unsqueeze_(0)  # Take max across channels and add an extra dimension

        return img, masked_img, mask

def visualize_batch(dataloader, num_imgs=4):
    # Get a batch of data
    imgs, masked_imgs, masks = next(iter(dataloader))
    imgs = imgs.cpu().numpy().transpose((0, 2, 3, 1))  # Move to CPU and convert to numpy
    masked_imgs = masked_imgs.cpu().numpy().transpose((0, 2, 3, 1))
    masks = masks.cpu().numpy().transpose((0, 2, 3, 1))

    fig, axs = plt.subplots(num_imgs, 3, figsize=(15, num_imgs * 5))
    for i in range(num_imgs):
        axs[i, 0].imshow(imgs[i])
        axs[i, 0].title.set_text('Original Image')

        axs[i, 1].imshow(masked_imgs[i])
        axs[i, 1].title.set_text('Masked Image')

        axs[i, 2].imshow(masks[i, :, :, 0], cmap='gray')  # We only need to display one channel of the mask
        axs[i, 2].title.set_text('Mask')

    plt.show()


def visualize(model, model_path, val_dataloader, device):
    # Load the pre-trained weights into the model
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    # Create an iterator from the dataloader
    data_iter = iter(val_dataloader)

    for i in range(50):
        # Take a batch from the dataloader

        imgs, masked_imgs, masks = next(data_iter)

        masked_img_copy = torch.clone(masked_imgs)

        batch_size = masked_imgs.shape[0]
        tensor_image_batch = tensor_image.repeat(batch_size, 1, 1, 1)
        expanded_mask = masks.expand_as(masked_imgs)
        masked_imgs[expanded_mask == 1] = tensor_image_batch[expanded_mask == 1]

        imgs, masked_imgs, masks = imgs.to(device), masked_imgs.to(device), masks.to(device)

        with torch.no_grad():
            output = model(masked_imgs, masks)

        # Plot the images
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))

        axs[0].imshow(imgs[0].cpu().numpy().transpose(1, 2, 0))
        axs[0].set_title('Original Image')

        axs[1].imshow(masked_imgs[0].cpu().numpy().transpose(1, 2, 0))
        axs[1].set_title('Masked Image')

        axs[2].imshow(masks[0].cpu().numpy().transpose(1, 2, 0))
        axs[2].set_title('Mask')

        axs[3].imshow(output[0].detach().cpu().numpy().transpose(1, 2, 0))


        axs[3].set_title('Reconstructed Image')

        for ax in axs:
            ax.axis('off')

        plt.show()


insert_image = cv2.imread("/home/pabllo/summer_school_ws/src/MissingDataChallenge/trained_model/average_image.png")
tensor_image = torch.tensor(insert_image/255, dtype=torch.float32).permute(2, 0, 1)


# Define the transform (normalize to [0,1] since images are 8-bit)
transform = transforms.Compose([
    transforms.ToTensor()
])

# Specify the directories and text files
img_dir = '/home/pabllo/summer_school_ws/src/cats/MissingDataOpenData/originals'
masked_img_dir = '/home/pabllo/summer_school_ws/src/cats/MissingDataOpenData/masked'
mask_dir = '/home/pabllo/summer_school_ws/src/cats/MissingDataOpenData/masks'
train_txt_file = '/home/pabllo/summer_school_ws/src/cats/MissingDataOpenData/data_splits/training.txt'
val_txt_file = '/home/pabllo/summer_school_ws/src/cats/MissingDataOpenData/data_splits/validation_200.txt'

# Create the datasets
train_dataset = InpaintingDataset(img_dir, masked_img_dir, mask_dir, train_txt_file, transform=transform)
val_dataset = InpaintingDataset(img_dir, masked_img_dir, mask_dir, val_txt_file, transform=transform)

# Create the dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)


# Verify the model's shape
model = UNet()
#verify_model_shape(model)
#visualize_batch(train_dataloader)

#visualize(model,"/home/pabllo/summer_school_ws/src/cats/unet_attention_activ.ckpt", train_dataloader, "cuda")
#visualize(model,"/home/pabllo/summer_school_ws/src/cats/unet_fourier_percept.ckpt", train_dataloader, "cuda")
visualize(model,"/home/pabllo/summer_school_ws/src/MissingDataChallenge/weighed_model_2.ckpt", val_dataloader, "cuda")
