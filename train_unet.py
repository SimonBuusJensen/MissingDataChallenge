import torch
import argparse
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
from unet_test_2 import UNet
from inpaint_config import InPaintConfig

class InpaintingDataset(Dataset):
    def __init__(self, img_dir, masked_img_dir, mask_dir, txt_file, transform=None):
        self.img_dir = img_dir
        self.masked_img_dir = masked_img_dir
        self.mask_dir = mask_dir
        self.transform = transform

        with open(txt_file, 'r') as f:
            self.file_list = f.read().splitlines()

        self.augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        ])

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

        #img = self.augmentation(img)
        #masked_img = self.augmentation(masked_img)
        #mask = self.augmentation(mask)

        if self.transform:
            img = self.transform(img)
            masked_img = self.transform(masked_img)
            # Convert the 3-channel mask image to a 1-channel binary mask
            mask = self.transform(mask)
            mask = torch.max(mask, dim=0)[0].unsqueeze_(0)  # Take max across channels and add an extra dimension

        return img, masked_img, mask

class WeightedPerceptualLoss(nn.Module):
    def __init__(self):
        super(WeightedPerceptualLoss, self).__init__()
        # Using VGG16 model for Perceptual Loss
        vgg = models.vgg16(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg.features)[:23]).eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, x, y, mask, weight=10):
        x_features = self.feature_extractor(x)
        y_features = self.feature_extractor(y)

        # Calculate the squared differences
        diff = (x_features - y_features) ** 2

        # Resize the mask to match the feature dimensions.
        # Assumes mask is 1 for inpainting regions and 0 elsewhere.
        # Using interpolate for resizing the mask to the needed dimensions.
        mask_resized = torch.nn.functional.interpolate(mask, size=x_features.shape[2:], mode='nearest')

        # Apply weights
        weighted_diff = diff * (1 + mask_resized * (weight - 1))

        return torch.mean(weighted_diff)

def weighted_mse_loss(input, target, mask, weight=10):
    # Calculate the standard MSE
    mse = (input - target) ** 2

    # Apply weights. Assuming mask has 1s where inpainting is needed and 0s elsewhere.
    weighted_mse = mse * (1 + mask * (weight - 1))

    return torch.mean(weighted_mse)

if __name__ == "__main__":

    args = argparse.ArgumentParser(description='InpaintImages')
    config = InPaintConfig(args)
    settings = config.settings

    data_dir = settings["dirs"]["input_data_dir"]
    img_dir = os.path.join(data_dir, "originals")
    masked_img_dir = os.path.join(data_dir, "masked") 
    mask_dir = os.path.join(data_dir, "masks")
    train_txt_file = os.path.join(data_dir, "data_splits", "training.txt")
    val_txt_file = os.path.join(data_dir, "data_splits", "validation_200.txt")

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Create the datasets
    train_dataset = InpaintingDataset(img_dir, masked_img_dir, mask_dir, train_txt_file, transform=transform)
    val_dataset = InpaintingDataset(img_dir, masked_img_dir, mask_dir, val_txt_file, transform=transform)

    train_batch_size = 10
    # Create the dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=train_batch_size, shuffle=False, num_workers=4)

    # Training loop
    num_epochs = 12

    n_total_steps = len(train_dataloader)
    validate_every_n_steps = 50
    # Placeholder for the best validation loss
    best_val_loss = np.inf

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = UNet().to(device)
    # model.load_state_dict(torch.load(f"{data_dir}/weighed_model_3.ckpt"))

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    #perceptual_loss = PerceptualLoss().to(device)

    weighted_perceptual_loss = WeightedPerceptualLoss().to(device)

    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, verbose=True)
    
    insert_image = cv2.imread(f"{data_dir}/trained_model/average_image.png")
    tensor_image = torch.tensor(insert_image/255, dtype=torch.float32).permute(2, 0, 1)


    for epoch in range(num_epochs):
        for i, (img, masked_img, mask) in enumerate(train_dataloader):


            batch_size = masked_img.shape[0]
            tensor_image_batch = tensor_image.repeat(batch_size, 1, 1, 1)
            expanded_mask = mask.expand_as(masked_img)
            masked_img[expanded_mask == 1] = tensor_image_batch[expanded_mask == 1]

            """
            img_np = masked_img[0].cpu().numpy()
            img_np = img_np.transpose(1, 2, 0)
            img_np = (img_np * 255).astype(np.uint8)

            cv2.imshow('Image', img_np)
            cv2.waitKey(0)
            """

            # Move the data to the GPU
            img, masked_img, mask = img.to(device), masked_img.to(device), mask.to(device)

            model.train()
            optimizer.zero_grad()


            output = model(masked_img, mask)

            weighted_mse = weighted_mse_loss(output,img,mask)

            weighted_percept = weighted_perceptual_loss(output,img,mask)


            total_loss = weighted_mse + weighted_percept

            total_loss.backward()
            optimizer.step()

            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Training Loss: {total_loss.item()}, MSE: {weighted_mse.item()}, WeightedPerceptualLoss: {weighted_percept.item()}")

            # Validate the model every validate_every_n_steps steps
            if (i + 1) % validate_every_n_steps == 0:
                model.eval()
                with torch.no_grad():
                    val_losses = []
                    for img, masked_img, mask in val_dataloader:


                        batch_size = masked_img.shape[0]
                        tensor_image_batch = tensor_image.repeat(batch_size, 1, 1, 1)
                        expanded_mask = mask.expand_as(masked_img)
                        masked_img[expanded_mask == 1] = tensor_image_batch[expanded_mask == 1]
                        # Move the data to the GPU
                        img, masked_img, mask = img.to(device), masked_img.to(device), mask.to(device)


                        output = model(masked_img, mask)

                        weighted_mse = weighted_mse_loss(output,img,mask)

                        weighted_percept = weighted_perceptual_loss(output,img,mask)

                        # Calculate total loss
                        total_loss = weighted_mse + weighted_percept
                        val_losses.append(total_loss.item())

                    avg_val_loss = np.mean(val_losses)
                    scheduler.step(avg_val_loss)
                    print(f"Step [{i+1}/{n_total_steps}], Avg Validation Loss: {avg_val_loss}")

                    # Checkpointing
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        print(f"Improved validation loss {best_val_loss}: Saving model ...")
                        torch.save(model.state_dict(), os.path.join(data_dir, 'weighed_model_3.ckpt'))
