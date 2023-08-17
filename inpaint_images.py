import argparse
from skimage import io
import os
import pathlib
import numpy as np
from inpaint_config import InPaintConfig
from inpaint_tools import read_file_list
from tqdm import tqdm




def inpaint_one_image_meanimage(in_image, mask_image, avg_image):
    mask_image = np.squeeze(mask_image)
    inpainted_mask = np.copy(avg_image)
    inpainted_mask[mask_image == 0] = 0

    inpaint_image = inpainted_mask + in_image
    return inpaint_image

def inpaint_one_image_meanimagebutbetter(in_image, mask_image, avg_image, *args, **kwargs):
    mask_image = np.squeeze(mask_image)

    m = np.mean(in_image[mask_image == 0], axis=0)

    in_image[mask_image == 255] = m

    # center the avg image around 0
    avg_image = avg_image - np.mean(avg_image, axis=0)
    avg_image = avg_image + m
    # clip the avg image to 0-255
    avg_image = np.clip(avg_image, 0, 255)
    in_image[mask_image == 255] = avg_image[mask_image == 255]

    return in_image.astype(np.uint8)

def inpaint_one_image_patches(in_image, mask_image, avg_image):
    mask_image = np.copy(np.squeeze(mask_image))
    inpainted_mask = np.copy(in_image)

    # get list of masked indices
    mask_indices = np.argwhere(mask_image == 255)
    # randomize the list
    np.random.shuffle(mask_indices)

    # iterate over mask indices
    #while np.sum(mask_image) > 0:
        #print(np.sum(mask_image))
    for i, j in mask_indices:
            if mask_image[i, j] == 255:
                # take a 20x20 patch around the pixel
                for patch_size in range(10,100):
                    #print(patch_size)
                    patch = inpainted_mask[max(0, i-patch_size):min(in_image.shape[0], i+patch_size),
                                    max(0, j-patch_size):min(in_image.shape[1], j+patch_size), :]
                    # select non-masked pixels
                    patch = patch[mask_image[max(0, i-patch_size):min(in_image.shape[0], i+patch_size),
                                                max(0, j-patch_size):min(in_image.shape[1], j+patch_size)] == 0]
                    if not len(patch):
                        continue

                    inpainted_mask[i, j] = patch.mean(axis=0)
                    mask_image[i, j] = 0
                    break
            
    return inpainted_mask

def inpaint_one_image_patches_avg(in_image, mask_image, avg_image):
    inpainted = inpaint_one_image_patches(in_image, mask_image, None)

    # center the avg image around 0
    
    avg_image = avg_image - np.mean(avg_image, axis=0)
    avg_image = avg_image*2
    avg_image = avg_image + inpainted
    # clip the avg image to 0-255
    avg_image = np.clip(avg_image, 0, 255)
    inpainted[mask_image == 255] = avg_image[mask_image == 255]

    return inpainted.astype(np.uint8)

def inpaint_one_image_symmetry(in_image, mask_img, avg_image):
    # create mask for left side

    
    tf_mask = np.copy(mask_img)

    # create a mask selecting the left side
    left_mask = np.zeros_like(mask_img)
    left_mask[:, :in_image.shape[1]//2] = 1
    # create a mask selecting the right side
    right_mask = np.zeros_like(mask_img)
    right_mask[:, in_image.shape[1]//2:] = 1

    flipped = np.flip(in_image, axis=1)
    mask_flipped = np.flip(mask_img, axis=1)

    
    inpainted = np.copy(in_image)
    # figure out which side has more masked pixels
    left_masked = np.sum(left_mask)
    right_masked = np.sum(right_mask)
    # flip the template and fill in the masked pixels
    if left_masked > right_masked:
        side_mask = right_mask
    else:
        side_mask = left_mask
    
    idx = np.nonzero(np.logical_and(side_mask == 1, mask_img == 255))
    #print(len(idx))
    inpainted[idx] = flipped[idx]
    tf_mask[idx] = mask_flipped[idx]

    patch_inpainted = inpaint_one_image_patches(inpainted, tf_mask, avg_image=None)

    inpainted[tf_mask == 255] = patch_inpainted[tf_mask == 255]
    
    return inpainted

def inpaint_one_image_nonet(in_image, mask_image, avg_image):
    import torch
    from unet_test_raw import UNet
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_path = "MissingDataOpenData/weighed_model_3.ckpt"
    model = UNet().to(device)
    
    #tensor_image = torch.tensor(avg_image/255, dtype=torch.float32).permute(2, 0, 1)
    

    # Load the pre-trained weights into the model
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    masked_imgs = transform(in_image).unsqueeze(0)
    masks = transform(mask_image).unsqueeze(0)

    tensor_image = torch.randn_like(masked_imgs)

    batch_size = masked_imgs.shape[0]
    #tensor_image_batch = tensor_image.repeat(batch_size, 1, 1, 1)
    expanded_mask = masks.expand_as(masked_imgs)
    masked_imgs[expanded_mask == 1] = tensor_image[expanded_mask == 1]

    masked_imgs, masks = masked_imgs.to(device), masks.to(device)

    with torch.no_grad():
        output = model(masked_imgs, masks)

    output = output[0].detach().cpu().numpy().transpose(1, 2, 0)

    result = masked_imgs[0].detach().cpu().numpy().transpose(1, 2, 0)
    #result[mask_image == 255] = output[mask_image == 255]
    result = output

    # turn to 0-255
    result = result * 255
    result = result.astype(np.uint8)
    return result

def inpaint_one_image_ynet(in_image, mask_image, avg_image):
    import torch
    from unet_test_raw import UNet
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_path = "results/trained_model/raw_unet_2.ckpt"
    model = UNet().to(device)
    
    tensor_image = torch.tensor(avg_image/255, dtype=torch.float32).permute(2, 0, 1)

    # Load the pre-trained weights into the model
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    masked_imgs = transform(in_image).unsqueeze(0)
    masks = transform(mask_image).unsqueeze(0)

    batch_size = masked_imgs.shape[0]
    tensor_image_batch = tensor_image.repeat(batch_size, 1, 1, 1)
    expanded_mask = masks.expand_as(masked_imgs)
    masked_imgs[expanded_mask == 1] = tensor_image_batch[expanded_mask == 1]

    masked_imgs, masks = masked_imgs.to(device), masks.to(device)

    with torch.no_grad():
        output = model(masked_imgs, masks)

    output = output[0].detach().cpu().numpy().transpose(1, 2, 0)

    result = masked_imgs[0].detach().cpu().numpy().transpose(1, 2, 0)
    #result[mask_image == 255] = output[mask_image == 255]
    result = output

    # turn to 0-255
    result = result * 255
    result = result.astype(np.uint8)
    return result

def inpaint_one_image_lama(in_image, mask_image, *args):
    from lama_cleaner.model.lama import LaMa
    import torch

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    lama = LaMa(device)
    from lama_cleaner.schema import Config
    config=Config(ldm_steps=25, hd_strategy="original", hd_strategy_crop_margin=100, hd_strategy_crop_trigger_size=0, hd_strategy_resize_limit=1)

    res = lama(in_image,mask_image,config).astype(np.uint8)
    # swap bgr to rgb
    res = res[:, :, ::-1]
    
    inpainted = np.copy(in_image)
    inpainted[mask_image == 255] = res[mask_image == 255]
    return inpainted

inpaint_func_dict = {
    "MeanImageInpaint": inpaint_one_image_meanimage,
    "MeanImageButBetter": inpaint_one_image_meanimagebutbetter,
    "PatchInpaint": inpaint_one_image_patches,
    "PatchInpaintAvg": inpaint_one_image_patches_avg,
    "YNet": inpaint_one_image_ynet,
    "WonkyCats": inpaint_one_image_symmetry,
    "PirateCats": inpaint_one_image_lama,
    "NoNet": inpaint_one_image_nonet
}


def inpaint_images(settings):
    input_data_dir = settings["dirs"]["input_data_dir"]
    output_data_dir = settings["dirs"]["output_data_dir"]
    data_set = settings["data_set"]
    model_dir = os.path.join(output_data_dir, "trained_model")
    method = settings["training_params"]["method"]

    inpainted_result_dir = os.path.join(output_data_dir, f"inpainted_{data_set}_{method}")
    pathlib.Path(inpainted_result_dir).mkdir(parents=True, exist_ok=True)

    inpaint_func = inpaint_func_dict[method]

    print(f"InPainting {data_set} and placing results in {inpainted_result_dir} with model from {model_dir}")

    avg_img_name = os.path.join(model_dir, "average_image.png")
    avg_img = io.imread(avg_img_name)

    file_list = os.path.join(input_data_dir, "data_splits", data_set + ".txt")
    file_ids = read_file_list(file_list)
    if file_ids is None:
        return

    print(f"Inpainting {len(file_ids)} images")

    for idx in tqdm(file_ids):
        in_image_name = os.path.join(input_data_dir, "masked", f"{idx}_stroke_masked.png")
        in_mask_name = os.path.join(input_data_dir, "masks", f"{idx}_stroke_mask.png")
        out_image_name = os.path.join(inpainted_result_dir, f"{idx}.png")

        im_masked = io.imread(in_image_name)
        im_mask = io.imread(in_mask_name)

        inpainted_image = inpaint_func(im_masked, im_mask, avg_img)
        io.imsave(out_image_name, inpainted_image)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='InpaintImages')
    config = InPaintConfig(args)
    if config.settings is not None:
        inpaint_images(config.settings)
