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
                for patch_size in range(5,100):
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


inpaint_func_dict = {
    "MeanImageInpaint": inpaint_one_image_meanimage,
    "MeanImageButBetter": inpaint_one_image_meanimagebutbetter,
    "PatchInpaint": inpaint_one_image_patches,
    "PatchInpaintAvg": inpaint_one_image_patches_avg,
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
