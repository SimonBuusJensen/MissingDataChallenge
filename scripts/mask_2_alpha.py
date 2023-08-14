


import os
import dotenv
from PIL import Image

dotenv.load_dotenv()
data_dir = dotenv.get_key(".env", "DATADIR")

"""
Script for opening a mask image and converting white pixels to transparent (alpha channel)
"""

mask_dir = f"{data_dir}/masks"
masked_transparent_dir = f"{data_dir}/masks_transparent"
os.makedirs(masked_transparent_dir, exist_ok=True)

# Iterate through all files in the directory
for filename in os.listdir(mask_dir):

    image_fn = f"{mask_dir}/{filename}"

    img = Image.open(image_fn)

    # Convert white pixels to transparent
    img = img.convert("RGBA")
    datas = img.getdata()

    newData = []
    for item in datas:
        if item[0] == 255 and item[1] == 255 and item[2] == 255:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)

    img.putdata(newData)

    # Save the image as PNG
    img.save(f"{masked_transparent_dir}/{os.path.basename(image_fn)}")