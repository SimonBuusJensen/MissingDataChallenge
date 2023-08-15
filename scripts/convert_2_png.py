import os
import dotenv

dotenv.load_dotenv()
data_dir = dotenv.get_key(".env", "DATADIR")

"""
Script for converting all JPG images to PNG images
"""

input_dir_name = f"{data_dir}/originals"
output_dir_name = f"{data_dir}/originals_png"

os.makedirs(output_dir_name, exist_ok=True)

# Loop through all files in the directory
if __name__ == "__main__":
    for filename in os.listdir(dir_name):
        if filename.endswith(".jpg"):
            file_name = filename.split(".")[0]
            
            print(f"Converting {file_name}")

            # Read the image
            from PIL import Image
            img = Image.open(f"{input_dir_name}/{file_name}.jpg")

            # Save the image as PNG
            img.save(f"{output_dir_name}/{file_name}.png")

            


