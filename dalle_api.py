import openai
import dotenv
import requests
import os
from PIL import Image
from io import BytesIO

# Read the API key from .env file
dotenv.load_dotenv()
openai.api_key = dotenv.get_key(".env", "OPENAIKEY")
data_dir = dotenv.get_key(".env", "DATADIR")

file_name="00000645_003" 

input_img = f"{data_dir}/masked/{file_name}_stroke_masked.png"
mask_img = f"{data_dir}/masks_transparent/{file_name}_stroke_mask.png"
assert os.path.exists(input_img) and os.path.exists(mask_img), f"Input or mask image does not exist: {input_img} {mask_img}"

output_dir = f"{data_dir}/inference"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_fn = os.path.join(output_dir, file_name + ".png")
assert not os.path.exists(output_fn), f"Already performed inference for {file_name}"

response = openai.Image.create_edit(
  image=open(input_img, "rb"),
  mask=open(mask_img, "rb"),
  prompt="A cat",
  n=1,
  size="512x512"
)
image_url = response['data'][0]['url']

print(image_url)

# Download the image
response = requests.get(image_url)
img = Image.open(BytesIO(response.content))

# Resize to 360x360
img = img.resize((360, 360), Image.LANCZOS)
img.save(output_fn)