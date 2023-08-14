import openai
import dotenv
import requests
from PIL import Image
from io import BytesIO

dotenv.load_dotenv()

# Read the API key from .env file
openai.api_key = dotenv.get_key(".env", "OPENAIKEY")
data_dir = dotenv.get_key(".env", "DATADIR")

file_name="00000001_005" 

response = openai.Image.create_edit(
  image=open(f"{data_dir}/{file_name}_stroke_masked.png", "rb"),
  mask=open(f"{data_dir}/masks/{file_name}_stroke_mask.png", "rb"),
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

img.save(f"{data_dir}/{file_name}.png")