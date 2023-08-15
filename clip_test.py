import torch
import clip
from PIL import Image
import dotenv

dotenv.load_dotenv()
data_dir = dotenv.get_key(".env", "DATADIR")

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image_fn = f"{data_dir}/originals_png/00000001_000.png"
image = preprocess(Image.open(image_fn)).unsqueeze(0).to(device)
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]