import torch
import clip
from PIL import Image
import dotenv
from matplotlib import pyplot as plt

dotenv.load_dotenv()
data_dir = dotenv.get_key(".env", "DATADIR")

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image_fn = f"{data_dir}/originals_png/00000001_000.png"

img = Image.open(image_fn)

preprocessed_image = preprocess(img).unsqueeze(0).to(device)
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

# Display the original and preprocessed image:
plt.figure(figsize=(16, 16))
plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(img)
plt.axis("off")
plt.subplot(1, 2, 2)
plt.title("Preprocessed")
plt.imshow(preprocessed_image[0].cpu().permute(1, 2, 0))
plt.axis("off")
plt.show()


with torch.no_grad():
    image_features = model.encode_image(preprocessed_image)

    # print size of image features and processed image
    print("Image features shape", image_features.shape)
    print("Preprocessed image shape", preprocessed_image.shape)

    text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(preprocessed_image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]