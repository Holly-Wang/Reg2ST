from PIL import Image
import torch
import requests
from transformers import AutoImageProcessor, AutoModel
import numpy as np

# Load an image
# image = Image.open(
#     requests.get(
#         "https://github.com/owkin/HistoSSLscaling/blob/main/assets/example.tif?raw=true",
#         stream=True
#     ).raw
# )

# image = Image.open('example.tif')

# Load phikon-v2
processor = AutoImageProcessor.from_pretrained("../pikonv2")
model = AutoModel.from_pretrained("../pikonv2")
model.eval()

# Process the image
# inputs = processor(image, return_tensors="pt")
for i in range(12):
    t2 = np.load(f"/d/wangx/patches/skin/{i}.npy")

    inputs = {"pixel_values":torch.from_numpy(t2).squeeze(dim=0)}
# Get the features
    with torch.inference_mode():
        outputs = model(**inputs)
        features = outputs.last_hidden_state[:, 0, :]  # (1, 1024) shape
        np.save(f"/d/wangx/phikonv2_embedding/skin/{i}.npy", features.numpy())

# # print(f"Processor inputs: {inputs.keys()}")
# print(inputs["pixel_values"].shape)
# # print(outputs.keys())
# print(features.shape)