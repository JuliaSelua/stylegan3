# encode_individual_noalign_recursive.py
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

from utils.helpers import normalize_to_neg_one_to_one
from utils.iresnet import iresnet100  # ElasticFace Backbone

SAMPLES_DIR = "out/id_samples"        # Ordner mit ID-Unterordnern
EMBEDDINGS_DIR = "out/embeddings"
IMAGE_SIZE = 112
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# ------------------- Backbone -------------------
face_backbone = iresnet100(num_features=512)
ckpt = torch.load(os.path.join("utils", "Elastic_R100_295672backbone.pth"), map_location="cpu")
face_backbone.load_state_dict(ckpt)
face_backbone = face_backbone.to(DEVICE)
face_backbone.eval()

# ------------------- Encode Images -------------------
embeddings = []
labels = []

for id_folder in sorted(os.listdir(SAMPLES_DIR)):
    id_path = os.path.join(SAMPLES_DIR, id_folder)
    if not os.path.isdir(id_path):
        continue

    for fname in sorted(os.listdir(id_path)):
        if not fname.endswith((".png", ".jpg")):
            continue

        input_path = os.path.join(id_path, fname)

        # Bild laden & mittig croppen
        img = Image.open(input_path).convert("RGB")
        w, h = img.size
        min_side = min(w, h)
        left = (w - min_side) // 2
        top = (h - min_side) // 2
        img_cropped = img.crop((left, top, left + min_side, top + min_side))
        img_resized = img_cropped.resize((IMAGE_SIZE, IMAGE_SIZE))

        img_tensor = transforms.functional.to_tensor(img_resized).unsqueeze(0)
        img_tensor = normalize_to_neg_one_to_one(img_tensor).to(DEVICE)

        # Embedding
        with torch.no_grad():
            emb = face_backbone(img_tensor)
            emb = torch.nn.functional.normalize(emb)

        embeddings.append(emb.cpu().numpy()[0])
        labels.append(id_folder)

embeddings = np.array(embeddings)
labels = np.array(labels)
np.save(os.path.join(EMBEDDINGS_DIR, "embeddings.npy"), embeddings)
np.save(os.path.join(EMBEDDINGS_DIR, "labels.npy"), labels)

print("Encoding complete.")
print(f"Saved {len(embeddings)} embeddings to {EMBEDDINGS_DIR}")

