# align_individual_fixed.py
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from facenet_pytorch import MTCNN
import numpy as np
from utils.alignment.arcface import norm_crop  # iDiff utility

# ------------------- Pfade -------------------
INPUT_DIR = "out/id_samples"           # Ordner mit generierten Einzelbildern
OUTPUT_DIR = "out/id_samples_aligned"  # Ordner für aligned Bilder
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALIGNED_SIZE = 112  # Größe nach norm_crop

# ------------------- MTCNN Setup -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(keep_all=True, min_face_size=20, post_process=False, device=device)

# ------------------- Main Loop -------------------
skipped = []

for fname in os.listdir(INPUT_DIR):
    if not (fname.endswith(".png") or fname.endswith(".jpg")):
        continue

    input_path = os.path.join(INPUT_DIR, fname)
    output_path = os.path.join(OUTPUT_DIR, fname)

    try:
        # Bild laden
        img = Image.open(input_path).convert("RGB")
        img_np = np.array(img)
        if img_np.dtype != np.uint8:
            img_np = (img_np * 255).astype(np.uint8)

        # Gesichter erkennen
        boxes, _, landmarks = mtcnn.detect([img_np], landmarks=True)

        if boxes is None or landmarks is None:
            # Fallback: einfache Resize
            aligned = transforms.functional.resize(img, ALIGNED_SIZE)
            print(f"{fname}: 0 faces detected, fallback resize applied")

        else:
            # Wähle Gesicht, das am nächsten zur Bildmitte liegt
            box_centers = np.mean(boxes[0], axis=1)
            img_center = np.array([img_np.shape[1]/2, img_np.shape[0]/2])
            idx = np.argmin(np.sum((box_centers - img_center)**2, axis=1))
            facial5points = landmarks[0][idx]

            # norm_crop
            aligned_img = norm_crop(img_np, landmark=facial5points, image_size=ALIGNED_SIZE, createEvalDB=True)
            aligned = torch.from_numpy(aligned_img).permute(2,0,1)/255.0

        # Speichern
        save_image(aligned, output_path)
        print(f"Aligned {fname}")

    except Exception as e:
        print(f"Skipped {fname} due to {e}")
        skipped.append(fname)

# ------------------- Log -------------------
if skipped:
    with open(os.path.join(OUTPUT_DIR, "skipped.txt"), "w") as f:
        for s in skipped:
            f.write(s + "\n")

print(f"Finished aligning. Skipped {len(skipped)} images.")
