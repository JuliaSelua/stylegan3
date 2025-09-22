# align_id_grids.py
import os
from PIL import Image
import torch
import torchvision
from torchvision.utils import save_image, make_grid
import torchvision.transforms as transforms
from facenet_pytorch import MTCNN
import numpy as np
from utils.alignment.arcface import norm_crop  # iDiff arcface utility

# Pfade
INPUT_DIR = "out/id_samples"  # Ordner mit deinen Grids
OUTPUT_DIR = "out/id_samples_aligned"  # Ordner für aligned Grids
os.makedirs(OUTPUT_DIR, exist_ok=True)

GRID_IMAGE_SIZE = 128  # Größe der Einzelbilder im Grid (wie beim Generieren)
ALIGNED_SIZE = 112     # Größe nach norm_crop

# Face Detector
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(keep_all=True, min_face_size=1, post_process=False, device=device)

# Hilfsfunktion: Split Grid in Einzelbilder
def split_grid(img_path, grid_size):
    img = Image.open(img_path).convert("RGB")
    img_tensor = transforms.functional.to_tensor(img)
    nrows = img_tensor.shape[1] // grid_size
    ncols = img_tensor.shape[2] // grid_size
    tiles = []
    for r in range(nrows):
        for c in range(ncols):
            tile = img_tensor[:, r*grid_size:(r+1)*grid_size, c*grid_size:(c+1)*grid_size]
            tiles.append(tile)
    return torch.stack(tiles)

# Main Loop
skipped = []

for fname in os.listdir(INPUT_DIR):
    if not (fname.endswith(".png") or fname.endswith(".jpg")):
        continue

    input_path = os.path.join(INPUT_DIR, fname)
    output_path = os.path.join(OUTPUT_DIR, fname)

    try:
        # 1. Split Grid
        tiles = split_grid(input_path, GRID_IMAGE_SIZE)
        aligned_tiles = []

        # 2. Align Faces
        for tile in tiles:
            img_np = (tile.permute(1,2,0).numpy() * 255).astype(np.uint8)
            boxes, _, landmarks = mtcnn.detect(img_np, landmarks=True)

            if landmarks is None:
                # Fallback: einfache Resize
                tile_resized = transforms.functional.resize(tile, ALIGNED_SIZE)
                aligned_tiles.append(tile_resized)
                continue

            # Wähle Gesicht, das am nächsten zur Bildmitte liegt
            box_centers = np.mean(boxes, axis=1)
            img_center = np.array([img_np.shape[1]/2, img_np.shape[0]/2])
            idx = np.argmin(np.sum((box_centers - img_center)**2, axis=1))
            facial5points = landmarks[idx]

            # norm_crop
            aligned_img = norm_crop(img_np, landmark=facial5points, image_size=ALIGNED_SIZE, createEvalDB=True)
            aligned_tiles.append(torch.from_numpy(aligned_img).permute(2,0,1)/255.0)

        # 3. Stack & Save Grid
        aligned_grid = make_grid(torch.stack(aligned_tiles), nrow=int(np.sqrt(len(aligned_tiles))), padding=0)
        save_image(aligned_grid, output_path)
        print(f"Aligned {fname}")

    except Exception as e:
        print(f"Skipped {fname} due to {e}")
        skipped.append(fname)

# Log skipped
if skipped:
    with open(os.path.join(OUTPUT_DIR, "skipped.txt"), "w") as f:
        for s in skipped:
            f.write(s + "\n")
