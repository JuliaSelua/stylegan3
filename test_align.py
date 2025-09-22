from PIL import Image
import numpy as np
import torch
from facenet_pytorch import MTCNN
from torchvision.utils import save_image
from utils.alignment.arcface import norm_crop  # iDiff utility

# ------------------- Einstellungen -------------------
INPUT_IMG = "out/id_samples/id00000/id00000_style00.png"
OUTPUT_IMG = "out/id_samples_aligned/id00000_style00_aligned.png"
ALIGNED_SIZE = 112

# ------------------- MTCNN Setup -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(keep_all=True, device=device)

# ------------------- Bild laden -------------------
img = Image.open(INPUT_IMG).convert("RGB")
img_np = np.array(img)

# ------------------- Gesicht erkennen -------------------
boxes, _, landmarks = mtcnn.detect([img_np], landmarks=True)

if boxes is None or landmarks is None:
    print("Kein Gesicht erkannt. Verwende Resize fallback.")
    aligned = img.resize((ALIGNED_SIZE, ALIGNED_SIZE))
else:
    # Gesicht auswählen (nächstes zur Bildmitte)
    box_centers = np.mean(boxes[0], axis=1)
    img_center = np.array([img_np.shape[1]/2, img_np.shape[0]/2])
    idx = np.argmin(np.sum((box_centers - img_center)**2, axis=1))
    facial5points = landmarks[0][idx]

    # NormCrop für Alignment
    aligned_img = norm_crop(img_np, landmark=facial5points, image_size=ALIGNED_SIZE)
    aligned = torch.from_numpy(aligned_img).permute(2,0,1)/255.0  # Tensor 3xHxW

# ------------------- Speichern -------------------
# Wenn Tensor, convertiere zu PIL
if isinstance(aligned, torch.Tensor):
    from torchvision.transforms.functional import to_pil_image
    aligned = to_pil_image(aligned)
    
aligned.save(OUTPUT_IMG)
print("Bild gespeichert:", OUTPUT_IMG)
