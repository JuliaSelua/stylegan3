import os
from PIL import Image
import numpy as np
from facenet_pytorch import MTCNN
from utils.alignment.arcface import norm_crop

INPUT_DIR = "out/id_samples"
OUTPUT_DIR = "out/id_samples_aligned"
ALIGNED_SIZE = 112

device = "cuda" if __import__('torch').cuda.is_available() else "cpu"
mtcnn = MTCNN(keep_all=True, min_face_size=20, post_process=False, device=device)

skipped = []

for root, dirs, files in os.walk(INPUT_DIR):
    for fname in sorted(files):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        input_path = os.path.join(root, fname)
        # Behalte die Unterordner-Struktur
        relative_path = os.path.relpath(input_path, INPUT_DIR)
        output_path = os.path.join(OUTPUT_DIR, relative_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        try:
            img = Image.open(input_path).convert("RGB")
            img_np = np.array(img)

            boxes, _, landmarks = mtcnn.detect([img_np], landmarks=True)

            if boxes is not None and boxes[0] is not None and len(boxes[0]) > 0:
                box_centers = np.mean(boxes[0], axis=1)
                img_center = np.array([img_np.shape[1]/2, img_np.shape[0]/2])
                idx = np.argmin(np.sum((box_centers - img_center)**2, axis=1))
                facial5points = landmarks[0][idx]

                aligned_np = norm_crop(img_np, landmark=facial5points, image_size=ALIGNED_SIZE, createEvalDB=True)

                if aligned_np.dtype != np.uint8:
                    aligned_np = np.clip(aligned_np, 0, 255).astype(np.uint8)

                aligned_img = Image.fromarray(aligned_np)
                print(f"{fname}: face aligned")

            else:
                aligned_img = img.resize((ALIGNED_SIZE, ALIGNED_SIZE))
                print(f"{fname}: 0 faces detected, fallback resize applied")

            aligned_img.save(output_path)
            print(f"Saved {output_path}")

        except Exception as e:
            print(f"Skipped {fname} due to {e}")
            skipped.append(os.path.relpath(input_path, INPUT_DIR))

if skipped:
    with open(os.path.join(OUTPUT_DIR, "skipped.txt"), "w") as f:
        for s in skipped:
            f.write(s + "\n")

print(f"Finished aligning. Skipped {len(skipped)} images.")
