import os
from PIL import Image
import numpy as np
from facenet_pytorch import MTCNN
from utils.alignment.arcface import norm_crop

INPUT_DIR = "out/id_samples"
OUTPUT_DIR = "out/id_samples_aligned"
ALIGNED_SIZE = 112
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda" if __import__('torch').cuda.is_available() else "cpu"
mtcnn = MTCNN(keep_all=True, min_face_size=20, post_process=False, device=device)

skipped = []

for root, _, files in os.walk(INPUT_DIR):
    for fname in files:
        if not (fname.endswith(".png") or fname.endswith(".jpg")):
            continue

        input_path = os.path.join(root, fname)

        # Unterordner-Struktur beibehalten
        rel_path = os.path.relpath(root, INPUT_DIR)
        output_subdir = os.path.join(OUTPUT_DIR, rel_path)
        os.makedirs(output_subdir, exist_ok=True)
        output_path = os.path.join(output_subdir, fname)

        try:
            img = Image.open(input_path).convert("RGB")
            img_np = np.array(img)

            boxes, _, landmarks = mtcnn.detect([img_np], landmarks=True)

            aligned_img = None
            # Sicherstellen, dass boxes[0] existiert und nicht leer ist
            if boxes is None or landmarks is None or boxes[0] is None or len(boxes[0]) == 0:
                aligned_img = img.resize((ALIGNED_SIZE, ALIGNED_SIZE))
                print(f"{fname}: 0 faces detected, fallback resize applied")
            else:
                # Gesicht, das am nächsten zur Bildmitte liegt
                box_centers = np.mean(boxes[0], axis=1)
                img_center = np.array([img_np.shape[1]/2, img_np.shape[0]/2])
                idx = np.argmin(np.sum((box_centers - img_center)**2, axis=1))
                facial5points = landmarks[0][idx]

                aligned_np = norm_crop(img_np, landmark=facial5points, image_size=ALIGNED_SIZE, createEvalDB=True)
                if aligned_np is None or aligned_np.size == 0:
                    aligned_img = img.resize((ALIGNED_SIZE, ALIGNED_SIZE))
                    print(f"{fname}: norm_crop failed, fallback resize applied")
                else:
                    aligned_img = Image.fromarray(aligned_np)

            aligned_img.save(output_path)
            print(f"Aligned {fname} -> {output_path}")

        except Exception as e:
            print(f"Skipped {fname} due to {e}")
            skipped.append(fname)

if skipped:
    with open(os.path.join(OUTPUT_DIR, "skipped.txt"), "w") as f:
        for s in skipped:
            f.write(s + "\n")

print(f"Finished aligning. Skipped {len(skipped)} images.")

