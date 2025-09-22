# align.py
import os
import glob
import cv2
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN

import torch

def align_face(img, mtcnn, image_size=112):
    """
    Detect face and align to 112x112.
    img: PIL image
    """
    # Detect face + landmarks
    boxes, probs, landmarks = mtcnn.detect(img, landmarks=True)

    if boxes is None or landmarks is None:
        return None  # no face found

    # Nehme nur die erste erkannte Person
    lm = landmarks[0]
    # landmarks: [ [x_left_eye,y_left_eye], [x_right_eye,y_right_eye], [nose], [left_mouth], [right_mouth] ]
    left_eye, right_eye = lm[0], lm[1]

    # Berechne Winkel für Rotation
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))

    # Mittelpunkt zwischen den Augen
    eyes_center = ((left_eye[0] + right_eye[0]) / 2,
                   (left_eye[1] + right_eye[1]) / 2)

    # Transformation: Rotieren + Skalieren + Croppen
    M = cv2.getRotationMatrix2D(eyes_center, angle, scale=1.0)
    aligned = cv2.warpAffine(np.array(img), M, (img.width, img.height), flags=cv2.INTER_CUBIC)

    # Crop auf quadratisches Gesicht (hier einfach center crop)
    # Du kannst es feiner machen mit Box vom Detector
    x, y, w, h = boxes[0]
    face = aligned[int(y):int(y+h), int(x):int(x+w)]
    face = cv2.resize(face, (image_size, image_size))

    return Image.fromarray(face)

def align_directory(src_dir, dst_dir, image_size=112, device='cuda'):
    os.makedirs(dst_dir, exist_ok=True)
    mtcnn = MTCNN(keep_all=False, device=device)

    id_dirs = sorted([d for d in glob.glob(os.path.join(src_dir, "id*")) if os.path.isdir(d)])

    for id_dir in id_dirs:
        id_name = os.path.basename(id_dir)
        out_id_dir = os.path.join(dst_dir, id_name)
        os.makedirs(out_id_dir, exist_ok=True)

        for img_path in glob.glob(os.path.join(id_dir, "*.png")):
            img = Image.open(img_path).convert("RGB")
            aligned = align_face(img, mtcnn, image_size=image_size)
            if aligned is not None:
                aligned.save(os.path.join(out_id_dir, os.path.basename(img_path)))
            else:
                print(f"⚠️ No face detected in {img_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True, help="Directory with raw generated images")
    parser.add_argument("--dst", type=str, required=True, help="Directory to save aligned faces")
    args = parser.parse_args()

    align_directory(args.src, args.dst)
