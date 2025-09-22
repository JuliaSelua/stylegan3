# align.py
import os
from PIL import Image
import torch
from facenet_pytorch import MTCNN
from torchvision.utils import save_image, make_grid
import torchvision.transforms as transforms

def align_images(input_dir, output_dir, image_size=112):
    os.makedirs(output_dir, exist_ok=True)

    mtcnn = MTCNN(keep_all=True, device='cuda')

    for fname in os.listdir(input_dir):
        if not (fname.endswith('.png') or fname.endswith('.jpg')):
            continue

        img_path = os.path.join(input_dir, fname)
        img = Image.open(img_path).convert('RGB')
        img_tensor = transforms.ToTensor()(img)

        # MTCNN erwartet Batch, hier nur 1 Bild
        boxes, _, landmarks = mtcnn.detect(img_tensor.unsqueeze(0).cuda(), landmarks=True)

        if landmarks is None or len(landmarks[0]) == 0:
            print(f"Keine Gesichter gefunden: {fname}, Bild wird nur resized.")
            img_resized = transforms.Resize((image_size, image_size))(img_tensor)
            save_image(img_resized, os.path.join(output_dir, fname))
            continue

        # Gesicht am nächsten zur Bildmitte
        box_centers = (boxes[0][:, :2] + boxes[0][:, 2:]) / 2
        img_center = torch.tensor([img_tensor.shape[2]/2, img_tensor.shape[1]/2])
        distances = ((torch.tensor(box_centers) - img_center)**2).sum(dim=1)
        best_idx = distances.argmin().item()
        best_landmark = landmarks[0][best_idx]

        # Normales Crop wie bei iDiff
        from utils.alignment.arcface import norm_crop  # muss aus iDiff-Repo importiert werden
        aligned = norm_crop(np.array(img), landmark=best_landmark, image_size=image_size)

        aligned_tensor = transforms.ToTensor()(Image.fromarray(aligned))
        save_image(aligned_tensor, os.path.join(output_dir, fname))
        print(f"Aligned: {fname}")

if __name__ == "__main__":
    input_dir = "/out/id_samples"
    output_dir = "/out/id_samples_aligned"
    align_images(input_dir, output_dir)

