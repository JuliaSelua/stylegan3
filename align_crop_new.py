import os
import cv2
from tqdm import tqdm
import argparse
from os.path import join as ojoin
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

from utils.align_trans import norm_crop
from facenet_pytorch import MTCNN
import torch

# MTCNN initialisieren
mtcnn = MTCNN(select_largest=True, min_face_size=60, post_process=False, device="cuda:0")


def load_syn_paths(datadir, num_imgs=0):
    img_files = sorted(f for f in os.listdir(datadir) if f.lower().endswith(('.png', '.jpg', '.jpeg')))
    img_files = img_files if num_imgs == 0 else img_files[:num_imgs]
    return [ojoin(datadir, f_name) for f_name in img_files]


def load_real_paths(datadir, num_imgs=0):
    img_paths = []
    id_folders = sorted(os.listdir(datadir))
    for id in id_folders:
        path = ojoin(datadir, id)
        if not os.path.isdir(path):
            continue
        img_files = sorted(f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg')))
        img_paths += [ojoin(path, f_name) for f_name in img_files]
    img_paths = img_paths if num_imgs == 0 else img_paths[:num_imgs]
    return img_paths


def is_folder_structure(datadir):
    img_path = sorted(os.listdir(datadir))[0]
    img_path = ojoin(datadir, img_path)
    return os.path.isdir(img_path)


class InferenceDataset(Dataset):
    def __init__(self, datadir, num_imgs=0, folder_structure=False):
        self.folder_structure = folder_structure
        if self.folder_structure:
            self.img_paths = load_real_paths(datadir, num_imgs)
        else:
            self.img_paths = load_syn_paths(datadir, num_imgs)
        print("Amount of images:", len(self.img_paths))

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img_file = os.path.split(img_path)[-1]
        if self.folder_structure:
            tmp = os.path.dirname(img_path)
            img_file = ojoin(os.path.basename(tmp), img_file)
        img = cv2.imread(self.img_paths[index])
        return img, img_file

    def __len__(self):
        return len(self.img_paths)


def align_images(in_folder, out_folder, batchsize, num_imgs=0, evalDB=False):
    os.makedirs(out_folder, exist_ok=True)
    is_folder = is_folder_structure(in_folder)
    train_dataset = InferenceDataset(in_folder, num_imgs=num_imgs, folder_structure=is_folder)
    train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=False, drop_last=False)

    skipped_imgs = []

    for img_batch, img_names in tqdm(train_loader):
        # Filter None/leer Bilder
        valid_imgs = []
        valid_names = []
        for img, name in zip(img_batch, img_names):
            if img is None:
                skipped_imgs.append(name)
                continue
            valid_imgs.append(img)
            valid_names.append(name)

        if len(valid_imgs) == 0:
            continue

        # Konvertiere OpenCV-Bilder in PIL Images für MTCNN
        pil_imgs = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in valid_imgs]

        # Detect Faces
        boxes, probs, landmarks = mtcnn.detect(pil_imgs, landmarks=True)

        for img, img_name, landmark in zip(valid_imgs, valid_names, landmarks):
            if landmark is None or not isinstance(landmark[0], (list, np.ndarray)):
                skipped_imgs.append(img_name)
                continue

            out_path = out_folder
            if is_folder:
                id_dir = os.path.split(img_name)[0]
                out_path = ojoin(out_folder, id_dir)
                os.makedirs(out_path, exist_ok=True)
                img_name = os.path.split(img_name)[1]

            facial5points = np.array(landmark[0], dtype=np.float32)
            warped_face = norm_crop(img, landmark=facial5points, image_size=112, createEvalDB=evalDB)
            cv2.imwrite(ojoin(out_path, img_name), warped_face)

    print("Skipped images:", skipped_imgs)
    print(f"Images with no Face: {len(skipped_imgs)}")


def main():
    parser = argparse.ArgumentParser(description="MTCNN alignment")
    parser.add_argument("--in_folder", type=str, required=True, help="folder with images")
    parser.add_argument("--out_folder", type=str, required=True, help="folder to save aligned images")
    parser.add_argument("--batchsize", type=int, default=32)
    parser.add_argument("--evalDB", type=int, default=0, help="1 for eval DB alignment")
    parser.add_argument("--num_imgs", type=int, default=0, help="amount of images to align; 0 for all images")
    args = parser.parse_args()

    align_images(
        args.in_folder,
        args.out_folder,
        args.batchsize,
        num_imgs=args.num_imgs,
        evalDB=args.evalDB == 1,
    )


if __name__ == "__main__":
    main()

