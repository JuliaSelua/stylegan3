# encode_individual_nohydra.py
"""
Encode aligned StyleGAN images using face recognition backbones
Output: embeddings.npy + labels.npy
"""

import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

from utils.helpers import normalize_to_neg_one_to_one, ensure_path_join
from utils.iresnet import iresnet100, iresnet50
from utils.irse import IR_101
from utils.synface_resnet import LResNet50E_IR
from utils.moco import MoCo

# ------------------- Settings -------------------
FRM_NAME = "elasticface"  # elasticface / curricularface / idiff-face / sface / usynthface / synface
SAMPLES_DIR = "out/id_samples_aligned"   # Ordner mit aligned Bildern pro ID-Ordner
EMBEDDINGS_DIR = "out/embeddings"        # Ausgabeordner
IMAGE_SIZE = 112
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# ------------------- Instantiate Face Backbone -------------------
if FRM_NAME == "elasticface":
    face_backbone = iresnet100(num_features=512)
    ckpt = torch.load(os.path.join("utils", "Elastic_R100_295672backbone.pth"), map_location="cpu")
    face_backbone.load_state_dict(ckpt)

elif FRM_NAME == "curricularface":
    face_backbone = IR_101([112, 112])
    ckpt = torch.load(os.path.join("utils", "CurricularFace_Backbone.pth"), map_location="cpu")
    face_backbone.load_state_dict(ckpt)

elif FRM_NAME == "idiff-face":
    face_backbone = iresnet50(num_features=512)
    ckpt = torch.load(os.path.join("utils", "54684backbone.pth"), map_location="cpu")
    face_backbone.load_state_dict(ckpt)

elif FRM_NAME == "sface":
    face_backbone = iresnet50(num_features=512)
    ckpt = torch.load(os.path.join("utils", "79232backbone.pth"), map_location="cpu")
    face_backbone.load_state_dict(ckpt)

elif FRM_NAME == "usynthface":
    face_backbone = iresnet50(num_features=512)
    face_backbone = MoCo(base_encoder=iresnet50, dim=512, K=32768)
    ckpt = torch.load(os.path.join("utils", "checkpoint_051.pth"), map_location="cpu")
    face_backbone.load_state_dict(ckpt, strict=False)
    face_backbone = face_backbone.encoder_q

elif FRM_NAME == "synface":
    face_backbone = LResNet50E_IR([112, 96])
    ckpt = torch.load(os.path.join("utils", "model_10k_50_idmix_9197.pth"), map_location="cpu")["state_dict"]
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

    print("Encoding ID:", id_folder)
    img_tensors = []

    for fname in sorted(os.listdir(id_path)):
        if not (fname.endswith(".png") or fname.endswith(".jpg")):
            continue

        img_path = os.path.join(id_path, fname)
        img = Image.open(img_path).convert("RGB")
        img_tensor = transforms.functional.to_tensor(img)
        img_tensor = transforms.functional.resize(img_tensor, IMAGE_SIZE)
        img_tensors.append(img_tensor)

    if not img_tensors:
        continue

    imgs = torch.stack(img_tensors)
    imgs = normalize_to_neg_one_to_one(imgs)
    imgs = imgs.to(DEVICE)

    with torch.no_grad():
        id_embeds = face_backbone(imgs)
        id_embeds = torch.nn.functional.normalize(id_embeds)

    for embed in id_embeds.cpu().numpy():
        embeddings.append(embed)
        labels.append(id_folder)

# ------------------- Save -------------------
torch.save(np.array(embeddings), os.path.join(EMBEDDINGS_DIR, "embeddings.npy"))
torch.save(np.array(labels), os.path.join(EMBEDDINGS_DIR, "labels.npy"))

print("Encoding complete. Embeddings saved to", EMBEDDINGS_DIR)

