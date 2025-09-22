# encode_individual.py
"""
Encode aligned StyleGAN images using face recognition backbones
Output: embeddings.npy + labels.npy
"""

import os
from typing import Any
import hydra
import torch
import torchvision.transforms as transforms
from pytorch_lightning.lite import LightningLite
from PIL import Image
from omegaconf import OmegaConf, DictConfig
import numpy as np

from utils.helpers import ensure_path_join, normalize_to_neg_one_to_one

import sys
from utils.iresnet import iresnet100, iresnet50
from utils.irse import IR_101
from utils.synface_resnet import LResNet50E_IR
from utils.moco import MoCo

sys.path.insert(0, 'IDiff-Face/')


class EncoderLite(LightningLite):
    def run(self, cfg) -> Any:
        face_backbone = None

        for frm_name in cfg.encode.frm_names:
            print(f"Starting Encoding Process for FRM: {frm_name}")
            del face_backbone

            # instantiate face recognition backbone
            if frm_name == "elasticface":
                face_backbone = iresnet100(num_features=512)
                ckpt = torch.load(os.path.join("utils", "Elastic_R100_295672backbone.pth"), map_location="cpu")
                face_backbone.load_state_dict(ckpt)

            elif frm_name == "curricularface":
                face_backbone = IR_101([112, 112])
                ckpt = torch.load(os.path.join("utils", "CurricularFace_Backbone.pth"), map_location="cpu")
                face_backbone.load_state_dict(ckpt)

            elif frm_name == "idiff-face":
                face_backbone = iresnet50(num_features=512)
                ckpt = torch.load(os.path.join("utils", "54684backbone.pth"), map_location="cpu")
                face_backbone.load_state_dict(ckpt)

            elif frm_name == "sface":
                face_backbone = iresnet50(num_features=512)
                ckpt = torch.load(os.path.join("utils", "79232backbone.pth"), map_location="cpu")
                face_backbone.load_state_dict(ckpt)

            elif frm_name == "usynthface":
                face_backbone = iresnet50(num_features=512)
                face_backbone = MoCo(base_encoder=iresnet50, dim=512, K=32768)
                ckpt = torch.load(os.path.join("utils", "checkpoint_051.pth"), map_location="cpu")
                face_backbone.load_state_dict(ckpt, strict=False)
                face_backbone = face_backbone.encoder_q

            elif frm_name == "synface":
                face_backbone = LResNet50E_IR([112, 96])
                ckpt = torch.load(os.path.join("utils", "model_10k_50_idmix_9197.pth"), map_location="cpu")["state_dict"]
                face_backbone.load_state_dict(ckpt)

            # push face recognition backbone to device
            face_backbone = self.setup(face_backbone)
            face_backbone.eval()

            for model_name in cfg.encode.model_names:
                for contexts_name in cfg.encode.contexts_names:

                    # build paths
                    if cfg.encode.aligned:
                        samples_dir = ensure_path_join("samples", "aligned", model_name, contexts_name)
                        embeddings_dir = ensure_path_join("samples", "aligned", "embeddings", model_name, contexts_name, frm_name)
                    else:
                        samples_dir = ensure_path_join("samples", model_name, contexts_name)
                        embeddings_dir = ensure_path_join("samples", "embeddings", model_name, contexts_name, frm_name)

                    if not os.path.isdir(samples_dir):
                        print(f"Samples directory {samples_dir} does not exist! Skipping.")
                        continue

                    os.makedirs(embeddings_dir, exist_ok=True)

                    # encode images
                    embeddings, labels = self.encode_images(face_backbone, samples_dir, image_size=112)

                    # save embeddings + labels
                    torch.save(embeddings, os.path.join(embeddings_dir, "embeddings.npy"))
                    torch.save(labels, os.path.join(embeddings_dir, "labels.npy"))

    def encode_images(self, face_backbone, samples_dir, image_size=112):
        embeddings = []
        id_labels = []

        for id_folder in sorted(os.listdir(samples_dir)):
            id_path = os.path.join(samples_dir, id_folder)
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
                img_tensor = transforms.functional.resize(img_tensor, image_size)
                img_tensors.append(img_tensor)

            if not img_tensors:
                continue

            imgs = torch.stack(img_tensors)
            imgs = normalize_to_neg_one_to_one(imgs)
            imgs = imgs.cuda()

            with torch.no_grad():
                id_embeds = face_backbone(imgs)
                id_embeds = torch.nn.functional.normalize(id_embeds)

            for embed in id_embeds.cpu().numpy():
                embeddings.append(embed)
                id_labels.append(id_folder)

        return np.array(embeddings), np.array(id_labels)


@hydra.main(config_path='configs', config_name='encode_config', version_base=None)
def encode(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    sampler = EncoderLite(devices=[0], accelerator="auto")
    sampler.run(cfg)


if __name__ == "__main__":
    encode()
