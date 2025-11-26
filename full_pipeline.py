import os
import sys
import torch
import numpy as np
from PIL import Image
import multiprocessing as mp
from functools import partial
import argparse
from tqdm import tqdm

# StyleGAN imports
import legacy

# Face alignment + embedding
from facenet_pytorch import MTCNN
from utils.alignment.arcface import norm_crop
from utils.helpers import normalize_to_neg_one_to_one
from utils.iresnet import iresnet100

# Evaluation
import numpy as np
from pyeer.eer_info import get_eer_stats
from pyeer.report import generate_eer_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ============================================================
# 1) -------- GENERATION -------------------------------------
# ============================================================

def generate_images_parallel(G, outdir, total, batch_size, device):
    os.makedirs(outdir, exist_ok=True)

    n_batches = (total + batch_size - 1) // batch_size

    for b in tqdm(range(n_batches), desc="Generating", ncols=120):
        bs = min(batch_size, total - b*batch_size)

        z = torch.randn(bs, G.z_dim, device=device)
        z2 = torch.randn(bs, G.z_dim, device=device)

        with torch.no_grad():
            imgs = G(z, None, z2=z2)
            imgs = (imgs.clamp(-1, 1) + 1) * 127.5
            imgs = imgs.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

        for i in range(bs):
            idx = b*batch_size + i
            Image.fromarray(imgs[i], "RGB").save(
                os.path.join(outdir, f"{idx:08d}.png")
            )

# ============================================================
# 2) -------- ALIGNMENT (PARALLEL) ----------------------------
# ============================================================

def align_single(img_path, outdir, mtcnn, size):
    try:
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)

        boxes, _, landmarks = mtcnn.detect([img_np], landmarks=True)

        # Wenn Gesicht gefunden wurde → norm_crop
        if landmarks is not None and landmarks[0] is not None:
            lms = np.array(landmarks[0][0], dtype=np.float32)
            aligned = norm_crop(img_np, lms, image_size=size, createEvalDB=True)
            aligned = np.clip(aligned, 0, 255).astype(np.uint8)
        else:
            aligned = np.array(img.resize((size, size)))

        rel = os.path.basename(img_path)
        Image.fromarray(aligned).save(os.path.join(outdir, rel))

    except Exception as e:
        return img_path, str(e)

    return None

def align_parallel(input_dir, outdir, device, size=112, workers=12):
    os.makedirs(outdir, exist_ok=True)

    mtcnn = MTCNN(keep_all=True, min_face_size=20, device=device)

    img_paths = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(".png")
    ]

    pool = mp.Pool(workers)
    fn = partial(align_single, outdir=outdir, mtcnn=mtcnn, size=size)

    errors = list(tqdm(pool.imap(fn, img_paths), total=len(img_paths), desc="Aligning", ncols=120))
    pool.close()
    pool.join()

    errors = [e for e in errors if e is not None]
    return errors

# ============================================================
# 3) -------- EMBEDDING (GPU BATCHES) -------------------------
# ============================================================

def embed_images(input_dir, embed_dir, device, batch_size=256):
    os.makedirs(embed_dir, exist_ok=True)

    # Backbone
    model = iresnet100(num_features=512)
    ckpt = torch.load("utils/Elastic_R100_295672backbone.pth", map_location="cpu")
    model.load_state_dict(ckpt)
    model = model.to(device).eval()

    img_files = sorted([
        os.path.join(input_dir, f) for f in os.listdir(input_dir)
        if f.lower().endswith(".png")
    ])

    embeddings = []
    labels = []

    def load_tensor(path):
        img = Image.open(path).convert("RGB").resize((112, 112))
        t = torch.from_numpy(np.array(img)).permute(2,0,1).float() / 255.0
        return t

    for b in tqdm(range(0, len(img_files), batch_size), desc="Embedding", ncols=120):
        batch = img_files[b:b+batch_size]
        imgs = torch.stack([load_tensor(p) for p in batch])
        imgs = normalize_to_neg_one_to_one(imgs).to(device)

        with torch.no_grad():
            emb = model(imgs)
            emb = torch.nn.functional.normalize(emb)

        embeddings.append(emb.cpu())
        labels.extend(batch)

    embeddings = torch.cat(embeddings, dim=0)
    torch.save(embeddings, os.path.join(embed_dir, "embeddings.pt"))
    torch.save(labels, os.path.join(embed_dir, "labels.pt"))

    return embeddings, labels

# ============================================================
# 4) -------- EVALUATION -------------------------------------
# ============================================================

def evaluate_embeddings(embeddings, labels, outdir, suffix=""):
    os.makedirs(outdir, exist_ok=True)

    emb = embeddings.numpy()
    labels_np = np.array(labels)

    norm = emb / np.linalg.norm(emb, axis=1, keepdims=True)

    # Genuine scores
    genuine = []
    for lbl in np.unique(labels_np):
        idx = np.where(labels_np == lbl)[0]
        if len(idx) > 1:
            for i in range(len(idx)):
                for j in range(i+1, len(idx)):
                    genuine.append(np.dot(norm[idx[i]], norm[idx[j]]))

    # Imposter scores
    imposter = []
    for _ in range(200000):
        i, j = np.random.randint(0, len(norm), 2)
        if labels_np[i] != labels_np[j]:
            imposter.append(np.dot(norm[i], norm[j]))

    # Save raw
    np.savetxt(os.path.join(outdir, "genuine_scores.txt"), genuine)
    np.savetxt(os.path.join(outdir, "imposter_scores.txt"), imposter)

    # Plot
    plt.figure(figsize=(8,6))
    plt.hist(genuine, bins=100, alpha=0.5, label="Genuine")
    plt.hist(imposter, bins=100, alpha=0.5, label="Imposter")
    plt.legend()
    plt.xlim(-1,1)
    plt.title("Genuine vs Imposter")
    plt.savefig(os.path.join(outdir, "similarity_hist.png"), dpi=300)
    plt.close()

    # EER
    eer = get_eer_stats(genuine, imposter)
    generate_eer_report([eer], ["synthetic"], os.path.join(outdir, "eer_report.html"))

# ============================================================
# MAIN --------------------------------------------------------
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--num_images", type=int, default=10000)
    parser.add_argument("--gen_batch", type=int, default=64)
    parser.add_argument("--align_workers", type=int, default=12)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load StyleGAN
    with open(args.network, "rb") as f:
        G = legacy.load_network_pkl(f)["G_ema"].to(device)

    gen_dir = os.path.join(args.outdir, "generated_pngs")
    align_dir = os.path.join(args.outdir, "aligned")
    embed_dir = os.path.join(args.outdir, "embeddings")
    eval_dir = os.path.join(args.outdir, "evaluation")

    # ---- 1) GENERATE ----
    generate_images_parallel(G, gen_dir, args.num_images, args.gen_batch, device)

    # ---- 2) ALIGN ----
    errors = align_parallel(gen_dir, align_dir, device, workers=args.align_workers)

    # ---- 3) EMBEDDING ----
    emb, labels = embed_images(align_dir, embed_dir, device)

    # ---- 4) EVALUATE ----
    evaluate_embeddings(emb, labels, eval_dir)

    print("All done!")
