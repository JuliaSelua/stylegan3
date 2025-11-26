import os
import torch
import numpy as np
from PIL import Image
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
import argparse

# StyleGAN
import legacy

# Face alignment + embedding
from facenet_pytorch import MTCNN
from utils.alignment.arcface import norm_crop
from utils.helpers import normalize_to_neg_one_to_one
from utils.iresnet import iresnet100

# Evaluation
from pyeer.eer_info import get_eer_stats
from pyeer.report import generate_eer_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ============================================================
# 1) -------- GENERATION (IDs + Styles) ---------------------
# ============================================================

def generate_images_id_style(G, outdir, n_ids, n_styles, device):
    base_dir = os.path.join(outdir, "generated_images")
    os.makedirs(base_dir, exist_ok=True)

    for seed in tqdm(range(n_ids), desc="Generating IDs", ncols=120):
        torch.manual_seed(seed)
        np.random.seed(seed)

        id_dir = os.path.join(base_dir, f"id{seed:05d}")
        os.makedirs(id_dir, exist_ok=True)

        # z for ID
        z_id = torch.randn(1, G.z_dim, device=device)

        for style_idx in range(n_styles):
            z_style = torch.randn(1, G.z_dim, device=device)
            with torch.no_grad():
                img = G(z_id, None, z2=z_style)
                img = (img.clamp(-1, 1) + 1) * 127.5
                img = img.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)[0]

            filename = os.path.join(id_dir, f"id{seed:05d}_style{style_idx:02d}.png")
            Image.fromarray(img, "RGB").save(filename)

# ============================================================
# 2) -------- ALIGNMENT (PARALLEL) ----------------------------
# ============================================================

def align_single(img_path, outdir, mtcnn, size):
    try:
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)

        boxes, _, landmarks = mtcnn.detect([img_np], landmarks=True)

        if landmarks is not None and landmarks[0] is not None:
            lms = np.array(landmarks[0][0], dtype=np.float32)
            aligned = norm_crop(img_np, lms, image_size=size, createEvalDB=True)
            aligned = np.clip(aligned, 0, 255).astype(np.uint8)
        else:
            aligned = np.array(img.resize((size, size)))

        rel = os.path.relpath(img_path, start=os.path.dirname(outdir))
        out_path = os.path.join(outdir, rel)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        Image.fromarray(aligned).save(out_path)

    except Exception as e:
        return img_path, str(e)
    return None

def align_parallel(input_dir, outdir, device, size=112, workers=12):
    os.makedirs(outdir, exist_ok=True)
    mtcnn = MTCNN(keep_all=True, min_face_size=20, device=device)

    img_paths = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(".png"):
                img_paths.append(os.path.join(root, f))

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

    model = iresnet100(num_features=512)
    ckpt = torch.load("utils/Elastic_R100_295672backbone.pth", map_location="cpu")
    model.load_state_dict(ckpt)
    model = model.to(device).eval()

    img_files = []
    labels = []
    for id_folder in sorted(os.listdir(input_dir)):
        id_path = os.path.join(input_dir, id_folder)
        if not os.path.isdir(id_path):
            continue
        for fname in sorted(os.listdir(id_path)):
            if fname.lower().endswith(".png"):
                img_files.append(os.path.join(id_path, fname))
                labels.append(id_folder)

    embeddings = []

    def load_tensor(path):
        img = Image.open(path).convert("RGB").resize((112, 112))
        t = torch.from_numpy(np.array(img)).permute(2,0,1).float() / 255.0
        return t

    for b in tqdm(range(0, len(img_files), batch_size), desc="Embedding", ncols=120):
        batch_paths = img_files[b:b+batch_size]
        imgs = torch.stack([load_tensor(p) for p in batch_paths])
        imgs = normalize_to_neg_one_to_one(imgs).to(device)

        with torch.no_grad():
            emb = model(imgs)
            emb = torch.nn.functional.normalize(emb)

        embeddings.append(emb.cpu())

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

    genuine = []
    for lbl in np.unique(labels_np):
        idx = np.where(labels_np == lbl)[0]
        if len(idx) > 1:
            for i in range(len(idx)):
                for j in range(i+1, len(idx)):
                    genuine.append(np.dot(norm[idx[i]], norm[idx[j]]))

    imposter = []
    for _ in range(200000):
        i, j = np.random.randint(0, len(norm), 2)
        if labels_np[i] != labels_np[j]:
            imposter.append(np.dot(norm[i], norm[j]))

    np.savetxt(os.path.join(outdir, "genuine_scores.txt"), genuine)
    np.savetxt(os.path.join(outdir, "imposter_scores.txt"), imposter)

    plt.figure(figsize=(8,6))
    plt.hist(genuine, bins=100, alpha=0.5, label="Genuine")
    plt.hist(imposter, bins=100, alpha=0.5, label="Imposter")
    plt.legend()
    plt.xlim(-1,1)
    plt.title("Genuine vs Imposter")
    plt.savefig(os.path.join(outdir, "similarity_hist.png"), dpi=300)
    plt.close()

    eer = get_eer_stats(genuine, imposter)
    generate_eer_report([eer], ["synthetic"], os.path.join(outdir, "eer_report.html"))

# ============================================================
# MAIN --------------------------------------------------------
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--n_ids", type=int, default=10000)
    parser.add_argument("--n_styles", type=int, default=50)
    parser.add_argument("--align_workers", type=int, default=12)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load StyleGAN
    with open(args.network, "rb") as f:
        G = legacy.load_network_pkl(f)["G_ema"].to(device)

    gen_dir = os.path.join(args.outdir, "generated_images")
    align_dir = os.path.join(args.outdir, "aligned")
    embed_dir = os.path.join(args.outdir, "embeddings")
    eval_dir = os.path.join(args.outdir, "evaluation")

    # ---- 1) GENERATE ----
    generate_images_id_style(G, gen_dir, args.n_ids, args.n_styles, device)

    # ---- 2) ALIGN ----
    errors = align_parallel(gen_dir, align_dir, device, workers=args.align_workers)
    if errors:
        print(f"Skipped {len(errors)} images due to errors.")

    # ---- 3) EMBEDDING ----
    emb, labels = embed_images(align_dir, embed_dir, device)

    # ---- 4) EVALUATE ----
    evaluate_embeddings(emb, labels, eval_dir)

    print("All done!")
