import os
import torch
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm  # Fortschrittsbalken

# StyleGAN
import legacy

# ============================================================
# GENERATION FUNCTION
# ============================================================

def generate_images_id_style(G, outdir, n_ids, n_styles, device, gen_batch=16):
    base_dir = os.path.join(outdir, "generated_images")
    os.makedirs(base_dir, exist_ok=True)

    # Fortschrittsbalken nur über IDs
    for seed in tqdm(range(n_ids), desc="Generating IDs", ncols=120):
        torch.manual_seed(seed)
        np.random.seed(seed)

        id_dir = os.path.join(base_dir, f"id{seed:05d}")
        os.makedirs(id_dir, exist_ok=True)

        z_id = torch.randn(1, G.z_dim, device=device)
        z_styles = torch.randn(n_styles, G.z_dim, device=device)

        for i in range(0, n_styles, gen_batch):
            batch_z = z_styles[i:i+gen_batch]
            with torch.no_grad():
                imgs = G(z_id.repeat(len(batch_z), 1), None, z2=batch_z)
                imgs = (imgs.clamp(-1, 1) + 1) * 127.5
                imgs = imgs.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

            for j, img in enumerate(imgs):
                idx = i + j
                filename = os.path.join(id_dir, f"id{seed:05d}_style{idx:02d}.png")
                Image.fromarray(img, "RGB").save(filename)

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", required=True, help="Pfad zum StyleGAN-Netzwerk-PKL")
    parser.add_argument("--outdir", required=True, help="Ausgabeverzeichnis")
    parser.add_argument("--n_ids", type=int, default=10000, help="Anzahl der IDs")
    parser.add_argument("--n_styles", type=int, default=50, help="Anzahl der Styles pro ID")
    parser.add_argument("--gen_batch", type=int, default=16, help="Batch-Größe für die StyleGAN-Generierung")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # StyleGAN laden
    with open(args.network, "rb") as f:
        G = legacy.load_network_pkl(f)["G_ema"].to(device)

    generate_images_id_style(G, args.outdir, args.n_ids, args.n_styles, device, gen_batch=args.gen_batch)
    print("Generation complete!")
