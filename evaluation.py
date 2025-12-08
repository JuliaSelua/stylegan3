import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pyeer.eer_info import get_eer_stats
from pyeer.report import generate_eer_report
from facenet_pytorch import MTCNN
from utils.alignment.arcface import norm_crop
from utils.helpers import normalize_to_neg_one_to_one
from utils.iresnet import iresnet100
# andere Backbones falls nötig: iresnet50, IR_101, LResNet50E_IR, MoCo

# ------------------- Argumente -------------------
parser = argparse.ArgumentParser(description="Full pipeline: Align, Encode, Evaluate")
parser.add_argument("--input_dir", type=str, required=True, help="Ordner mit StyleGAN-Bildern")
parser.add_argument("--suffix", type=str, default="", help="Suffix für Dateien/Plot-Titel")
args = parser.parse_args()

SUFFIX = f"{args.suffix}" if args.suffix else ""

INPUT_DIR = args.input_dir

ALIGNED_DIR = os.path.join(INPUT_DIR, "id_samples_aligned")
EMBEDDINGS_DIR = os.path.join(INPUT_DIR, "embeddings")
EVAL_DIR = os.path.join(INPUT_DIR, "evaluation")
if os.path.exists(os.path.join(INPUT_DIR, "generated_images")):
    INPUT_DIR = os.path.join(INPUT_DIR, "generated_images")
ALIGNED_SIZE = 112
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(ALIGNED_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

# ------------------- Alignment -------------------
print("=== Alignment ===")
mtcnn = MTCNN(keep_all=True, min_face_size=20, post_process=True, device=DEVICE)
skipped = []

for root, dirs, files in os.walk(INPUT_DIR):
    for fname in sorted(files):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        input_path = os.path.join(root, fname)
        relative_path = os.path.relpath(input_path, INPUT_DIR)
        output_path = os.path.join(ALIGNED_DIR, relative_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        try:
            img = Image.open(input_path).convert("RGB")
            img_np = np.array(img)
            boxes, _, landmarks = mtcnn.detect([img_np], landmarks=True)

            if boxes is not None and boxes[0] is not None and boxes[0].shape[0] > 0 \
               and landmarks is not None and landmarks[0] is not None and landmarks[0].shape[1:] == (5,2):
                idx = 0
                facial5points = np.array(landmarks[0][idx], dtype=np.float32)
                aligned_np = norm_crop(img_np, facial5points, image_size=ALIGNED_SIZE, createEvalDB=True)
                aligned_np = np.clip(aligned_np, 0, 255).astype(np.uint8)
                aligned_img = Image.fromarray(aligned_np)
            else:
                aligned_img = img.resize((ALIGNED_SIZE, ALIGNED_SIZE))

            aligned_img.save(output_path)

        except Exception as e:
            print(f"Skipped {fname} due to {e}")
            skipped.append(os.path.relpath(input_path, INPUT_DIR))

if skipped:
    with open(os.path.join(ALIGNED_DIR, "skipped.txt"), "w") as f:
        for s in skipped:
            f.write(s + "\n")
print(f"Alignment complete. Skipped {len(skipped)} images.")

# ------------------- Encoding -------------------
print("=== Encoding ===")
face_backbone = iresnet100(num_features=512)
ckpt = torch.load(os.path.join("utils", "Elastic_R100_295672backbone.pth"), map_location="cpu")
face_backbone.load_state_dict(ckpt)
face_backbone = face_backbone.to(DEVICE).eval()

embeddings = []
labels = []

for id_folder in sorted(os.listdir(ALIGNED_DIR)):
    id_path = os.path.join(ALIGNED_DIR, id_folder)
    if not os.path.isdir(id_path):
        continue

    img_tensors = []
    for fname in sorted(os.listdir(id_path)):
        if not fname.lower().endswith((".png", ".jpg")):
            continue
        img_path = os.path.join(id_path, fname)
        img = Image.open(img_path).convert("RGB")
        img = img.resize((ALIGNED_SIZE, ALIGNED_SIZE))
        img_tensor = torch.from_numpy(np.array(img)).permute(2,0,1).float() / 255.0
        img_tensors.append(img_tensor)

    if not img_tensors:
        continue

    imgs = torch.stack(img_tensors)
    imgs = normalize_to_neg_one_to_one(imgs).to(DEVICE)

    with torch.no_grad():
        id_embeds = face_backbone(imgs)
        id_embeds = torch.nn.functional.normalize(id_embeds)

    embeddings.append(id_embeds.cpu())
    labels.extend([id_folder]*id_embeds.shape[0])

embeddings = torch.cat(embeddings, dim=0)
torch.save(embeddings, os.path.join(EMBEDDINGS_DIR, "embeddings.pt"))
torch.save(labels, os.path.join(EMBEDDINGS_DIR, "labels.pt"))
print("Encoding complete.")

# ------------------- Evaluation -------------------
print("=== Evaluation ===")
SUBSAMPLE_SIZE = 1_000_000
GENUINE_SAMPLE_STEP = 16
HIST_BINS = np.arange(-1, 1, 0.01)

embeddings_np = embeddings.numpy()
labels_np = np.array(labels)
normed_embeddings = embeddings_np / np.linalg.norm(embeddings_np, axis=1, keepdims=True)

# Genuine
genuine_scores = []
unique_labels = np.unique(labels_np)
for lbl in unique_labels:
    idxs = np.where(labels_np == lbl)[0]
    for i in range(0, len(idxs), GENUINE_SAMPLE_STEP):
        for j in range(i+1, len(idxs)):
            genuine_scores.append(np.dot(normed_embeddings[idxs[i]], normed_embeddings[idxs[j]]))

# Imposter
imposter_scores = []
num_embeddings = len(labels_np)
for _ in range(min(SUBSAMPLE_SIZE, num_embeddings*10)):
    i = np.random.randint(0, num_embeddings)
    possible_j = np.where(labels_np != labels_np[i])[0]
    j = np.random.choice(possible_j)
    imposter_scores.append(np.dot(normed_embeddings[i], normed_embeddings[j]))

np.savetxt(os.path.join(EVAL_DIR, "genuine_scores.txt"), genuine_scores)
np.savetxt(os.path.join(EVAL_DIR, "imposter_scores.txt"), imposter_scores)

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

# Histogramm / Plot
plt.figure(figsize=(8,6))
plt.hist(genuine_scores, bins=HIST_BINS, alpha=0.5, color="green", label="Genuine")
plt.hist(imposter_scores, bins=HIST_BINS, alpha=0.5, color="red", label="Imposter")
plt.xlim(-1,1)
plt.legend()
plt.xlabel("Cosine similarity")
plt.ylabel("Count")
plt.title(f"Genuine vs Imposter Similarity Distribution {SUFFIX}")
plt.tight_layout()
plt.savefig(os.path.join(EVAL_DIR, f"similarity_histogram_{SUFFIX}.png"), dpi=300)
plt.show()
print("Evaluation complete. Histogram saved.")


# --- Seaborn-Plot hinzufügen ---
import seaborn as sns
import pandas as pd

# Berechne EER
eer_stats = get_eer_stats(genuine_scores, imposter_scores)

# Report erzeugen
eer_dict = {"synthetic_vs_synthetic": eer_stats}
report_path = os.path.join(EVAL_DIR, "pyeer_report.html")
generate_eer_report(list(eer_dict.values()), list(eer_dict.keys()), report_path)
print("EER report saved to:", report_path)

df = pd.DataFrame({
    "score": np.concatenate([genuine_scores, imposter_scores]),
    "label": ["genuine"]*len(genuine_scores) + ["imposter"]*len(imposter_scores)
})

sns.set_theme(style="whitegrid")
plt.figure(figsize=(10,6))

ax = sns.histplot(
    data=df,
    x="score",
    hue="label",
    stat="probability",
    common_norm=False,
    bins=50,
    kde=True,
    palette={"genuine": "#009D81", "imposter": "#0083CC"},
    alpha=0.6
)

legend = ax.get_legend()
if legend is not None:
    legend.set_title(None)


# Achsenlimits setzen
#ax.set_xlim(-1.1, 1.1)   # x-Achse von -1 bis 1
#ax.set_ylim(0, 0.8)  # y-Achse von 0 bis 0.8

# Legende nur für genuine / imposter
#plt.legend(title=None)

# EER-Linie ohne Legende
plt.axvline(x=eer_stats.eer_th, color='orange', linestyle='--', label="_nolegend_")

plt.xlabel("Cosine similarity", fontsize=14, fontweight='bold')
plt.ylabel("Probability", fontsize=14, fontweight='bold')
#plt.title(f"Distribution of Genuine vs. Imposter Scores {SUFFIX}", fontsize=16, fontweight='bold')
plt.tight_layout()
plot_path = os.path.join(EVAL_DIR, f"genuine_vs_imposter_distribution_{SUFFIX}.png")
plt.savefig(plot_path, dpi=300)
#plt.show()
print("Seaborn plot saved to", plot_path)

