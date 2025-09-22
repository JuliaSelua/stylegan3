# evaluate_individual.py
import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from pyeer.eer_info import get_eer_stats
from pyeer.report import generate_eer_report

# ------------------- Pfade -------------------
EMBEDDINGS_PATH = "out/embeddings/embeddings.npy"
LABELS_PATH = "out/embeddings/labels.npy"
OUTPUT_DIR = "out/evaluation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------- Laden -------------------
embeddings = np.load(EMBEDDINGS_PATH, allow_pickle=True)
labels = np.load(LABELS_PATH, allow_pickle=True)

# Falls es npz-Dateien sind:
if isinstance(embeddings, np.lib.npyio.NpzFile):
    print("Embeddings keys:", embeddings.files)
    embeddings = embeddings["arr_0"]  # oder der passende Key

if isinstance(labels, np.lib.npyio.NpzFile):
    print("Labels keys:", labels.files)
    labels = labels["arr_0"]

print(f"Loaded {embeddings.shape[0]} embeddings of size {embeddings.shape[1]}")
print(f"Unique IDs: {len(np.unique(labels))}")


# ------------------- Helpers -------------------
def generate_genuine_pairs(labels):
    """Alle Paare mit derselben ID"""
    for label in np.unique(labels):
        idxs = np.where(labels == label)[0]
        for i, j in product(idxs, idxs):
            if i < j:  # keine Duplikate
                yield i, j

def generate_random_imposter_pairs(labels, n=10000):
    """Random Paare mit unterschiedlicher ID"""
    seen = set()
    while len(seen) < n:
        i, j = np.random.choice(len(labels), 2, replace=False)
        if labels[i] != labels[j]:
            seen.add((i, j))
            yield i, j

# ------------------- Score Berechnung -------------------
genuine_scores = []
for i, j in generate_genuine_pairs(labels):
    cos_sim = np.dot(embeddings[i], embeddings[j])
    genuine_scores.append(cos_sim)

imposter_scores = []
for i, j in generate_random_imposter_pairs(labels, n=len(genuine_scores)):
    cos_sim = np.dot(embeddings[i], embeddings[j])
    imposter_scores.append(cos_sim)

print(f"Genuine pairs: {len(genuine_scores)}")
print(f"Imposter pairs: {len(imposter_scores)}")

# ------------------- Plotten -------------------
plt.figure(figsize=(8,6))
plt.hist(genuine_scores, bins=np.linspace(-1, 1, 50), alpha=0.5, label="Genuine", color="green")
plt.hist(imposter_scores, bins=np.linspace(-1, 1, 50), alpha=0.5, label="Imposter", color="red")
plt.xlabel("Cosine similarity")
plt.ylabel("Frequency")
plt.legend()
plt.title("Synthetic vs Synthetic")
plt.savefig(os.path.join(OUTPUT_DIR, "synthetic_vs_synthetic.png"), dpi=200)

# ------------------- EER Stats -------------------
eer_stats = get_eer_stats(genuine_scores, imposter_scores)
print("EER stats:", eer_stats)

# Optional: HTML-Report
report_path = os.path.join(OUTPUT_DIR, "pyeer_report.html")
generate_eer_report([eer_stats], ["synthetic_vs_synthetic"], report_path)

# ------------------- Save Raw Scores -------------------
np.savetxt(os.path.join(OUTPUT_DIR, "genuine_scores.txt"), genuine_scores)
np.savetxt(os.path.join(OUTPUT_DIR, "imposter_scores.txt"), imposter_scores)

print("Done. Results saved in:", OUTPUT_DIR)
