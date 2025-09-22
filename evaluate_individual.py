# evaluate_individual_fixed.py
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ------------------- Pfade -------------------
EMBEDDINGS_PATH = "out/embeddings/embeddings.npy"
LABELS_PATH = "out/embeddings/labels.npy"
EVAL_DIR = "evaluation"
os.makedirs(EVAL_DIR, exist_ok=True)

# ------------------- Load embeddings and labels -------------------
embeddings = np.load(EMBEDDINGS_PATH)
labels = np.load(LABELS_PATH)

print("Loaded embeddings shape:", embeddings.shape)
print("Loaded labels shape:", labels.shape)

if embeddings.shape[0] == 0:
    print("No embeddings found! Exiting.")
    exit()

# ------------------- Compute genuine & imposter pairs -------------------
genuine_scores = []
imposter_scores = []

num_samples = len(labels)
for i in range(num_samples):
    for j in range(i + 1, num_samples):
        sim = cosine_similarity(
            embeddings[i].reshape(1, -1),
            embeddings[j].reshape(1, -1)
        )[0][0]

        if labels[i] == labels[j]:
            genuine_scores.append(sim)
        else:
            imposter_scores.append(sim)

print("Genuine pairs:", len(genuine_scores))
print("Imposter pairs:", len(imposter_scores))

# ------------------- Save scores -------------------
np.savetxt(os.path.join(EVAL_DIR, "genuine_scores.txt"), genuine_scores)
np.savetxt(os.path.join(EVAL_DIR, "imposter_scores.txt"), imposter_scores)

print(f"Saved genuine_scores.txt and imposter_scores.txt in {EVAL_DIR}/")

