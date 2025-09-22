import os
import numpy as np
import torch

# === Lade Embeddings ===
emb_path = "out/embeddings/embeddings.npy"
labels_path = "out/embeddings/labels.npy"

embeddings = torch.load(emb_path, map_location="cpu")
labels = torch.load(labels_path, map_location="cpu")

# --- Normalisiere auf numpy 2D Array ---
if isinstance(embeddings, list):
    # Liste von Vektoren
    embeddings = np.stack([e.detach().cpu().numpy() if torch.is_tensor(e) else np.array(e) for e in embeddings])
elif torch.is_tensor(embeddings):
    embeddings = embeddings.detach().cpu().numpy()
elif isinstance(embeddings, np.lib.npyio.NpzFile):
    # Falls es np.savez war
    print("Embeddings keys:", embeddings.files)
    embeddings = embeddings[embeddings.files[0]]
elif isinstance(embeddings, np.ndarray):
    pass
else:
    raise TypeError(f"Unsupported embeddings type: {type(embeddings)}")

# Labels ebenfalls angleichen
if isinstance(labels, list):
    labels = np.array(labels)
elif torch.is_tensor(labels):
    labels = labels.cpu().numpy()

print(f"Loaded embeddings shape: {embeddings.shape}")
print(f"Loaded labels shape: {labels.shape}")

# === Beispiel: Genuine & Imposter Scores ===
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

genuine_scores = []
imposter_scores = []

for i in range(len(embeddings)):
    for j in range(i + 1, len(embeddings)):
        sim = cosine_similarity(embeddings[i], embeddings[j])
        if labels[i] == labels[j]:
            genuine_scores.append(sim)
        else:
            imposter_scores.append(sim)

print(f"Genuine pairs: {len(genuine_scores)}")
print(f"Imposter pairs: {len(imposter_scores)}")

# Speichern für spätere Auswertung / Plotten
os.makedirs("evaluation", exist_ok=True)
np.savetxt("evaluation/genuine_scores.txt", genuine_scores)
np.savetxt("evaluation/imposter_scores.txt", imposter_scores)

print("Saved genuine_scores.txt and imposter_scores.txt in evaluation/")
