# scripts/split_dataset.py
# ─────────────────────────────────────────────
# Divise vos 2500 images en train / val / test
# ─────────────────────────────────────────────

import os
import shutil
import random

def split_dataset(source_dir, output_dir, ratios=(0.7, 0.2, 0.1), seed=42):

    # Vérifier que le dossier source existe
    if not os.path.exists(source_dir):
        print(f"❌ Dossier introuvable : {source_dir}")
        return

    # Récupérer toutes les images
    extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    images = [f for f in os.listdir(source_dir) if f.endswith(extensions)]

    if len(images) == 0:
        print("❌ Aucune image trouvée dans le dossier source")
        return

    print(f"📦 {len(images)} images trouvées")

    # Mélanger aléatoirement (reproductible grâce au seed)
    random.seed(seed)
    random.shuffle(images)

    # Calculer les indices
    n       = len(images)
    n_train = int(n * ratios[0])
    n_val   = int(n * ratios[1])

    splits = {
        'train': images[:n_train],
        'val'  : images[n_train:n_train + n_val],
        'test' : images[n_train + n_val:]
    }

    # Créer les dossiers et copier
    for split, fichiers in splits.items():
        dossier = os.path.join(output_dir, 'images', split)
        os.makedirs(dossier, exist_ok=True)

        for fname in fichiers:
            src = os.path.join(source_dir, fname)
            dst = os.path.join(dossier, fname)
            shutil.copy2(src, dst)

        print(f"✅ {split:6s} : {len(fichiers):4d} images → {dossier}")

    print("\n🎉 Division terminée avec succès !")


# ── Lancer ────────────────────────────────────
if __name__ == "__main__":

    # ⚠️ MODIFIEZ CE CHEMIN avec l'emplacement de vos 2500 images
    SOURCE  = "C:/Users/K-ACHGUER/Desktop/Master DevOps and Cloud Computing/Semester 1/Intelligence Artificielle et Innovation Technologique/AI"
    OUTPUT  = "./dataset"

    split_dataset(
        source_dir = SOURCE,
        output_dir = OUTPUT
    )