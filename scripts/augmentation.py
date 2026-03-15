# scripts/augmentation.py
# ─────────────────────────────────────────────
# Augmentation du dataset Calyptus
# ─────────────────────────────────────────────

import albumentations as A
import cv2
import os
import numpy as np
from tqdm import tqdm


# ── Pipeline d'augmentation ───────────────────
def get_pipeline():
    return A.Compose([

        # Géométrie
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=30,
                 border_mode=cv2.BORDER_CONSTANT, p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.2,
            rotate_limit=15,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.4
        ),

        # Couleur & Luminosité
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=0.5
        ),
        A.HueSaturationValue(
            hue_shift_limit=15,
            sat_shift_limit=30,
            val_shift_limit=20,
            p=0.4
        ),

        # Bruit & Flou
        A.GaussNoise(p=0.3),
        A.MotionBlur(blur_limit=5, p=0.2),

        # Météo
        A.RandomShadow(p=0.3),
        A.RandomFog(
            fog_coef_lower=0.1,
            fog_coef_upper=0.3,
            p=0.2
        ),

    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels'],
        min_visibility=0.3
    ))


# ── Lire les labels YOLO ──────────────────────
def lire_labels(label_path):
    bboxes, classes = [], []
    if not os.path.exists(label_path):
        return classes, bboxes
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) == 5:
                classes.append(int(parts[0]))
                bboxes.append([float(x) for x in parts[1:]])
    return classes, bboxes


# ── Sauvegarder les labels YOLO ───────────────
def sauvegarder_labels(label_path, classes, bboxes):
    with open(label_path, 'w') as f:
        for cls, bbox in zip(classes, bboxes):
            f.write(f"{cls} {' '.join(map(str, bbox))}\n")


# ── Augmenter le dossier train ────────────────
def augmenter(images_dir, labels_dir,
              output_images_dir, output_labels_dir,
              n_augments=3):
    """
    n_augments=3 → 1781 × (1+3) = 7124 images
    """
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    pipeline = get_pipeline()

    images = [f for f in os.listdir(images_dir)
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    compteur = 0

    for fname in tqdm(images, desc="🔄 Augmentation en cours"):
        name      = os.path.splitext(fname)[0]
        img_path  = os.path.join(images_dir, fname)
        lbl_path  = os.path.join(labels_dir, name + '.txt')

        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        classes, bboxes = lire_labels(lbl_path)

        # ── Copier l'original ──────────────────
        cv2.imwrite(
            os.path.join(output_images_dir, fname),
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        )
        sauvegarder_labels(
            os.path.join(output_labels_dir, name + '.txt'),
            classes, bboxes
        )

        # ── Générer les augmentations ──────────
        for i in range(n_augments):
            try:
                aug = pipeline(
                    image        = img,
                    bboxes       = bboxes,
                    class_labels = classes
                )

                aug_name = f"{name}_aug{i+1}"

                cv2.imwrite(
                    os.path.join(output_images_dir, aug_name + '.jpg'),
                    cv2.cvtColor(aug['image'], cv2.COLOR_RGB2BGR)
                )
                sauvegarder_labels(
                    os.path.join(output_labels_dir, aug_name + '.txt'),
                    aug['class_labels'],
                    aug['bboxes']
                )
                compteur += 1

            except Exception as e:
                print(f"⚠️ Erreur {fname} aug{i+1}: {e}")

    print(f"\n✅ {len(images)} originales + {compteur} augmentées")
    print(f"📦 Total train : {len(images) + compteur} images")


# ── Copier val et test sans augmentation ──────
def copier_sans_augmentation(src_dir, dst_dir):
    import shutil
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir)
    print(f"📋 Copié : {dst_dir}")


# ══════════════════════════════════════════════
#  LANCER L'AUGMENTATION
# ══════════════════════════════════════════════
if __name__ == "__main__":

    print("=" * 50)
    print("   🔄 AUGMENTATION DU DATASET CALYPTUS")
    print("=" * 50)

    # ── Augmenter le train ─────────────────────
    print("\n📁 Augmentation du dossier TRAIN...")
    augmenter(
        images_dir        = 'dataset_final/train/images',
        labels_dir        = 'dataset_final/train/labels',
        output_images_dir = 'dataset_augmented/train/images',
        output_labels_dir = 'dataset_augmented/train/labels',
        n_augments        = 3
    )

    # ── Copier valid sans augmentation ────────
    print("\n📁 Copie du dossier VALID...")
    copier_sans_augmentation(
        'dataset_final/valid',
        'dataset_augmented/valid'
    )

    # ── Copier test sans augmentation ─────────
    print("\n📁 Copie du dossier TEST...")
    copier_sans_augmentation(
        'dataset_final/test',
        'dataset_augmented/test'
    )

    # ── Créer data.yaml pour le dataset augmenté
    yaml_content = """# Dataset Calyptus - Augmented
path: ./dataset_augmented
train: train/images
val: valid/images
test: test/images

nc: 1
names: ['calyptus']
"""
    with open('dataset_augmented/data.yaml', 'w') as f:
        f.write(yaml_content)

    # ── Résumé final ───────────────────────────
    from pathlib import Path
    print("\n" + "=" * 50)
    print("   📊 RÉSUMÉ FINAL")
    print("=" * 50)
    for split in ['train', 'valid', 'test']:
        img_dir = Path(f'dataset_augmented/{split}/images')
        n = len(list(img_dir.glob('*'))) if img_dir.exists() else 0
        print(f"   {split:6s} : {n} images")
    print("=" * 50)
    print("\n🎉 Augmentation terminée avec succès !")