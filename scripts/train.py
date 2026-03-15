# scripts/train.py
# ─────────────────────────────────────────────
# Entraînement YOLOv8 — Détection Calyptus
# ─────────────────────────────────────────────

import torch
from ultralytics import YOLO
import os

# ── Vérification environnement ────────────────
print("=" * 50)
print("   🔍 VÉRIFICATION ENVIRONNEMENT")
print("=" * 50)
print(f"   PyTorch version  : {torch.__version__}")
print(f"   GPU disponible   : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"   GPU              : {torch.cuda.get_device_name(0)}")
    device = 'cuda'
else:
    print("   ⚠️  Pas de GPU — entraînement sur CPU")
    device = 'cpu'
print("=" * 50)

# ── Charger le modèle ─────────────────────────
print("\n📥 Chargement du modèle YOLOv8n...")
model = YOLO('yolov8n.pt')
print("✅ Modèle chargé !\n")

# ── Lancer l'entraînement ─────────────────────
print("🚀 Début de l'entraînement...\n")

results = model.train(
    # Dataset
    data    = 'dataset_augmented/data.yaml',

    # Durée
    epochs  = 50,
    patience= 10,

    # Taille & Batch
    imgsz   = 640,
    batch   = 8,

    # Optimisation
    optimizer    = 'AdamW',
    lr0          = 0.001,
    weight_decay = 0.0005,
    warmup_epochs= 3,

    # Hardware
    device  = device,

    # Sauvegarde
    project = 'runs/detect',
    name    = 'calyptus_v1',
    save    = True,

    # Suivi
    plots   = True,
    verbose = True,
)

print("\n" + "=" * 50)
print("   ✅ ENTRAÎNEMENT TERMINÉ !")
print("=" * 50)
print("   🏆 Meilleur modèle :")
print("   runs/detect/calyptus_v1/weights/best.pt")