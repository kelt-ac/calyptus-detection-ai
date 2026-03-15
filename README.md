# 🌿 Calyptus Detection AI

Détection automatique du Calyptus par Deep Learning (YOLOv8)  
Projet — Module Technologie Innovante & Intelligence Artificielle

![Python](https://img.shields.io/badge/Python-3.12-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.55-red)
![mAP50](https://img.shields.io/badge/mAP50-91.1%25-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📋 Description du Projet

Ce projet implémente un système de détection automatique du **Calyptus**
(plante invasive) en utilisant le Deep Learning et le traitement d'image.
Le modèle est basé sur **YOLOv8s** entraîné sur un dataset de 2507 images
annotées manuellement via Roboflow.

---

## 🏆 Résultats

| Métrique | Score |
|----------|-------|
| mAP@50 | **91.1%** |
| Précision | **85.7%** |
| Rappel | **88.7%** |
| F1-Score | **87.2%** |
| mAP@50-95 | **71.3%** |
| Vitesse inférence | **~10ms/image** |


---

## 🔄 Pipeline du Projet
```
Images Collectées (2500)
        ↓
Annotation (Roboflow) → 2507 images annotées
        ↓
Augmentation (Albumentations ×4) → 7850 images
        ↓
Entraînement (YOLOv8s | 150 epochs | Tesla T4)
        ↓
Évaluation (mAP50 = 91.1%)
        ↓
Déploiement (Streamlit)
```

---

## 🛠️ Technologies Utilisées

| Outil | Rôle |
|-------|------|
| **YOLOv8s** | Architecture de détection |
| **Roboflow** | Annotation des images |
| **Albumentations** | Augmentation des données |
| **Google Colab** | Entraînement GPU |
| **Kaggle** | Entraînement GPU (Tesla T4) |
| **Streamlit** | Interface web |
| **OpenCV** | Traitement d'image |
| **Python 3.12** | Langage de programmation |

---

## 📁 Structure du Projet
```
calyptus-detection-ai/
├── 📁 docs/
│   └── rapport_evaluation.txt
├── 📁 scripts/
│   ├── split_dataset.py
│   ├── augmentation.py
│   ├── train.py
│   ├── verify_annotations.py
│   └── progression.py
├── 📄 app.py               ← Application Streamlit
├── 📄 calyptus.yaml        ← Configuration YOLOv8
├── 📄 requirements.txt     ← Dépendances Python
└── 📄 README.md
```

---

## 🚀 Installation & Utilisation

### 1. Cloner le repository
```bash
git clone https://github.com/kelt-ac/calyptus-detection-ai.git
cd calyptus-detection-ai
```

### 2. Créer l'environnement virtuel
```bash
python -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate    # Mac/Linux
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 4. Télécharger le modèle best.pt
```
Télécharger best.pt depuis Google Drive
https://drive.google.com/file/d/1eppvmG-JWdfDpof1rfR1bN2kmIPQfZdv/view?usp=sharing
→ Placer dans : runs/detect/calyptus_v5/weights/best.pt
```

### 5. Lancer l'application
```bash
streamlit run app.py
```

### 6. Ouvrir dans le navigateur
```
http://localhost:8501
```

---

## 📊 Détails de l'Entraînement

| Paramètre | Valeur |
|-----------|--------|
| Modèle | YOLOv8s |
| Epochs | 150 |
| Batch size | 16 |
| Image size | 640×640 |
| Optimizer | AdamW |
| Learning rate | 0.0005 |
| GPU | Tesla T4 |
| Durée | 5.08 heures |
| Dataset train | 7124 images |
| Dataset valid | 479 images |
| Dataset test | 247 images |

---

## 📈 Progression de l'Entraînement

| Epoch | mAP50 |
|-------|-------|
| 1 | 0.741 |
| 30 | 0.670 |
| 60 | 0.682 |
| 90 | 0.863 |
| 150 | **0.925** |

---

## 🖼️ Utilisation de l'Application
```
1. Ouvrir http://localhost:8501
2. Ajuster le seuil de confiance (défaut: 0.50)
3. Uploader une image JPG/PNG
4. Voir les détections en temps réel
5. Télécharger l'image annotée
```

---

## 📦 Dataset

Le dataset est disponible sur Google Drive :
- 2507 images originales annotées
- 7850 images après augmentation
- Format : YOLOv8 (YOLO txt)
- Classes : 1 (calyptus)

> ⚠️ Le dataset n'est pas inclus dans ce repository
> (trop volumineux). Contactez l'équipe pour y accéder.

---

## 📄 Licence

Ce projet est sous licence **MIT**.  
Voir le fichier [LICENSE](LICENSE) pour plus de détails.

---

## 🎓 Contexte Académique

Projet réalisé dans le cadre du module :  
**Technologie Innovante & Intelligence Artificielle**  
Master DevOps and Cloud Computing — Semestre 1  
Faculté Polydisciplinaire de Larache — 2025/2026

