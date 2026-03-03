# 🌿 Calyptus Detection AI

Détection automatique du Calyptus par Deep Learning (YOLOv8)  
Projet — Module Technologie Innovante & Intelligence Artificielle

---

## 📋 Pipeline du Projet
```
Images → Annotation → Augmentation → Entraînement → Évaluation → Déploiement
```

## 🚀 Installation
```bash
git clone https://github.com/kelt-ac/calyptus-detection-ai.git
cd calyptus-detection-ai
pip install -r requirements.txt
```

## ▶️ Utilisation
```bash
# Entraîner le modèle
python train.py

# Évaluer
python evaluate.py

# Lancer l'interface web
streamlit run app_streamlit.py

# Lancer la démo rapide
python app_gradio.py
```

## 🛠️ Technologies
- **Modèle** : YOLOv8m (Ultralytics)
- **Augmentation** : Albumentations
- **Interface** : Streamlit / Gradio
- **API** : FastAPI
