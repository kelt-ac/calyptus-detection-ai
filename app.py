# app.py — Interface Streamlit Détection Calyptus
# ─────────────────────────────────────────────────

import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import time
import io
import pandas as pd

# ── Configuration page ─────────────────────────
st.set_page_config(
    page_title = "Détecteur de Calyptus",
    page_icon  = "🌿",
    layout     = "wide"
)

# ── CSS personnalisé ───────────────────────────
st.markdown("""
<style>
    .titre {
        text-align: center;
        color: #2e7d32;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sous-titre {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .success-box {
        background: #e8f5e9;
        border: 1px solid #4caf50;
        border-radius: 8px;
        padding: 1rem;
        color: #1b5e20;
    }
    .warning-box {
        background: #fff3e0;
        border: 1px solid #ff9800;
        border-radius: 8px;
        padding: 1rem;
        color: #e65100;
    }
</style>
""", unsafe_allow_html=True)

# ── Titre ──────────────────────────────────────
st.markdown(
    '<div class="titre">Detecteur de Calyptus</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="sous-titre">Detection automatique par YOLOv8 — Projet IA Technologie Innovante</div>',
    unsafe_allow_html=True
)
st.markdown("---")


# ── Charger le modèle (mis en cache) ──────────
@st.cache_resource
def charger_modele():
    try:
        model = YOLO('runs/detect/calyptus_v5/weights/best.pt')
        return model
    except Exception as e:
        st.error(f"Erreur chargement modele : {e}")
        return None

model = charger_modele()

if model is None:
    st.error("Modele non trouve ! Verifiez le chemin de best.pt")
    st.stop()


# ── Fonction de détection ──────────────────────
def detecter(image_pil, seuil, iou):
    img_np = np.array(image_pil)
    debut  = time.time()

    results = model.predict(
        img_np,
        conf    = seuil,
        iou     = iou,
        verbose = False
    )[0]

    duree     = time.time() - debut
    img_annot = img_np.copy()
    detections = []

    for i, box in enumerate(results.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        aire = (x2 - x1) * (y2 - y1)

        # Dessiner la boite
        cv2.rectangle(img_annot, (x1,y1), (x2,y2),
                      (34, 139, 34), 3)

        # Label avec fond
        label = f"#{i+1} {conf*100:.1f}%"
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(img_annot,
                      (x1, y1-th-12), (x1+tw+8, y1),
                      (34, 139, 34), -1)
        cv2.putText(img_annot, label, (x1+4, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2)

        detections.append({
            'N'         : i + 1,
            'Confiance' : f"{conf*100:.1f}%",
            'Position'  : f"({x1},{y1})-({x2},{y2})",
            'Aire (px)' : aire
        })

    return Image.fromarray(img_annot), detections, duree


# ══════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════
with st.sidebar:
    st.header("Parametres")

    seuil = st.slider(
        "Seuil de confiance",
        min_value = 0.10,
        max_value = 0.95,
        value     = 0.50,
        step      = 0.05,
        help      = "Plus eleve = moins de faux positifs"
    )

    iou = st.slider(
        "Seuil IoU",
        min_value = 0.10,
        max_value = 0.95,
        value     = 0.45,
        step      = 0.05,
        help      = "Controle la suppression des doublons"
    )

    st.markdown("---")
    st.header("Performances du Modele")
    st.metric("mAP50",     "91.1%", "+2.1%")
    st.metric("Precision", "85.7%", "+1.8%")
    st.metric("Recall",    "88.7%", "+3.2%")
    st.metric("F1-Score",  "87.2%", "+2.5%")

    st.markdown("---")
    st.header("Informations")
    st.info("""
    Modele   : YOLOv8s
    Dataset  : 2507 images
    Augmente : 7850 images
    Epochs   : 150
    GPU      : Tesla T4
    Duree    : 5.08 heures
    """)


# ══════════════════════════════════════════════
#  ZONE PRINCIPALE
# ══════════════════════════════════════════════
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("Image d'entree")

    source = st.radio(
        "Source :",
        ["Uploader une image", "Camera (webcam)"],
        horizontal=True
    )

    image_pil = None

    if source == "Uploader une image":
        fichier = st.file_uploader(
            "Choisir une image",
            type=['jpg', 'jpeg', 'png', 'webp']
        )
        if fichier:
            image_pil = Image.open(fichier).convert('RGB')
            st.image(
                image_pil,
                caption        = "Image originale",
                use_column_width = True
            )
    else:
        photo = st.camera_input("Prendre une photo")
        if photo:
            image_pil = Image.open(photo).convert('RGB')


with col2:
    st.subheader("Resultat de detection")

    if image_pil is not None:
        with st.spinner("Analyse en cours..."):
            img_result, detections, duree = detecter(
                image_pil, seuil, iou
            )

        st.image(
            img_result,
            caption          = "Image annotee",
            use_column_width = True
        )

        n = len(detections)

        if n > 0:
            st.success(
                f"✅ {n} calyptus detecte(s) "
                f"en {duree*1000:.0f} ms"
            )
        else:
            st.warning(
                "❌ Aucun calyptus detecte — "
                "essayez de baisser le seuil"
            )

        # Tableau des détections
        if detections:
            st.markdown("### Details des detections")
            df = pd.DataFrame(detections)
            st.dataframe(df, use_container_width=True)

        # Bouton téléchargement
        buf = io.BytesIO()
        img_result.save(buf, format='JPEG', quality=95)
        st.download_button(
            label     = "Telecharger l'image annotee",
            data      = buf.getvalue(),
            file_name = "calyptus_detection.jpg",
            mime      = "image/jpeg"
        )

    else:
        st.info("Uploadez une image pour commencer l'analyse")


# ══════════════════════════════════════════════
#  STATISTIQUES SESSION
# ══════════════════════════════════════════════
st.markdown("---")
st.subheader("A propos du projet")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Images annotees",  "2507")
c2.metric("Images augmentees","7850")
c3.metric("Meilleur mAP50",   "91.1%")
c4.metric("Vitesse inference", "~10ms")