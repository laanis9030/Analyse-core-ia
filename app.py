import streamlit as st
import numpy as np
import tensorflow as tf
from scipy.signal import stft
import scipy.io as sio
import matplotlib.pyplot as plt
import os
import gdown

# ==========================================
# ⚙️ CONFIGURATION DE LA PAGE
# ==========================================
st.set_page_config(
    page_title="PredictCore AI | Analyse MLOps",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 🧠 TÉLÉCHARGEMENT ET CHARGEMENT DU MODÈLE
# ==========================================
@st.cache_resource
def load_ai_model():
    model_path = "predict_core_model.keras"
    file_id = '1wYE7yoEwSSAjHqiyM2Uf8_l0xvHUGVDf' # L'ID de ton fichier sur Google Drive
    
    # 1. Vérifie si le modèle est déjà là. Sinon, le télécharge.
    if not os.path.exists(model_path):
        with st.spinner("⏳ Téléchargement du cerveau de l'IA depuis Google Drive... (Cela peut prendre 1 à 2 minutes la première fois)"):
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, model_path, quiet=False)
            
    # 2. Charge le modèle en mémoire
    return tf.keras.models.load_model(model_path)

try:
    model = load_ai_model()
    modele_charge = True
except Exception as e:
    modele_charge = False
    st.error(f"🚨 Impossible de charger le modèle. Erreur : {e}")

# ==========================================
# 🛠️ FONCTIONS DE TRAITEMENT DU SIGNAL
# ==========================================
def process_signal_to_spectrogram(signal_brut):
    """Transforme un signal 1D en spectrogramme 65x65x1 pour le CNN"""
    if len(signal_brut) > 2048:
        signal_brut = signal_brut[:2048]
    elif len(signal_brut) < 2048:
        signal_brut = np.pad(signal_brut, (0, 2048 - len(signal_brut)))
        
    _, _, Zxx = stft(signal_brut, fs=12000, nperseg=128, noverlap=96)
    spectro = np.abs(Zxx)
    
    spectro = (spectro - spectro.min()) / (spectro.max() - spectro.min() + 1e-8)
    
    spectro_input = np.expand_dims(spectro, axis=0)
    spectro_input = np.expand_dims(spectro_input, axis=-1)
    
    return spectro, spectro_input

def generate_dummy_signal():
    """Génère un faux signal bruité pour la démonstration"""
    t = np.linspace(0, 1, 2048)
    signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.random.randn(2048)
    return signal

# ==========================================
# 🎨 INTERFACE UTILISATEUR (FRONTEND)
# ==========================================
st.title("⚙️ PredictCore AI : Diagnostic Multi-Tâches")
st.markdown("""
Bienvenue sur le tableau de bord de maintenance prédictive. 
Cette intelligence artificielle analyse les vibrations des roulements pour détecter simultanément :
**Le type de défaut**, **le diamètre de la fissure**, et **l'indice de santé global**.
""")

st.sidebar.header("📁 Importation des données")
uploaded_file = st.sidebar.file_uploader("Chargez un fichier signal (.mat)", type=["mat"])

use_dummy = st.sidebar.button("Ou générer un signal de test")

st.sidebar.markdown("---")
st.sidebar.info("Modèle : Custom ResNet (4.2M Paramètres)")

# ==========================================
# 🚀 MOTEUR DE PRÉDICTION
# ==========================================
signal_data = None

if uploaded_file is not None:
    try:
        mat_data = sio.loadmat(uploaded_file)
        time_keys = [k for k in mat_data.keys() if 'time' in k]
        if time_keys:
            signal_data = mat_data[time_keys[0]].flatten()
            st.success("Fichier chargé avec succès !")
        else:
            st.error("Aucune donnée temporelle trouvée dans ce fichier .mat.")
    except Exception as e:
        st.error(f"Erreur de lecture du fichier : {e}")
elif use_dummy:
    signal_data = generate_dummy_signal()
    st.info("Utilisation d'un signal de démonstration généré aléatoirement.")

if signal_data is not None and modele_charge:
    st.markdown("---")
    st.header("📊 Résultats de l'Analyse IA")
    
    col1, col2 = st.columns([1, 2])
    
    with st.spinner('Analyse par réseau de neurones en cours...'):
        spectro_visu, x_pred = process_signal_to_spectrogram(signal_data)
        
        preds = model.predict(x_pred)
        pred_class = np.argmax(preds[0][0])
        pred_fissure = preds[1][0][0]
        pred_health = preds[2][0][0]
        
        classes_dict = {
            0: ("Sain (Normal)", "✅", "green"),
            1: ("Défaut Bague Interne", "⚠️", "orange"),
            2: ("Défaut Bille", "🔴", "red"),
            3: ("Défaut Bague Externe", "🚨", "red")
        }
        
        etat_texte, icone, couleur = classes_dict.get(pred_class, ("Inconnu", "❓", "gray"))
    
    with col1:
        st.subheader("Diagnostic Actuel")
        st.markdown(f"### {icone} {etat_texte}")
        
        st.metric(label="Diamètre de la Fissure estimé", value=f"{pred_fissure:.3f} pouces")
        
        health_clamped = max(0.0, min(100.0, pred_health))
        st.metric(label="Indice de Santé (SoH)", value=f"{health_clamped:.1f} %")
        st.progress(int(health_clamped) / 100)

    with col2:
        st.subheader("Empreinte Vibratoire (Spectrogramme)")
        fig, ax = plt.subplots(figsize=(6, 4))
        cax = ax.imshow(spectro_visu, aspect='auto', origin='lower', cmap='jet')
        ax.set_title("Transformation de Fourier à Court Terme (STFT)")
        ax.set_xlabel("Temps")
        ax.set_ylabel("Fréquence")
        fig.colorbar(cax, ax=ax, label="Amplitude Nomalisée")
        st.pyplot(fig)
