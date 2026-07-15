# =====================================================================
#  🛡️ ANALYSE-CORE V2 - NASA PREDICTIVE MAINTENANCE DASHBOARD (app.py)
# =====================================================================

import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os
import gdown
import torchvision.transforms as T

# 1. CONFIGURATION DE LA PAGE & THÈME
st.set_page_config(
    page_title="Analyse-Core v2 - NASA Diagnostic",
    page_icon="⚡",
    layout="wide"
)

# Injection CSS personnalisée pour un design Cyberpunk / Sci-Fi haut de gamme
st.markdown("""
    <style>
        /* Fond global et polices */
        .stApp {
            background: linear-gradient(135deg, #0a0b10 0%, #12131c 100%);
            color: #e2e8f0;
            font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        }
        
        /* Titre principal néon */
        .neon-title {
            font-size: 2.8rem;
            font-weight: 800;
            background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 0 20px rgba(0, 242, 254, 0.3);
            margin-bottom: 0.5rem;
        }
        
        /* Cartes de diagnostic (Glassmorphism) */
        .diagnostic-card {
            background: rgba(30, 41, 59, 0.45);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        }
        
        /* Badge d'état de santé */
        .badge-normal {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.4) 100%);
            border: 1px solid #10b981;
            color: #34d399;
            padding: 10px 20px;
            border-radius: 8px;
            font-weight: bold;
            text-align: center;
            text-shadow: 0 0 10px rgba(16, 185, 129, 0.5);
        }
        .badge-warning {
            background: linear-gradient(135deg, rgba(245, 158, 11, 0.2) 0%, rgba(217, 119, 6, 0.4) 100%);
            border: 1px solid #f59e0b;
            color: #fbbf24;
            padding: 10px 20px;
            border-radius: 8px;
            font-weight: bold;
            text-align: center;
            text-shadow: 0 0 10px rgba(245, 158, 11, 0.5);
        }
        .badge-danger {
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(220, 38, 38, 0.4) 100%);
            border: 1px solid #ef4444;
            color: #fca5a5;
            padding: 10px 20px;
            border-radius: 8px;
            font-weight: bold;
            text-align: center;
            text-shadow: 0 0 10px rgba(239, 68, 68, 0.5);
        }
        
        /* Custom sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: #0d0e15 !important;
            border-right: 1px solid rgba(255, 255, 255, 0.05);
        }
    </style>
""", unsafe_allow_html=True)

# 2. DEFINITION DE L'ARCHITECTURE DU MODÈLE
class AnalyseCoreIMS(nn.Module):
    def __init__(self):
        super(AnalyseCoreIMS, self).__init__()
        resnet = models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        in_features = 512
        
        # Têtes Multi-Tâches
        self.rul_head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.health_head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.fault_head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 4)
        )
        
    def forward(self, img):
        features = self.backbone(img)
        features = torch.flatten(features, 1)
        return self.rul_head(features), self.health_head(features), self.fault_head(features)

# 3. CHARGEMENT SÉCURISÉ DU MODÈLE VIA TON LIEN GOOGLE DRIVE
GOOGLE_DRIVE_FILE_ID = "1BiF5lYgM7ChVaABQT3e3_stEjkugd95M"
MODEL_LOCAL_PATH = "analyse_core_ims_v2.pt"

@st.cache_resource
def load_predictive_model():
    if not os.path.exists(MODEL_LOCAL_PATH):
        with st.spinner("📥 Connexion sécurisée au stockage Cloud de Google Drive..."):
            url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
            try:
                gdown.download(url, MODEL_LOCAL_PATH, quiet=False)
            except Exception as e:
                st.error(f"Erreur de téléchargement : {e}. Assurez-vous que le lien Drive est accessible.")
            
    model = AnalyseCoreIMS()
    # Chargement sur CPU pour garantir la compatibilité complète sur Streamlit Cloud
    model.load_state_dict(torch.load(MODEL_LOCAL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

# --- HEADER DE L'APPLICATION ---
st.markdown("<h1 class='neon-title'>🛡️ ANALYSE-CORE v2</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #8a8f98; font-size: 1.1rem; margin-top:-10px;'>Système de Diagnostic Multi-Tâches IA & Maintenance Prédictive (Dataset NASA IMS)</p>", unsafe_allow_html=True)
st.markdown("<hr style='border: 0; height: 1px; background: linear-gradient(90deg, rgba(0,242,254,0.8), rgba(0,0,0,0)); margin-bottom: 30px;'>", unsafe_allow_html=True)

# --- CONFIGURATION SIDEBAR DE CONTRÔLE ---
st.sidebar.markdown("<h2 style='color: #00f2fe; font-size: 1.5rem; margin-bottom: 20px;'>🛠️ Salle de Contrôle</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='color: #8a8f98; font-size: 0.9rem;'>Fournissez les vibrations de l'accéléromètre à analyser.</p>", unsafe_allow_html=True)

option = st.sidebar.selectbox(
    "Méthode de capture du signal",
    ["🔄 Générer un signal physique de test", "📤 Téléverser un fichier brut (.npy, .txt)"]
)

sig = None

if option == "📤 Téléverser un fichier brut (.npy, .txt)":
    uploaded_file = st.sidebar.file_uploader("Fichier accélérométrique (min. 2048 points)", type=["npy", "txt"])
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".npy"):
            sig = np.load(uploaded_file)
        else:
            sig = np.loadtxt(uploaded_file)
else:
    test_type = st.sidebar.selectbox("Sélectionner une signature physique :", [
        "Nominale (Machine Saine)", 
        "Anomalie Bague Interne (Début d'usure)", 
        "Défaut de Bille (Dégradation moyenne)", 
        "Défaillance Bague Externe (État critique)"
    ])
    
    if st.sidebar.button("⚡ Injecter le signal dans l'analyseur"):
        t = np.linspace(0, 2048/20000, 2048)
        noise = np.random.normal(0, 0.15, 2048)
        if test_type == "Nominale (Machine Saine)":
            sig = noise
        elif test_type == "Anomalie Bague Interne (Début d'usure)":
            sig = noise + 0.45 * np.sin(2 * np.pi * 120 * t) + 0.3 * np.sin(2 * np.pi * 240 * t)
        elif test_type == "Défaut de Bille (Dégradation moyenne)":
            sig = noise + 0.65 * np.sin(2 * np.pi * 80 * t) + 0.5 * np.sin(2 * np.pi * 160 * t)
        else: # Défaillance Bague Externe
            sig = noise + 1.6 * np.sin(2 * np.pi * 230 * t) + 1.2 * np.sin(2 * np.pi * 460 * t)
        st.sidebar.success(f"Signal '{test_type}' injecté !")

# --- COEUR DU PIPELINE DE TRAITEMENT ET AFFICHAGE ---
if sig is not None:
    if len(sig) < 2048:
        st.error("⚠️ Erreur : Le signal doit contenir au moins 2048 points d'échantillonnage.")
    else:
        # Prélèvement du segment de 2048 points et normalisation
        segment = sig[:2048]
        segment = (segment - np.mean(segment)) / (np.std(segment) + 1e-10)
        
        # 1. Génération du Spectrogramme STFT de haute précision
        f, t_spec, Sxx = signal.spectrogram(segment, fs=20000, nperseg=256, noverlap=128)
        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        Sxx_norm = (Sxx_db - Sxx_db.min()) / (Sxx_db.max() - Sxx_db.min() + 1e-10)
        
        # Préparation du tenseur pour le ResNet18 [1, 3, 128, 128]
        img_tensor = torch.tensor(Sxx_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        img_tensor = T.Resize((128, 128))(img_tensor).repeat(1, 3, 1, 1)
        
        # 2. Inférence de l'IA
        model = load_predictive_model()
        with torch.no_grad():
            pred_rul, pred_health, pred_fault = model(img_tensor)
            
        rul_val = pred_rul.item() * 100.0
        health_val = pred_health.item() * 100.0
        fault_idx = torch.argmax(pred_fault, dim=1).item()
        
        # 3. INTERFACE DE DIAGNOSTIC DYNAMIQUE
        col_metrics, col_plots = st.columns([1, 1.3])
        
        with col_metrics:
            st.markdown("<h3 style='color: #00f2fe; margin-bottom: 20px;'>📊 Diagnostic de l'IA</h3>", unsafe_allow_html=True)
            
            # Bloc d'état dynamique de la machine
            st.markdown("<div class='diagnostic-card'>", unsafe_allow_html=True)
            st.markdown("<p style='color: #8a8f98; margin-bottom: 5px; font-weight:600;'>CLASSIFICATION DU COMPORTEMENT</p>", unsafe_allow_html=True)
            
            classes = [
                ("🟢 MACHINE SAINE", "Aucune dégradation détectée. Fonctionnement nominal de l'équipement.", "badge-normal"),
                ("⚠️ DEFAUT BAGUE INTERNE", "Usure naissante sur la bague interne du roulement. À surveiller lors des prochains entretiens.", "badge-warning"),
                ("⚠️ DEFAUT ELEMENTS ROULANTS (BILLES)", "Ecaillage ou micro-fissure sur une ou plusieurs billes. Surveillance renforcée.", "badge-warning"),
                ("🚨 DEFAUT BAGUE EXTERNE (CRITIQUE)", "Défaillance structurelle majeure de la bague externe. Arrêt d'urgence de la ligne préconisé !", "badge-danger")
            ]
            
            title, desc, badge_style = classes[fault_idx]
            st.markdown(f"<div class='{badge_style}' style='font-size: 1.3rem; letter-spacing: 1px;'>{title}</div>", unsafe_allow_html=True)
            st.markdown(f"<p style='color: #d1d5db; margin-top: 15px; font-size: 0.95rem; line-height:1.5;'>{desc}</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Blocs RUL et Etat de Santé
            st.markdown("<div class='diagnostic-card'>", unsafe_allow_html=True)
            
            # Health
            st.markdown(f"<div style='display:flex; justify-content:space-between; margin-bottom:5px;'><span style='font-weight:600; color:#e2e8f0;'>❤️ ÉTAT DE SANTÉ ACTUEL</span><span style='color:#00f2fe; font-weight:bold; font-size:1.1rem;'>{health_val:.1f}%</span></div>", unsafe_allow_html=True)
            st.progress(health_val / 100.0)
            
            st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)
            
            # RUL
            st.markdown(f"<div style='display:flex; justify-content:space-between; margin-bottom:5px;'><span style='font-weight:600; color:#e2e8f0;'>⏳ DURÉE DE VIE UTILE RESTANTE (RUL)</span><span style='color:#f59e0b; font-weight:bold; font-size:1.1rem;'>{rul_val:.1f}%</span></div>", unsafe_allow_html=True)
            st.progress(rul_val / 100.0)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col_plots:
            st.markdown("<h3 style='color: #00f2fe; margin-bottom: 20px;'>📈 Signaux & Spectres de Fréquence</h3>", unsafe_allow_html=True)
            
            # Affichage graphique haut de gamme avec Matplotlib sous thème sombre
            plt.style.use('dark_background')
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.5))
            fig.patch.set_facecolor('#0a0b10')
            
            # Waveform
            ax1.plot(segment[:600], color='#00f2fe', alpha=0.85, linewidth=1.5)
            ax1.set_facecolor('#11131e')
            ax1.set_title("VIBRATIONS BRUTES (ÉCHANTILLON TEMPOREL)", color='#8a8f98', fontsize=10, loc='left', pad=8)
            ax1.grid(True, color='rgba(255,255,255,0.05)')
            ax1.tick_params(colors='#8a8f98', labelsize=8)
            
            # Spectrogram
            im = ax2.pcolormesh(t_spec * 1000, f, Sxx_db, shading='gouraud', cmap='magma')
            ax2.set_facecolor('#11131e')
            ax2.set_title("SPECTROGRAMME STFT (RÉPARTITION SPECTRO-TEMPORELLE)", color='#8a8f98', fontsize=10, loc='left', pad=8)
            ax2.set_ylabel("Fréquence (Hz)", color='#8a8f98', fontsize=8)
            ax2.set_xlabel("Temps (ms)", color='#8a8f98', fontsize=8)
            ax2.tick_params(colors='#8a8f98', labelsize=8)
            
            # Colorbar stylisée
            cbar = fig.colorbar(im, ax=ax2, orientation='horizontal', pad=0.2, shrink=0.7)
            cbar.set_label("Densité de puissance (dB)", color='#8a8f98', fontsize=8)
            cbar.ax.tick_params(labelsize=7, colors='#8a8f98')
            
            plt.tight_layout()
            st.pyplot(fig)
else:
    # Page d'accueil lorsque aucun signal n'est encore analysé
    st.markdown("<div class='diagnostic-card' style='text-align: center; padding: 60px 20px; border-style: dashed; border-color: rgba(0, 242, 254, 0.4);'>", unsafe_allow_html=True)
    st.markdown("<h2 style='color:#00f2fe; margin-bottom: 15px;'>📡 Système Prêt à l'Analyse</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #8a8f98; font-size:1.1rem; max-width:600px; margin: 0 auto 30px auto;'>L'intelligence artificielle multi-tâches d'Analyse-Core attend de recevoir les signaux vibratoires pour démarrer le monitoring en temps réel.</p>", unsafe_allow_html=True)
    st.markdown("<p style='color: #34d399; font-weight:bold;'>💡 Choisissez une signature dans le panneau latéral gauche et cliquez sur 'Injecter' !</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
