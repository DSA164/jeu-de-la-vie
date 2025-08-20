# 🌐 Web Visualisation — Choix d’architecture pour le Jeu de la Vie

Ce document explique **les choix techniques** retenus pour l’application de visualisation web du **Jeu de la Vie**.  
Objectif : **afficher l’évolution en temps réel** d’une grande grille, avec une **latence minimale** et un **coût CPU** contenu.

---

## 🎯 Résumé des choix

1) **NumPy** pour la représentation et l’évolution de la grille  
   → simple, rapide (vectorisation), interopérable avec M/L (PyTorch/JAX), facile à **afficher** (`imshow`, `st.image`…).

2) **Streamlit** pour la présentation  
   → mise en page rapide, contrôles (boutons, sliders), partage facile, code concis.

3) **Web-streaming** pour l’animation temps réel  
   → **éviter** le cycle `streamlit.rerun()` (cher et latent), **pousser** des frames vidéo vers le navigateur (latence plus faible, rendu fluide).

---

## 1) Pourquoi **NumPy** pour l’état et le calcul

- **Modèle de données** : grille 2D binaire `A` (dtype `bool`/`uint8`) — mémoire compacte, opérations vectorisées.
- **Calcul des voisins** :  
  - soit par **décalages** (`np.roll`)  
  - soit par **convolution 2D** (kernels `3×3`) via `scipy.signal/ndimage` (implémentations C performantes).
- **Interop M/L** :  
  - conversion **zéro-friction** vers PyTorch/JAX/CuPy si besoin d’accélération GPU/TPU.  
  - possible de **brancher un modèle** (détection de motifs, policy RL, etc.) directement sur la matrice `A`.

**Affichage natif** facile :  
- `matplotlib.pyplot.imshow(A)` (contrôle colormap/axes)  
- `streamlit.image(A)` (plus **rapide** que `pyplot` pour des frames successives)  
- `plotly.imshow(A)` (si besoin d’interactions)

> Recommandation : pour l’animation **haute fréquence**, privilégier `st.image` avec un **placeholder** (`ph = st.empty()`), puis `ph.image(...)`.

---

## 2) Pourquoi **Streamlit** pour la présentation

- **Simplicité** : mise en page en quelques lignes, widgets intégrés (sliders/vitesses, checkboxes, select).  
- **Expérience** : déploiement facile (cloud ou serveur perso), partage via URL, support de cache/ressources (`st.cache_resource`).  
- **Organisation** : séparation claire **UI** / **moteur de simulation** (session state + ressources partagées).

**Patrons de code utiles** :
- `st.sidebar` pour les **paramètres** (taille de la grille, densité initiale, vitesse, règles custom).  
- `st.tabs` pour séparer **Visualisation** / **Stats** / **Présets**.  
- `st.empty()` pour un **conteneur** d’image mis à jour à chaque frame.

---

## 3) Web-streaming pour l’animation (sans `streamlit.rerun()`)

### Problème
- Un loop du style `while running: step(); st.image(...); time.sleep(...)` **réexécute le script** à chaque itération, déclenchant (ou forçant) des reruns.  
- Les reruns répétés **coûtent cher** (re-render complet) et **augmentent la latence** côté utilisateur.

### Solution retenue : **pousser des frames** jusqu’au client
- Utiliser un **webstream** vidéo afin d’**envoyer** l’image produite à chaque step **sans rerun global** de la page.  
- Deux approches viables :
  1. **`streamlit-webrtc`** : WebRTC en **SENDONLY** (le serveur envoie les frames).  
     - Très adapté pour du **flux continu** (30–60 fps selon charge).  
     - Latence faible, gestion réseau robuste (NAT, etc.). **[CHOIX RETENU]**
  2. **WebSocket** (ex. FastAPI + `websockets`) : pousser des tableaux encodés en PNG/JSON-B64 vers un canvas côté navigateur.  
     - Plus “bas niveau”, nécessite un **front JS** dédié.

> **Choix par défaut** : `streamlit-webrtc` (simple, efficace, intégration Streamlit).

---

## 4) Architecture logique (vue d’ensemble)

```
┌──────────────────┐                           ┌─────────────────────────┐
│ Navigateur       │      contrôle (UI)        │ Streamlit               │
│ (Streamlit UI)   │◄──────────────────────────│ Widgets / Mise en page  │
└───────▲──────────┘                           └───────────┬─────────────┘
        │ frames vidéo WebRTC                              │
        │ (poussées)                                       │
        │                                                  ▼
┌───────┴───────────┐                              ┌──────────────────────────┐
│ WebRTC Player     │   calc étape (NumPy/Scipy)   │ Moteur de simulation     │
│ (streamlit-webrtc)│◄─────────────────────────────│ (état A, step(), kernel) │
└───────────────────┘                              └──────────────────────────┘
```

 - Le **moteur** fait évoluer `A` (NumPy + convolution).  
 - Le **producteur de frames** convertit `A` en image (grayscale ou palette).  
 - Le **player WebRTC** pousse ces frames au navigateur — **aucun rerun global** requis.

---

## 5) Exemples minimaux

### 5.1. Baseline (sans WebRTC) — utile pour debug & prototypage
```python
import time
import numpy as np
import streamlit as st

H, W = 512, 512
A = (np.random.rand(H, W) > 0.8).astype(np.uint8)
ph = st.empty()

while True:
    # ... step(A) avec np.roll ou convolution (remplacez par votre fonction)
    # A = step_conv(A)

    # Affichage rapide (grayscale, valeurs 0–255)
    ph.image(A * 255, clamp=True, channels="GRAY", use_column_width=True)
    time.sleep(0.03)  # ~30 fps
```

### 5.2. Streaming WebRTC
 Installation : ```bash pip install streamlit-webrtc av

```python
import av
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# ---- Moteur de simu (ex: convolution SciPy ou décalages NumPy) ----
def step_np_roll(A: np.ndarray) -> np.ndarray:
    N = (
        np.roll(np.roll(A,  1, 0),  1, 1) +
        np.roll(np.roll(A,  1, 0),  0, 1) +
        np.roll(np.roll(A,  1, 0), -1, 1) +
        np.roll(np.roll(A,  0, 0),  1, 1) +
        np.roll(np.roll(A,  0, 0), -1, 1) +
        np.roll(np.roll(A, -1, 0),  1, 1) +
        np.roll(np.roll(A, -1, 0),  0, 1) +
        np.roll(np.roll(A, -1, 0), -1, 1)
    )
    return ((N == 3) | ((A == 1) & (N == 2))).astype(np.uint8)

# ---- Video Processor (pousse des frames) ----
class GameOfLifeProcessor(VideoProcessorBase):
    def __init__(self):
        H, W = 512, 512
        self.A = (np.random.rand(H, W) > 0.8).astype(np.uint8)

    def recv(self, frame):
        # 1) étape de simulation
        self.A = step_np_roll(self.A)

        # 2) rendu en niveaux de gris 8 bits
        img = (self.A * 255).astype(np.uint8)

        # 3) AV frame (format 'gray' → converti en 'bgr24' si besoin)
        #    Certains navigateurs attendent du BGR/RGB :
        img_bgr = np.stack([img, img, img], axis=-1)  # (H, W, 3)
        return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")

st.set_page_config(page_title="Jeu de la Vie — WebRTC", layout="wide")
st.title("Jeu de la Vie — Streaming WebRTC")

webrtc_streamer(
    key="life",
    mode=WebRtcMode.SENDONLY,                 # on ENVOIE seulement
    video_processor_factory=GameOfLifeProcessor,
    media_stream_constraints={"video": True, "audio": False},
)
```

## 6. Performance & Qualité
- **Précalc** : réutiliser les buffers (éviter les reallocations).
- **Dtypes** : uint8 pour l’état + rendu rapide (0/255).
- **Convolution** : scipy.signal.convolve2d ou GPU (PyTorch/CuPy) selon la taille.
- **Palette** : pour un rendu plus lisible, mapper 0/1 → [[0,0,0],[255,255,255]] (ou couleurs custom).
- **Framerate** : limiter à ~30–60 fps pour éviter de saturer le CPU réseau.
- **Taille de frame** : adapter la résolution au débit (downscale pour le streaming si nécessaire).
