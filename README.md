# 🧬 Jeu de la Vie – Optimisation & IA

**Projet pédagogique** autour du célèbre *Jeu de la Vie* de John Conway, conçu pour explorer :  
- l’**optimisation matricielle** (NumPy, calcul vectorisé, GPU, etc.)  
- l’**application d’algorithmes d’intelligence artificielle** sur des systèmes dynamiques  

---

## 🚀 Objectifs

- Fournir une implémentation claire et modulaire du Jeu de la Vie.  
- Expérimenter différentes stratégies d’**optimisation matricielle**.  
- Explorer l’utilisation de techniques issues de l’**IA / ML** (ex. détection de motifs, apprentissage de règles, optimisation de performance).  
- Servir de **support pédagogique** pour étudiants, chercheurs et passionnés.  

## Documentation

### [Optimisation matricielle pour le Jeu de la Vie](./optimisation_matricielle.md)
Ajoute un document expliquant :
- Représentation en **matrice binaire** (`uint8`/`bool`) et gestion des **bords périodiques** (*wrap*).
- Calcul des voisins **vectorisé** avec `numpy.roll` (sans boucle Python).
- **Convolution 2D** : `scipy.signal.convolve2d` et `scipy.ndimage.convolve` (implémentations C rapides).
- **FFT** (`scipy.signal.fftconvolve`) pour **très grandes grilles**.
- **Accélération GPU** : PyTorch/JAX/CuPy (conv2d + padding circulaire) pour un parallélisme massif.

Contenu : exemples de fonctions `step(...)` (np.roll, SciPy, ndimage, FFT, PyTorch/CuPy), bonnes pratiques perf (dtype, réutilisation de buffers), mini-benchmark de cohérence/temps, et tableau comparatif des approches.


### [Optimisation pour l'affichage en faible latense sur une page web](./optimisation_web.md)
Ajoute un document expliquant:
- NumPy pour la grille et le calcul (vectorisation, compat M/L)
- Streamlit pour la présentation et les contrôles
- WebRTC (streamlit-webrtc) pour pousser des frames sans streamlit.rerun()

Contenu: architecture, recommandations perf (dtype, buffer reuse, fps), exemples minimaux

---

## 📂 Structure du projet

- A définir...

---

## ⚙️ Installation

Cloner le dépôt et installer les dépendances :

```bash
git clone https://github.com/ton-compte/jeu-de-la-vie.git
cd jeu-de-la-vie

# Initialiser l'environnement local en Python 3.13 et installer les deps
uv venv -p 3.13
uv sync

# (macOS) si vous utilisez WebRTC : installez ffmpeg
brew install ffmpeg

# Lancer l'app
uv run streamlit run Life.py
```

---

## 🧪 Applications prévues

 - Comparaison des performances entre différentes approches (boucles Python vs NumPy vs Torch).
 - Utilisation de modèles ML pour prédire l’évolution de la grille.
 - Génération de patterns optimisés via algorithmes génétiques ou renforcement.
 - Visualisation interactive des états.

---

## 📜 Licence
Distribué sous licence Apache 2.0.
Voir le fichier [LICENSE](./LICENSE) pour plus de détails.

---

🤝 Contribution
Les contributions sont les bienvenues !
Forkez le repo
Créez une branche (git checkout -b feature/ma-fonctionnalite)
Proposez une PR
