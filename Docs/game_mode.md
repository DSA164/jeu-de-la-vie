# 🎮 Modes de jeu – Jeu de la Vie

Ce document décrit les différents **modes de simulation** disponibles dans le projet.

---

## 🟦 Mode binaire (classique)

- Chaque cellule prend une valeur **0 (morte)** ou **1 (vivante)**.  
- Les règles suivent le *Game of Life* de John Conway :  

  - Une cellule **vivante** reste vivante si elle a **2 ou 3 voisines vivantes**, sinon elle meurt.  
  - Une cellule **morte** devient vivante si elle a **exactement 3 voisines vivantes**.  

➡️ Ce mode est adapté pour :
- la compréhension des bases,  
- l’étude de structures classiques (planeurs, oscillateurs, etc.),  
- les comparaisons de performance entre implémentations.  

---

## 🦠 Mode avancé (valeurs continues)

- Chaque cellule peut prendre une valeur flottante comprise entre **0 et 1**.  
- Interprétation possible :  
  - `0.0` = cellule morte  
  - `1.0` = cellule pleinement vivante  
  - valeurs intermédiaires = état partiellement actif (intensité, probabilité, énergie…).  

### Exemple de règles envisagées
- Une cellule met à jour sa valeur selon une **fonction de transition continue** :  
  - pondération des voisins par leur intensité,  
  - application d’une fonction sigmoïde, tanh ou seuil flou,  
  - possibilité d’introduire du **bruit** ou des **paramètres d’apprentissage**.  

### Applications
- Étude de dynamiques plus riches que Conway (graduel au lieu de binaire).  
- Utilisation comme **banc d’essai pour des algorithmes d’IA** (optimisation, apprentissage de règles, renforcement).  
- Visualisation plus expressive (couleur ou intensité lumineuse).  

---

## 🔮 Extensions possibles

Ces modes pourront évoluer vers :
- **Automates multi-états** (plusieurs niveaux discrets de vie/mort).  
- **Automates probabilistes** (transition basée sur une distribution de probabilité).  
- **Jeux hybrides** (mélange de cellules binaires et continues).  
- **Paramétrage IA** : apprentissage automatique des règles au lieu de les coder manuellement.  

---

## 📌 Résumé

| Mode         | Valeurs possibles | Exemple de règles                 | Usage principal.           |
|--------------|-------------------|-----------------------------------|----------------------------|
| **Binaire**  | 0 ou 1            | Règles de Conway                  | Base, patterns classiques  |
| **Avancé**   | [0, 1] (continu)  | Transition par fonction continue  | Recherche, IA, extensions  |
