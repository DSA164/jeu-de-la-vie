---
# 🚀 Calcul matriciel optimisé pour le **Jeu de la vie**

Le **Jeu de la vie de Conway** peut être exprimé en **calcul matriciel** pour accélérer drastiquement son exécution.  
Voici les différentes approches possibles, du simple **NumPy** au **GPU**.

---

## 1. 🎲 Représentation matricielle
On représente la grille comme une **matrice binaire** `A` (numpy array, bool/int8) :  
- `1` → cellule vivante  
- `0` → cellule morte

> ℹ️ Les exemples ci-dessous utilisent des **bords périodiques** (“tore”).  

---

## 2. ➕ Calcul vectorisé des voisins (`numpy.roll`)
Au lieu de boucler cellule par cellule, on additionne les **versions décalées** de la matrice :

```python
import numpy as np

def step(A: np.ndarray) -> np.ndarray:
    # Somme des voisins via décalages (bords périodiques)
    N = (
        np.roll(np.roll(A,  1, axis=0),  1, axis=1) +  # haut-gauche
        np.roll(np.roll(A,  1, axis=0),  0, axis=1) +  # haut
        np.roll(np.roll(A,  1, axis=0), -1, axis=1) +  # haut-droit
        np.roll(np.roll(A,  0, axis=0),  1, axis=1) +  # gauche
        np.roll(np.roll(A,  0, axis=0), -1, axis=1) +  # droite
        np.roll(np.roll(A, -1, axis=0),  1, axis=1) +  # bas-gauche
        np.roll(np.roll(A, -1, axis=0),  0, axis=1) +  # bas
        np.roll(np.roll(A, -1, axis=0), -1, axis=1)    # bas-droit
    )

    # Règles de Conway (matriciel)
    return ((N == 3) | ((A == 1) & (N == 2))).astype(np.uint8)
