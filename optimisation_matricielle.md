🚀 Calcul matriciel optimisé pour le Jeu de la vie
Le Jeu de la vie de Conway peut être exprimé en calcul matriciel pour accélérer drastiquement son exécution.

Voici les différentes approches possibles, du simple numpy au GPU.
1. 🎲 Représentation matricielle
On représente la grille comme une matrice binaire A (numpy array, bool/int8) :
1 → cellule vivante
0 → cellule morte
2. ➕ Calcul vectorisé des voisins (numpy.roll)
Au lieu de boucler cellule par cellule, on additionne les versions décalées de la matrice :
import numpy as np

def step(A):
    # Somme des voisins avec décalages
    N = (
        np.roll(np.roll(A,  1, 0),  1, 1) +  # haut gauche
        np.roll(np.roll(A,  1, 0),  0, 1) +  # haut
        np.roll(np.roll(A,  1, 0), -1, 1) +  # haut droit
        np.roll(np.roll(A,  0, 0),  1, 1) +  # gauche
        np.roll(np.roll(A,  0, 0), -1, 1) +  # droite
        np.roll(np.roll(A, -1, 0),  1, 1) +  # bas gauche
        np.roll(np.roll(A, -1, 0),  0, 1) +  # bas
        np.roll(np.roll(A, -1, 0), -1, 1)    # bas droit
    )

    # Règles du jeu (matriciel)
    return ((N == 3) | ((A == 1) & (N == 2))).astype(np.uint8)
⚡ Avantage : calcul vectorisé sans boucle explicite.

3. 🎛️ Optimisation par convolution
On peut utiliser une convolution 2D avec un noyau 3x3 rempli de 1, sauf au centre :
from scipy.signal import convolve2d

KERNEL = np.array([[1,1,1],
                   [1,0,1],
                   [1,1,1]])

def step_conv(A):
    N = convolve2d(A, KERNEL, mode="same", boundary="wrap")
    return ((N == 3) | ((A == 1) & (N == 2))).astype(np.uint8)
✔ scipy.signal.convolve2d est implémenté en C → beaucoup plus rapide que les décalages.
✔ Possibilité d’utiliser la FFT pour de grandes grilles.
4. ⚡ GPU & parallélisation
CuPy : API compatible numpy → on remplace np par cupy.
PyTorch / JAX : on peut faire la même convolution en tensor GPU et exploiter le parallélisme massif.
Pour des grilles énormes : calcul distribué avec MPI ou Dask.
5. 📊 Résumé des approches
Approche	Avantages	Limites
Boucle Python	Simple	Très lent
Décalages + np.roll	Full vectorisé, rapide	Plus lent que convolution sur grandes grilles
Convolution (scipy.signal)	Super rapide, natif C	Dépend d’une lib externe
GPU (cupy, torch, jax)	Ultra rapide sur grilles énormes	Nécessite GPU

