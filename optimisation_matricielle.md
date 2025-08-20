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
```

---

## 3. 🎛️ Optimisation par convolution

### 3.1. Variante SciPy
On utilise une convolution 2D avec un noyau 3×3 rempli de 1, sauf au centre :
```python
import numpy as np
from scipy.signal import convolve2d

KERNEL = np.array([[1, 1, 1],
                   [1, 0, 1],
                   [1, 1, 1]], dtype=np.uint8)

def step_conv(A: np.ndarray) -> np.ndarray:
    # boundary="wrap" → bords périodiques ; mode="same" → conservation de la taille
    N = convolve2d(A, KERNEL, mode="same", boundary="wrap")
    return ((N == 3) | ((A == 1) & (N == 2))).astype(np.uint8)
```
Atouts :
`scipy.signal.convolve2d` est implémenté en C → très rapide.
Peut utiliser la FFT (selon la taille) pour accélérer les grandes grilles.

### 3.2. Convolution via SciPy/ndimage (alternative)
`ndimage.convolve` est optimisée en C et gère aussi les bords périodiques.

```python
```

```python
import numpy as np
from scipy import ndimage as ndi

KERNEL = np.array([[1, 1, 1],
                   [1, 0, 1],
                   [1, 1, 1]], dtype=np.uint8)

def step_conv_ndimage(A: np.ndarray) -> np.ndarray:
    N = ndi.convolve(A, KERNEL, mode="wrap")
    return ((N == 3) | ((A == 1) & (N == 2))).astype(np.uint8)
```

### 3.3. Convolution FFT (utile sur très grandes grilles)
La convolution par FFT devient avantageuse pour des tailles importantes. Avec `scipy.signal.fftconvolve` :

```python
import numpy as np
from scipy.signal import fftconvolve

# Même kernel 3x3 (on peut rester en int, mais la FFT travaille en float)
KERNEL = np.array([[1, 1, 1],
                   [1, 0, 1],
                   [1, 1, 1]], dtype=np.float32)

def step_conv_fft(A: np.ndarray) -> np.ndarray:
    A_f = A.astype(np.float32)
    # 'same' pour conserver la taille ; pour wrap, on peut appliquer un padding circulaire manuel
    N = fftconvolve(A_f, KERNEL, mode="same")
    N = np.rint(N).astype(np.int32)  # remet en entier
    return ((N == 3) | ((A == 1) & (N == 2))).astype(np.uint8)
```


💡 Pour un véritable wrap avec FFT, on peut replier les bords (tiling) avant FFT, ou utiliser des librairies spécialisées. Sur de petites grilles, préférez `convolve2d`.

### 3.4. Convolution PyTorch (CPU/GPU)
Utilise les tensors et l’API `conv2d`. Pour un bord périodique, on applique un padding circulaire puis une conv sans padding.

```python
import numpy as np
import torch
import torch.nn.functional as F

# Sélection automatique CPU/GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Kernel 3x3 (1→vivant, 0→mort) sans le centre
KERNEL_T = torch.tensor([[1, 1, 1],
                         [1, 0, 1],
                         [1, 1, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)

@torch.no_grad()
def step_conv_torch(A_np: np.ndarray) -> np.ndarray:
    # A_np : (H, W) binaire {0,1}
    A = torch.from_numpy(A_np.astype(np.float32)).to(device).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

    # Padding circulaire de 1 pixel (gauche/droite/haut/bas)
    A_pad = F.pad(A, pad=(1, 1, 1, 1), mode="circular")

    # Convolution sans padding (déjà padée)
    N = F.conv2d(A_pad, KERNEL_T, padding=0)  # (1,1,H,W)

    # Règle de Conway (en float/bool tensor)
    alive_next = (N == 3) | ((A == 1) & (N == 2))
    return alive_next.squeeze(0).squeeze(0).to(torch.uint8).cpu().numpy()
```

✅ Remplacez device = torch.device("cuda" ...) pour forcer CPU/GPU.
Avec CUDA, cette version peut être très rapide sur de grandes grilles.

### 3.5. Convolution CuPy (GPU NVIDIA, API NumPy-like)
CuPy offre une API compatible NumPy. On peut également utiliser cupyx.scipy.signal.convolve2d.

```python
# pip install cupy-cuda12x  (selon votre version CUDA)
import cupy as cp
from cupyx.scipy.signal import convolve2d as cp_convolve2d

KERNEL_CP = cp.array([[1, 1, 1],
                      [1, 0, 1],
                      [1, 1, 1]], dtype=cp.uint8)

def step_conv_cupy(A_np: np.ndarray) -> np.ndarray:
    A = cp.array(A_np, dtype=cp.uint8)
    N = cp_convolve2d(A, KERNEL_CP, mode="same", boundary="wrap")
    next_state = ((N == 3) | ((A == 1) & (N == 2))).astype(cp.uint8)
    return cp.asnumpy(next_state)


```
