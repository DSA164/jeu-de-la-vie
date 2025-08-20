#-------------------------------------------------#
#      Outils de gestion de la grille de jeu.     #
#-------------------------------------------------#

import numpy as np
from typing import Tuple, Literal

GameModes = Literal['classic', 'advanced_rgb', 'advanced_float']

def init_grid(rows: int = 128, cols: int = 128, game_mode: GameModes = 'classic') -> np.ndarray:
    """
    Initialise une grille selon le mode de jeu.

    - classic        : grille 2D uint8, valeurs {0,1}
    - advanced_rgb   : grille 3D uint8, shape (rows, cols, 3), canaux 0..255
    - advanced_float : grille 2D float32, valeurs continues [0,1] (initialisées à 0)

    Returns
    -------
    np.ndarray
    """
    if rows <= 0 or cols <= 0:
        raise ValueError("rows et cols doivent être > 0")
    elif game_mode == "classic":
        return np.zeros((rows, cols), dtype=np.uint8)

    elif game_mode == "advanced_rgb":
        return np.zeros((rows, cols, 3), dtype=np.uint8)

    elif game_mode == "advanced_float":
        return np.zeros((rows, cols), dtype=np.float32)
    else:
        raise ValueError(f"Mode inconnu: {game_mode!r}. "
                        "Choisir parmi {'classic', 'advanced_rgb', 'advanced_float'}.")



if __name__ == "__main__":
    grid_classic = init_grid(4, 4, "classic")
    print("Grille 'classic':\n", grid_classic)
    print("------------------------")

    grid_rgb = init_grid(4, 4, "advanced_rgb")
    print("Grille 'advanced_rgb':\n", grid_rgb)
    print("------------------------")

    grid_float = init_grid(4, 4, "advanced_float")
    print("Grille 'advanced_float':\n", grid_float)
    
    wrong_grid = init_grid(4, 4, "avancé") 