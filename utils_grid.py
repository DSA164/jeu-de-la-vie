#-------------------------------------------------#
#      Outils de gestion de la grille de jeu.     #
#-------------------------------------------------#

import numpy as np
from typing import Tuple, Literal, Dict, Any

GameModes = Literal['classic', 'advanced_rgb', 'advanced_float']


#   Initilisation de la grille de jeu
#---------------------------------------
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



#   Detection du mode de jeu en fonction du type de grille
#------------------------------------------------------------
def detect_game_mode(
    grid: np.ndarray,
    *,
    float_eps: float = 1e-6,
    return_info: bool = False,
) -> GameModes | Tuple[GameModes, Dict[str, Any]]:
    """
    Détecte le game_mode d'une grille.

    Règles:
    - advanced_rgb   : shape (H, W, 3) et dtype uint8 (image RGB 0..255)
    - advanced_float : shape (H, W), dtype float*, et valeurs dans [0,1] (avec tolérance)
    - classic        : shape (H, W), dtype bool ou int*, et valeurs sous-ensemble de {0,1}

    Paramètres
    ----------
    grid : np.ndarray
    float_eps : tolérance pour la plage [0,1] en mode float
    return_info : si True, renvoie (mode, infos) avec diagnostics

    Renvoie
    -------
    GameMode ou (GameMode, infos)
    """
    if not isinstance(grid, np.ndarray):
        raise TypeError("grid doit être un np.ndarray")

    info: Dict[str, Any] = {
        "shape": grid.shape,
        "dtype": str(grid.dtype),
        "ndim": grid.ndim,
    }

    # --- Cas RGB ---
    if grid.ndim == 3 and grid.shape[2] == 3:
        if grid.dtype == np.uint8:
            mode: GameModes = "advanced_rgb"
            return (mode, info) if return_info else mode
        else:
            # RGB attendu en uint8 uniquement pour ce projet
            raise ValueError(
                f"Grille 3D avec 3 canaux détectée mais dtype {grid.dtype!s}. "
                "Le mode 'advanced_rgb' attend dtype uint8."
            )

    # --- Cas 2D ---
    if grid.ndim == 2:
        # Stats rapides
        # (np.nanmin/np.nanmax au cas où il y aurait des NaN)
        minv = float(np.nanmin(grid))
        maxv = float(np.nanmax(grid))
        info.update({"min": minv, "max": maxv})

        # Floats continus [0,1] (avec tolérance)
        if np.issubdtype(grid.dtype, np.floating):
            if minv >= -float_eps and maxv <= 1.0 + float_eps:
                mode = "advanced_float"
                return (mode, info) if return_info else mode
            else:
                raise ValueError(
                    f"Grille float 2D détectée avec valeurs hors [0,1]: min={minv:.3g}, max={maxv:.3g}."
                )

        # Bool ou ints {0,1} -> classic
        if grid.dtype == np.bool_:
            mode = "classic"
            return (mode, info) if return_info else mode

        if np.issubdtype(grid.dtype, np.integer):
            # Vérifier que les valeurs sont bien {0,1}
            # Utiliser min/max pour éviter np.unique sur de grands tableaux
            if minv >= 0 and maxv <= 1:
                mode = "classic"
                return (mode, info) if return_info else mode
            else:
                raise ValueError(
                    f"Grille int 2D détectée mais valeurs >1 (min={minv:.3g}, max={maxv:.3g}). "
                    "Ce projet attend {0,1} pour 'classic'."
                )

        # Autres dtypes non couverts
        raise ValueError(
            f"Grille 2D détectée avec dtype {grid.dtype!s} non supporté pour l’auto-détection."
        )

    # --- Autres shapes non supportées ---
    raise ValueError(
        f"Shape {grid.shape} (ndim={grid.ndim}) non supportée. "
        "Attendu: (H,W) ou (H,W,3)."
    )

if __name__ == "__main__":
    interline = "\n------------------------\n"
    grid_classic = init_grid(4, 4, "classic")
    print("Grille 'classic':\n", grid_classic)
    print(f"Detection du mode: {detect_game_mode(grid_classic)}")
    print(interline)

    grid_rgb = init_grid(4, 4, "advanced_rgb")
    print("Grille 'advanced_rgb':\n", grid_rgb)
    print(f"Detection du mode: {detect_game_mode(grid_rgb)}")
    print(interline)

    grid_float = init_grid(4, 4, "advanced_float")
    print("Grille 'advanced_float':\n", grid_float)
    print(f"Detection du mode: {detect_game_mode(grid_float)}")
    print(interline)
    
    wrong_grid = init_grid(4, 4, "avancé") 
    print(f"Detection du mode: {detect_game_mode(wrong_grid)}")
    