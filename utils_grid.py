#-------------------------------------------------#
#      Outils de gestion de la grille de jeu.     #
#-------------------------------------------------#

import numpy as np
from enum import StrEnum
from typing import Tuple, Literal, Dict, Any, Optional, Sequence
import plotly.express as px
import plotly.graph_objects as go

GameModes = Literal['classic', 'advanced_rgb', 'advanced_float']
#ColorSelections = Literal['Veridis', 'Cividis', 'Plasma', 'Magma', 'Plotly3', 'deep', 'tempo', 'RdBu', 'RdPu', 'BuGn', 'BuPu', 'PuBuGn', 'Greens']

class ColorScale(StrEnum):
    Viridis  = "Viridis"
    Cividis  = "Cividis"
    Plasma   = "Plasma"
    RdBu     = "RdBu"
    RdPu     = "RdPu"
    Greens   = "Greens"

# pour l’UI (runtime, ordonnée)
PALETTES: tuple[ColorScale, ...] = tuple(ColorScale)


#   Initilisation de la grille de jeu
#---------------------------------------
def create_grid(rows: int = 128, cols: int = 128, game_mode: GameModes = 'classic') -> np.ndarray:
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
    
#   Fonction de'affichage de la grille (plotly)
#------------------------------------------------    
def show_grid(
    grid: np.ndarray,
    title: Optional[str] = None,
    show_axes: bool = False,
    show_cell_grid: bool = False,             # quadrillage visible (classic seulement)
    figsize: Optional[Tuple[int, int]] = None, # (width_px, height_px)
    show_colorbar: bool = True,
    color_continuous: ColorScale | str = ColorScale.Viridis,
    #color_continuous: ColorSelections = 'Veridis'
    
):
    """
    Affiche la grille avec Plotly en auto-détectant le mode si non fourni.

    - classic        : 2D, uint8/bool/int  avec valeurs {0,1} -> Heatmap binaire
    - advanced_float : 2D, float in [0,1]  -> imshow continu (Viridis)
    - advanced_rgb   : 3D, (H,W,3) uint8   -> Image RGB

    Retourne: plotly.graph_objects.Figure
    """
 
    mode = detect_game_mode(grid)  # suppose la fonction fournie précédemment

    # Dimensions / taille par défaut
    if grid.ndim == 2:
        rows, cols = grid.shape
    elif grid.ndim == 3 and grid.shape[2] == 3:
        rows, cols = grid.shape[:2]
    else:
        raise ValueError("Shape incompatible, attendu (H,W) ou (H,W,3).")

    if figsize is None:
        px_per_cell = 8
        width = min(max(cols * px_per_cell, 320), 1600)
        height = min(max(rows * px_per_cell, 240), 1200)
    else:
        width, height = figsize

    # --- Rendu selon mode ---
    if mode == "advanced_rgb":
        if grid.dtype != np.uint8:
            raise TypeError("advanced_rgb attend dtype uint8 (0..255).")
        fig = go.Figure(go.Image(z=grid))
        show_colorbar = False

    elif mode == "advanced_float":
        fig = px.imshow(
            grid,
            color_continuous_scale=color_continuous,  # choix parmis la palettes de dégradé définis dans 'ColorSelections'
            zmin=0.0, zmax=1.0,
            origin="upper",
        )
        fig.update_traces(zsmooth=False)

    elif mode == "classic":
        z = grid.astype(np.uint8)
        colorscale = [[0.0, "#ffffff"], [1.0, "#111111"]]
        fig = go.Figure(go.Heatmap(
            z=z,
            colorscale=colorscale,
            zmin=0, zmax=1,
            showscale=False,
            zsmooth=False,
            xgap=1 if show_cell_grid else 0,
            ygap=1 if show_cell_grid else 0,
            hovertemplate="(row=%{y}, col=%{x}) → %{z}<extra></extra>",
        ))
        show_colorbar = False

    else:
        raise ValueError(f"Mode inconnu: {mode}")

    # Axes / layout
    fig.update_yaxes(autorange="reversed")
    if not show_axes:
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)

    fig.update_layout(
        title=title or "",
        width=width, height=height,
        margin=dict(l=10, r=10, t=40 if title else 10, b=10),
        coloraxis_showscale=bool(show_colorbar),
    )
    return fig


#   Fonction de selection de la palette de couleur (advanced)
#--------------------------------------------------------------
def add_palette_selector(
    fig: go.Figure,
    palettes: Sequence[ColorScale] = PALETTES,
    *,
    initial: Optional[ColorScale] = None,
    label: str = "Palette",
) -> go.Figure:
    if initial is not None:
        fig.update_layout(coloraxis=dict(colorscale=initial.value))

    buttons = [
        dict(
            label=pal.value,
            method="relayout",
            args=[{"coloraxis.colorscale": pal.value}],
        )
        for pal in palettes
    ]

    fig.update_layout(
        updatemenus=[dict(
            type="buttons", direction="left",
            x=0.5, xanchor="center", y=1.08, yanchor="bottom",
            buttons=buttons, showactive=True,
            active=(palettes.index(initial) if (initial and initial in palettes) else 0),
        )],
        annotations=[dict(text=label, x=0.02, xanchor="left", y=1.12, yanchor="bottom", showarrow=False)]
        + list(fig.layout.annotations or ()),
    )
    return fig


if __name__ == "__main__":
    interline = "\n------------------------\n"
    
    # TEST DE L'INITIALISATION DES GRILLES
    grid_classic = create_grid(4, 4, "classic")
    print("Grille 'classic':\n", grid_classic)
    print(f"Detection du mode: {detect_game_mode(grid_classic)}")
    print(interline)

    grid_rgb = create_grid(4, 4, "advanced_rgb")
    print("Grille 'advanced_rgb':\n", grid_rgb)
    print(f"Detection du mode: {detect_game_mode(grid_rgb)}")
    print(interline)

    grid_float = create_grid(4, 4, "advanced_float")
    print("Grille 'advanced_float':\n", grid_float)
    print(f"Detection du mode: {detect_game_mode(grid_float)}")
    print(interline)

    
    # TEST DE L'AFFICHAGE DES GRILLES


    # Advanced RGB : chaque cellule = couleur RGB aléatoire
    g3 = np.random.randint(0, 256, size=(40, 40, 3), dtype=np.uint8)
    show_grid(g3, title="RGB aléatoire").show()

    # --- Gallerie unifiée des formes (bell_one / bell_two) ---
    import os
    from plotly.subplots import make_subplots

    forms_dir = os.path.join("Docs", "Forms")
    path_one = os.path.join(forms_dir, "bell_one.csv")
    path_two = os.path.join(forms_dir, "bell_two.csv")

    # Chargement (advanced: float 0..1)
    g1 = np.loadtxt(path_one, delimiter=",", dtype=np.float32)
    g2 = np.loadtxt(path_two, delimiter=",", dtype=np.float32)

    # Version classic via seuil 0.5
    b1 = (g1 > 0.5).astype(np.uint8)
    b2 = (g2 > 0.5).astype(np.uint8)

    fig = make_subplots(
        rows=2, cols=2,
        horizontal_spacing=0.06, vertical_spacing=0.10
    )

    # ---- Ligne 1 : Advanced (avec colorbar) ----
    fig.add_trace(
        go.Heatmap(z=g1, zmin=0, zmax=1, coloraxis="coloraxis", showscale=True),
        row=1, col=1
    )
    fig.add_trace(
        go.Heatmap(z=g2, zmin=0, zmax=1, coloraxis="coloraxis2", showscale=True),
        row=1, col=2
    )

    # ---- Ligne 2 : Classic (binaire, sans colorbar) ----
    binary_colors = [[0.0, "#ffffff"], [1.0, "#111111"]]
    fig.add_trace(
        go.Heatmap(z=b1, zmin=0, zmax=1, colorscale=binary_colors, showscale=False, xgap=1, ygap=1),
        row=2, col=1
    )
    fig.add_trace(
        go.Heatmap(z=b2, zmin=0, zmax=1, colorscale=binary_colors, showscale=False, xgap=1, ygap=1),
        row=2, col=2
    )

    # Axes invisibles + origine en haut à gauche
    for i in range(1, 5):
        fig.layout[f"xaxis{i}"].visible = False
        fig.layout[f"yaxis{i}"].visible = False
        fig.layout[f"yaxis{i}"].autorange = "reversed"

    # Palettes initiales (advanced)
    fig.update_layout(
        coloraxis=dict(colorscale=ColorScale.Viridis.value),
        coloraxis2=dict(colorscale=ColorScale.Cividis.value),
        width=1100, height=900,
        margin=dict(l=120, r=200, t=100, b=40),  # marges pour titres à gauche et menu à droite
        title="Bell forms — Gallery"
    )

    # ===== En-têtes colonnes (Jelly 1 / Jell 2) =====
    # Centres de domaines X des 2 colonnes (on prend ceux de la ligne du haut)
    xc1 = sum(fig.layout.xaxis.domain) / 2.0      # col 1
    xc2 = sum(fig.layout.xaxis2.domain) / 2.0     # col 2
    fig.add_annotation(text="<b>Jelly&nbsp;1</b>", x=xc1, y=1.02, xref="paper", yref="paper",
                    showarrow=False, yanchor="bottom")
    fig.add_annotation(text="<b>Jelly&nbsp;2</b>", x=xc2, y=1.02, xref="paper", yref="paper",
                    showarrow=False, yanchor="bottom")

    # ===== En-têtes lignes (Advanced / Classic) =====
    # Centres de domaines Y des 2 lignes (on prend ceux de la colonne de gauche)
    yr1 = sum(fig.layout.yaxis.domain) / 2.0      # row 1
    yr2 = sum(fig.layout.yaxis3.domain) / 2.0     # row 2
    fig.add_annotation(text="<b>Advanced</b>", x=0, y=yr1, xref="paper", yref="paper",
                    xanchor="right", showarrow=False, xshift=-10)
    fig.add_annotation(text="<b>Classic</b>",  x=0, y=yr2, xref="paper", yref="paper",
                    xanchor="right", showarrow=False, xshift=-10)

    # ===== Menu palettes à droite (vertical) =====
    buttons = [
        dict(
            label=pal.value,
            method="relayout",
            args=[{
                "coloraxis.colorscale": pal.value,
                "coloraxis2.colorscale": pal.value
            }],
        ) for pal in PALETTES
    ]
    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            direction="down",           # liste verticale
            x=1.1, xanchor="left",     # à droite
            y=1.0,  yanchor="top",
            buttons=buttons,
            showactive=True,
            pad={"r": 8, "t": 6, "b": 6}
        )],
        annotations=(
            list(fig.layout.annotations or []) + [
                dict(text="Palette", x=1.1, xanchor="left",
                    y=1.0, yanchor="bottom", showarrow=False)
            ]
        )
    )

    fig.show()
