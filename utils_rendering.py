#---------------------------------------------------------#
#      Moteur de rendu de jeu sous form de streaming.     #
#---------------------------------------------------------#


import numpy as np
import threading
from dataclasses import dataclass

from utils_grid import init_grid, create_grid
from utils_life_rules import evolution


# ----------- petit moteur thread-safe pour le rendu -----------
@dataclass
class GoLParams:
    rows: int = 60
    cols: int = 100
    rule: str = "B3/S23"
    periodic: bool = True
    life_density: float = 0.2
    fps: int = 10
    cell_size: int = 8  # pixels par cellule
    game_mode: str = "classic"  # on reste sur classic ici


class GoLEngine:
    def __init__(self, params: GoLParams):
        self.lock = threading.Lock()
        self.params = params
        self.running = True

        self.grid = create_grid(rows=params.rows, cols=params.cols, game_mode=params.game_mode)
        self.grid = init_grid(grid=self.grid, life_density=params.life_density)

    def reset(self):
        with self.lock:
            self.grid = create_grid(rows=self.params.rows, cols=self.params.cols, game_mode=self.params.game_mode)
            self.grid = init_grid(grid=self.grid, life_density=self.params.life_density)

    def update_params(self, **kwargs):
        with self.lock:
            for k, v in kwargs.items():
                if hasattr(self.params, k):
                    setattr(self.params, k, v)

    def toggle_running(self, value: bool):
        with self.lock:
            self.running = value

    def step_and_render_bgr(self) -> np.ndarray:
        """
        Avance d'un pas (si running=True) puis renvoie l'image BGR (np.uint8) upscalée.
        """
        with self.lock:
            if self.running:
                self.grid = evolution(
                    grid=self.grid,
                    rule=self.params.rule,
                    periodic_border=self.params.periodic
                )

            # rendu simple : binaire -> niveaux de gris 0..255
            img = (self.grid.astype(np.float32) * 255.0)
            img = img.clip(0, 255).astype(np.uint8)

            # upscale visuel par répétition de pixels (pas d'OpenCV nécessaire)
            if self.params.cell_size > 1:
                k = self.params.cell_size
                img = np.kron(img, np.ones((k, k), dtype=np.uint8))

            # option: fines lignes de grille si cell_size >= 6
            if self.params.cell_size >= 6:
                img[::self.params.cell_size, :] = np.minimum(img[::self.params.cell_size, :], 64)
                img[:, ::self.params.cell_size] = np.minimum(img[:, ::self.params.cell_size], 64)

            # passer en BGR (3 canaux)
            img_bgr = np.stack([img, img, img], axis=-1)
            return img_bgr

