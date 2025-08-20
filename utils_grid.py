#-------------------------------------------------#
#      Outils de gestion de la grille de jeu.     #
#-------------------------------------------------#

import numpy as np

def grid(row: int = 128, col: int = 128, game_mode: str = 'classic'):
    # jeu classique ou cellule vivante = 1 et morte = 0
    if game_mode == 'classic':
        game_grid = np.zeros((row, col), dtype=np.uint8) 
    # jeu avancé ou cellule peu prendre des états intermédiaire codée en RGB sur 256 niveaux
    elif game_mode == 'advanced':
        game_grid = np.zeros((row, col, 3), dtype=np.uint8)
    return game_grid



if __name__ == "__main__":
    grid_classic = grid(4, 4)
    print(f"grille classique: \n {grid_classic}")
    print('------------------------')
    grid_advanced = grid(4, 4, game_mode='advanced')
    print(f"grille avancée: \n {grid_advanced}")