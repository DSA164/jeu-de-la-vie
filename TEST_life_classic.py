#---------------------------------------------------------#
#      Script de test des fonction du mode classique.     #
#---------------------------------------------------------#

import time
import os

from utils_grid import init_grid, create_grid
from utils_life_rules import evolution

grid_test = create_grid(rows=10, cols=30, game_mode='classic')
grid_test = init_grid(grid = grid_test, life_density = 0.2)

for i in range(100):
    os.system("cls" if os.name == "nt" else "clear")  # efface l'écran
    grid_test = evolution(grid=grid_test, rule='B3/S23', periodic_border=True)
    print(grid_test)
    time.sleep(0.3)