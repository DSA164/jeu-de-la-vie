#--------------------------------------------------------------------------#
#      Outils de spécification des règles le vie sur la grille de jeu.     #
#--------------------------------------------------------------------------#

import numpy as np
from typing import Tuple, Iterable

#      MODE CLASSIC (BINAIRE)     
#---------------------------------

# Règle de base definie empiriquement par John Horton Conway en 1970:
# B3/S23 'Birth'/'Survival' 
# => naissance d'une cellule si et seulement si 3 cellules adjacentes sont vivantes (=1)
# => survie d'une cellule vivnate si 2 ou 3 cellules adjacentes sont vivantes
# => mort de la cellule dans tous les autres cas.




# Bornes des valeurs autorisées 
ALLOWED_COUNTS: tuple[int, ...] = tuple(range(0, 9))   

# Normalisation, validation & formatage
def _normalize_counts(counts: Iterable[int]) -> tuple[int, ...]:
    """Tri + dédoublonnage + cast int."""
    return tuple(sorted({int(c) for c in counts}))


# passage d'un format 'B3/S23' à (B = (3,), S = (2,3))
def format_BS_rule_to_inter_list(BS_rule: str = 'B3/S23', ALLOWED_COUNTS=ALLOWED_COUNTS) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    #if BS_rule
    B_rule, S_rule = BS_rule.split('/')
    B_unsorted = (n for n in B_rule[1:])
    S_unsorted = (n for n in S_rule[1:])
    return _normalize_counts(B_unsorted), _normalize_counts(S_unsorted)
    
    
# passage d'un format (B = (3,), S = (2,3)) à 'B3/S23'  
def format_BS_rule_to_string(B: Iterable[int], S: Iterable[int]) -> str:
    """'B3/S23', 'B36/S23', etc."""
    Bn = _normalize_counts(B)
    Sn = _normalize_counts(S)
    return f"B{''.join(map(str, Bn))}/S{''.join(map(str, Sn))}"


def evolution(grid: np.ndarray, rule: str = 'B3/S23') -> np.ndarray:
    # regle par defaut telle que définie par John Horton Conway (B3/S23)
    B, S = format_BS_rule_to_inter_list(rule)
    # Somme des 8 voisins
    neightbors = (
        np.roll(np.roll(grid,  1, axis=0),  1, axis=1) +  # haut-gauche
        np.roll(np.roll(grid,  1, axis=0),  0, axis=1) +  # haut
        np.roll(np.roll(grid,  1, axis=0), -1, axis=1) +  # haut-droit
        np.roll(np.roll(grid,  0, axis=0),  1, axis=1) +  # gauche
        np.roll(np.roll(grid,  0, axis=0), -1, axis=1) +  # droite
        np.roll(np.roll(grid, -1, axis=0),  1, axis=1) +  # bas-gauche
        np.roll(np.roll(grid, -1, axis=0),  0, axis=1) +  # bas
        np.roll(np.roll(grid, -1, axis=0), -1, axis=1)    # bas-droit
    )
    
    # Règle B/S (test éléments par élément)
    birth = (grid == 0) & np.isin(neightbors, list(B))
    survive = (grid == 1) & np.isin(neightbors, list(S))
    
    return (birth | survive).astype(np.uint8)


#      MODE ADVANCED (CONTINU)     
#---------------------------------



if __name__ == "__main__":
    interline = "\n------------------------------------------------------------------------------------------------------------\n"
    print("Test de la fonction 'normalise' avec B = (1, 3, 2, 5, 5, 3) ==> resultat: ", _normalize_counts((1, 3, 2, 5, 5, 3)))
    print("Test de la fonction 'format_BS_rule_to_inter_list' avec B3/S23 ==> resultat: ", format_BS_rule_to_inter_list("B3/S23"))
    print("Test de la fonction 'format_BS_rule_to_inter_list' avec B3113/S423 ==> resultat: ", format_BS_rule_to_inter_list("B3113/S423"))
    print(interline)
    A = np.array([[0,0,0,0,0], [0,1,0,1,0], [0,0,0,0,0], [0,0,1,0,0], [0,0,0,0,0]])
    print("Test de la fonction 'evolution' avec B3/S23 et la grille A:")
    print(A)
    print('Resultat:')
    print(evolution(A))
    print(interline)
    B = np.array([[1,0,1], [0,0,0], [0,1,0]])
    print("Test de la fonction 'evolution' avec B3/S23 et la grille B [Verification de la périodicité des frontrières]:")
    print(B)
    print('Resultat:')
    print(evolution(B))
    
    