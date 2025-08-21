#--------------------------------------------------------------------------#
#      Outils de spécification des règles le vie sur la grille de jeu.     #
#--------------------------------------------------------------------------#

import numpy as np
from typing import Tuple, Iterable
import re

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

# véfifie le format de la string BSrule ex: 'B3/S23' -> True, 'B3_S23' -> False,
def format_BS_rule_match(BS_rule: str = 'B3/S23') -> Tuple[bool, str]:
    if not re.match(r'^B[0-8]*\/S[0-8]*$', BS_rule):
        return False, "La règle doit être au format 'Bxxxx/Syyyy', où xxxx et yyyy sont des chiffres de 0 à 8"
    else:
        return True, ""

# passage d'un format 'B3/S23' à (B = (3,), S = (2,3))
def format_BS_rule_to_inter_list(BS_rule: str = 'B3/S23', ALLOWED_COUNTS=ALLOWED_COUNTS) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    _format, msg = format_BS_rule_match(BS_rule)
    if not _format:
        print(msg)
        return None, None
    
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


def evolution(grid: np.ndarray, rule: str = 'B3/S23', periodic_border:bool = True) -> np.ndarray:
    """
    Jeu de la Vie (Conway) avec bords périodiques (torus) ou non (extérieur mort).
    - grid : array 2D binaire (0/1)
    - rule : 'B.../S...' (ex: 'B3/S23')
    - periodic_border : True -> wrap (torique) ; False -> padding 0 (limité)
    """
    # regle par defaut telle que définie par John Horton Conway (B3/S23)
    B, S = format_BS_rule_to_inter_list(rule)
    
    # gp = grid padded (H+2, W+2)
    if periodic_border:
        # bords périodiques (configuration torique)
        gp = np.pad(grid, ((1, 1), (1, 1)), mode='wrap')  # (H+2, W+2)
    else:
        # bords limitants : padding de 0 tout autour
        gp = np.pad(grid, ((1, 1), (1, 1)), mode='constant', constant_values=0)

    # Somme des 8 voisins (H, W)
    neighbors = (
        gp[:-2, :-2] + gp[:-2, 1:-1] + gp[:-2, 2:] +   # ligne du haut
        gp[1:-1, :-2]               + gp[1:-1, 2:] +   # même ligne (gauche/droite)
        gp[2:,   :-2] + gp[2:, 1:-1] + gp[2:,   2:]    # ligne du bas
    )
        
    # Règle B/S (test éléments par élément)
    birth = (grid == 0) & np.isin(neighbors, list(B))
    survive = (grid == 1) & np.isin(neighbors, list(S))
    
    return (birth | survive).astype(np.uint8)


#      MODE ADVANCED (CONTINU)     
#---------------------------------



#      TEST DES FONCTIONS     
#-----------------------------
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
    
    print("Test de la fonction 'evolution' avec B3/S23 et la grille B avec la périodicité des frontières déactivée:")
    print(B)
    print('Resultat:')
    print(evolution(B, periodic_border=False))
    
    print(interline)
    print("Test erreur dans le format de la rule ex : E34B12")
    format_BS_rule_to_inter_list('E34B12')
    
    