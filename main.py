"""
main.py
Plano de Investigacao - Sistemas Lineares
Disciplina: Calculo Numerico
"""

import time

import matplotlib.pyplot as plt
import numpy as np

from cholesky import resolver_cholesky
from condicao import experimento_hilbert, perturbar_b
from gauss import resolver_gauss
from lu import fatoracao_lu, subst_prog, subst_retro
from thomas import montar_tridiagonal, thomas


# Secao 1: Verificacao basica
A = np.array(
    [
        [3, 2, 4],
        [1, 1, 2],
        [4, 3, -2],
    ],
    dtype=float,
)
b = np.array([1, 2, 3], dtype=float)

x = resolver_gauss(A, b)
print(f"Secao 1 - Solucao: {x}")
print(f"Secao 1 - Residuo: {np.linalg.norm(A @ x - b):.2e}")

# TODO: Secao 2 - Efeito do pivoteamento
# (implemente aqui o experimento com float32)

# TODO: Secao 3 - Fatoracao LU, multiplos b

# TODO: Secao 4 - Thomas vs. Gauss

# TODO: Secao 5 - Custo computacional

# TODO: Secao 6 - Condicionamento
for n in [4, 6, 8, 10, 12]:
    kappa, erro = experimento_hilbert(n)
    print(f"n={n:2d} kappa={kappa:.2e} erro={erro:.2e}")

# TODO: Secao 7 - PageRank

plt.tight_layout()
plt.savefig("resultados.pdf", dpi=150)
plt.show()
