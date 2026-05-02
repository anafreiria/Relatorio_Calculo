"""Projeto integrador: PageRank numerico.

O PDF pede este arquivo na entrega, mas nao fornece um codigo-base pronto.
Use este arquivo para implementar as questoes Q7.1, Q7.2 e Q7.3.
"""

import numpy as np

from gauss import resolver_gauss
from lu import fatoracao_lu, subst_prog, subst_retro


def matriz_transicao(G):
    """TODO: construa P normalizando as colunas de G."""
    G = np.array(G, dtype=float)
    return G


def pagerank(G, alpha=0.85):
    """TODO: monte e resolva (I - alpha P.T) pi = (1-alpha)/n e."""
    P = matriz_transicao(G)
    n = P.shape[0]
    A = np.eye(n) - alpha * P.T
    b = np.full(n, (1 - alpha) / n)
    pi = resolver_gauss(A, b)
    return pi
