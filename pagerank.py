"""Projeto integrador: PageRank numerico."""

import time

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from gauss import resolver_gauss
from lu import fatoracao_lu, subst_prog, subst_retro


def matriz_transicao(G):
    """Constroi P normalizando as colunas da matriz de adjacencia G.

    A entrada G[i, j] = 1 indica um link da pagina j para a pagina i.
    Colunas sem links de saida sao tratadas como links uniformes para todas
    as paginas, evitando colunas nulas.
    """
    G = np.array(G, dtype=float)
    n = G.shape[0]
    P = np.zeros_like(G, dtype=float)
    soma_colunas = G.sum(axis=0)

    for j in range(n):
        if soma_colunas[j] == 0:
            P[:, j] = 1.0 / n
        else:
            P[:, j] = G[:, j] / soma_colunas[j]

    return P


def sistema_pagerank(G, alpha=0.85):
    """Monta (I - alpha P.T) pi = (1-alpha)/n * e."""
    P = matriz_transicao(G)
    n = P.shape[0]
    A = np.eye(n) - alpha * P.T
    b = np.full(n, (1.0 - alpha) / n)
    return A, b, P


def resolver_lu_pagerank(A, b):
    """Resolve o sistema de PageRank usando a LU implementada no trabalho."""
    L, U = fatoracao_lu(A)
    y = subst_prog(L, b)
    return subst_retro(U, y)


def normalizar_rank(pi):
    """Normaliza o vetor de PageRank para soma 1."""
    pi = np.array(pi, dtype=float)
    soma = pi.sum()
    if soma != 0:
        pi = pi / soma
    return pi


def pagerank_gauss(G, alpha=0.85):
    """Calcula PageRank resolvendo o sistema com resolver_gauss."""
    A, b, P = sistema_pagerank(G, alpha)
    pi = normalizar_rank(resolver_gauss(A, b))
    return pi, A, b, P


def pagerank_lu(G, alpha=0.85):
    """Calcula PageRank resolvendo o sistema com a LU do trabalho."""
    A, b, P = sistema_pagerank(G, alpha)
    pi = normalizar_rank(resolver_lu_pagerank(A, b))
    return pi, A, b, P


def grafo_aleatorio(n=20, p=0.3, seed=0):
    """Gera grafo aleatorio com probabilidade de aresta p."""
    rng = np.random.default_rng(seed)
    G = (rng.random((n, n)) < p).astype(float)
    np.fill_diagonal(G, 0.0)
    return G


def comparar_metodos(G, alpha=0.85):
    """Compara Gauss, LU propria e scipy.linalg.lu_solve."""
    A, b, _ = sistema_pagerank(G, alpha)
    resultados = {}

    t0 = time.perf_counter()
    pi_gauss = normalizar_rank(resolver_gauss(A, b))
    tempo = time.perf_counter() - t0
    resultados["Gauss"] = {
        "tempo": tempo,
        "rank": pi_gauss,
        "residuo": np.linalg.norm(A @ pi_gauss - b),
    }

    t0 = time.perf_counter()
    pi_lu = normalizar_rank(resolver_lu_pagerank(A, b))
    tempo = time.perf_counter() - t0
    resultados["LU propria"] = {
        "tempo": tempo,
        "rank": pi_lu,
        "residuo": np.linalg.norm(A @ pi_lu - b),
    }

    t0 = time.perf_counter()
    fator = lu_factor(A)
    pi_scipy = normalizar_rank(lu_solve(fator, b))
    tempo = time.perf_counter() - t0
    resultados["SciPy lu_solve"] = {
        "tempo": tempo,
        "rank": pi_scipy,
        "residuo": np.linalg.norm(A @ pi_scipy - b),
    }

    return resultados


def condicoes_alpha(G, alphas):
    """Calcula kappa_2(I - alpha P.T) para varios valores de alpha."""
    valores = []
    for alpha in alphas:
        A, _, _ = sistema_pagerank(G, alpha)
        valores.append(np.linalg.cond(A))
    return np.array(valores)
