"""Eliminacao de Gauss com pivoteamento parcial."""

import numpy as np


def gauss(A, b):
    """Eliminacao de Gauss com pivoteamento parcial.
    Retorna (A_triangular, b_modificado).
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)

    for k in range(n - 1):
        # Pivoteamento parcial: busca maior |a_ik| abaixo da linha k
        p = np.argmax(np.abs(A[k:, k])) + k
        A[[k, p]] = A[[p, k]]
        b[[k, p]] = b[[p, k]]

        for i in range(k + 1, n):
            m = A[i, k] / A[k, k]  # multiplicador
            A[i, k:] -= m * A[k, k:]
            b[i] -= m * b[k]

    return A, b


def subst_retro(A, b):
    """Substituicao retroativa para sistema triangular superior."""
    n = len(b)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - A[i, i + 1:] @ x[i + 1:]) / A[i, i]
    return x


def resolver_gauss(A, b):
    """Pipeline completo: Gauss + substituicao retroativa."""
    Au, bu = gauss(A, b)
    return subst_retro(Au, bu)
