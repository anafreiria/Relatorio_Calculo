"""
main.py
Plano de Investigacao - Sistemas Lineares
Disciplina: Calculo Numerico
"""
import numpy as np
import time
import matplotlib.pyplot as plt


from cholesky import resolver_cholesky
from condicao import experimento_hilbert, perturbar_b
from gauss import gauss, resolver_gauss
from lu import fatoracao_lu, subst_prog, subst_retro
from thomas import montar_tridiagonal, thomas


print("\n============================================================")
print("SECAO 1 - ELIMINACAO DE GAUSS E PIVOTEAMENTO")
print("============================================================")

# 1.1 - Verificacao basica
# Q1.1:
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
x_esperado = np.array([-3, 5, 0], dtype=float)
residuo = np.linalg.norm(A @ x - b)

print("\nQ1.1 - Verificacao basica")
print(f"Solucao obtida: {x}")
print(f"Solucao esperada: {x_esperado}")
print(f"A solucao confere? {np.allclose(x, x_esperado)}")
print(f"Residuo ||Ax - b||2: {residuo:.2e}")


# Q1.2 - Imprimir a matriz triangular superior U
Au, bu = gauss(A, b)

print("\nQ1.2 - Matriz triangular superior U")
print("Matriz U:")
print(np.array2string(Au, precision=4, suppress_small=True))

print("\nVetor b modificado:")
print(np.array2string(bu, precision=4, suppress_small=True))

print("\nLinha trocada pelo pivoteamento: linha 1 com linha 3")

# 1.2 - Efeito do pivoteamento na precisao
# Q1.3 - Experimento com float32
def gauss_sem_pivoteamento(A, b):
    """Versao sem pivoteamento para comparar com o metodo original."""
    A = np.array(A, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    n = len(b)

    for k in range(n - 1):
        for i in range(k + 1, n):
            m = A[i, k] / A[k, k]
            A[i, k:] -= m * A[k, k:]
            b[i] -= m * b[k]

    return A, b


def gauss_com_pivoteamento_float32(A, b):
    """Versao com pivoteamento parcial usando aritmetica float32."""
    A = np.array(A, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    n = len(b)

    for k in range(n - 1):
        p = np.argmax(np.abs(A[k:, k])) + k
        A[[k, p]] = A[[p, k]]
        b[[k, p]] = b[[p, k]]

        for i in range(k + 1, n):
            m = A[i, k] / A[k, k]
            A[i, k:] -= m * A[k, k:]
            b[i] -= m * b[k]

    return A, b


def subst_retro_float32(A, b):
    """Substituicao retroativa usando float32."""
    n = len(b)
    x = np.zeros(n, dtype=np.float32)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - A[i, i + 1:] @ x[i + 1:]) / A[i, i]
    return x


A_mal = np.array(
    [
        [0.0003, 3],
        [1, 1],
    ],
    dtype=np.float32,
)
b_mal = np.array([2.0001, 1], dtype=np.float32)
x_exato = np.array([1 / 3, 2 / 3], dtype=np.float32)

Au_sem, bu_sem = gauss_sem_pivoteamento(A_mal, b_mal)
x_sem = subst_retro_float32(Au_sem, bu_sem)
erro_sem = np.abs((x_sem - x_exato) / x_exato)

Au_com, bu_com = gauss_com_pivoteamento_float32(A_mal, b_mal)
x_com = subst_retro_float32(Au_com, bu_com)
erro_com = np.abs((x_com - x_exato) / x_exato)

print("\nQ1.3 - Tabela de comparacao com float32")
print("Metodo                 x1 calculado        Erro relativo x1        Erro relativo x2")
print(f"Sem pivoteamento       {x_sem[0]:.8f}           {erro_sem[0]:.8e}          {erro_sem[1]:.8e}")
print(f"Com piv. parcial       {x_com[0]:.8f}           {erro_com[0]:.8e}          {erro_com[1]:.8e}")



# Q1.4 - Variações de coeficientes
valores_a11 = np.array([1e-1, 1e-3, 1e-6, 1e-9], dtype=np.float32)
erros_x1_sem = []
erros_x1_com = []

for a11 in valores_a11:
    A_var = np.array(
        [
            [a11, 3],
            [1, 1],
        ],
        dtype=np.float32,
    )

    # Manteremos a solucao exata (1/3, 2/3)^T para todos os valores de a11.
    b_var = A_var @ x_exato

    Au_sem, bu_sem = gauss_sem_pivoteamento(A_var, b_var)
    x_sem_var = subst_retro_float32(Au_sem, bu_sem)
    erro_x1_sem = abs((x_sem_var[0] - x_exato[0]) / x_exato[0])
    erros_x1_sem.append(erro_x1_sem)

    Au_com, bu_com = gauss_com_pivoteamento_float32(A_var, b_var)
    x_com_var = subst_retro_float32(Au_com, bu_com)
    erro_x1_com = abs((x_com_var[0] - x_exato[0]) / x_exato[0])
    erros_x1_com.append(erro_x1_com)

print("\nQ1.4 - Erro relativo em x1 variando a11")
print("a11             Erro sem pivoteamento        Erro com pivoteamento")
for a11, erro_sem_i, erro_com_i in zip(valores_a11, erros_x1_sem, erros_x1_com):
    print(f"{a11:.0e}           {erro_sem_i:.8e}              {erro_com_i:.8e}")

plt.figure()
plt.loglog(valores_a11, erros_x1_sem, marker="o", label="Sem pivoteamento")
plt.loglog(valores_a11, erros_x1_com, marker="s", label="Com piv. parcial")
plt.title("Q1.4 - Erro relativo em x1")
plt.xlabel("a11")
plt.ylabel("Erro relativo em x1")
plt.grid(True, which="both")
plt.legend()
plt.savefig("grafico_q1_4.png", dpi=150)

# 1.3 - Sistemas singulares e quase singulares
# Q1.5 - Exercicio com sistema singular
A_singular = np.array(
    [
        [1, -3, 1],
        [6, -18, 4],
        [-1, 3, -1],
    ],
    dtype=float,
)
b_singular = np.array([1, 2, 4], dtype=float)

print("\nQ1.5 - Sistema singular")
try:
    x_singular = resolver_gauss(A_singular, b_singular)
    print(f"Solucao obtida: {x_singular}")
except ValueError as erro:
    print("O metodo detectou singularidade.")
    print(f"Mensagem: {erro}")


# RAISSA
print("\n============================================================")
print("SECAO 2 - Fatoracao LU")
print("============================================================")

# Q2.1 - 
# Q2.2 - 

# TODO: Secao 4 - Thomas vs. Gauss

# TODO: Secao 5 - Custo computacional

# print("\n============================================================")
# print("SECAO 6 - CONDICIONAMENTO")
# print("============================================================")

# # TODO: Secao 6 - Condicionamento
# print("\nQ6.1 - Matriz de Hilbert")
# for n in [4, 6, 8, 10, 12]:
#     kappa, erro = experimento_hilbert(n)
#     print(f"n={n:2d} kappa={kappa:.2e} erro={erro:.2e}")

# # RAISANA
# print("\n============================================================")
# print("SECAO 7 - PageRank")
# print("============================================================")

# plt.tight_layout()
# plt.savefig("resultados.pdf", dpi=150)
# plt.show()

# # RAISANA
# print("\n============================================================")
# print("SECAO 8 - Desafio")
# print("============================================================")