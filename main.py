"""
Alunas: Ana Flávia Freiria Rodrigues e Raissa Nunes Peret
Plano de Investigacao - Sistemas Lineares
Disciplina: Cálculo Numérico
"""
import numpy as np
import time
import matplotlib.pyplot as plt
from math import sin
import matplotlib
matplotlib.use("Agg")
from scipy.linalg import hilbert
from scipy.sparse import diags, eye as sparse_eye, kron
from scipy.sparse.linalg import spsolve

from cholesky import cholesky, resolver_cholesky
from condicao import experimento_hilbert, perturbar_b
from gauss import gauss, resolver_gauss
from lu import fatoracao_lu, subst_prog, subst_retro
from pagerank import (
    comparar_metodos,
    condicoes_alpha,
    grafo_aleatorio,
    matriz_transicao,
    pagerank_gauss,
    pagerank_lu,
    sistema_pagerank,
)
from thomas import montar_tridiagonal, thomas

import matplotlib
matplotlib.use("Agg")  

from scipy.linalg import lu as scipy_lu

# ANA
print("=" * 60)
print("SECAO 1 — Eliminação de Gauss com pivoteamento parcial")
print("=" * 60)


# # Q1.1 - Verificacao basica
# print("\n--- Q1.1: Verificacao basica ---")
# A = np.array(
#     [
#         [3, 2, 4],
#         [1, 1, 2],
#         [4, 3, -2],
#     ],
#     dtype=float,
# )
# b = np.array([1, 2, 3], dtype=float)

# x = resolver_gauss(A, b)
# x_esperado = np.array([-3, 5, 0], dtype=float)
# residuo = np.linalg.norm(A @ x - b)

# print(f"Solucao obtida: {x}")
# print(f"Solucao esperada: {x_esperado}")
# print(f"A solucao confere? {np.allclose(x, x_esperado)}")
# print(f"Residuo ||Ax - b||2: {residuo:.2e}")

# # Q1.2 - Pivoteamento parcial
# print("\n--- Q1.2: Matriz triangular superior U ---")
# Au, bu = gauss(A, b)

# print("Matriz U:")
# print(np.array2string(Au, precision=4, suppress_small=True))

# print("\nVetor b modificado:")
# print(np.array2string(bu, precision=4, suppress_small=True))

# print("\nLinha trocada pelo pivoteamento: linha 1 com linha 3")

# # Q1.3 - Comparacao com float32
# print("\n--- Q1.3: Tabela de comparacao com float32 ---")


# def gauss_sem_pivoteamento(A, b):
#     """Versao sem pivoteamento para comparar com o metodo original."""
#     A = np.array(A, dtype=np.float32)
#     b = np.array(b, dtype=np.float32)
#     n = len(b)

#     for k in range(n - 1):
#         for i in range(k + 1, n):
#             m = A[i, k] / A[k, k]
#             A[i, k:] -= m * A[k, k:]
#             b[i] -= m * b[k]

#     return A, b


# def gauss_com_pivoteamento_float32(A, b):
#     """Versao com pivoteamento parcial usando aritmetica float32."""
#     A = np.array(A, dtype=np.float32)
#     b = np.array(b, dtype=np.float32)
#     n = len(b)

#     for k in range(n - 1):
#         p = np.argmax(np.abs(A[k:, k])) + k
#         A[[k, p]] = A[[p, k]]
#         b[[k, p]] = b[[p, k]]

#         for i in range(k + 1, n):
#             m = A[i, k] / A[k, k]
#             A[i, k:] -= m * A[k, k:]
#             b[i] -= m * b[k]

#     return A, b


# def subst_retro_float32(A, b):
#     """Substituicao retroativa usando float32."""
#     n = len(b)
#     x = np.zeros(n, dtype=np.float32)
#     for i in range(n - 1, -1, -1):
#         x[i] = (b[i] - A[i, i + 1:] @ x[i + 1:]) / A[i, i]
#     return x


# A_mal = np.array(
#     [
#         [0.0003, 3],
#         [1, 1],
#     ],
#     dtype=np.float32,
# )
# b_mal = np.array([2.0001, 1], dtype=np.float32)
# x_exato = np.array([1 / 3, 2 / 3], dtype=np.float32)

# Au_sem, bu_sem = gauss_sem_pivoteamento(A_mal, b_mal)
# x_sem = subst_retro_float32(Au_sem, bu_sem)
# erro_sem = np.abs((x_sem - x_exato) / x_exato)

# Au_com, bu_com = gauss_com_pivoteamento_float32(A_mal, b_mal)
# x_com = subst_retro_float32(Au_com, bu_com)
# erro_com = np.abs((x_com - x_exato) / x_exato)

# print("Metodo                 x1 calculado        Erro relativo x1        Erro relativo x2")
# print(f"Sem pivoteamento       {x_sem[0]:.8f}           {erro_sem[0]:.8e}          {erro_sem[1]:.8e}")
# print(f"Com piv. parcial       {x_com[0]:.8f}           {erro_com[0]:.8e}          {erro_com[1]:.8e}")

# # Q1.4 - Erro relativo em x1 variando a11
# print("\n--- Q1.4: Erro relativo em x1 variando a11 ---")

# valores_a11 = np.array([1e-1, 1e-3, 1e-6, 1e-9], dtype=np.float32)
# erros_x1_sem = []
# erros_x1_com = []

# for a11 in valores_a11:
#     A_var = np.array(
#         [
#             [a11, 3],
#             [1, 1],
#         ],
#         dtype=np.float32,
#     )

#     # Manteremos a solucao exata (1/3, 2/3)^T para todos os valores de a11.
#     b_var = A_var @ x_exato

#     Au_sem, bu_sem = gauss_sem_pivoteamento(A_var, b_var)
#     x_sem_var = subst_retro_float32(Au_sem, bu_sem)
#     erro_x1_sem = abs((x_sem_var[0] - x_exato[0]) / x_exato[0])
#     erros_x1_sem.append(erro_x1_sem)

#     Au_com, bu_com = gauss_com_pivoteamento_float32(A_var, b_var)
#     x_com_var = subst_retro_float32(Au_com, bu_com)
#     erro_x1_com = abs((x_com_var[0] - x_exato[0]) / x_exato[0])
#     erros_x1_com.append(erro_x1_com)

# print("a11             Erro sem pivoteamento        Erro com pivoteamento")
# for a11, erro_sem_i, erro_com_i in zip(valores_a11, erros_x1_sem, erros_x1_com):
#     print(f"{a11:.0e}           {erro_sem_i:.8e}              {erro_com_i:.8e}")

# plt.figure()
# plt.loglog(valores_a11, erros_x1_sem, marker="o", label="Sem pivoteamento")
# plt.loglog(valores_a11, erros_x1_com, marker="s", label="Com piv. parcial")
# plt.title("Q1.4 - Erro relativo em x1")
# plt.xlabel("a11")
# plt.ylabel("Erro relativo em x1")
# plt.grid(True, which="both")
# plt.legend()
# plt.savefig("grafico_q1_4.png", dpi=150)

# # Q1.5 - Sistema singular
# print("\n--- Q1.5: Sistema singular ---")

# A_singular = np.array(
#     [
#         [1, -3, 1],
#         [6, -18, 4],
#         [-1, 3, -1],
#     ],
#     dtype=float,
# )
# b_singular = np.array([1, 2, 4], dtype=float)

# try:
#     x_singular = resolver_gauss(A_singular, b_singular)
#     print(f"Solucao obtida: {x_singular}")
# except ValueError as erro:
#     print("O metodo detectou singularidade.")
#     print(f"Mensagem: {erro}")


# RAISSA
print()
print("=" * 60)
print("SECAO 2 — Fatoracao LU (Doolittle)")
print("=" * 60)


# # Q2.1 
# print()
# print("--- Q2.1: Construcao de L e U e verificacao ---")
 
# A_lu = np.array(
#     [
#         [2.0,  1.0,  1.0],
#         [4.0, -6.0,  0.0],
#         [-2.0,  7.0,  2.0],
#     ]
# )
 
# L, U = fatoracao_lu(A_lu)
 
# print("\nMatriz L (triangular inferior):")
# print(np.array2string(L, precision=6, suppress_small=True))
 
# print("\nMatriz U (triangular superior):")
# print(np.array2string(U, precision=6, suppress_small=True))
 
# erro_fat = np.linalg.norm(L @ U - A_lu, "fro")
# print(f"\nErro de fatoracao ||LU - A||_F = {erro_fat:.2e}")
 
# # Verificacao visual da estrutura triangular
# tril_L = np.allclose(L, np.tril(L))
# triu_U = np.allclose(U, np.triu(U))
# diag_L_uns = np.allclose(np.diag(L), np.ones(L.shape[0]))
# print(f"L e triangular inferior? {tril_L}")
# print(f"Diagonal de L sao todos 1? {diag_L_uns}")
# print(f"U e triangular superior? {triu_U}")


# # Q2.2 
# print()
# print("--- Q2.2: Multiplos lados direitos com L e U reutilizados ---")
 
# b1 = np.array([1.0, 2.0, 3.0])
# b2 = np.array([0.0, 1.0, -1.0])
 
# # L e U ja calculado acima, reutiliza sem refatorar
# y1 = subst_prog(L, b1)
# x1 = subst_retro(U, y1)
 
# y2 = subst_prog(L, b2)
# x2 = subst_retro(U, y2)
 
# res1 = np.linalg.norm(A_lu @ x1 - b1)
# res2 = np.linalg.norm(A_lu @ x2 - b2)
 
# print(f"\nb1 = {b1}  =>  x1 = {x1}")
# print(f"Residuo ||A x1 - b1||_2 = {res1:.2e}")
# print(f"\nb2 = {b2}  =>  x2 = {x2}")
# print(f"Residuo ||A x2 - b2||_2 = {res2:.2e}")
 

# # Q2.3 — Vantagem com multiplos lados direitos (benchmarking)
# print()
# print("--- Q2.3: Gauss repetido vs. LU reutilizado (n=200, k=50) ---")
 
# np.random.seed(42)
# n_bench = 200
# k_bench = 50
# A_bench = np.random.randn(n_bench, n_bench)
# # Garante invertibilidade
# A_bench += n_bench * np.eye(n_bench)
# Bs_bench = [np.random.randn(n_bench) for _ in range(k_bench)]
 
# # Estrategia (a): Gauss repetido
# t0 = time.perf_counter()
# for bk in Bs_bench:
#     resolver_gauss(A_bench, bk)
# t_gauss = time.perf_counter() - t0
 
# # Estrategia (b): LU reutilizado
# t0 = time.perf_counter()
# L_bench, U_bench = fatoracao_lu(A_bench)
# for bk in Bs_bench:
#     yk = subst_prog(L_bench, bk)
#     subst_retro(U_bench, yk)
# t_lu = time.perf_counter() - t0
 
# print(f"\n{'Estrategia':<25} {'Tempo total (s)':>18} {'Por sistema (ms)':>18}")
# print("-" * 63)
# print(f"{'Gauss repetido':<25} {t_gauss:>18.4f} {t_gauss / k_bench * 1000:>18.3f}")
# print(f"{'LU reutilizado':<25} {t_lu:>18.4f} {t_lu   / k_bench * 1000:>18.3f}")
# print(f"\nFator de aceleracao: {t_gauss / t_lu:.2f}x")
 

# # Q2.4 — Fatoracao PLU com pivoteamento 
# print()
# print("--- Q2.4: Fatoracao PLU com pivoteamento ---")
 
# A_plu = np.array([[0.0, 1.0], [2.0, 3.0]])
 
# # (a) Testa fatoracao_lu sem pivoteamento
# print("\n(a) Tentando fatoracao_lu (sem pivoteamento) em A = [[0,1],[2,3]]:")
# try:
#     L_sem, U_sem = fatoracao_lu(A_plu)
#     print(f"    Resultado L:\n{L_sem}")
#     print(f"    Resultado U:\n{U_sem}")
#     print(f"    Erro ||LU-A||_F = {np.linalg.norm(L_sem @ U_sem - A_plu, 'fro'):.2e}")
#     # Verificar se U[0,0] e zero (divisao por zero iminente)
#     if abs(U_sem[0, 0]) < 1e-14:
#         print("    ATENCAO: pivo U[0,0] nulo — resultado invalido (NaN/Inf esperado)!")
#     else:
#         print("    Concluiu sem excecao (verifique se o resultado e confiavel).")
# except Exception as e:
#     print(f"    ERRO capturado: {e}")
 
# # (b) scipy.linalg.lu
# print("\n(b) Usando scipy.linalg.lu (com pivoteamento):")
# P_sp, L_sp, U_sp = scipy_lu(A_plu)
 
# print(f"\n    Matriz de permutacao P:\n{P_sp}")
# print(f"\n    Triangular inferior L:\n{L_sp}")
# print(f"\n    Triangular superior U:\n{U_sp}")
# print(f"\n    Verificacao P @ A = L @ U:")
# print(f"    P @ A =\n{P_sp @ A_plu}")
# print(f"    L @ U =\n{L_sp @ U_sp}")
# err_plu = np.linalg.norm(P_sp @ A_plu - L_sp @ U_sp, "fro")
# print(f"\n    Erro ||PA - LU||_F = {err_plu:.2e}")
 
# # Demonstracao: como resolver Ax = b com PLU
# b_plu = np.array([1.0, 2.0])
# # PA = LU  =>  Ax = b  =>  PAx = Pb  =>  LUx = Pb
# Pb = P_sp @ b_plu
# y_plu = subst_prog(L_sp, Pb)
# x_plu = subst_retro(U_sp, y_plu)
# print(f"\n    Solucao de Ax = {b_plu} via PLU: x = {x_plu}")
# print(f"    Residuo ||Ax - b||_2 = {np.linalg.norm(A_plu @ x_plu - b_plu):.2e}")
 


# ANA
print()
print("=" * 60)
print("SECAO 3 - Fatoracao de Cholesky para matrizes SPD")
print("=" * 60)

# # Q3.1 - Identificando matrizes SPD
# print()
# print("--- Q3.1: Teste de Cholesky e autovalores ---")

# matrizes_cholesky = {
#     "A1": np.array(
#         [
#             [4, 2],
#             [2, 3],
#         ],
#         dtype=float,
#     ),
#     "A2": np.array(
#         [
#             [1, 2],
#             [2, 1],
#         ],
#         dtype=float,
#     ),
#     "A3": np.array(
#         [
#             [4, 2, 2],
#             [2, 3, 0],
#             [2, 0, 3],
#         ],
#         dtype=float,
#     ),
# }

# previsoes_cholesky = {
#     "A1": "SPD, pois e simetrica e espera-se autovalores positivos.",
#     "A2": "Nao SPD, pois deve possuir autovalor negativo.",
#     "A3": "SPD, pois e simetrica e espera-se autovalores positivos.",
# }

# for nome, A_chol in matrizes_cholesky.items():
#     print("\n" + "-" * 60)
#     print(f"{nome}")
#     print("-" * 60)
#     print("Matriz A:")
#     print(np.array2string(A_chol, precision=4, suppress_small=True))
#     print(f"\nPrevisao: {previsoes_cholesky[nome]}")

#     autovalores = np.linalg.eigvalsh(A_chol)
#     print("\nAutovalores:")
#     print(np.array2string(autovalores, precision=6, suppress_small=True))

#     try:
#         L_chol = cholesky(A_chol)
#         print("\nResultado: Cholesky OK. A matriz e SPD.")
#         print("\nMatriz L:")
#         print(np.array2string(L_chol, precision=6, suppress_small=True))
#         erro_chol = np.linalg.norm(L_chol @ L_chol.T - A_chol, "fro")
#         print(f"\nErro ||L L^T - A||_F = {erro_chol:.2e}")
#     except ValueError as erro:
#         print("\nResultado: Cholesky falhou. A matriz nao e SPD.")
#         print(f"Mensagem: {erro}")

# # Q3.2 - Cholesky vs. LU: custo
# print()
# print("--- Q3.2: Cholesky vs. LU em matrizes SPD aleatorias ---")

# np.random.seed(0)
# tamanhos_cholesky = np.array([50, 100, 200, 500])
# tempos_cholesky = []
# tempos_lu_cholesky = []

# for n in tamanhos_cholesky:
#     B = np.random.randn(n, n)
#     A_spd = B.T @ B + n * np.eye(n)

#     t0 = time.perf_counter()
#     cholesky(A_spd)
#     tempo_chol = time.perf_counter() - t0
#     tempos_cholesky.append(tempo_chol)

#     t0 = time.perf_counter()
#     fatoracao_lu(A_spd)
#     tempo_lu = time.perf_counter() - t0
#     tempos_lu_cholesky.append(tempo_lu)

# print(f"\n{'n':>6} {'Cholesky (s)':>18} {'LU (s)':>18} {'Razao LU/Cholesky':>22}")
# print("-" * 70)
# for n, tempo_chol, tempo_lu in zip(tamanhos_cholesky, tempos_cholesky, tempos_lu_cholesky):
#     print(f"{n:>6d} {tempo_chol:>18.6f} {tempo_lu:>18.6f} {tempo_lu / tempo_chol:>22.2f}")

# plt.figure()
# plt.loglog(tamanhos_cholesky, tempos_cholesky, marker="o", label="Cholesky")
# plt.loglog(tamanhos_cholesky, tempos_lu_cholesky, marker="s", label="LU")
# plt.title("Q3.2 - Cholesky vs. LU")
# plt.xlabel("n")
# plt.ylabel("Tempo (s)")
# plt.grid(True, which="both")
# plt.legend()
# plt.savefig("grafico_q3_2.png", dpi=150)

# # Q3.3 - Regressao linear via equacoes normais e Cholesky
# print()
# print("--- Q3.3: Regressao linear por equacoes normais ---")

# np.random.seed(1)

# # Matriz de design X em R^{100 x 5}: coluna de uns + 4 colunas aleatorias
# X = np.column_stack(
#     [
#         np.ones(100),
#         np.random.randn(100, 4),
#     ]
# )

# beta_estrela = np.array([2, -1, 3, 0.5, -2], dtype=float)
# epsilon = 0.05 * np.random.randn(100)
# y = X @ beta_estrela + epsilon

# # Equacoes normais: (X^T X) beta = X^T y
# A_normal = X.T @ X
# b_normal = X.T @ y

# beta_cholesky, L_normal = resolver_cholesky(A_normal, b_normal)
# beta_lstsq = np.linalg.lstsq(X, y, rcond=None)[0]

# residuo_cholesky = np.linalg.norm(X @ beta_cholesky - y)
# residuo_lstsq = np.linalg.norm(X @ beta_lstsq - y)
# diferenca_betas = np.linalg.norm(beta_cholesky - beta_lstsq)

# print("\nbeta*:")
# print(np.array2string(beta_estrela, precision=6, suppress_small=True))

# print("\nBeta estimado por Cholesky:")
# print(np.array2string(beta_cholesky, precision=6, suppress_small=True))

# print("\nBeta estimado por np.linalg.lstsq:")
# print(np.array2string(beta_lstsq, precision=6, suppress_small=True))

# print(f"\nResiduo ||X beta_cholesky - y||2 = {residuo_cholesky:.6e}")
# print(f"Residuo ||X beta_lstsq - y||2    = {residuo_lstsq:.6e}")
# print(f"Diferenca ||beta_cholesky - beta_lstsq||2 = {diferenca_betas:.6e}")

# RAISSA - thomas
print()
print("=" * 60)
print("SECAO 4 - Algoritmo de Thomas para sistemas tridiagonais")
print("=" * 60)
# # Q4.1 — Sistema tridiagonal 5x5 e verificacao
# print()
# print("--- Q4.1: Sistema tridiagonal 5x5 e verificacao ---")
 
# a41 = [-1.0, -1.0, -1.0, -1.0]   # subdiagonal
# b41 = [4.0, 4.0, 4.0, 4.0, 4.0]  # diagonal principal
# c41 = [-1.0, -1.0, -1.0, -1.0]   # superdiagonal
# d41 = [1.0, 0.0, 0.0, 0.0, 1.0]  # lado direito
 
# x41 = thomas(a41, b41, c41, d41)
# A41 = montar_tridiagonal(a41, b41, c41)
# residuo41 = np.linalg.norm(A41 @ x41 - np.array(d41, dtype=float))
 
# print(f"Solucao x = {x41}")
# print(f"Residuo ||Ax - d||_2 = {residuo41:.2e}")
# print("Este sistema surge na discretizacao por diferencas finitas")
# print("da equacao de Poisson 1D: -u''(x) = f(x) com c.c. de Dirichlet.")
 
# # Q4.2 — Thomas vs. Gauss: escalonamento
# print()
# print("--- Q4.2: Thomas vs. Gauss — escalonamento ---")
# print(f"\n{'n':>7}  {'Thomas (ms)':>12}  {'Gauss (ms)':>12}  {'Razao':>8}")
# print("-" * 48)
 
# sizes_42 = [100, 500, 1000, 5000, 10000]
# t_thomas_42 = []
# t_gauss_42 = []
 
# for n in sizes_42:
#     a = np.full(n - 1, -1.0)
#     b_diag = np.full(n, 4.0)
#     c = np.full(n - 1, -1.0)
#     d = np.ones(n)
 
#     # Tempo Thomas
#     t0 = time.perf_counter()
#     thomas(a, b_diag, c, d)
#     t_th = (time.perf_counter() - t0) * 1000
#     t_thomas_42.append(t_th)
 
#     # Tempo Gauss (apenas ate n=1000, custo O(n^3) inviavel para maiores)
#     if n <= 1000:
#         A_dense = montar_tridiagonal(a, b_diag, c)
#         t0 = time.perf_counter()
#         resolver_gauss(A_dense, d.copy())
#         t_g = (time.perf_counter() - t0) * 1000
#         t_gauss_42.append(t_g)
#         razao = t_g / t_th
#         print(f"{n:>7d}  {t_th:>12.3f}  {t_g:>12.3f}  {razao:>7.1f}x")
#     else:
#         # Extrapola via O(n^3) a partir de n=1000
#         t_g_extrap = t_gauss_42[2] * (n / 1000) ** 3
#         t_gauss_42.append(t_g_extrap)
#         razao = t_g_extrap / t_th
#         print(f"{n:>7d}  {t_th:>12.3f}  {t_g_extrap:>11.0f}* {razao:>7.0f}x")
 
# print("(*) valor extrapolado via O(n^3) a partir de n=1000")
 
# # Q4.3 — Comparacao de memoria para n=10000
# print()
# print("--- Q4.3: Comparacao de memoria para n=10000 ---")
# n_mem = 10000
# MB_densa = n_mem**2 * 8 / (2**20)
# MB_tri = 3 * n_mem * 8 / (2**20)
# print(f"Matriz densa (float64):          {MB_densa:.2f} MB")
# print(f"Representacao tridiagonal:        {MB_tri:.4f} MB")
# print(f"Fator de reducao de memoria:      {MB_densa / MB_tri:.0f}x")
# print("Para n>=30000, a matriz densa ultrapassaria 7 GB -- inviavel.")
# print("O Thomas resolve sistemas com n>1.000.000 em segundos.")

# ANA
print()
print("=" * 60)
print("SECAO 5 - Custo computacional empirico")
print("=" * 60)


# def medir_tempo(funcao):
#     """Mede o tempo de execucao de uma chamada."""
#     t0 = time.perf_counter()
#     resultado = funcao()
#     tempo = time.perf_counter() - t0
#     return tempo, resultado



# # Q5.1 - Lei de escala
# print()
# print("--- Q5.1: Lei de escala do metodo de Gauss ---")

# np.random.seed(5)
# tamanhos_gauss = np.array([10, 20, 50, 100, 200, 500])
# tempos_gauss_q5 = []

# for n in tamanhos_gauss:
#     A_q5 = np.random.randn(n, n) + n * np.eye(n)
#     b_q5 = np.random.randn(n)

#     tempo, _ = medir_tempo(lambda: resolver_gauss(A_q5, b_q5))
#     tempos_gauss_q5.append(tempo)

# print(f"\n{'n':>6} {'Tempo resolver_gauss (s)':>28}")
# print("-" * 38)
# for n, tempo in zip(tamanhos_gauss, tempos_gauss_q5):
#     print(f"{n:>6d} {tempo:>28.6f}")

# coeficientes = np.polyfit(np.log(tamanhos_gauss), np.log(tempos_gauss_q5), 1)
# alpha = coeficientes[0]
# c = np.exp(coeficientes[1])
# tempos_ajuste = c * tamanhos_gauss**alpha

# print(f"\nLei ajustada: T(n) = {c:.3e} * n^{alpha:.3f}")
# print(f"Expoente alpha encontrado: {alpha:.3f}")

# plt.figure()
# plt.loglog(tamanhos_gauss, tempos_gauss_q5, marker="o", label="Tempo medido")
# plt.loglog(tamanhos_gauss, tempos_ajuste, linestyle="--", label=f"Ajuste alpha={alpha:.2f}")
# plt.title("Q5.1 - Lei de escala do metodo de Gauss")
# plt.xlabel("n")
# plt.ylabel("Tempo (s)")
# plt.grid(True, which="both")
# plt.legend()
# plt.savefig("grafico_q5_1.png", dpi=150)


# # Q5.2 - Comparacao com previsao teorica
# print()
# print("--- Q5.2: Comparacao com previsao teorica ---")

# n_ref = 200
# total_mult = n_ref**3
# produto = 1.0

# t0 = time.perf_counter()
# for i in range(total_mult):
#     produto *= 1.0000000001
# tempo_mult = time.perf_counter() - t0
# R = total_mult / tempo_mult

# tempos_teoricos = (2 * tamanhos_gauss**3 / 3) / R
# eficiencias = np.array(tempos_gauss_q5) / tempos_teoricos

# print(f"\nEstimativa de R: {R:.3e} multiplicacoes por segundo")
# print(f"Medida feita com {total_mult} multiplicacoes em loop Python")

# print(f"\n{'n':>6} {'Tmed (s)':>14} {'Tteo (s)':>14} {'Tmed/Tteo':>14}")
# print("-" * 52)
# for n, t_med, t_teo, eficiencia in zip(tamanhos_gauss, tempos_gauss_q5, tempos_teoricos, eficiencias):
#     print(f"{n:>6d} {t_med:>14.6f} {t_teo:>14.6f} {eficiencia:>14.6f}")



# # Q5.3 - Comparacao de metodos
# print()
# print("--- Q5.3: Comparacao de metodos para matriz SPD n=300 ---")

# np.random.seed(6)
# n_comp = 300
# B_comp = np.random.randn(n_comp, n_comp)
# A_comp = B_comp.T @ B_comp + n_comp * np.eye(n_comp)
# b_comp = np.random.randn(n_comp)

# tempo_gauss_comp, _ = medir_tempo(lambda: resolver_gauss(A_comp, b_comp))
# tempo_lu_comp, _ = medir_tempo(lambda: fatoracao_lu(A_comp))
# tempo_chol_comp, _ = medir_tempo(lambda: resolver_cholesky(A_comp, b_comp))
# tempo_numpy_comp, _ = medir_tempo(lambda: np.linalg.solve(A_comp, b_comp))

# flops_gauss = 2 * n_comp**3 / 3
# flops_lu = 2 * n_comp**3 / 3
# flops_chol = n_comp**3 / 3
# flops_numpy = 2 * n_comp**3 / 3

# metodos_q5 = [
#     ("Gauss com pivoteamento", tempo_gauss_comp, flops_gauss),
#     ("LU Doolittle", tempo_lu_comp, flops_lu),
#     ("Cholesky", tempo_chol_comp, flops_chol),
#     ("numpy.linalg.solve", tempo_numpy_comp, flops_numpy),
# ]

# print(f"\n{'Metodo':<24} {'Tempo (s)':>12} {'Flops teoricos':>18} {'Razao vs. Cholesky':>22}")
# print("-" * 82)
# for nome, tempo, flops in metodos_q5:
#     print(f"{nome:<24} {tempo:>12.6f} {flops:>18.3e} {tempo / tempo_chol_comp:>22.2f}")


# RAISSA
print()
print("=" * 60)
print("SECAO 6 - Condicionamento e sensibilidade à pertubacao")
print("=" * 60)


# # --- Q6.1: Matriz de Hilbert ---
# print()
# print("--- Q6.1: Matriz de Hilbert ---")
# print(f"\n{'n':>4} {'kappa':>14} {'Erro relativo':>16} {'Digitos corretos':>18}")
# print("-" * 58)
 
# ns_q61 = [4, 6, 8, 10, 12]
# kappas_q61 = []
# erros_q61 = []
 
# for n in ns_q61:
#     kappa, erro = experimento_hilbert(n)
#     digitos = max(0.0, 16 - np.log10(kappa))
#     kappas_q61.append(kappa)
#     erros_q61.append(erro)
#     print(f"{n:>4d} {kappa:>14.4e} {erro:>16.4e} {digitos:>18.2f}")
 
# print()
# print("Observacao: para n=12 os digitos corretos chegam a 0,")
# print("indicando que a solucao e completamente nao confiavel.")
# print("O resultado torna-se nao confiavel a partir de n=10.")
 

# # Grafico Q6.1
# fig, axes = plt.subplots(1, 2, figsize=(11, 4))
 
# axes[0].semilogy(ns_q61, kappas_q61, marker="o", color="#e07b54", linewidth=2)
# axes[0].set_title("Q6.1 — Numero de condicao κ₂(Hₙ)", fontsize=11)
# axes[0].set_xlabel("n")
# axes[0].set_ylabel("κ₂(Hₙ)")
# axes[0].grid(True, alpha=0.4)
 
# axes[1].semilogy(ns_q61, erros_q61, marker="s", color="#5b8fc9", linewidth=2)
# axes[1].set_title("Q6.1 — Erro relativo na solucao", fontsize=11)
# axes[1].set_xlabel("n")
# axes[1].set_ylabel("Erro relativo")
# axes[1].grid(True, alpha=0.4)
 
# plt.tight_layout()
# plt.savefig("grafico_q6_1.png", dpi=150)
# plt.close()
# print("\nGrafico salvo em grafico_q6_1.png")
 
# # --- Q6.2: Amplificacao de erros ---
# print()
# print("--- Q6.2: Amplificacao de erros ---")
 
# np.random.seed(7)
 
# # H6
# H6 = hilbert(6)
# b6 = H6 @ np.ones(6)
# amps_H6 = perturbar_b(H6, b6)
# kappa_H6 = np.linalg.cond(H6)
 
# # I6
# I6 = np.eye(6)
# b_I6 = np.ones(6)
# amps_I6 = perturbar_b(I6, b_I6)
# kappa_I6 = np.linalg.cond(I6)
 
# print(f"\nA = H6:  kappa = {kappa_H6:.4e}")
# print(f"  Amplificacao maxima:  {amps_H6.max():.4e}")
# print(f"  Amplificacao media:   {amps_H6.mean():.4e}")
# print(f"  Max <= kappa? {amps_H6.max() <= kappa_H6}")
 
# print(f"\nA = I6:  kappa = {kappa_I6:.4e}")
# print(f"  Amplificacao maxima:  {amps_I6.max():.4e}")
# print(f"  Amplificacao media:   {amps_I6.mean():.4e}")
 
# # Grafico Q6.2
# fig, axes = plt.subplots(1, 2, figsize=(11, 4))
 
# axes[0].hist(amps_H6, bins=15, color="#e07b54", edgecolor="white")
# axes[0].axvline(kappa_H6, color="red", linestyle="--",
#                 label=f"κ(H₆) = {kappa_H6:.2e}")
# axes[0].set_title("Q6.2 — Amplificacao com A = H₆", fontsize=11)
# axes[0].set_xlabel("Fator de amplificacao")
# axes[0].set_ylabel("Frequencia")
# axes[0].legend()
# axes[0].grid(True, alpha=0.3)
 
# axes[1].hist(amps_I6, bins=5, color="#5b8fc9", edgecolor="white")
# axes[1].axvline(1.0, color="red", linestyle="--", label="κ(I₆) = 1.0")
# axes[1].set_title("Q6.2 — Amplificacao com A = I₆", fontsize=11)
# axes[1].set_xlabel("Fator de amplificacao")
# axes[1].set_ylabel("Frequencia")
# axes[1].legend()
# axes[1].grid(True, alpha=0.3)
 
# plt.tight_layout()
# plt.savefig("grafico_q6_2.png", dpi=150)
# plt.close()
# print("\nGrafico salvo em grafico_q6_2.png")
 
# # --- Q6.3: Bem-condicionada vs. Mal-condicionada ---
# print()
# print("--- Q6.3: Impacto de perturbacao em A ---")
 
# np.random.seed(42)
 
# # Matriz bem-condicionada: diagonal com entradas proximas de 1
# A_bem = np.diag([1.0, 1.1, 0.9, 1.05, 0.95])
# kappa_bem = np.linalg.cond(A_bem)
 
# # Matriz mal-condicionada: Hilbert 5x5
# A_mal = hilbert(5)
# kappa_mal = np.linalg.cond(A_mal)
 
# print(f"\nA_bem (diagonal ~1):  kappa = {kappa_bem:.4f}")
# print(f"A_mal (Hilbert H5):   kappa = {kappa_mal:.4e}")
 
# eps_A = 1e-6
# np.random.seed(0)
# dA = eps_A * np.random.randn(5, 5)
 
# x_exato_bem = np.ones(5)
# b_bem = A_bem @ x_exato_bem
# x_bem_pert = np.linalg.solve(A_bem + dA, b_bem)
# erro_bem = np.linalg.norm(x_bem_pert - x_exato_bem) / np.linalg.norm(x_exato_bem)
 
# x_exato_mal = np.ones(5)
# b_mal = A_mal @ x_exato_mal
# x_mal_pert = np.linalg.solve(A_mal + dA, b_mal)
# erro_mal = np.linalg.norm(x_mal_pert - x_exato_mal) / np.linalg.norm(x_exato_mal)
 
# print(f"\nPerturbacao ||δA||/||A|| ≈ {eps_A:.0e}")
# print(f"A_bem: erro relativo na solucao = {erro_bem:.4e}")
# print(f"A_mal: erro relativo na solucao = {erro_mal:.4e}")
# print(f"Razao de erros (mal/bem): {erro_mal / erro_bem:.1f}x")
# print()
# print("Implicacao pratica: em A_mal uma perturbacao de 1e-6 em A")
# print(f"provoca um erro de {erro_mal:.1e} na solucao (~{erro_mal*100:.1f}%).")

# RAISSANA
print()
print("=" * 60)
print("SECAO 7 - PageRank Numerico")
print("=" * 60)

# # Q7.1 - Mini-rede de 4 paginas
# print()
# print("--- Q7.1: PageRank para mini-rede de 4 paginas ---")

# # Substitua G4 pela matriz de adjacencia fornecida na aula, se ela for diferente.
# # Convencao: G[i, j] = 1 indica link da pagina j para a pagina i.
# G4 = np.array(
#     [
#         [0, 1, 1, 0],
#         [0, 0, 1, 1],
#         [1, 0, 0, 1],
#         [0, 0, 1, 0],
#     ],
#     dtype=float,
# )

# alpha = 0.85
# P4 = matriz_transicao(G4)
# A4, b4, _ = sistema_pagerank(G4, alpha)
# pi_gauss_4, _, _, _ = pagerank_gauss(G4, alpha)
# pi_lu_4, _, _, _ = pagerank_lu(G4, alpha)

# print("\nMatriz de adjacencia G:")
# print(np.array2string(G4, precision=4, suppress_small=True))

# print("\nMatriz de transicao P:")
# print(np.array2string(P4, precision=4, suppress_small=True))

# print("\nSistema A pi = b, com A = I - 0.85 P^T:")
# print("A:")
# print(np.array2string(A4, precision=4, suppress_small=True))
# print("b:")
# print(np.array2string(b4, precision=4, suppress_small=True))

# print("\nPageRank por Gauss:")
# print(np.array2string(pi_gauss_4, precision=6, suppress_small=True))

# print("\nPageRank por LU:")
# print(np.array2string(pi_lu_4, precision=6, suppress_small=True))

# print(f"\n||pi_gauss||_1 = {np.linalg.norm(pi_gauss_4, 1):.6f}")
# print(f"||pi_lu||_1    = {np.linalg.norm(pi_lu_4, 1):.6f}")
# print(f"Diferenca ||pi_gauss - pi_lu||_2 = {np.linalg.norm(pi_gauss_4 - pi_lu_4):.6e}")

# ordem_paginas = np.argsort(-pi_gauss_4) + 1
# print(f"Ordem das paginas por rank decrescente: {ordem_paginas}")


# # Q7.2 - Grafo aleatorio com n = 20
# print()
# print("--- Q7.2: PageRank para grafo aleatorio n=20 ---")

# G20 = grafo_aleatorio(n=20, p=0.3, seed=0)
# resultados_q72 = comparar_metodos(G20, alpha=0.85)
# rank_scipy = resultados_q72["SciPy lu_solve"]["rank"]

# print(f"\n{'Metodo':<18} {'Tempo (s)':>12} {'Residuo':>14} {'Dif. vs SciPy':>16}")
# print("-" * 64)
# for nome, dados in resultados_q72.items():
#     diferenca = np.linalg.norm(dados["rank"] - rank_scipy)
#     print(f"{nome:<18} {dados['tempo']:>12.6f} {dados['residuo']:>14.6e} {diferenca:>16.6e}")

# print("\nTop 5 paginas pelo PageRank SciPy:")
# top5 = np.argsort(-rank_scipy)[:5] + 1
# print(top5)


# # Q7.3 - Condicionamento quando alpha -> 1
# print()
# print("--- Q7.3: Condicionamento para diferentes valores de alpha ---")

# alphas_q73 = np.array([0.5, 0.7, 0.85, 0.95, 0.99])
# kappas_q73 = condicoes_alpha(G20, alphas_q73)

# print(f"\n{'alpha':>8} {'kappa2(I - alpha P^T)':>28}")
# print("-" * 40)
# for alpha_i, kappa_i in zip(alphas_q73, kappas_q73):
#     print(f"{alpha_i:>8.2f} {kappa_i:>28.6e}")

# plt.figure(figsize=(6, 4))
# plt.semilogy(alphas_q73, kappas_q73, marker="o")
# plt.title("Q7.3 - Condicionamento do sistema PageRank")
# plt.xlabel("alpha")
# plt.ylabel("kappa2(I - alpha P^T)")
# plt.grid(True, which="both", alpha=0.4)
# plt.tight_layout()
# plt.savefig("grafico_q7_3.png", dpi=150)
# plt.close()
# print("\nGrafico salvo em grafico_q7_3.png")

# RAISANA
print()
print("=" * 60)
print("SECAO 8 - Desafio")
print("=" * 60)


# # Q8.1 - Analise de estabilidade em cascata
# print()
# print("--- Q8.1: Analise de estabilidade em cascata ---")

# epsilons = np.array([1e-1, 1e-3, 1e-6, 1e-9, 1e-12])
# x_exato_81 = np.array([1.0, 1.0])
# kappas_81 = []
# residuos_81 = []
# erros_81 = []

# print(f"\n{'epsilon':>12} {'kappa(Aeps)':>16} {'residuo':>14} {'erro relativo':>16}")
# print("-" * 64)
# for eps in epsilons:
#     A_eps = np.array(
#         [
#             [1.0, 1.0],
#             [1.0, 1.0 + eps],
#         ]
#     )
#     b_eps = A_eps @ x_exato_81
#     x_calc = np.linalg.solve(A_eps, b_eps)

#     kappa_eps = np.linalg.cond(A_eps)
#     residuo_eps = np.linalg.norm(A_eps @ x_calc - b_eps)
#     erro_eps = np.linalg.norm(x_calc - x_exato_81) / np.linalg.norm(x_exato_81)

#     kappas_81.append(kappa_eps)
#     residuos_81.append(residuo_eps)
#     erros_81.append(erro_eps)
#     print(f"{eps:>12.0e} {kappa_eps:>16.6e} {residuo_eps:>14.6e} {erro_eps:>16.6e}")

# plt.figure(figsize=(6, 4))
# plt.loglog(epsilons, kappas_81, marker="o", label="kappa(Aeps)")
# plt.loglog(epsilons, erros_81, marker="s", label="erro relativo")
# plt.gca().invert_xaxis()
# plt.title("Q8.1 - Estabilidade quando epsilon -> 0")
# plt.xlabel("epsilon")
# plt.ylabel("valor em escala log")
# plt.grid(True, which="both", alpha=0.4)
# plt.legend()
# plt.tight_layout()
# plt.savefig("grafico_q8_1.png", dpi=150)
# plt.close()
# print("\nGrafico salvo em grafico_q8_1.png")


# # Q8.2 - Bloco tridiagonal e EDPs 2D
# print()
# print("--- Q8.2: Bloco tridiagonal e EDPs 2D ---")

# m = 10
# n_82 = m * m
# T = diags(
#     diagonals=[-np.ones(m - 1), 4 * np.ones(m), -np.ones(m - 1)],
#     offsets=[-1, 0, 1],
#     format="csr",
# )
# I = sparse_eye(m, format="csr")
# S = diags(
#     diagonals=[-np.ones(m - 1), -np.ones(m - 1)],
#     offsets=[-1, 1],
#     shape=(m, m),
#     format="csr",
# )
# A_sparse = kron(I, T, format="csr") + kron(S, I, format="csr")
# b_82 = np.ones(n_82)

# A_dense_82 = A_sparse.toarray()
# t0 = time.perf_counter()
# x_gauss_82 = resolver_gauss(A_dense_82, b_82)
# tempo_gauss_82 = time.perf_counter() - t0
# res_gauss_82 = np.linalg.norm(A_dense_82 @ x_gauss_82 - b_82)

# t0 = time.perf_counter()
# x_sparse_82 = spsolve(A_sparse, b_82)
# tempo_sparse_82 = time.perf_counter() - t0
# res_sparse_82 = np.linalg.norm(A_sparse @ x_sparse_82 - b_82)

# mem_densa_82 = A_dense_82.nbytes / 2**20
# mem_esparsa_82 = (
#     A_sparse.data.nbytes + A_sparse.indices.nbytes + A_sparse.indptr.nbytes
# ) / 2**20

# print(f"\nGrade m x m: {m} x {m}")
# print(f"Tamanho do sistema: n = {n_82}")
# print(f"Tempo Gauss denso: {tempo_gauss_82:.6f} s, residuo = {res_gauss_82:.6e}")
# print(f"Tempo spsolve esparso: {tempo_sparse_82:.6f} s, residuo = {res_sparse_82:.6e}")
# print(f"Memoria matriz densa: {mem_densa_82:.4f} MB")
# print(f"Memoria matriz esparsa CSR: {mem_esparsa_82:.4f} MB")


# # Q8.3 - Implementacao vetorizada da LU
# print()
# print("--- Q8.3: Implementacao vetorizada da fatoracao LU ---")


# def fatoracao_lu_vetorizada(A, tol=1e-12):
#     """Fatoracao LU com atualizacoes vetorizadas de linhas e colunas."""
#     U = np.array(A, dtype=float, copy=True)
#     n = U.shape[0]
#     L = np.eye(n)

#     for k in range(n - 1):
#         if abs(U[k, k]) < tol:
#             raise ValueError(f"Pivo nulo ou muito pequeno em k={k}")
#         L[k + 1:, k] = U[k + 1:, k] / U[k, k]
#         U[k + 1:, k:] -= np.outer(L[k + 1:, k], U[k, k:])
#         U[k + 1:, k] = 0.0

#     return L, U


# np.random.seed(8)
# n_83 = 100
# A_83 = np.random.randn(n_83, n_83) + n_83 * np.eye(n_83)

# t0 = time.perf_counter()
# L_original, U_original = fatoracao_lu(A_83)
# tempo_lu_original = time.perf_counter() - t0

# t0 = time.perf_counter()
# L_vet, U_vet = fatoracao_lu_vetorizada(A_83)
# tempo_lu_vet = time.perf_counter() - t0

# erro_original_83 = np.linalg.norm(L_original @ U_original - A_83, "fro")
# erro_vet_83 = np.linalg.norm(L_vet @ U_vet - A_83, "fro")
# ganho_83 = tempo_lu_original / tempo_lu_vet

# print(f"\nTempo LU original: {tempo_lu_original:.6f} s")
# print(f"Tempo LU vetorizada: {tempo_lu_vet:.6f} s")
# print(f"Ganho de desempenho: {ganho_83:.2f}x")
# print(f"Erro ||LU - A||_F original: {erro_original_83:.6e}")
# print(f"Erro ||LU - A||_F vetorizada: {erro_vet_83:.6e}")



# #GERAÇÃO DE GRÁFICOS
# # ------------------------------------------------------------------------------------------------
# # Graficos — Q2.3: comparativo de tempos
# fig, ax = plt.subplots(figsize=(6, 4))
# categorias = ["Gauss repetido", "LU reutilizado"]
# tempos = [t_gauss, t_lu]
# cores = ["#e07b54", "#5b8fc9"]
# bars = ax.bar(categorias, tempos, color=cores, width=0.4, edgecolor="white")
# ax.bar_label(bars, fmt="%.3f s", padding=4, fontsize=10)
# ax.set_title(f"Q2.3 — Gauss vs. LU reutilizado\n(n={n_bench}, k={k_bench} sistemas)", fontsize=11)
# ax.set_ylabel("Tempo total (s)")
# ax.set_ylim(0, max(tempos) * 1.25)
# ax.text(
#     0.97, 0.85,
#     f"Aceleracao: {t_gauss / t_lu:.1f}x",
#     transform=ax.transAxes,
#     ha="right", va="top",
#     fontsize=10,
#     bbox=dict(boxstyle="round,pad=0.3", fc="#f0f4fb", ec="#5b8fc9"),
# )
# plt.tight_layout()
# plt.savefig("grafico_q2_3.png", dpi=300)
# plt.close()
# print("\nGrafico de comparativo de tempos salvo")
# print("\n" + "-" * 60)
# print("Execucao concluida com sucesso.")
# print("=" * 60)

# # ---- Grafico Q4.2 -----------------------------------------------
# fig, ax = plt.subplots(figsize=(8, 5))
# n_medidos = sizes_42[:3]
# t_g_medidos = t_gauss_42[:3]
 
# ax.loglog(sizes_42, t_thomas_42, marker="o", label="Thomas O(n)", linewidth=2)
# ax.loglog(n_medidos, t_g_medidos, marker="s", color="red",
#           label="Gauss O(n³) medido", linewidth=2)
# ax.loglog(sizes_42[2:], t_gauss_42[2:], marker="s", color="red",
#           linestyle="--", label="Gauss O(n³) extrapolado")
 
# ax.set_title("Q4.2 — Thomas vs. Gauss: Escalonamento", fontsize=13)
# ax.set_xlabel("n (tamanho do sistema)")
# ax.set_ylabel("Tempo (ms)")
# ax.legend()
# ax.grid(True, which="both", alpha=0.4)
# plt.tight_layout()
# plt.savefig("grafico_q4_2.png", dpi=150)
# plt.close()
# print("\nGrafico thomas_vs_gauss salvo ")
