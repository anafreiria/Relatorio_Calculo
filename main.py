"""
main.py
Plano de Investigacao - Sistemas Lineares
Disciplina: Calculo Numerico
"""
import numpy as np
import time
import matplotlib.pyplot as plt


from cholesky import cholesky, resolver_cholesky
from condicao import experimento_hilbert, perturbar_b
from gauss import gauss, resolver_gauss
from lu import fatoracao_lu, subst_prog, subst_retro
from thomas import montar_tridiagonal, thomas

import matplotlib
matplotlib.use("Agg")  

from scipy.linalg import lu as scipy_lu


# print("\n============================================================")
# print("SECAO 1 - ELIMINACAO DE GAUSS E PIVOTEAMENTO")
# print("============================================================")

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


# Q2.1 
print()
print("--- Q2.1: Construcao de L e U e verificacao ---")
 
A_lu = np.array(
    [
        [2.0,  1.0,  1.0],
        [4.0, -6.0,  0.0],
        [-2.0,  7.0,  2.0],
    ]
)
 
L, U = fatoracao_lu(A_lu)
 
print("\nMatriz L (triangular inferior):")
print(np.array2string(L, precision=6, suppress_small=True))
 
print("\nMatriz U (triangular superior):")
print(np.array2string(U, precision=6, suppress_small=True))
 
erro_fat = np.linalg.norm(L @ U - A_lu, "fro")
print(f"\nErro de fatoracao ||LU - A||_F = {erro_fat:.2e}")
 
# Verificacao visual da estrutura triangular
tril_L = np.allclose(L, np.tril(L))
triu_U = np.allclose(U, np.triu(U))
diag_L_uns = np.allclose(np.diag(L), np.ones(L.shape[0]))
print(f"L e triangular inferior? {tril_L}")
print(f"Diagonal de L sao todos 1? {diag_L_uns}")
print(f"U e triangular superior? {triu_U}")


# Q2.2 
print()
print("--- Q2.2: Multiplos lados direitos com L e U reutilizados ---")
 
b1 = np.array([1.0, 2.0, 3.0])
b2 = np.array([0.0, 1.0, -1.0])
 
# L e U ja calculado acima, reutiliza sem refatorar
y1 = subst_prog(L, b1)
x1 = subst_retro(U, y1)
 
y2 = subst_prog(L, b2)
x2 = subst_retro(U, y2)
 
res1 = np.linalg.norm(A_lu @ x1 - b1)
res2 = np.linalg.norm(A_lu @ x2 - b2)
 
print(f"\nb1 = {b1}  =>  x1 = {x1}")
print(f"Residuo ||A x1 - b1||_2 = {res1:.2e}")
print(f"\nb2 = {b2}  =>  x2 = {x2}")
print(f"Residuo ||A x2 - b2||_2 = {res2:.2e}")
 

# Q2.3 — Vantagem com multiplos lados direitos (benchmarking)
print()
print("--- Q2.3: Gauss repetido vs. LU reutilizado (n=200, k=50) ---")
 
np.random.seed(42)
n_bench = 200
k_bench = 50
A_bench = np.random.randn(n_bench, n_bench)
# Garante invertibilidade
A_bench += n_bench * np.eye(n_bench)
Bs_bench = [np.random.randn(n_bench) for _ in range(k_bench)]
 
# Estrategia (a): Gauss repetido
t0 = time.perf_counter()
for bk in Bs_bench:
    resolver_gauss(A_bench, bk)
t_gauss = time.perf_counter() - t0
 
# Estrategia (b): LU reutilizado
t0 = time.perf_counter()
L_bench, U_bench = fatoracao_lu(A_bench)
for bk in Bs_bench:
    yk = subst_prog(L_bench, bk)
    subst_retro(U_bench, yk)
t_lu = time.perf_counter() - t0
 
print(f"\n{'Estrategia':<25} {'Tempo total (s)':>18} {'Por sistema (ms)':>18}")
print("-" * 63)
print(f"{'Gauss repetido':<25} {t_gauss:>18.4f} {t_gauss / k_bench * 1000:>18.3f}")
print(f"{'LU reutilizado':<25} {t_lu:>18.4f} {t_lu   / k_bench * 1000:>18.3f}")
print(f"\nFator de aceleracao: {t_gauss / t_lu:.2f}x")
 

# Q2.4 — Fatoracao PLU com pivoteamento 
print()
print("--- Q2.4: Fatoracao PLU com pivoteamento ---")
 
A_plu = np.array([[0.0, 1.0], [2.0, 3.0]])
 
# (a) Testa fatoracao_lu sem pivoteamento
print("\n(a) Tentando fatoracao_lu (sem pivoteamento) em A = [[0,1],[2,3]]:")
try:
    L_sem, U_sem = fatoracao_lu(A_plu)
    print(f"    Resultado L:\n{L_sem}")
    print(f"    Resultado U:\n{U_sem}")
    print(f"    Erro ||LU-A||_F = {np.linalg.norm(L_sem @ U_sem - A_plu, 'fro'):.2e}")
    # Verificar se U[0,0] e zero (divisao por zero iminente)
    if abs(U_sem[0, 0]) < 1e-14:
        print("    ATENCAO: pivo U[0,0] nulo — resultado invalido (NaN/Inf esperado)!")
    else:
        print("    Concluiu sem excecao (verifique se o resultado e confiavel).")
except Exception as e:
    print(f"    ERRO capturado: {e}")
 
# (b) scipy.linalg.lu
print("\n(b) Usando scipy.linalg.lu (com pivoteamento):")
P_sp, L_sp, U_sp = scipy_lu(A_plu)
 
print(f"\n    Matriz de permutacao P:\n{P_sp}")
print(f"\n    Triangular inferior L:\n{L_sp}")
print(f"\n    Triangular superior U:\n{U_sp}")
print(f"\n    Verificacao P @ A = L @ U:")
print(f"    P @ A =\n{P_sp @ A_plu}")
print(f"    L @ U =\n{L_sp @ U_sp}")
err_plu = np.linalg.norm(P_sp @ A_plu - L_sp @ U_sp, "fro")
print(f"\n    Erro ||PA - LU||_F = {err_plu:.2e}")
 
# Demonstracao: como resolver Ax = b com PLU
b_plu = np.array([1.0, 2.0])
# PA = LU  =>  Ax = b  =>  PAx = Pb  =>  LUx = Pb
Pb = P_sp @ b_plu
y_plu = subst_prog(L_sp, Pb)
x_plu = subst_retro(U_sp, y_plu)
print(f"\n    Solucao de Ax = {b_plu} via PLU: x = {x_plu}")
print(f"    Residuo ||Ax - b||_2 = {np.linalg.norm(A_plu @ x_plu - b_plu):.2e}")
 
# ------------------------------------------------------------------------------
# Graficos — Q2.3: comparativo de tempos
fig, ax = plt.subplots(figsize=(6, 4))
categorias = ["Gauss repetido", "LU reutilizado"]
tempos = [t_gauss, t_lu]
cores = ["#e07b54", "#5b8fc9"]
bars = ax.bar(categorias, tempos, color=cores, width=0.4, edgecolor="white")
ax.bar_label(bars, fmt="%.3f s", padding=4, fontsize=10)
ax.set_title(f"Q2.3 — Gauss vs. LU reutilizado\n(n={n_bench}, k={k_bench} sistemas)", fontsize=11)
ax.set_ylabel("Tempo total (s)")
ax.set_ylim(0, max(tempos) * 1.25)
ax.text(
    0.97, 0.85,
    f"Aceleracao: {t_gauss / t_lu:.1f}x",
    transform=ax.transAxes,
    ha="right", va="top",
    fontsize=10,
    bbox=dict(boxstyle="round,pad=0.3", fc="#f0f4fb", ec="#5b8fc9"),
)
plt.tight_layout()
plt.savefig("resultadosDeLU.pdf", dpi=150)
plt.close()
print("\nGrafico salvo em resultados.pdf")
print("\n" + "-" * 60)
print("Execucao concluida com sucesso.")
print("=" * 60)

# ANA
# print()
# print("=" * 60)
# print("SECAO 3 - Fatoracao de Cholesky para matrizes SPD")
# print("=" * 60)

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

# RAISSA
print()
print("=" * 60)
print("SECAO 4 - Algoritmo de Thomas para sistemas tridiagonais")
print("=" * 60)

# # ANA
# print()
# print("=" * 60)
# print("SECAO 5 - Custo computacional empirico")
# print("=" * 60)


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

# print("\n--- Q6.1: Matriz de Hilbert ---")
# for n in [4, 6, 8, 10, 12]:
#     kappa, erro = experimento_hilbert(n)
#     print(f"n={n:2d} kappa={kappa:.2e} erro={erro:.2e}")

# # RAISANA
# print()
# print("=" * 60)
# print("SECAO 7 - PageRank Numérico")
# print("=" * 60)

# plt.tight_layout()
# plt.savefig("resultados.pdf", dpi=150)
# plt.show()

# # RAISANA
# print()
# print("=" * 60)
# print("SECAO 8 - Desafio")
# print("=" * 60)
