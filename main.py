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
from lu import fatoracao_lu, resolver_lu, subst_prog, subst_retro
from thomas import montar_tridiagonal, thomas

import matplotlib
matplotlib.use("Agg")  

from scipy.linalg import lu as scipy_lu


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

print()
print("=" * 60)
print("SECAO 3 — Fatoracao LU (Doolittle)")
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
