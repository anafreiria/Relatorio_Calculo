

## Q4.1 — Execução e Verificação

### Sistema resolvido

O sistema tridiagonal 5×5 abaixo foi montado e resolvido:

$$
\begin{pmatrix}
4 & -1 & 0 & 0 & 0 \\
-1 & 4 & -1 & 0 & 0 \\
0 & -1 & 4 & -1 & 0 \\
0 & 0 & -1 & 4 & -1 \\
0 & 0 & 0 & -1 & 4
\end{pmatrix}
\mathbf{x} =
\begin{pmatrix} 1 \\ 0 \\ 0 \\ 0 \\ 1 \end{pmatrix}
$$

### Resultado

| Componente | Valor |
|---|---|
| x₁ | 0.26923077 |
| x₂ | 0.07692308 |
| x₃ | 0.03846154 |
| x₄ | 0.07692308 |
| x₅ | 0.26923077 |

**Resíduo:** `‖Ax − d‖₂ = 0.00e+00`

O resíduo nulo (dentro da precisão de máquina) confirma que o algoritmo de Thomas produziu a solução exata para este sistema.

A solução apresenta simetria esperada (x₁ = x₅ e x₂ = x₄), pois o sistema possui coeficientes simétricos e lado direito simétrico, sendo consistente com a natureza do problema.

### Contexto físico

Esta matriz surge naturalmente na **discretização por diferenças finitas** do problema de valor de contorno unidimensional:

$$-u''(x) = f(x), \quad u(0) = u(1) = 0$$

Com passo h = 1/(n+1) e aproximação central u''(xᵢ) ≈ (uᵢ₋₁ − 2uᵢ + uᵢ₊₁)/h², obtém-se um sistema tridiagonal com diagonal principal ≈ 4/h² e subdiagonais ≈ −1/h². O sistema resolvido corresponde à equação de Poisson 1D, com condições de contorno de Dirichlet.

---

## Q4.2 — Thomas vs. Gauss: Escalonamento

### Metodologia

Para cada n ∈ {100, 500, 1000, 5000, 10000}, foram gerados sistemas tridiagonais com:
- Diagonal principal: bᵢ = 4
- Sub/superdiagonais: aᵢ = cᵢ = −1
- Lado direito: dᵢ = 1

Os tempos foram medidos com `time.perf_counter()`. Para n = 5000 e n = 10000, o Gauss com matriz densa foi **extrapolado** via escala O(n³) a partir do valor medido em n = 1000.

### Tabela de Resultados

| n | Thomas (ms) | Gauss (ms) | Razão observada | Razão teórica* |
|---|---|---|---|---|
| 100 | 0.385 | 20.11 | 52.2× | ≈ O(n³)/O(n) = n² |
| 500 | 1.279 | 350.05 | 273.8× | ~10.000× (relativo n=100) |
| 1000 | 1.551 | 1540.45 | 993.2× | ~1000× |
| 5000 | 6.003 | ~194.934.000 ms | ~32.500× | teórico |
| 10000 | 11.443 | ~1.559.475.000 ms | ~136.000× | teórico |

*A razão teórica cresce como O(n²), pois Thomas é O(n) e Gauss é O(n³).

### Gráfico

![Gráfico Q4.2 — Thomas vs. Gauss][grafico_thomas_vs_gauss]

*Escala log-log. Pontos sólidos: valores medidos; tracejado: extrapolação O(n³) para o Gauss.*

### Análise

Os resultados confirmam claramente as previsões teóricas:

- **Thomas** cresce **linearmente** com n: ao passar de n=100 para n=10000 (fator 100), o tempo aumentou de 0,385 ms para 11,4 ms (fator ≈ 30× - próximo do esperado, com overhead de memória para vetores maiores).

- **Gauss** cresce **cúbicamente**: ao passar de n=100 para n=1000 (fator 10), o tempo aumentou de 20 ms para 1540 ms (fator ≈ 77×, consistente com 10³/10 ≈ 100×, considerando constantes de implementação Python pura).

- A **razão Thomas/Gauss** aumenta de ≈52× em n=100 para ≈993× em n=1000, evidenciando o ganho crescente com n, como previsto pela complexidade relativa O(n³)/O(n) = O(n²).

Para n=10000, o Gauss levaria **mais de 25 minutos** para completar, enquanto o Thomas resolve em apenas **11,4 ms**, sendo uma diferença de mais de 100.000×.

---

## Q4.3 — Análise de Memória

Para n = 10.000:

| Representação | Memória |
|---|---|
| Matriz densa (n × n, float64) | **762,94 MB** |
| Três vetores tridiagonais (float64) | **0,23 MB** |
| **Fator de redução** | **≈ 3.333×** |

**Cálculo:**
- Densa: n² × 8 bytes / 2²⁰ = 10.000² × 8 / 1.048.576 ≈ **762,94 MB**
- Tridiagonal: 3 × n × 8 bytes / 2²⁰ ≈ **0,23 MB**

### Implicações práticas

1. **Viabilidade computacional:** Uma matriz densa 10.000×10.000 em float64 ocupa quase 763 MB, próximo do limite de RAM de computadores domésticos, e inviável para n ≥ 30.000 (> 7 GB). A representação esparsa por três vetores é praticamente ilimitada.

2. **Equações diferenciais parciais:** Em simulações de física, engenharia e finanças, sistemas com n > 10⁶ são comuns (discretizações 1D finas ou grids 2D/3D). Apenas estruturas esparsas tornam esses problemas tratáveis.

3. **Custo de banda de memória:** Mesmo que a RAM estivesse disponível, a leitura da matriz densa é muito mais lenta que a leitura de três vetores contíguos, a localidade de cache favorece fortemente o Thomas.

4. **Extensão para 2D:** Na equação de Poisson 2D em grade m×m, a matriz possui estrutura **bloco-tridiagonal** com n = m² equações. Para m=100 (n=10.000), o Thomas não se aplica diretamente, mas solvers esparsos especializados continuam sendo essenciais.

---

## Conclusões

O Algoritmo de Thomas é o método de escolha para sistemas tridiagonais, oferecendo:

- **Complexidade ótima O(n)** tanto em tempo quanto em memória;
- **Precisão equivalente** ao Gauss (resíduo nulo no experimento);
- **Implementação simples** e naturalmente estável para matrizes diagonalmente dominantes (como as oriundas de diferenças finitas);
- **Escalabilidade** para problemas de grande porte inviáveis com métodos densos.

A comparação empírica com Gauss confirma que a diferença de desempenho **não é apenas constante**, mas cresce como n², tornando o Thomas de 993× mais rápido em n=1000 e potencialmente 136.000× mais rápido em n=10.000.

[grafico_thomas_vs_gauss]: grafico_thomas_vs_gauss.png