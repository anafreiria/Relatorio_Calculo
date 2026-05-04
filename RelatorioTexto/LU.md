# Seção Fatoração LU (Doolittle)

## Q2.1 — Construção e verificação de L e U

### Matriz utilizada
$$
A = \begin{bmatrix} 2 & 1 & 1 \\ 4 & -6 & 0 \\ -2 & 7 & 2 \end{bmatrix}
$$

### Saída do terminal

```
Matriz L (triangular inferior):
[[ 1.  0.  0.]
 [ 2.  1.  0.]
 [-1. -1.  1.]]

Matriz U (triangular superior):
[[ 2.  1.  1.]
 [ 0. -8. -2.]
 [ 0.  0.  1.]]

Erro de fatoracao ||LU - A||_F = 0.00e+00
L e triangular inferior? Sim
Diagonal de L sao todos 1? Sim
U e triangular superior? Sim
```

### Interpretação

A fatoração de Doolittle decompõe $A = LU$ onde:

- **L** é triangular inferior com **diagonal unitária** (1s na diagonal), propriedade que distingue Doolittle de Crout.
- **U** é a matriz triangular superior resultante da eliminação.

Os elementos abaixo da diagonal de $L$ são exatamente os **multiplicadores** usados durante a eliminação gaussiana:

| Elemento | Valor | Significado |
|----------|-------|-------------|
| $L_{2,1} = 2$ | $4/2 = 2$ | linha 2 ← linha 2 − 2 × linha 1 |
| $L_{3,1} = -1$ | $-2/2 = -1$ | linha 3 ← linha 3 − (−1) × linha 1 |
| $L_{3,2} = -1$ | $8/(-8) = -1$ | linha 3 ← linha 3 − (−1) × linha 2 |

O **erro de fatoração** $\|LU - A\|_F = 0{,}00 \times 10^{0}$ indica que o produto $LU$ reconstrói $A$ com precisão de máquina (float64, ~15–16 dígitos significativos), confirmando que nenhuma informação foi perdida no processo.

---

## Q2.2 — Dois lados direitos com L e U reutilizados

### Saída do terminal

```
b1 = [1. 2. 3.]  =>  x1 = [-1. -1.  4.]
Residuo ||A x1 - b1||_2 = 0.00e+00

b2 = [ 0.  1. -1.]  =>  x2 = [ 0.0625 -0.125  0.]
Residuo ||A x2 - b2||_2 = 0.00e+00
```

### Verificação manual — x1

$$
A \mathbf{x}_1 = \begin{bmatrix}2&1&1\\4&-6&0\\-2&7&2\end{bmatrix} \begin{bmatrix}-1\\-1\\4\end{bmatrix} = \begin{bmatrix}2(-1)+1(-1)+1(4)\\4(-1)+(-6)(-1)+0\\-2(-1)+7(-1)+2(4)\end{bmatrix} = \begin{bmatrix}1\\2\\3\end{bmatrix} = \mathbf{b}_1 \ \checkmark
$$

### Interpretação

A fatoração $A = LU$ foi calculada **uma única vez**. Para cada novo lado direito $\mathbf{b}_i$, o custo de resolução é apenas $\mathcal{O}(n^2)$, em dois passos:

1. **Substituição progressiva:** resolve $L\mathbf{y} = \mathbf{b}$ (varre $L$ de cima para baixo)
2. **Substituição retroativa:** resolve $U\mathbf{x} = \mathbf{y}$ (varre $U$ de baixo para cima)

Ambos os resíduos são **zero em precisão de máquina**, confirmando que as soluções são exatas. Esta é a principal vantagem prática da Fatoração LU: quando a mesma matriz $A$ precisa ser resolvida para vários vetores $\mathbf{b}$ distintos, paga-se o custo $\mathcal{O}(n^3)$ da fatoração apenas uma vez e $\mathcal{O}(n^2)$ por cada sistema adicional.

---

## Q2.3 — Vantagem com múltiplos lados direitos (benchmarking)

**Configuração:** $n = 200$, $k = 50$ sistemas, `np.random.seed(42)`

### Saída do terminal

```
Estrategia                   Tempo total (s)   Por sistema (ms)
---------------------------------------------------------------
Gauss repetido                        2.6702             53.404
LU reutilizado                        0.1005              2.010

Fator de aceleracao: 26.57x
```

### Tabela de resultados

| Estratégia | Tempo total (s) | Tempo por sistema (ms) |
|---|---|---|
| Gauss repetido | 2,6702 | 53,404 |
| LU reutilizado | 0,1005 | 2,010 |
| **Fator de aceleração** | **26,57×** | — |

### Justificativa teórica, contagem de flops

**Gauss repetido** (fatoração + substituições a cada chamada):

$$
\text{Total} = k \cdot \frac{2}{3}n^3 = 50 \cdot \frac{2}{3}(200)^3 \approx 2{,}67 \times 10^8 \ \text{flops}
$$

**LU reutilizado** (fatoração uma vez + $k$ pares de substituições):

$$
\text{Total} = \frac{2}{3}n^3 + k \cdot 2n^2 = 5{,}33 \times 10^6 + 50 \cdot 2 \cdot (200)^2 \approx 9{,}33 \times 10^6 \ \text{flops}
$$

**Fator teórico esperado:**

$$
\frac{2{,}67 \times 10^8}{9{,}33 \times 10^6} \approx 28{,}6\times
$$

### Interpretação

O fator medido (**26,57×**) está em excelente acordo com a previsão teórica (**28,6×**), com diferença de apenas ~7%. Essa discrepância é esperada e deve-se a:

- *Overhead* de chamadas de função em Python puro
- Efeitos de cache da CPU (dados já carregados na segunda rodada)
- Variação de carga do sistema operacional durante a medição
- Alocações e cópias de memória adicionais em cada chamada de `resolver_gauss`

O gráfico gerado (`resultados.pdf`) ilustra visualmente o contraste: a barra do Gauss repetido (2,670 s) domina amplamente sobre a barra do LU reutilizado (0,101 s), com o fator de aceleração 26,6× anotado.

> **Conclusão prática:** sempre que a mesma matriz $A$ precisar ser resolvida para múltiplos vetores $\mathbf{b}$ (simulações, análise de sensibilidade, controle em tempo real), a estratégia LU reutilizado é amplamente superior. Para $k$ sistemas, o ganho cresce linearmente com $k$.

---

## Q2.4 — Fatoração PLU com pivoteamento (`scipy.linalg.lu`)

### Matriz testada

$$
A = \begin{bmatrix} 0 & 1 \\ 2 & 3 \end{bmatrix}
$$

### (a) `fatoracao_lu` sem pivoteamento — saída do terminal

```
RuntimeWarning: divide by zero encountered in scalar divide
  L[i, k] = (A[i, k] - L[i, :k] @ U[:k, k]) / U[k, k]

Resultado L:
[[ 1.  0.]
 [inf  1.]]
Resultado U:
[[  0.   1.]
 [  0. -inf]]

Erro ||LU - A||_F = nan
ATENCAO: pivo U[0,0] nulo — resultado invalido (NaN/Inf esperado)!
```

**Interpretação:** O elemento $a_{00} = 0$ torna-se o primeiro pivô $U_{00} = 0$. Na iteração seguinte, o algoritmo calcula:

$$
L_{1,0} = \frac{A_{1,0}}{U_{0,0}} = \frac{2}{0} = +\infty
$$

O Python/NumPy **não lança exceção**, emite apenas um `RuntimeWarning` e prossegue com `inf`, contaminando toda a fatoração. O resultado é completamente inválido!

> **Importante:** $\det(A) = 0 \cdot 3 - 1 \cdot 2 = -2 \neq 0$, portanto $A$ é **invertível**. O problema **não é singularidade**,  é a ausência de pivoteamento. A Eliminação de Gauss sem pivoteamento falha mesmo para matrizes invertíveis quando o elemento diagonal é zero.

---

### (b) `scipy.linalg.lu` com pivoteamento 

```
Matriz de permutacao P:
[[0. 1.]
 [1. 0.]]

Triangular inferior L:
[[1. 0.]
 [0. 1.]]

Triangular superior U:
[[2. 3.]
 [0. 1.]]

Verificacao P @ A = L @ U:
P @ A = [[2. 3.] [0. 1.]]
L @ U = [[2. 3.] [0. 1.]]

Erro ||PA - LU||_F = 0.00e+00

Solucao de Ax = [1. 2.] via PLU: x = [-0.5  1. ]
Residuo ||Ax - b||_2 = 0.00e+00
```

### Papel da matriz de permutação P

$P$ é uma matriz de permutação (ortogonal: $P^T = P^{-1}$) que registra as **trocas de linha** realizadas pelo pivoteamento parcial. Ela garante que:

1. Nenhum pivô seja zero (evita divisão por zero)
2. O pivô escolhido seja o **maior em valor absoluto** na coluna correspondente, minimizando a propagação de erros de arredondamento

No exemplo, $P$ troca as duas linhas de $A$, posicionando o elemento 2 (maior em módulo) na posição de pivô:

$$
PA = \begin{bmatrix}0&1\\1&0\end{bmatrix}\begin{bmatrix}0&1\\2&3\end{bmatrix} = \begin{bmatrix}2&3\\0&1\end{bmatrix} = LU \ \checkmark
$$

### Como usar P para resolver Ax = b

Dado que $PA = LU$, o sistema $A\mathbf{x} = \mathbf{b}$ é resolvido em três etapas:

$$
A\mathbf{x} = \mathbf{b} \xrightarrow{\times P} (PA)\mathbf{x} = P\mathbf{b} \implies LU\mathbf{x} = P\mathbf{b}
$$

| Passo | Operação | Custo |
|-------|----------|-------|
| 1 | Calcula $P\mathbf{b}$ (permuta linhas de $\mathbf{b}$) | $\mathcal{O}(n)$ |
| 2 | Resolve $L\mathbf{y} = P\mathbf{b}$ (substituição progressiva) | $\mathcal{O}(n^2)$ |
| 3 | Resolve $U\mathbf{x} = \mathbf{y}$ (substituição retroativa) | $\mathcal{O}(n^2)$ |

**Demonstração numérica** para $\mathbf{b} = [1, 2]^T$:

$$
P\mathbf{b} = \begin{bmatrix}2\\1\end{bmatrix} \xrightarrow{L\mathbf{y}=P\mathbf{b}} \mathbf{y} = \begin{bmatrix}2\\1\end{bmatrix} \xrightarrow{U\mathbf{x}=\mathbf{y}} \mathbf{x} = \begin{bmatrix}-0{,}5\\1{,}0\end{bmatrix}
$$

Verificação: $A\mathbf{x} = \begin{bmatrix}0(-0{,}5)+1(1)\\2(-0{,}5)+3(1)\end{bmatrix} = \begin{bmatrix}1\\2\end{bmatrix} = \mathbf{b} \ \checkmark$

Resíduo: $\|A\mathbf{x} - \mathbf{b}\|_2 = 0{,}00 \times 10^{0}$

---

## Lições aprendidas 

| # | Lição |
|---|-------|
| 1 | A Fatoração de Doolittle é exata em precisão de máquina: $\|LU - A\|_F = 0$ |
| 2 | Reutilizar $L$ e $U$ reduz o custo de $k \cdot \mathcal{O}(n^3)$ para $\mathcal{O}(n^3) + k \cdot \mathcal{O}(n^2)$ — ganho medido de **26,57×** |
| 3 | Um pivô nulo não implica singularidade, mas causa falha numérica catastrófica na LU sem pivoteamento |
| 4 | A fatoração PLU incorpora pivoteamento parcial via $P$: resolve-se $LU\mathbf{x} = P\mathbf{b}$, não $A\mathbf{x} = \mathbf{b}$ diretamente |
| 5 | Para uso geral, prefira sempre PLU (`scipy.linalg.lu` ou `numpy.linalg.solve`) à LU de Doolittle pura |