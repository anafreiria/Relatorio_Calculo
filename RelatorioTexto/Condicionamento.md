
## Q6.1 — Matriz de Hilbert

### Experimento

Para cada $n \in \{4, 6, 8, 10, 12\}$, o sistema $H_n x = b$ foi resolvido com $b = H_n \mathbf{1}$
(solução exata $x^* = \mathbf{1}$). Foram medidos o número de condição $\kappa_2(H_n)$ e o
erro relativo $\|x_{\text{calc}} - x^*\|_2 / \|x^*\|_2$.

### Tabela de resultados

| n  | $\kappa_2(H_n)$  | Erro relativo   | Dígitos corretos |
|----|-----------------|-----------------|-----------------|
| 4  | 1.5514 × 10⁴    | 3.8618 × 10⁻¹³  | 11.81           |
| 6  | 1.4951 × 10⁷    | 3.1069 × 10⁻¹⁰  | 8.83            |
| 8  | 1.5258 × 10¹⁰   | 2.2563 × 10⁻⁷   | 5.82            |
| 10 | 1.6025 × 10¹³   | 1.6583 × 10⁻⁴   | 2.80            |
| 12 | 1.8023 × 10¹⁶   | 1.8113 × 10⁻¹   | 0.00            |

> **Fórmula dos dígitos corretos:** $d \approx 16 - \log_{10}(\kappa_2)$,
> pois o arredondamento em ponto flutuante (float64) introduz erros da ordem de $10^{-16}$,
> e o número de condição indica quantas vezes esse erro é amplificado.

### Análise

A matriz de Hilbert $H_n$ com $H_{ij} = 1/(i+j-1)$ é simétricamente positiva definida (SPD),
mas é notoriamente mal-condicionada. O número de condição cresce de forma aproximadamente
geométrica com $n$ (quase uma década a cada dois passos), o que demonstra o
comportamento explosivo do condicionamento.

- Para $n = 4$: $\kappa \approx 1.55 \times 10^4$ → ainda restam ~12 dígitos corretos, resultado confiável.
- Para $n = 8$: $\kappa \approx 1.53 \times 10^{10}$ → restam apenas ~6 dígitos corretos.
- Para $n = 10$: $\kappa \approx 1.60 \times 10^{13}$ → apenas ~3 dígitos corretos. Resultado pouco confiável.
- Para $n = 12$: $\kappa \approx 1.64 \times 10^{16}$ → dígitos corretos $\approx 0$. **Resultado completamente não confiável.**

**Conclusão:** O resultado torna-se completamente não confiável a partir de **$n = 12$**,
e começa a ser questionável a partir de $n = 10$ (menos de 3 dígitos corretos).
Isso ilustra que a presença de uma estrutura matemática simples (entradas racionais,
simetria, positividade) não garante boa resolubilidade numérica.

---

## Q6.2 — Amplificação de Erros

A função `perturbar_b` foi aplicada 100 vezes, com perturbações aleatórias de
nível $\delta = 10^{-6}$ no vetor $b$, para duas matrizes:

- $A = H_6$ (mal-condicionada, $\kappa \approx 1.49 \times 10^7$)
- $A = I_6$ (identidade, $\kappa = 1$)

O fator de amplificação é definido como:

$$\text{amp} = \frac{\|\delta x\|_2 / \|x\|_2}{\|\delta b\|_2 / \|b\|_2}$$

### Resultados

| Matriz | $\kappa_2(A)$  | Amplificação máxima | Amplificação média |
|--------|---------------|--------------------|--------------------|
| $H_6$  | 1.4951 × 10⁷  | 1.1316 × 10⁷       | 4.8185 × 10⁶      |
| $I_6$  | 1.0000        | 1.0000             | 1.0000             |

### Análise

O resultado central da teoria diz que, para uma perturbação $\delta b$ no lado direito:

$$\frac{\|\delta x\|_2}{\|x\|_2} \leq \kappa_2(A) \cdot \frac{\|\delta b\|_2}{\|b\|_2}$$

Os experimentos confirmam isso: a amplificação máxima observada (1.13 × 10⁷) está
dentro do limite teórico $\kappa(H_6) = 1.49 \times 10^7$, ou seja, o resultado
numérico está em acordo com a teoria.

Para a **identidade $I_6$**, a amplificação é exatamente 1 em todos os ensaios: perturbações
em $b$ se transferem diretamente para $x$ sem amplificação, o que é esperado, pois
$I \cdot \delta x = \delta b$ implica $\|\delta x\| = \|\delta b\|$.

Para $H_6$, a amplificação média (~$4.8 \times 10^6$) indica que uma perturbação
relativa de $10^{-6}$ em $b$ pode produzir um erro relativo de até ~48% na solução,
tornando os resultados completamente sem sentido mesmo para perturbações
microscópicas nos dados de entrada.

---

## Q6.3 — Perturbação em $A$: Bem-condicionada vs. Mal-condicionada


Foram criadas duas matrizes $5 \times 5$:

- **$A_{\text{bem}}$**: diagonal com entradas $\{1.0,\ 1.1,\ 0.9,\ 1.05,\ 0.95\}$, → $\kappa \approx 1.22$
- **$A_{\text{mal}}$**: matriz de Hilbert $H_5$, → $\kappa \approx 4.77 \times 10^5$

Para cada matriz, a solução exata é $x^* = \mathbf{1}$. Em seguida, introduz-se uma
perturbação aleatória $\delta A$ com $\|\delta A\| \approx 10^{-6}$ e resolve-se
$(A + \delta A)x = b$. O erro relativo resultante na solução é medido.

### Resultados

| Matriz       | $\kappa_2(A)$  | $\|\delta A\|_F / \|A\|_F$ | Erro relativo na solução |
|--------------|---------------|---------------------------|--------------------------|
| $A_{\text{bem}}$ | 1.22      | ~$10^{-6}$                | 3.59 × 10⁻⁶              |
| $A_{\text{mal}}$ (H₅) | 4.77 × 10⁵ | ~$10^{-6}$         | 8.21 × 10⁻²             |

**Razão entre os erros:** 22.868×

### Análise:

O erro relativo na solução se relaciona com a perturbação em $A$ por:

$$\frac{\|\delta x\|}{\|x\|} \lesssim \kappa_2(A) \cdot \frac{\|\delta A\|}{\|A\|}$$

Para **$A_{\text{bem}}$** ($\kappa \approx 1.22$): a perturbação de $10^{-6}$ em $A$ gera
um erro relativo da mesma ordem ($3.6 \times 10^{-6}$) na solução. O sistema se comporta
de forma **estável e previsível**.

Para **$A_{\text{mal}}$** ($\kappa \approx 4.77 \times 10^5$): a mesma perturbação de $10^{-6}$
amplificada pelo número de condição resulta em erro de **$\approx 8.2\%$** na solução.
Em termos práticos, os coeficientes de $x$ perderam quase 6 ordens de magnitude de precisão.

### Implicações práticas :

Em aplicações reais, modelagem estrutural, circuitos elétricos, simulações de fluidos,
os dados de entrada ($A$ e $b$) sempre carregam erros de medição, discretização e
representação em ponto flutuante. O número de condição determina se esses erros são
toleráveis:

- $\kappa \sim 1$: sistema robusto, pode-se confiar nos resultados.
- $\kappa \sim 10^6$: uma medição com precisão de 6 casas decimais pode produzir
  uma solução completamente incorreta.
- $\kappa > 10^{12}$: em float64, o resultado numérico já não reflete a solução real.

**Boa prática:** antes de resolver um sistema linear, calcule $\kappa_2(A)$. Valores acima
de $10^6 / 10^8$ são um sinal de alerta. Alternativas incluem pré-condicionamento,
regularização (Tikhonov) ou reformulação do problema.


> **Moral:** o número de condição $\kappa_2(A)$ é o principal indicador de confiabilidade
> de uma solução numérica. Métodos diretos como Gauss, LU e Cholesky são numericamente
> estáveis, os erros residuais surgem do condicionamento do **problema**, não do **algoritmo**.