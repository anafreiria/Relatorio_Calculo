# Metodos Diretos para Sistemas Lineares

Projeto desenvolvido para a disciplina de **Calculo Numerico**, com implementacoes em Python de metodos diretos para resolucao de sistemas lineares e experimentos numericos sobre estabilidade, condicionamento, custo computacional e aplicacoes.

O trabalho foi desenvolvido por **Ana Flavia Freiria Rodrigues** e **Raissa Nunes Peret**.

## Objetivo

O objetivo do projeto e estudar, implementar e comparar metodos diretos de solucao de sistemas lineares da forma:

```text
A x = b
```

Os experimentos mostram como cada metodo se comporta em diferentes situacoes: matrizes bem ou mal condicionadas, matrizes simetricas definidas positivas, sistemas tridiagonais, multiplos lados direitos e problemas aplicados como PageRank.

## Metodos implementados

- **Eliminacao de Gauss com pivoteamento parcial**
  - Resolve sistemas lineares densos.
  - Usa pivoteamento para melhorar a estabilidade numerica.
  - Detecta sistemas singulares ou quase singulares.

- **Fatoracao LU pelo metodo de Doolittle**
  - Decompoe uma matriz em `A = L U`.
  - Permite reutilizar a fatoracao para resolver varios sistemas com a mesma matriz `A`.
  - E comparada com uma fatoracao PLU usando `scipy.linalg.lu`.

- **Fatoracao de Cholesky**
  - Aplicada a matrizes simetricas definidas positivas.
  - Resolve sistemas do tipo `A = L L^T`.
  - E usada tambem em regressao linear via equacoes normais.

- **Algoritmo de Thomas**
  - Metodo especializado para sistemas tridiagonais.
  - Explora a estrutura da matriz para obter menor custo de tempo e memoria.
  - E comparado com Gauss em sistemas de tamanhos crescentes.

- **Analise de condicionamento**
  - Usa matrizes de Hilbert para estudar sensibilidade numerica.
  - Mede numero de condicao, erro relativo e amplificacao de perturbacoes.

- **PageRank numerico**
  - Modela o PageRank como um sistema linear.
  - Compara solucoes via Gauss, LU propria e `scipy.linalg.lu_solve`.
  - Analisa o condicionamento do sistema conforme o parametro `alpha` se aproxima de 1.

## Estrutura do projeto

```text
.
├── main.py          # Script principal com todos os experimentos
├── gauss.py         # Eliminacao de Gauss e substituicao retroativa
├── lu.py            # Fatoracao LU, substituicao progressiva e retroativa
├── cholesky.py      # Fatoracao de Cholesky e resolucao por Cholesky
├── thomas.py        # Algoritmo de Thomas para sistemas tridiagonais
├── condicao.py      # Experimentos de condicionamento numerico
├── pagerank.py      # Funcoes auxiliares para o PageRank numerico
└── README.md        # Documentacao do projeto
```

## Dependencias

O projeto usa Python 3 e as seguintes bibliotecas:

- `numpy`
- `scipy`
- `matplotlib`

Para instalar as dependencias:

```bash
pip install numpy scipy matplotlib
```

Se preferir usar ambiente virtual:

```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy scipy matplotlib
```

## Como executar

Execute o script principal:

```bash
python3 main.py
```

O programa imprime no terminal os resultados das secoes do trabalho, incluindo tabelas, residuos, erros relativos, tempos de execucao e comparacoes entre metodos.

Durante a execucao, tambem sao gerados graficos em arquivos `.png`.

## Graficos gerados

Ao executar `main.py`, os seguintes graficos podem ser criados no diretorio do projeto:

- `grafico_q1_4.png`: erro relativo em `x1` ao variar o pivo inicial.
- `grafico_q2_3.png`: comparacao entre Gauss repetido e LU reutilizado.
- `grafico_q3_2.png`: tempo de Cholesky versus LU.
- `grafico_q4_2.png`: escalonamento de Thomas versus Gauss.
- `grafico_q5_1.png`: lei de escala empirica do metodo de Gauss.
- `grafico_q6_1.png`: condicionamento e erro relativo em matrizes de Hilbert.
- `grafico_q6_2.png`: amplificacao de erros em matriz de Hilbert e identidade.
- `grafico_q7_3.png`: condicionamento do sistema PageRank para diferentes valores de `alpha`.
- `grafico_q8_1.png`: estabilidade numerica quando `epsilon` tende a zero.

## Organizacao dos experimentos

O arquivo `main.py` esta dividido nas seguintes secoes:

1. **Eliminacao de Gauss com pivoteamento parcial**
2. **Fatoracao LU**
3. **Fatoracao de Cholesky**
4. **Algoritmo de Thomas**
5. **Custo computacional empirico**
6. **Condicionamento e sensibilidade a perturbacoes**
7. **PageRank numerico**
8. **Desafios adicionais**

Cada secao executa testes especificos, compara resultados com referencias numericas e calcula metricas como residuo, erro relativo, numero de condicao e tempo de execucao.

## Exemplos de uso dos modulos

### Gauss com pivoteamento parcial

```python
import numpy as np
from gauss import resolver_gauss

A = np.array([[3, 2, 4], [1, 1, 2], [4, 3, -2]], dtype=float)
b = np.array([1, 2, 3], dtype=float)

x = resolver_gauss(A, b)
print(x)
```

### Fatoracao LU

```python
import numpy as np
from lu import fatoracao_lu, subst_prog, subst_retro

A = np.array([[2, 1, 1], [4, -6, 0], [-2, 7, 2]], dtype=float)
b = np.array([1, 2, 3], dtype=float)

L, U = fatoracao_lu(A)
y = subst_prog(L, b)
x = subst_retro(U, y)
print(x)
```

### Cholesky

```python
import numpy as np
from cholesky import resolver_cholesky

A = np.array([[4, 2], [2, 3]], dtype=float)
b = np.array([1, 1], dtype=float)

x, L = resolver_cholesky(A, b)
print(x)
```

### Thomas

```python
from thomas import thomas

a = [-1, -1, -1, -1]
b = [4, 4, 4, 4, 4]
c = [-1, -1, -1, -1]
d = [1, 0, 0, 0, 1]

x = thomas(a, b, c, d)
print(x)
```

## Observacoes numericas

- O pivoteamento parcial melhora a estabilidade da eliminacao de Gauss quando aparecem pivos pequenos.
- A fatoracao LU e vantajosa quando varios sistemas compartilham a mesma matriz `A` e possuem lados direitos diferentes.
- Cholesky e mais eficiente que LU para matrizes simetricas definidas positivas.
- Thomas e muito mais eficiente que Gauss para matrizes tridiagonais, tanto em tempo quanto em memoria.
- Matrizes de Hilbert sao exemplos classicos de matrizes mal condicionadas.
- No PageRank, valores de `alpha` proximos de 1 tendem a aumentar o numero de condicao do sistema.

## Resultado esperado

Ao final da execucao, o terminal deve exibir:

```text
Execucao concluida com sucesso.
```

Os arquivos de grafico ficam salvos no mesmo diretorio do projeto.
