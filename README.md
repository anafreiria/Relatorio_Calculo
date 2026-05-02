# Relatorio_Calculo

Este repositório contém os códigos-fonte em Python para o trabalho de Cálculo Numérico sobre sistemas de equações lineares e métodos diretos.

O objetivo do diretório é organizar as implementações e experimentos computacionais usados no relatório da disciplina, incluindo eliminação de Gauss, fatoração LU, fatoração de Cholesky, algoritmo de Thomas, condicionamento de matrizes e PageRank numérico.

## Arquivos

- `gauss.py`: eliminação de Gauss com pivoteamento parcial e substituição retroativa.
- `lu.py`: fatoração LU pelo método de Doolittle.
- `cholesky.py`: fatoração de Cholesky para matrizes simétricas positivas definidas.
- `thomas.py`: algoritmo de Thomas para sistemas tridiagonais.
- `condicao.py`: experimentos com matriz de Hilbert e sensibilidade a perturbações.
- `pagerank.py`: base para o projeto integrador de PageRank numérico.
- `main.py`: script principal para executar os experimentos do trabalho.

## Como executar

Instale as bibliotecas necessárias:

```bash
pip install numpy scipy matplotlib
```

Depois execute:

```bash
python3 main.py
```

O `main.py` serve como ponto de partida para reproduzir os resultados, tabelas e gráficos usados no relatório.

## Observação

Os códigos seguem os exemplos fornecidos no documento da atividade. As seções marcadas com `TODO` devem ser completadas com os experimentos e análises pedidos no trabalho.
