# Programação Genética para Regressão Simbólica

Computação Natural — DCC/UFMG.

Marco Tulio Mata -- https://github.com/marcotulio956/

## Conteúdo

```
TP1_GP.ipynb          documentação + experimentos (Introdução, Implementação, Experimentos, Conclusões, Bibliografia)
gpsr/                 biblioteca (todo o código do algoritmo)
  tree.py             árvores (lista prefixa), primitivas protegidas, população inicial, avaliação/RMSE
  operators.py        cruzamento de subárvores, mutação de subárvore e pontual (respeitam profundidade máx.)
  selection.py        roleta, torneio e ε-lexicase (MAD dinâmico / estático / clássico)
  gp.py               GPConfig e laço evolutivo (elitismo, estatísticas por geração)
  experiments.py      repetições em paralelo, tabelas, testes estatísticos, efeito dos operadores, bloat
  data.py, plots.py, report.py
tests/                testes unitários dos componentes (pytest)
data/                 <base>-train.csv / <base>-test.csv  (última coluna = y)
```

## Como executar

```bash
pip install -r requirements.txt
jupyter lab symbolicregression_GA.ipynb       # abra a partir desta pasta e use "Run All"
```

* Coloque em `data/` os pares `synth1`, `synth2` e `concrete` (`*-train.csv` e `*-test.csv`, sem cabeçalho,
  última coluna = saída). O notebook usa automaticamente as bases que encontrar.
* `QUICK = True` (célula 3.0) faz uma execução rápida só para testar o notebook; **os resultados finais
  devem ser gerados com `QUICK = False`** (30 repetições por configuração, execução em paralelo).

## Uso direto da biblioteca

```python
from gpsr import GPConfig, GeneticProgramming, load_dataset

d = load_dataset("synth1", "data")
cfg = GPConfig(pop_size=100, generations=100, selection="lexicase", pc=0.9, pm=0.05, n_elite=1, seed=0)
res = GeneticProgramming(cfg).fit(d.X_train, d.y_train, d.X_test, d.y_test)
print(res.expression, res.train_rmse, res.test_rmse)
res.history            # estatísticas de cada geração
```

## Testes

```bash
python -m pytest tests
```

Os testes cobrem os componentes (avaliação, funções protegidas, geração de árvores, profundidade máxima nos
operadores, seleção, leitura de dados). Não executam o laço evolutivo.
