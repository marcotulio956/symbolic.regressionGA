"""Infraestrutura experimental: repetições (em paralelo), tabelas, testes e métricas de bloat.

Fluxo típico::

    results = compare_configs({"A": cfgA, "B": cfgB}, data, n_runs=30)   # dict nome -> RepeatedResult
    summary_table(results)            # média/desvio/mediana de RMSE de treino e teste
    operator_effects(results)         # crossover/mutação: filhos melhores/piores que os pais
    pairwise_tests(results)           # Mann-Whitney U com correção de Holm
"""
from __future__ import annotations

import itertools
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from .data import Dataset
from .gp import GeneticProgramming, GPConfig, GPResult

try:  # barra de progresso opcional
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


# --------------------------------------------------------------------------- #
# Execução de repetições
# --------------------------------------------------------------------------- #
def _run_job(job):
    cfg, X, y, Xt, yt, names, dname = job
    return GeneticProgramming(cfg).fit(X, y, Xt, yt, var_names=names, dataset=dname)


def _execute(jobs, n_jobs, desc=""):
    n_jobs = max(1, min(n_jobs or os.cpu_count() or 1, len(jobs)))
    if n_jobs == 1:
        it = map(_run_job, jobs)
        if tqdm is not None:
            it = tqdm(it, total=len(jobs), desc=desc, leave=False)
        return list(it)
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        it = pool.map(_run_job, jobs, chunksize=1)
        if tqdm is not None:
            it = tqdm(it, total=len(jobs), desc=desc, leave=False)
        return list(it)


@dataclass
class RepeatedResult:
    """Resultados de ``n_runs`` execuções independentes de uma mesma configuração."""

    name: str
    config: GPConfig
    runs: list
    dataset: str = ""
    _hist: pd.DataFrame = field(default=None, repr=False)

    # -- por execução ------------------------------------------------------------
    @property
    def final(self) -> pd.DataFrame:
        """Uma linha por execução: RMSE (treino/teste) do melhor indivíduo e estrutura."""
        rows = []
        for r in self.runs:
            rows.append(dict(
                seed=r.config.seed,
                train_rmse=r.train_rmse,
                test_rmse=r.test_rmse,
                gap=r.test_rmse - r.train_rmse,
                size=r.best.size,
                depth=r.best.depth,
                elapsed=r.elapsed,
                expression=r.expression,
            ))
        return pd.DataFrame(rows)

    def metric(self, name):
        return self.final[name].to_numpy()

    # -- por geração -------------------------------------------------------------
    @property
    def history(self) -> pd.DataFrame:
        """Estatísticas por geração de todas as execuções (coluna ``run`` identifica a execução)."""
        if self._hist is None:
            self._hist = pd.concat(
                [r.history.assign(run=i) for i, r in enumerate(self.runs)], ignore_index=True)
        return self._hist

    def curve(self, stat) -> pd.DataFrame:
        """Média e desvio-padrão (entre execuções) de ``stat`` em cada geração."""
        return self.history.groupby("gen")[stat].agg(["mean", "std"]).fillna(0.0)

    @property
    def best_run(self) -> GPResult:
        return min(self.runs, key=lambda r: r.train_rmse)


def run_repetitions(config, data: Dataset, n_runs=30, base_seed=0, n_jobs=None, name=None):
    """Executa ``n_runs`` repetições com sementes ``base_seed, base_seed+1, ...``."""
    name = name or "config"
    return compare_configs({name: config}, data, n_runs, base_seed, n_jobs)[name]


def compare_configs(configs: dict, data: Dataset, n_runs=30, base_seed=0, n_jobs=None,
                    verbose=True):
    """Roda cada configuração ``n_runs`` vezes (mesmas sementes para todas) e devolve
    ``{nome: RepeatedResult}``.

    Usar as **mesmas sementes** em todas as configurações faz com que a população inicial
    da repetição *i* seja idêntica entre elas (quando ``pop_size`` e inicialização são
    iguais), reduzindo a variância das comparações.
    """
    jobs, owner = [], []
    for cname, cfg in configs.items():
        for i in range(n_runs):
            jobs.append((cfg.replace(seed=base_seed + i), data.X_train, data.y_train,
                         data.X_test, data.y_test, data.var_names, data.name))
            owner.append(cname)
    t0 = time.time()
    out = _execute(jobs, n_jobs, desc=f"{data.name}: {len(configs)} config × {n_runs} runs")
    grouped = {c: [] for c in configs}
    for c, res in zip(owner, out):
        grouped[c].append(res)
    if verbose:
        print(f"[{data.name}] {len(jobs)} execuções em {time.time() - t0:.1f}s")
    return {c: RepeatedResult(c, configs[c], runs, data.name) for c, runs in grouped.items()}


def sweep(base: GPConfig, param: str, values, data: Dataset, n_runs=30, base_seed=0,
          n_jobs=None, verbose=True):
    """Varia **um** parâmetro mantendo os demais fixos. Nomes dos resultados: ``param=valor``."""
    configs = {f"{param}={v}": base.replace(**{param: v}) for v in values}
    return compare_configs(configs, data, n_runs, base_seed, n_jobs, verbose)


# --------------------------------------------------------------------------- #
# Tabelas
# --------------------------------------------------------------------------- #
def summary_table(results: dict) -> pd.DataFrame:
    """Média, desvio-padrão e mediana do RMSE de treino e de teste (melhor indivíduo de cada
    execução), tamanho/profundidade médios do melhor indivíduo e tempo médio por execução."""
    rows = {}
    for name, rep in results.items():
        f = rep.final
        rows[name] = {
            "train_mean": f.train_rmse.mean(), "train_std": f.train_rmse.std(ddof=1),
            "train_median": f.train_rmse.median(),
            "test_mean": f.test_rmse.mean(), "test_std": f.test_rmse.std(ddof=1),
            "test_median": f.test_rmse.median(),
            "gap_mean": f.gap.mean(),
            "best_size": f["size"].mean(), "best_depth": f.depth.mean(),
            "time_s": f.elapsed.mean(), "n_runs": len(f),
        }
    return pd.DataFrame(rows).T


def checkpoint_table(results: dict, gens, stat="best_ever") -> pd.DataFrame:
    """Valor médio ± desvio de ``stat`` em gerações selecionadas (texto ``média ± desvio``).

    Útil para decidir o número de gerações: uma única execução com ``generations = G_max``
    contém a trajetória de todas as gerações menores.
    """
    rows = {}
    for name, rep in results.items():
        c = rep.curve(stat)
        rows[name] = {f"g={g}": f"{c.loc[g, 'mean']:.4g} ± {c.loc[g, 'std']:.2g}"
                      for g in gens if g in c.index}
    return pd.DataFrame(rows).T


def operator_effects(results: dict) -> pd.DataFrame:
    """Contribuição dos operadores (somando todas as gerações e execuções).

    * **Crossover**: cada filho é comparado com a **média** do RMSE dos dois pais (como pede
      o enunciado) e também com o **melhor** pai.
    * **Mutação**: cada indivíduo mutado é comparado com o indivíduo antes da mutação.
    * Valores em porcentagem do total de filhos / mutações avaliados.
    """
    rows = {}
    for name, rep in results.items():
        h = rep.history
        h = h[h.gen >= 1]
        s = h[["xo_n", "xo_failed", "xo_better", "xo_equal", "xo_worse", "xo_better_best",
               "mut_n", "mut_better", "mut_equal", "mut_worse"]].sum()
        pct = lambda a, b: 100.0 * a / b if b else np.nan  # noqa: E731
        rows[name] = {
            "xo_filhos": int(s.xo_n),
            "xo_melhor_que_media_pais_%": pct(s.xo_better, s.xo_n),
            "xo_igual_%": pct(s.xo_equal, s.xo_n),
            "xo_pior_que_media_pais_%": pct(s.xo_worse, s.xo_n),
            "xo_melhor_que_melhor_pai_%": pct(s.xo_better_best, s.xo_n),
            "xo_falhas_%": pct(s.xo_failed, s.xo_failed + s.xo_n / 2),
            "mut_indivs": int(s.mut_n),
            "mut_melhor_%": pct(s.mut_better, s.mut_n),
            "mut_igual_%": pct(s.mut_equal, s.mut_n),
            "mut_pior_%": pct(s.mut_worse, s.mut_n),
        }
    return pd.DataFrame(rows).T


def bloat_table(results: dict) -> pd.DataFrame:
    """Medidas simples de *bloat* (crescimento de tamanho sem ganho de fitness).

    * ``size_growth``: tamanho médio da população na última geração / na geração 0;
    * ``final_avg_size`` / ``final_avg_depth``: tamanho e profundidade médios finais;
    * ``final_best_size``: tamanho do melhor indivíduo encontrado;
    * ``fitness_gain``: RMSE do melhor da geração 0 / RMSE do melhor final (>1 = melhorou);
    * ``size_vs_fitness``: ``size_growth / fitness_gain`` – valores altos indicam que o tamanho
      cresceu mais do que a qualidade (sinal de bloat).
    """
    rows = {}
    for name, rep in results.items():
        h = rep.history
        g0, g1 = h[h.gen == h.gen.min()], h[h.gen == h.gen.max()]
        size_growth = (g1.avg_size.to_numpy() / g0.avg_size.to_numpy()).mean()
        gain = (g0.best_ever.to_numpy() / np.maximum(g1.best_ever.to_numpy(), 1e-12)).mean()
        rows[name] = {
            "size_growth": size_growth,
            "final_avg_size": g1.avg_size.mean(),
            "final_avg_depth": g1.avg_depth.mean(),
            "final_best_size": g1.best_ever_size.mean(),
            "fitness_gain": gain,
            "size_vs_fitness": size_growth / gain,
        }
    return pd.DataFrame(rows).T


# --------------------------------------------------------------------------- #
# Estatística
# --------------------------------------------------------------------------- #
def _holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def pairwise_tests(results: dict, metric="test_rmse") -> pd.DataFrame:
    """Teste de Mann-Whitney U (bilateral) entre todos os pares de configurações, com p-valor
    ajustado por Holm. Não assume normalidade (RMSE costuma ter cauda longa)."""
    names = list(results)
    rows = []
    for a, b in itertools.combinations(names, 2):
        x, y = results[a].metric(metric), results[b].metric(metric)
        try:
            p = stats.mannwhitneyu(x, y, alternative="two-sided").pvalue
        except ValueError:  # amostras idênticas
            p = 1.0
        rows.append(dict(a=a, b=b, mediana_a=np.median(x), mediana_b=np.median(y), p=p))
    df = pd.DataFrame(rows)
    if len(df):
        df["p_holm"] = _holm(df.p)
        df["signif_5%"] = df.p_holm < 0.05
    return df


def kruskal_pvalue(results: dict, metric="test_rmse") -> float:
    """p-valor do teste de Kruskal-Wallis entre todas as configurações."""
    samples = [rep.metric(metric) for rep in results.values()]
    try:
        return float(stats.kruskal(*samples).pvalue)
    except ValueError:
        return 1.0


def best_config(results: dict, metric="train_rmse", agg="mean") -> str:
    """Nome da configuração com menor ``agg`` (mean|median) de ``metric``.

    Por padrão usa o RMSE de **treino**, para que a escolha de parâmetros não use o conjunto
    de teste.
    """
    f = (np.mean if agg == "mean" else np.median)
    return min(results, key=lambda k: f(results[k].metric(metric)))
