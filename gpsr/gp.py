"""Laço evolutivo de Programação Genética para regressão simbólica."""
from __future__ import annotations

import dataclasses
import random
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .operators import mutate, subtree_crossover
from .selection import make_selector
from .tree import (
    DEFAULT_FUNCTIONS,
    BIG,
    Individual,
    PrimitiveSet,
    ramped_half_and_half,
    rmse,
)


@dataclass
class GPConfig:
    """Todos os parâmetros do algoritmo (nada fica *hardcoded* no laço evolutivo)."""

    pop_size: int = 100
    generations: int = 100
    max_depth: int = 7                 # profundidade máxima (raiz = 1)
    init_depth: tuple = (2, 5)         # profundidades da população inicial (ramped half-and-half)
    pc: float = 0.9                    # probabilidade de cruzamento
    pm: float = 0.05                   # probabilidade de mutação (por filho)
    selection: str = "tournament"      # 'roulette' | 'tournament' | 'lexicase'
    tournament_size: int = 2
    lexicase_epsilon: str = "dynamic"  # 'dynamic' | 'static' | 'none'
    n_elite: int = 1                   # nº de indivíduos copiados direto (0 = sem elitismo)
    mutation: str = "subtree"          # 'subtree' | 'point' | 'mixed'
    mut_depth: int = 4                 # profundidade máx. da subárvore sorteada na mutação
    functions: tuple = DEFAULT_FUNCTIONS
    const_range: tuple = (-5.0, 5.0)
    const_prob: float = 0.3
    seed: int = 0

    def replace(self, **changes):
        return dataclasses.replace(self, **changes)

    def to_dict(self):
        return dataclasses.asdict(self)


@dataclass
class GPResult:
    """Resultado de uma execução."""

    config: GPConfig
    history: pd.DataFrame            # uma linha por geração (0 = população inicial)
    best: Individual                 # melhor indivíduo (por RMSE de treino) de toda a execução
    train_rmse: float
    test_rmse: float
    elapsed: float
    dataset: str = ""

    @property
    def expression(self):
        return str(self.best)


def _cmp(child, ref):
    """+1 se o filho é melhor (menor RMSE) que a referência, -1 se pior, 0 se empata."""
    tol = 1e-12 * max(1.0, abs(ref))
    if child < ref - tol:
        return 1
    if child > ref + tol:
        return -1
    return 0


class GeneticProgramming:
    """Programação Genética (população geracional com elitismo opcional)."""

    def __init__(self, config: GPConfig | None = None, **kwargs):
        self.cfg = config if config is not None else GPConfig()
        if kwargs:
            self.cfg = self.cfg.replace(**kwargs)

    # ------------------------------------------------------------------ #
    def fit(self, X, y, X_test=None, y_test=None, var_names=None, dataset=""):
        cfg = self.cfg
        t0 = time.time()
        rng = random.Random(cfg.seed)
        nprng = np.random.default_rng(cfg.seed)

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        cols = np.ascontiguousarray(X.T)
        has_test = X_test is not None and y_test is not None
        if has_test:
            cols_te = np.ascontiguousarray(np.asarray(X_test, dtype=float).T)
            y_te = np.asarray(y_test, dtype=float)

        pset = PrimitiveSet(X.shape[1], cfg.functions, cfg.const_range, cfg.const_prob,
                            var_names=var_names)
        selector = make_selector(cfg.selection, cfg.tournament_size, cfg.lexicase_epsilon)
        need_errors = cfg.selection == "lexicase"

        rows = []
        with np.errstate(all="ignore"):
            # ---- população inicial ----------------------------------------------------
            pop = ramped_half_and_half(pset, rng, cfg.pop_size, *cfg.init_depth)
            for ind in pop:
                ind.evaluate(cols, y)

            best_ever = min(pop, key=lambda i: i.fitness).copy()
            best_test = self._test_rmse(best_ever, cols_te, y_te) if has_test else np.nan
            rows.append(self._record(0, pop, best_ever, best_test, {}))

            # ---- gerações -------------------------------------------------------------
            for g in range(1, cfg.generations + 1):
                pop, stats = self._next_generation(pop, selector, need_errors, pset, rng,
                                                   nprng, cols, y)
                gen_best = min(pop, key=lambda i: i.fitness)
                if gen_best.fitness < best_ever.fitness:
                    best_ever = gen_best.copy()
                    if has_test:
                        best_test = self._test_rmse(best_ever, cols_te, y_te)
                rows.append(self._record(g, pop, best_ever, best_test, stats))

        history = pd.DataFrame(rows)
        return GPResult(
            config=cfg,
            history=history,
            best=best_ever,
            train_rmse=best_ever.fitness,
            test_rmse=float(best_test),
            elapsed=time.time() - t0,
            dataset=dataset,
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _test_rmse(ind, cols_te, y_te):
        from .tree import predict_genes

        return rmse(predict_genes(ind.genes, cols_te, y_te.shape[0]), y_te)

    def _next_generation(self, pop, selector, need_errors, pset, rng, nprng, cols, y):
        cfg = self.cfg
        n = len(pop)
        fit = np.fromiter((i.fitness for i in pop), dtype=float, count=n)
        errors = np.vstack([i.errors for i in pop]) if need_errors else None
        n_elite = min(cfg.n_elite, n)
        order = np.argsort(fit, kind="stable")
        elites = [pop[i] for i in order[:n_elite]]

        selector.prepare(fit, errors)
        n_children = n - n_elite
        n_pairs = (n_children + 1) // 2
        parents = selector.select_many(2 * n_pairs, nprng).reshape(n_pairs, 2)

        st = dict(xo_n=0, xo_failed=0, xo_better=0, xo_worse=0, xo_equal=0, xo_better_best=0,
                  mut_n=0, mut_better=0, mut_worse=0, mut_equal=0)
        children = []
        for a, b in parents:
            p1, p2 = pop[a], pop[b]
            if rng.random() < cfg.pc:
                c1, c2, ok = subtree_crossover(p1, p2, rng, cfg.max_depth)
                if ok:
                    c1.evaluate(cols, y)
                    c2.evaluate(cols, y)
                    ref_mean = 0.5 * (p1.fitness + p2.fitness)
                    ref_best = min(p1.fitness, p2.fitness)
                    for c in (c1, c2):
                        st["xo_n"] += 1
                        r = _cmp(c.fitness, ref_mean)
                        st["xo_better" if r > 0 else "xo_worse" if r < 0 else "xo_equal"] += 1
                        if _cmp(c.fitness, ref_best) > 0:
                            st["xo_better_best"] += 1
                else:
                    st["xo_failed"] += 1
            else:
                c1, c2 = p1.copy(), p2.copy()
            for k, c in enumerate((c1, c2)):
                if rng.random() < cfg.pm:
                    m = mutate(c, pset, rng, cfg.mutation, cfg.max_depth, cfg.mut_depth)
                    m.evaluate(cols, y)
                    st["mut_n"] += 1
                    r = _cmp(m.fitness, c.fitness)
                    st["mut_better" if r > 0 else "mut_worse" if r < 0 else "mut_equal"] += 1
                    c = m
                children.append(c)
        return elites + children[:n_children], st

    # ------------------------------------------------------------------ #
    @staticmethod
    def _record(g, pop, best_ever, best_test, stats):
        fit = np.fromiter((i.fitness for i in pop), dtype=float, count=len(pop))
        sizes = np.fromiter((i.size for i in pop), dtype=float, count=len(pop))
        depths = np.fromiter((i.depth for i in pop), dtype=float, count=len(pop))
        n_unique = len({i.key for i in pop})
        row = dict(
            gen=g,
            best=fit.min(),
            worst=fit.max(),
            mean=fit.mean(),
            median=float(np.median(fit)),
            std=fit.std(),
            best_ever=best_ever.fitness,
            test_best_ever=best_test,
            n_unique=n_unique,
            n_repeated=len(pop) - n_unique,
            avg_size=sizes.mean(),
            avg_depth=depths.mean(),
            best_ever_size=best_ever.size,
            best_ever_depth=best_ever.depth,
        )
        for k in ("xo_n", "xo_failed", "xo_better", "xo_worse", "xo_equal", "xo_better_best",
                  "mut_n", "mut_better", "mut_worse", "mut_equal"):
            row[k] = stats.get(k, 0)
        return row
