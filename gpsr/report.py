"""Relatório padrão de uma comparação de configurações (tabelas, testes e gráficos)."""
from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np

from . import plots as P
from .experiments import kruskal_pvalue, pairwise_tests, summary_table

try:
    from IPython.display import Markdown, display
except Exception:  # pragma: no cover
    display = None
    Markdown = None

DEFAULT_CURVES = (
    ("best_ever", "Melhor RMSE acumulado (treino)", True),
    ("test_best_ever", "RMSE de teste do melhor acumulado", True),
    ("mean", "Fitness média da população (RMSE)", True),
    ("n_repeated", "Nº de indivíduos repetidos", False),
)


def _show(obj):
    if display is not None:
        display(obj)
    else:  # pragma: no cover
        print(obj)


def _title(text):
    if display is not None:
        display(Markdown(text))
    else:  # pragma: no cover
        print(text)


def curve_grid(results, curves=DEFAULT_CURVES, title=None, ncols=2):
    """Grade de gráficos de evolução (um painel por estatística)."""
    n = len(curves)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, min(ncols, n), figsize=(6.2 * min(ncols, n), 3.6 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (stat, label, logy) in zip(axes, curves):
        P.plot_curves(results, stat, ax=ax, title=label, logy=logy, ylabel="")
    for ax in axes[n:]:
        ax.set_visible(False)
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="semibold", color=P.INK)
    fig.tight_layout()
    return fig


def report(results, title="", curves=DEFAULT_CURVES, boxes=("train_rmse", "test_rmse"),
           tests=True, logy_boxes=True):
    """Mostra: tabela-resumo, testes estatísticos, caixas (treino/teste) e curvas de evolução."""
    if title:
        _title(f"#### {title}")
    _show(summary_table(results).round(4))

    if tests and len(results) > 1:
        for m in boxes:
            print(f"Kruskal-Wallis ({m}): p = {kruskal_pvalue(results, m):.4g}")
            _show(pairwise_tests(results, m).round(4))

    if boxes:
        fig, axes = plt.subplots(1, len(boxes), figsize=(max(4.8, 1.3 * len(results) + 2.5)
                                                          * len(boxes), 3.9))
        axes = np.atleast_1d(axes).ravel()
        for ax, m in zip(axes, boxes):
            P.plot_boxes(results, m, ax=ax, title=m, logy=logy_boxes)
        fig.tight_layout()
        plt.show()

    if curves:
        curve_grid(results, curves)
        plt.show()
