"""Gráficos dos experimentos (matplotlib).

Paleta categórica fixa (a ordem é atribuída por configuração, nunca reciclada):
azul, laranja, verde-água, amarelo, magenta, verde, violeta, vermelho.
Se houver mais de 8 séries o gráfico recusa — reduza o número de configurações.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6",
               "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
NEUTRAL = "#c3c2b7"


def set_style():
    """Aplica o estilo (fundo claro, grade discreta, linhas finas)."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK2,
        "legend.frameon": False, "legend.fontsize": 9,
        "lines.linewidth": 1.6, "figure.dpi": 110,
        "font.family": "sans-serif",
    })


set_style()


def _colors(n):
    if n > len(CATEGORICAL):
        raise ValueError(f"No máximo {len(CATEGORICAL)} séries por gráfico (recebi {n}); "
                         "agrupe/filtre as configurações.")
    return CATEGORICAL[:n]


def _ax(ax, figsize):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    return fig, ax


# --------------------------------------------------------------------------- #
def plot_curves(results, stat="best_ever", ax=None, title=None, ylabel=None, logy=False,
                band=True, figsize=(6.4, 3.8)):
    """Evolução média (entre execuções) de uma estatística por geração, com faixa ±1 desvio.

    ``stat``: best, mean, median, worst, best_ever, test_best_ever, n_repeated, n_unique,
    avg_size, avg_depth, best_ever_size, ...
    """
    fig, ax = _ax(ax, figsize)
    for color, (name, rep) in zip(_colors(len(results)), results.items()):
        c = rep.curve(stat)
        ax.plot(c.index, c["mean"], color=color, label=name)
        if band and len(rep.runs) > 1:
            lo, hi = c["mean"] - c["std"], c["mean"] + c["std"]
            if logy:
                lo = np.maximum(lo, c["mean"] * 0.05)
            ax.fill_between(c.index, lo, hi, color=color, alpha=0.12, linewidth=0)
    ax.set_xlabel("geração")
    ax.set_ylabel(ylabel or stat)
    if logy:
        ax.set_yscale("log")
    if title:
        ax.set_title(title)
    if len(results) > 1:
        ax.legend()
    return fig


def plot_population_stats(rep, ax=None, title=None, logy=True, figsize=(6.4, 3.8)):
    """Melhor, média e pior fitness da população, geração a geração (média entre execuções)."""
    fig, ax = _ax(ax, figsize)
    for color, (stat, label) in zip(_colors(3), [("best", "melhor"), ("mean", "média"),
                                                 ("worst", "pior")]):
        c = rep.curve(stat)
        ax.plot(c.index, c["mean"], color=color, label=label)
    ax.set_xlabel("geração")
    ax.set_ylabel("RMSE (treino)")
    if logy:
        ax.set_yscale("log")
    if title:
        ax.set_title(title)
    ax.legend()
    return fig


def plot_boxes(results, metric="test_rmse", ax=None, title=None, logy=False,
               figsize=None):
    """Distribuição (entre execuções) de uma métrica final, uma caixa por configuração."""
    n = len(results)
    cols = _colors(n)
    fig, ax = _ax(ax, figsize or (max(4.5, 1.3 * n + 2), 3.8))
    data = [rep.metric(metric) for rep in results.values()]
    bp = ax.boxplot(data, patch_artist=True, widths=0.5, showfliers=False,
                    medianprops=dict(color=INK, linewidth=1.6),
                    whiskerprops=dict(color=MUTED, linewidth=1.0),
                    capprops=dict(color=MUTED, linewidth=1.0))
    rng = np.random.default_rng(0)
    for i, (box, c, d) in enumerate(zip(bp["boxes"], cols, data), start=1):
        box.set(facecolor=c, alpha=0.25, edgecolor=c, linewidth=1.2)
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(d)), d, s=22, color=c, edgecolor=SURFACE,
                   linewidth=0.8, zorder=3)
    ax.set_xticks(range(1, n + 1))
    ax.set_xticklabels(list(results), rotation=20 if n > 3 else 0, ha="right" if n > 3 else "center")
    ax.set_ylabel(metric)
    ax.grid(axis="x", visible=False)
    if logy:
        ax.set_yscale("log")
    if title:
        ax.set_title(title)
    return fig


def plot_train_test(results, ax=None, title=None, logy=False, figsize=None):
    """Média ± desvio do RMSE de treino e de teste por configuração (detecção de overfitting)."""
    n = len(results)
    fig, ax = _ax(ax, figsize or (max(4.5, 1.3 * n + 2), 3.8))
    x = np.arange(n)
    w = 0.38
    for off, metric, color, lab in [(-w / 2 - 0.01, "train_rmse", CATEGORICAL[0], "treino"),
                                    (w / 2 + 0.01, "test_rmse", CATEGORICAL[1], "teste")]:
        m = [rep.metric(metric).mean() for rep in results.values()]
        s = [rep.metric(metric).std(ddof=1) if len(rep.runs) > 1 else 0 for rep in results.values()]
        ax.bar(x + off, m, w, color=color, label=lab, linewidth=0)
        ax.errorbar(x + off, m, yerr=s, fmt="none", ecolor=INK2, elinewidth=1.0, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(list(results), rotation=20 if n > 3 else 0, ha="right" if n > 3 else "center")
    ax.set_ylabel("RMSE")
    ax.grid(axis="x", visible=False)
    if logy:
        ax.set_yscale("log")
    ax.legend()
    if title:
        ax.set_title(title)
    return fig


def plot_operator_effects(results, kind="crossover", ax=None, title=None, figsize=None):
    """Barras empilhadas (100%): filhos melhores / iguais / piores (crossover) ou mutações
    que melhoram / mantêm / pioram o indivíduo (mutação)."""
    from .experiments import operator_effects

    df = operator_effects(results)
    if kind == "crossover":
        parts = [("xo_melhor_que_media_pais_%", "melhor que a média dos pais", CATEGORICAL[0]),
                 ("xo_igual_%", "igual", NEUTRAL),
                 ("xo_pior_que_media_pais_%", "pior que a média dos pais", CATEGORICAL[1])]
    else:
        parts = [("mut_melhor_%", "melhora o indivíduo", CATEGORICAL[0]),
                 ("mut_igual_%", "igual", NEUTRAL),
                 ("mut_pior_%", "piora o indivíduo", CATEGORICAL[1])]
    n = len(df)
    fig, ax = _ax(ax, figsize or (6.4, max(2.2, 0.55 * n + 1.4)))
    left = np.zeros(n)
    y = np.arange(n)
    for col, lab, color in parts:
        v = df[col].fillna(0).to_numpy()
        ax.barh(y, v, left=left, color=color, label=lab, height=0.6, edgecolor=SURFACE,
                linewidth=2)
        left += v
    ax.set_yticks(y)
    ax.set_yticklabels(df.index)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("% do total")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3)
    ax.set_title(title or ("Efeito do crossover" if kind == "crossover" else "Efeito da mutação"))
    return fig


def plot_prediction(result, data, figsize=(8.6, 3.8)):
    """Valor previsto × valor real (treino e teste) para o melhor indivíduo de uma execução."""
    ind = result.best
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, (X, y, lab, c) in zip(axes, [(data.X_train, data.y_train, "treino", CATEGORICAL[0]),
                                         (data.X_test, data.y_test, "teste", CATEGORICAL[1])]):
        p = ind.predict(X)
        ax.scatter(y, p, s=20, color=c, edgecolor=SURFACE, linewidth=0.6, zorder=3)
        lo, hi = min(y.min(), p.min()), max(y.max(), p.max())
        ax.plot([lo, hi], [lo, hi], color=MUTED, linewidth=1.0, linestyle="--")
        ax.set_xlabel("y real")
        ax.set_ylabel("y previsto")
        ax.set_title(f"{lab} (RMSE = {np.sqrt(np.mean((p - y) ** 2)):.4g})")
    fig.tight_layout()
    return fig
