"""Leitura das bases de dados (``<nome>-train.csv`` / ``<nome>-test.csv``).

Formato dos arquivos: CSV sem cabeçalho; a **última coluna** é a saída desejada (y).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Dataset:
    """Base de dados já separada em treino e teste."""

    name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    var_names: list = field(default_factory=list)
    standardized: bool = False

    @property
    def n_features(self):
        return self.X_train.shape[1]

    def __repr__(self):
        return (f"Dataset({self.name!r}, atributos={self.n_features}, "
                f"treino={len(self.y_train)}, teste={len(self.y_test)}, "
                f"padronizado={self.standardized})")


def _read_csv(path):
    df = pd.read_csv(path, header=None)
    # Se a primeira linha não for numérica, o arquivo tem cabeçalho.
    if df.iloc[0].map(lambda v: isinstance(v, str)).any():
        df = pd.read_csv(path, header=0)
    return df.to_numpy(dtype=float)


def available_datasets(data_dir="data"):
    """Nomes das bases que possuem os dois arquivos em ``data_dir``.

    Além do formato plano ``data/<nome>-*.csv``, aceita as bases organizadas
    em subpastas ``data/<nome>/<nome>-*.csv``.
    """
    data_dir = Path(data_dir)
    names = []
    for f in sorted(data_dir.glob("**/*-train.csv")):
        name = f.name[: -len("-train.csv")]
        if (f.parent / f"{name}-test.csv").exists():
            names.append(name)
    return sorted(set(names))


def load_dataset(name, data_dir="data", standardize=False):
    """Carrega uma base.

    Parameters
    ----------
    standardize : se ``True``, padroniza os **atributos** (z-score) usando média e desvio
        do conjunto de **treino** — as mesmas constantes são aplicadas ao teste, então não
        há vazamento de informação. A saída y nunca é alterada.
    """
    data_dir = Path(data_dir)
    train_files = list(data_dir.glob(f"**/{name}-train.csv"))
    test_files = list(data_dir.glob(f"**/{name}-test.csv"))
    if not train_files or not test_files:
        raise FileNotFoundError(f"Base {name!r} não encontrada em {data_dir}")
    tr = _read_csv(train_files[0])
    te = _read_csv(test_files[0])
    X_tr, y_tr = tr[:, :-1], tr[:, -1]
    X_te, y_te = te[:, :-1], te[:, -1]
    if standardize:
        mu = X_tr.mean(axis=0)
        sd = X_tr.std(axis=0)
        sd[sd == 0] = 1.0
        X_tr = (X_tr - mu) / sd
        X_te = (X_te - mu) / sd
    names = [f"x{i}" for i in range(X_tr.shape[1])]
    return Dataset(name, X_tr, y_tr, X_te, y_te, names, standardize)


def load_all(data_dir="data", standardize=()):
    """Carrega todas as bases disponíveis. ``standardize`` lista os nomes a padronizar."""
    return {n: load_dataset(n, data_dir, standardize=n in set(standardize))
            for n in available_datasets(data_dir)}
