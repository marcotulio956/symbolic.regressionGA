"""Métodos de seleção de pais: roleta, torneio e ε-lexicase.

Todos seguem a mesma interface (problema de *minimização*, fitness = RMSE):

    selector.prepare(fitness, errors)       # uma vez por geração
    idx = selector.select_many(m, nprng)    # índices (na população) de m pais
"""
from __future__ import annotations

import numpy as np


class RouletteSelection:
    """Seleção por roleta.

    Como o RMSE deve ser *minimizado*, a roleta usa o fitness transformado
    ``w = 1 / (1 + RMSE)``: quanto menor o erro, maior a fatia da roleta.
    """

    name = "roulette"

    def prepare(self, fitness, errors=None):
        self.cum = np.cumsum(1.0 / (1.0 + fitness))

    def select_many(self, m, nprng):
        r = nprng.random(m) * self.cum[-1]
        idx = np.searchsorted(self.cum, r, side="right")
        return np.minimum(idx, len(self.cum) - 1)


class TournamentSelection:
    """Seleção por torneio de tamanho ``k`` (sorteio com reposição)."""

    name = "tournament"

    def __init__(self, k=2):
        self.k = k

    def prepare(self, fitness, errors=None):
        self.fit = fitness

    def select_many(self, m, nprng):
        cand = nprng.integers(0, len(self.fit), size=(m, self.k))
        winners = np.argmin(self.fit[cand], axis=1)
        return cand[np.arange(m), winners]


class LexicaseSelection:
    """Seleção ε-lexicase (La Cava et al., 2019).

    Para cada pai: os casos de treino são embaralhados e, caso a caso, só ficam
    os candidatos com erro ``<= melhor_erro + ε``. Para quando sobra um único
    candidato; se os casos acabarem, sorteia-se um dos restantes.

    ``epsilon``:
      * ``"dynamic"`` – ε = MAD do erro entre os candidatos *atuais* (como no enunciado);
      * ``"static"``  – ε = MAD do erro na população inteira (calculado uma vez por geração);
      * ``"none"``    – lexicase clássico (ε = 0, só passa quem iguala o melhor erro).

    Usamos ``<=`` (e não ``<`` da Eq. 1) para que o melhor indivíduo sempre passe,
    inclusive quando MAD = 0.
    """

    name = "lexicase"

    def __init__(self, epsilon="dynamic"):
        if epsilon not in ("dynamic", "static", "none"):
            raise ValueError("epsilon deve ser 'dynamic', 'static' ou 'none'")
        self.epsilon = epsilon

    def prepare(self, fitness, errors):
        self.ET = np.ascontiguousarray(errors.T)  # (casos, indivíduos)
        self.n_cases, self.n = self.ET.shape
        self.all_idx = np.arange(self.n)
        if self.epsilon == "static":
            med = np.median(self.ET, axis=1, keepdims=True)
            self.mad = np.median(np.abs(self.ET - med), axis=1)

    def _select_one(self, nprng):
        cands = self.all_idx
        for c in nprng.permutation(self.n_cases):
            col = self.ET[c][cands]
            best = col.min()
            if self.epsilon == "dynamic":
                lam = np.median(np.abs(col - np.median(col)))
            elif self.epsilon == "static":
                lam = self.mad[c]
            else:
                lam = 0.0
            cands = cands[col <= best + lam]
            if cands.size == 1:
                return int(cands[0])
        return int(cands[nprng.integers(cands.size)])

    def select_many(self, m, nprng):
        return np.array([self._select_one(nprng) for _ in range(m)], dtype=int)


def make_selector(name, tournament_size=2, lexicase_epsilon="dynamic"):
    """Fábrica de seletores a partir do nome (``roulette``, ``tournament``, ``lexicase``)."""
    if name == "roulette":
        return RouletteSelection()
    if name == "tournament":
        return TournamentSelection(tournament_size)
    if name == "lexicase":
        return LexicaseSelection(lexicase_epsilon)
    raise ValueError(f"Seleção desconhecida: {name!r}")
