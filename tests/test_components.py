"""Testes unitários dos componentes (não executam nenhum treinamento / laço evolutivo)."""

""" AI DISCLAIMER: O uso LLM foi feito para auxiliar na criação dos testes unitarios e consequentes refactors em gpsr/"""

import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gpsr import (GPConfig, Individual, PrimitiveSet, generate_tree, load_dataset, make_selector,
                  mutate, ramped_half_and_half, subtree_crossover, to_infix, tree_depth)
from gpsr.tree import FUNCTIONS, Constant, Variable, node_depths, subtree_end

F = FUNCTIONS


def _ind(*toks):
    return Individual(list(toks))


def test_evaluation_prefix_order():
    X = np.array([[3.0, 1.0], [5.0, 2.0]])
    ind = _ind(F["sub"], Variable(0), Variable(1))          # x0 - x1
    assert np.allclose(ind.predict(X), [2.0, 3.0])
    ind = _ind(F["mul"], Variable(0), F["add"], Variable(1), Constant(2.0))  # x0*(x1+2)
    assert np.allclose(ind.predict(X), [9.0, 20.0])
    assert str(ind) == "(x0 * (x1 + 2))"


def test_constant_only_tree_broadcasts():
    ind = _ind(Constant(4.0))
    assert ind.predict(np.zeros((5, 2))).shape == (5,)


def test_protected_functions_are_finite():
    X = np.array([[0.0, -1.0, 1e-9]])
    for name in F:
        f = F[name]
        genes = [f] + [Variable(0)] * f.arity
        out = _ind(*genes).predict(X.T if False else np.array([[0.0], [-1.0], [1e-9]]))
        assert np.all(np.isfinite(out))
    div = _ind(F["div"], Variable(0), Variable(1))
    assert np.allclose(div.predict(np.array([[1.0, 0.0]])), [1.0])


def test_structure_helpers():
    g = [F["add"], F["mul"], Variable(0), Constant(1.0), Variable(1)]
    assert node_depths(g) == [1, 2, 3, 3, 2]
    assert tree_depth(g) == 3
    assert subtree_end(g, 1) == 4 and subtree_end(g, 0) == 5


@pytest.mark.parametrize("d", [1, 2, 4, 7])
def test_full_and_grow_depth(d):
    rng = random.Random(0)
    pset = PrimitiveSet(2)
    for _ in range(50):
        assert tree_depth(generate_tree(pset, rng, d, "full")) == d
        assert tree_depth(generate_tree(pset, rng, d, "grow")) <= d


def test_ramped_half_and_half():
    rng = random.Random(1)
    pset = PrimitiveSet(2)
    pop = ramped_half_and_half(pset, rng, 100, 2, 5)
    depths = {i.depth for i in pop}
    assert len(pop) == 100 and max(depths) <= 5 and {2, 5} <= depths
    for ind in pop:
        assert subtree_end(ind.genes, 0) == ind.size  # prefixa bem formada


def test_crossover_and_mutation_respect_max_depth():
    rng = random.Random(2)
    pset = PrimitiveSet(3)
    pop = ramped_half_and_half(pset, rng, 60, 2, 7)
    for _ in range(300):
        a, b = rng.sample(pop, 2)
        c1, c2, _ok = subtree_crossover(a, b, rng, max_depth=7)
        for c in (c1, c2):
            assert c.depth <= 7 and subtree_end(c.genes, 0) == c.size
        for kind in ("subtree", "point", "mixed"):
            m = mutate(a, pset, rng, kind, max_depth=7)
            assert m.depth <= 7 and subtree_end(m.genes, 0) == m.size


def test_crossover_does_not_mutate_parents():
    rng = random.Random(3)
    pset = PrimitiveSet(2)
    a, b = ramped_half_and_half(pset, rng, 2, 3, 4)
    ga, gb = list(a.genes), list(b.genes)
    for _ in range(50):
        subtree_crossover(a, b, rng, 7)
        mutate(a, pset, rng, "subtree", 7)
    assert a.genes == ga and b.genes == gb


def test_selection_pressure():
    nprng = np.random.default_rng(0)
    fit = np.linspace(0.1, 10.0, 50)
    errors = np.abs(np.random.default_rng(1).normal(size=(50, 30)))
    for name in ("roulette", "tournament"):
        sel = make_selector(name, tournament_size=3)
        sel.prepare(fit, errors)
        idx = sel.select_many(20000, nprng)
        assert idx.min() >= 0 and idx.max() < 50
        assert fit[idx].mean() < fit.mean()  # favorece fitness (RMSE) baixo


def test_lexicase_picks_dominant_individual():
    nprng = np.random.default_rng(0)
    errors = np.random.default_rng(1).uniform(1, 2, size=(20, 15))
    errors[7] = 0.0                       # indivíduo 7 é o melhor em todos os casos
    fit = np.sqrt((errors ** 2).mean(axis=1))
    for eps in ("dynamic", "static", "none"):
        sel = make_selector("lexicase", lexicase_epsilon=eps)
        sel.prepare(fit, errors)
        assert set(sel.select_many(50, nprng)) == {7}


def test_lexicase_all_equal_does_not_crash():
    errors = np.ones((10, 5))
    sel = make_selector("lexicase")
    sel.prepare(np.ones(10), errors)
    idx = sel.select_many(20, np.random.default_rng(0))
    assert idx.shape == (20,) and idx.max() < 10


def test_config_replace_is_immutable():
    c = GPConfig()
    c2 = c.replace(pc=0.6, pm=0.3)
    assert (c.pc, c.pm) == (0.9, 0.05) and (c2.pc, c2.pm) == (0.6, 0.3)


def test_load_synth1_shapes():
    data_dir = Path(__file__).resolve().parents[1] / "data"
    if not (data_dir / "synth1-train.csv").exists():
        pytest.skip("synth1 não disponível")
    d = load_dataset("synth1", data_dir)
    assert d.X_train.shape == (60, 2) and d.X_test.shape == (600, 2)
    d2 = load_dataset("synth1", data_dir, standardize=True)
    assert np.allclose(d2.X_train.mean(axis=0), 0, atol=1e-9)
    assert np.array_equal(d.y_train, d2.y_train)
