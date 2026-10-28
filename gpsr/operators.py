"""Operadores genéticos: cruzamento de subárvores e mutações.

Todos os operadores respeitam a profundidade máxima ``max_depth``: se um filho
violaria o limite, o sorteio é repetido (até ``tries`` vezes) e, se ainda assim
não der certo, os pais são devolvidos inalterados (e o cruzamento é marcado como
"falho").
"""
from __future__ import annotations

from .tree import (
    Constant,
    Individual,
    generate_tree,
    node_depths,
    subtree_end,
    tree_depth,
)


def _pick_node(genes, internal_idx, rng, p_internal):
    """Escolhe um ponto de corte: com prob. ``p_internal`` um nó interno (estilo Koza)."""
    if internal_idx and rng.random() < p_internal:
        return rng.choice(internal_idx)
    return rng.randrange(len(genes))


def subtree_crossover(p1, p2, rng, max_depth=7, tries=10, p_internal=0.9):
    """Cruzamento de subárvores.

    Sorteia um ponto em cada pai e troca as subárvores. Retorna ``(filho1, filho2, ok)``;
    ``ok`` é ``False`` quando nenhum par de pontos respeitou ``max_depth`` e os filhos
    são apenas cópias dos pais.
    """
    g1, g2 = p1.genes, p2.genes
    dep1, dep2 = node_depths(g1), node_depths(g2)
    in1 = [i for i, t in enumerate(g1) if t.arity]
    in2 = [i for i, t in enumerate(g2) if t.arity]
    for _ in range(tries):
        i = _pick_node(g1, in1, rng, p_internal)
        j = _pick_node(g2, in2, rng, p_internal)
        e1 = subtree_end(g1, i)
        e2 = subtree_end(g2, j)
        h1 = tree_depth(g1[i:e1])  # altura da subárvore de p1 que vai para p2
        h2 = tree_depth(g2[j:e2])
        if dep1[i] - 1 + h2 <= max_depth and dep2[j] - 1 + h1 <= max_depth:
            c1 = g1[:i] + g2[j:e2] + g1[e1:]
            c2 = g2[:j] + g1[i:e1] + g2[e2:]
            return Individual(c1), Individual(c2), True
    return p1.copy(), p2.copy(), False


def subtree_mutation(ind, pset, rng, max_depth=7, mut_depth=4):
    """Troca uma subárvore sorteada por outra gerada aleatoriamente (método *grow*)."""
    g = ind.genes
    dep = node_depths(g)
    i = rng.randrange(len(g))
    e = subtree_end(g, i)
    allowed = max(1, max_depth - dep[i] + 1)
    d = min(mut_depth, allowed)
    new = generate_tree(pset, rng, d, "grow")
    return Individual(g[:i] + new + g[e:])


def point_mutation(ind, pset, rng, sigma=None):
    """Mutação pontual: altera um único nó mantendo a estrutura da árvore.

    * função -> outra função de mesma aridade;
    * constante -> perturbação gaussiana (ou novo terminal, com prob. 0.5);
    * variável -> outro terminal sorteado.
    """
    g = list(ind.genes)
    i = rng.randrange(len(g))
    t = g[i]
    if t.arity:
        g[i] = rng.choice(pset.by_arity[t.arity])
    elif t.__class__ is Constant and rng.random() < 0.5:
        if sigma is None:
            lo, hi = pset.const_range
            sigma = 0.1 * (hi - lo)
        g[i] = Constant(t.value + rng.gauss(0.0, sigma))
    else:
        g[i] = pset.random_terminal(rng)
    return Individual(g)


def mutate(ind, pset, rng, kind="subtree", max_depth=7, mut_depth=4):
    """Aplica a mutação escolhida (``subtree``, ``point`` ou ``mixed``)."""
    if kind == "mixed":
        kind = "subtree" if rng.random() < 0.5 else "point"
    if kind == "subtree":
        return subtree_mutation(ind, pset, rng, max_depth, mut_depth)
    if kind == "point":
        return point_mutation(ind, pset, rng)
    raise ValueError(f"Mutação desconhecida: {kind!r}")
