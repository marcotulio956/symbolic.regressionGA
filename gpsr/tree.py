"""Representação de indivíduos (árvores), conjuntos de primitivas e avaliação.

Um indivíduo é uma árvore de expressão armazenada como uma **lista em pré-ordem**
(notação prefixa). Por exemplo, ``x0 * (x1 + 2.0)`` é guardada como::

    [mul, x0, add, x1, 2.0]

Essa representação linear torna cruzamento e mutação de subárvores simples
(fatiar listas) e a avaliação rápida (pilha, vetorizada com NumPy).

Convenção de profundidade: um nó isolado tem profundidade 1; a raiz de uma árvore
com 3 níveis tem profundidade total 3. O "tamanho máximo do indivíduo = 7" do
enunciado é interpretado como **profundidade máxima 7**.
"""
from __future__ import annotations

import numpy as np

# Valor usado para substituir saídas não finitas (NaN/inf) e limitar explosões numéricas.
BIG = 1e10


# --------------------------------------------------------------------------- #
# Tokens: funções, variáveis e constantes
# --------------------------------------------------------------------------- #
class Primitive:
    """Operador (nó interno) com aridade 1 ou 2."""

    __slots__ = ("name", "arity", "fn", "symbol", "key")

    def __init__(self, name, arity, fn, symbol=None):
        self.name = name
        self.arity = arity
        self.fn = fn
        self.symbol = symbol or name
        self.key = name

    def __repr__(self):
        return f"Primitive({self.name})"


class Variable:
    """Terminal que lê uma coluna (atributo) do conjunto de dados."""

    __slots__ = ("index", "name", "arity", "key")

    def __init__(self, index, name=None):
        self.index = int(index)
        self.name = name or f"x{index}"
        self.arity = 0
        self.key = f"v{index}"

    def __repr__(self):
        return f"Variable({self.name})"


class Constant:
    """Terminal com um valor numérico (constante efêmera aleatória)."""

    __slots__ = ("value", "name", "arity", "key")

    def __init__(self, value):
        self.value = float(value)
        self.name = f"{self.value:.4g}"
        self.arity = 0
        self.key = f"c{self.value!r}"

    def __repr__(self):
        return f"Constant({self.name})"


# --------------------------------------------------------------------------- #
# Funções (todas vetorizadas e "protegidas" contra domínio inválido)
# --------------------------------------------------------------------------- #
def _add(a, b):
    return a + b


def _sub(a, b):
    return a - b


def _mul(a, b):
    return a * b


def _div(a, b):
    """Divisão protegida: devolve 1 quando |b| é muito pequeno."""
    ok = np.abs(b) > 1e-6
    return np.where(ok, a / np.where(ok, b, 1.0), 1.0)


def _sin(a):
    return np.sin(a)


def _cos(a):
    return np.cos(a)


def _sqrt(a):
    """Raiz quadrada protegida: sqrt(|a|)."""
    return np.sqrt(np.abs(a))


def _log(a):
    """Logaritmo protegido: log(|a|), e 0 quando |a| é muito pequeno."""
    ok = np.abs(a) > 1e-6
    return np.where(ok, np.log(np.where(ok, np.abs(a), 1.0)), 0.0)


def _exp(a):
    """Exponencial com argumento limitado para evitar overflow."""
    return np.exp(np.clip(a, -20.0, 20.0))


FUNCTIONS = {
    "add": Primitive("add", 2, _add, "+"),
    "sub": Primitive("sub", 2, _sub, "-"),
    "mul": Primitive("mul", 2, _mul, "*"),
    "div": Primitive("div", 2, _div, "/"),
    "sin": Primitive("sin", 1, _sin, "sin"),
    "cos": Primitive("cos", 1, _cos, "cos"),
    "sqrt": Primitive("sqrt", 1, _sqrt, "sqrt"),
    "log": Primitive("log", 1, _log, "log"),
    "exp": Primitive("exp", 1, _exp, "exp"),
}

DEFAULT_FUNCTIONS = ("add", "sub", "mul", "div", "sin", "cos")


class PrimitiveSet:
    """Conjunto de funções e terminais usados para construir árvores.

    Parameters
    ----------
    n_vars : número de atributos de entrada (colunas de X).
    functions : nomes das funções (chaves de ``FUNCTIONS``).
    const_range : intervalo [a, b] de onde saem as constantes efêmeras.
    const_prob : probabilidade de um terminal sorteado ser constante (e não variável).
    grow_terminal_prob : prob. de escolher terminal em um nó interno no método *grow*.
    var_names : nomes dos atributos (apenas para impressão).
    """

    def __init__(self, n_vars, functions=DEFAULT_FUNCTIONS, const_range=(-5.0, 5.0),
                 const_prob=0.3, grow_terminal_prob=0.3, var_names=None):
        unknown = [f for f in functions if f not in FUNCTIONS]
        if unknown:
            raise ValueError(f"Funções desconhecidas: {unknown}. Opções: {sorted(FUNCTIONS)}")
        self.functions = [FUNCTIONS[f] for f in functions]
        self.variables = [
            Variable(i, var_names[i] if var_names is not None else None) for i in range(n_vars)
        ]
        self.const_range = tuple(const_range)
        self.const_prob = const_prob
        self.grow_terminal_prob = grow_terminal_prob
        self.by_arity = {}
        for f in self.functions:
            self.by_arity.setdefault(f.arity, []).append(f)

    def random_terminal(self, rng):
        if rng.random() < self.const_prob:
            return Constant(rng.uniform(*self.const_range))
        return rng.choice(self.variables)

    def random_function(self, rng):
        return rng.choice(self.functions)


# --------------------------------------------------------------------------- #
# Utilidades sobre listas em pré-ordem
# --------------------------------------------------------------------------- #
def subtree_end(genes, start):
    """Índice (exclusivo) do fim da subárvore que começa em ``start``."""
    need = 1
    i = start
    while need > 0:
        need += genes[i].arity - 1
        i += 1
    return i


def node_depths(genes):
    """Profundidade (raiz = 1) de cada nó, na ordem da lista."""
    out = []
    stack = []
    for t in genes:
        d = stack.pop() if stack else 1
        out.append(d)
        if t.arity:
            stack.extend([d + 1] * t.arity)
    return out


def tree_depth(genes):
    """Profundidade total da árvore (um único nó tem profundidade 1)."""
    return max(node_depths(genes))


def to_infix(genes, precision=4):
    """Converte a lista prefixa em uma expressão infixa legível."""

    def rec(i):
        t = genes[i]
        if t.arity == 0:
            return t.name, i + 1
        args = []
        j = i + 1
        for _ in range(t.arity):
            s, j = rec(j)
            args.append(s)
        if t.arity == 2 and t.symbol in "+-*/":
            return f"({args[0]} {t.symbol} {args[1]})", j
        return f"{t.symbol}({', '.join(args)})", j

    return rec(0)[0]


# --------------------------------------------------------------------------- #
# Geração de árvores (população inicial)
# --------------------------------------------------------------------------- #
def generate_tree(pset, rng, depth, method="grow"):
    """Gera uma árvore com profundidade máxima ``depth``.

    * ``full``: todos os ramos chegam até a profundidade ``depth``.
    * ``grow``: nós internos podem virar terminais antes de ``depth``
      (a raiz é sempre uma função quando ``depth > 1``).
    """
    genes = []

    def rec(d, is_root):
        stop = d <= 1
        if not stop and method == "grow" and not is_root:
            stop = rng.random() < pset.grow_terminal_prob
        if stop:
            genes.append(pset.random_terminal(rng))
            return
        f = pset.random_function(rng)
        genes.append(f)
        for _ in range(f.arity):
            rec(d - 1, False)

    rec(depth, True)
    return genes


def ramped_half_and_half(pset, rng, n, min_depth=2, max_depth=5):
    """População inicial *ramped half-and-half*.

    As profundidades ``min_depth..max_depth`` são percorridas ciclicamente e, a cada
    volta, alterna-se entre os métodos ``full`` e ``grow``. Assim metade dos
    indivíduos é gerada com cada método e todas as profundidades são representadas.
    """
    depths = list(range(min_depth, max_depth + 1))
    nd = len(depths)
    pop = []
    for i in range(n):
        d = depths[i % nd]
        method = "full" if (i // nd) % 2 == 0 else "grow"
        pop.append(Individual(generate_tree(pset, rng, d, method)))
    return pop


# --------------------------------------------------------------------------- #
# Avaliação
# --------------------------------------------------------------------------- #
def predict_genes(genes, cols, n_samples):
    """Avalia uma árvore (lista prefixa) nas colunas ``cols`` (shape n_vars x N)."""
    stack = []
    push = stack.append
    pop = stack.pop
    for t in reversed(genes):
        a = t.arity
        if a == 0:
            push(cols[t.index] if t.__class__ is Variable else t.value)
        elif a == 1:
            push(t.fn(pop()))
        else:
            x = pop()
            y = pop()
            push(t.fn(x, y))
    out = np.asarray(stack[0], dtype=float)
    if out.ndim == 0:
        out = np.full(n_samples, float(out))
    out = np.nan_to_num(out, nan=BIG, posinf=BIG, neginf=-BIG)
    np.clip(out, -BIG, BIG, out=out)
    return out


def rmse(y_pred, y_true):
    """Raiz do erro quadrático médio."""
    return float(np.sqrt(np.mean((np.asarray(y_pred) - np.asarray(y_true)) ** 2)))


class Individual:
    """Um indivíduo: genótipo (lista prefixa) + fitness (RMSE) + erro por caso.

    ``errors`` guarda o erro absoluto em cada exemplo de treino; ele é necessário
    para a seleção ε-lexicase e permite recalcular o RMSE (``sqrt(mean(errors**2))``).
    """

    __slots__ = ("genes", "fitness", "errors", "_key")

    def __init__(self, genes, fitness=None, errors=None):
        self.genes = genes
        self.fitness = fitness
        self.errors = errors
        self._key = None

    # -- propriedades estruturais ------------------------------------------------
    @property
    def size(self):
        return len(self.genes)

    @property
    def depth(self):
        return tree_depth(self.genes)

    @property
    def key(self):
        """Chave sintática: dois indivíduos com a mesma chave são idênticos."""
        if self._key is None:
            self._key = "|".join(t.key for t in self.genes)
        return self._key

    def copy(self):
        """Cópia que preserva fitness/erros já calculados (genes não são alterados in-place)."""
        c = Individual(list(self.genes), self.fitness, self.errors)
        c._key = self._key
        return c

    # -- avaliação -------------------------------------------------------------------
    def evaluate(self, cols, y):
        """Calcula o erro absoluto por caso e o RMSE em (cols, y)."""
        pred = predict_genes(self.genes, cols, y.shape[0])
        err = np.abs(pred - y)
        self.errors = err
        self.fitness = float(np.sqrt(np.mean(err * err)))
        return self.fitness

    def predict(self, X):
        """Predição para uma matriz X (N x n_vars)."""
        X = np.asarray(X, dtype=float)
        with np.errstate(all="ignore"):
            return predict_genes(self.genes, np.ascontiguousarray(X.T), X.shape[0])

    def score(self, X, y):
        """RMSE do indivíduo em (X, y)."""
        return rmse(self.predict(X), y)

    def __str__(self):
        return to_infix(self.genes)

    def __repr__(self):
        fit = "?" if self.fitness is None else f"{self.fitness:.4g}"
        return f"Individual(size={self.size}, fitness={fit}, expr={self})"
