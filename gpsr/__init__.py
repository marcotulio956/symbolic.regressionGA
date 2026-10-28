"""gpsr – Programação Genética para Regressão Simbólica (TP1 de Computação Natural)."""
from .data import Dataset, available_datasets, load_all, load_dataset
from .experiments import (
    RepeatedResult,
    best_config,
    bloat_table,
    checkpoint_table,
    compare_configs,
    kruskal_pvalue,
    operator_effects,
    pairwise_tests,
    run_repetitions,
    summary_table,
    sweep,
)
from . import plots
from .gp import GeneticProgramming, GPConfig, GPResult
from .operators import mutate, point_mutation, subtree_crossover, subtree_mutation
from .report import curve_grid, report
from .selection import (
    LexicaseSelection,
    RouletteSelection,
    TournamentSelection,
    make_selector,
)
from .tree import (
    FUNCTIONS,
    Individual,
    PrimitiveSet,
    generate_tree,
    ramped_half_and_half,
    to_infix,
    tree_depth,
)

__all__ = [n for n in dir() if not n.startswith("_")]
