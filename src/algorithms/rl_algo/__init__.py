"""Reinforcement-learning side of the project: a shared-Q-table "family brain".

Within a single life a shark picks actions with tabular Q-learning; all sharks
in a family read and write the *same* ``QTable`` (so a death-by-poison teaches
the whole family). Observations are bucketed into discrete states by
``discretize`` before they ever touch the table.

The genetic algorithm reuses ``FamilyRL`` through ``src.simulation``: a genome's
temperament biases which action the shared brain takes in ``ToyOcean``, and the
reward it earns becomes that genome's fitness.
"""

from .actions import Action
from .family_rl import FamilyRL
from .q_table import NUM_ACTIONS, QTable
from .state import State, discretize
from .toy_env import ToyOcean

__all__ = [
    "Action",
    "FamilyRL",
    "NUM_ACTIONS",
    "QTable",
    "State",
    "discretize",
    "ToyOcean",
]
