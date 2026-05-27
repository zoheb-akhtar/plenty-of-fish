"""Tabular Q-learning agent with a shared family Q-table."""

import random

from actions import Action
from q_table import NUM_ACTIONS, QTable
from state import State


class FamilyRL:
    """One Q-table shared by all sharks in a family."""

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        q_table: QTable | None = None,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q = q_table if q_table is not None else QTable()

    def select_action(self, state: State) -> Action:
        """Epsilon-greedy: explore randomly or exploit best Q."""
        if random.random() < self.epsilon:
            return Action(random.randint(0, NUM_ACTIONS - 1))
        return self.q.best_action(state)

    def update(
        self,
        state: State,
        action: Action,
        reward: float,
        next_state: State,
        done: bool,
    ) -> None:
        """Bellman Q-learning update for Q(s, a)."""
        current = self.q.get(state, action)
        if done:
            target = reward
        else:
            target = reward + self.gamma * self.q.max_q(next_state)
        new_q = current + self.alpha * (target - current)
        self.q.set(state, action, new_q)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
