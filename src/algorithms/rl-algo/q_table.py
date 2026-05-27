"""Tabular Q-values: state tuple -> action -> float."""

from collections import defaultdict

from actions import Action
from state import State

NUM_ACTIONS = len(Action)


class QTable:
    def __init__(self):
        self._table: dict[State, list[float]] = defaultdict(
            lambda: [0.0] * NUM_ACTIONS
        )

    def get(self, state: State, action: Action) -> float:
        return self._table[state][int(action)]

    def set(self, state: State, action: Action, value: float) -> None:
        self._table[state][int(action)] = value

    def best_action(self, state: State) -> Action:
        """Action with highest Q (ties -> lowest action index)."""
        q_values = self._table[state]
        best_idx = max(range(NUM_ACTIONS), key=lambda i: q_values[i])
        return Action(best_idx)

    def max_q(self, state: State) -> float:
        """Best Q in this state (for Bellman update on next state)."""
        return max(self._table[state])

    def __len__(self) -> int:
        return len(self._table)


if __name__ == "__main__":
    from state import discretize
    from toy_env import ToyOcean

    q = QTable()
    env = ToyOcean()
    obs = env.reset()
    s = discretize(obs)

    print("initial Q(ATTACK):", q.get(s, Action.ATTACK))
    q.set(s, Action.ATTACK, 10.0)
    print("after set:", q.get(s, Action.ATTACK))
    print("best action:", q.best_action(s).name)
    print("max_q:", q.max_q(s))
    print("states in table:", len(q))
