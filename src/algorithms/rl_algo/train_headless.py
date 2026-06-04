"""Headless Q-learning training on ToyOcean."""

from __future__ import annotations

import csv
from pathlib import Path

from actions import Action
from family_rl import FamilyRL
from state import discretize
from toy_env import ToyOcean

EPISODES = 2000
MAX_STEPS = 200
PRINT_EVERY = 50
ROLLING_WINDOW = 50
OUTPUT_DIR = Path(__file__).resolve().parent


def rolling_average(values: list[float], window: int) -> list[float]:
    if not values:
        return []
    out: list[float] = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        chunk = values[start : i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def train(
    episodes: int = EPISODES,
    max_steps: int = MAX_STEPS,
    print_every: int = PRINT_EVERY,
) -> tuple[FamilyRL, dict[str, list[float]]]:
    env = ToyOcean()
    agent = FamilyRL()

    history: dict[str, list[float]] = {
        "episode": [],
        "reward": [],
        "steps": [],
        "epsilon": [],
    }
    rewards_window: list[float] = []

    for episode in range(1, episodes + 1):
        obs = env.reset()
        state = discretize(obs)
        total_reward = 0.0
        steps = 0
        done = False

        while not done and steps < max_steps:
            action = agent.select_action(state)
            obs, reward, done = env.step(action)
            next_state = discretize(obs)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps += 1

        agent.decay_epsilon()

        history["episode"].append(float(episode))
        history["reward"].append(total_reward)
        history["steps"].append(float(steps))
        history["epsilon"].append(agent.epsilon)

        rewards_window.append(total_reward)
        if len(rewards_window) > print_every:
            rewards_window.pop(0)

        if episode % print_every == 0 or episode == 1:
            avg = sum(rewards_window) / len(rewards_window)
            print(
                f"episode {episode:4d} | "
                f"reward {total_reward:7.1f} | "
                f"steps {steps:3d} | "
                f"epsilon {agent.epsilon:.3f} | "
                f"avg_reward({len(rewards_window)}) {avg:7.1f} | "
                f"states {len(agent.q)}"
            )

    return agent, history


def save_history_csv(history: dict[str, list[float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = zip(
        history["episode"],
        history["reward"],
        history["steps"],
        history["epsilon"],
    )
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "steps", "epsilon"])
        writer.writerows(rows)


def plot_training(
    history: dict[str, list[float]],
    output_dir: Path = OUTPUT_DIR,
    rolling_window: int = ROLLING_WINDOW,
    show: bool = False,
) -> Path:
    import matplotlib.pyplot as plt

    episodes = [int(e) for e in history["episode"]]
    rewards = history["reward"]
    steps = history["steps"]
    epsilons = history["epsilon"]
    avg_rewards = rolling_average(rewards, rolling_window)

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "training_plot.png"

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    axes[0].plot(episodes, rewards, alpha=0.25, color="steelblue", label="episode reward")
    axes[0].plot(episodes, avg_rewards, color="darkblue", linewidth=2, label=f"{rolling_window}-ep avg")
    axes[0].axhline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("Total reward")
    axes[0].legend(loc="lower right")
    axes[0].set_title("Q-learning on ToyOcean")

    axes[1].plot(episodes, steps, color="seagreen", alpha=0.7)
    axes[1].set_ylabel("Steps per episode")

    axes[2].plot(episodes, epsilons, color="coral")
    axes[2].set_ylabel("Epsilon")
    axes[2].set_xlabel("Episode")

    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    if show:
        plt.show()
    plt.close(fig)

    return png_path


if __name__ == "__main__":
    agent, history = train()
    csv_path = OUTPUT_DIR / "training_log.csv"
    save_history_csv(history, csv_path)
    plot_path = plot_training(history, show=False)
    print(f"Saved {csv_path}")
    print(f"Saved {plot_path}")
