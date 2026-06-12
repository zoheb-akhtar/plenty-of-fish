# Plenty of Fish - A Shark Evolution Simulation

A shark "family" that gets better at surviving in two ways at once: a **genetic
algorithm (GA)** evolves shark *bodies* between lives, while **reinforcement
learning (RL)** teaches a *shared brain* how to behave within a life. Bodies and
behaviour improve together.

## Problem Overview

A shark lives on a tile grid (the `Ocean`) scattered with safe and poisonous
fish of different sizes. Each step it can move, attack, or rest; it spends energy
to move and dies if it starves or bites poison while weak. The goal is a family
of sharks whose traits (size, speed, caution, aggression) and behaviour
(when to hunt vs. hang back) together earn the most reward.

Two algorithms tackle this on two different timescales and meet through behaviour:

- **RL — within-life decisions.** A `FamilyRL` brain runs tabular Q-learning
  (epsilon-greedy, Bellman update). Every shark in the family reads and writes
  the *same* Q-table, so one shark's fatal poison bite teaches the whole family.
  Observations (energy + nearest fish) are bucketed into discrete states by
  `discretize()` before they touch the table.
- **GA — who reproduces.** The GA evolves genomes (tournament selection +
  arithmetic crossover + Gaussian mutation, with elitism). Only the *evolvable*
  traits — `size`, `speed`, `caution`, `aggression` — vary; the rest hold their
  defaults.

A genome's fitness is simply the reward its biased
behaviour earns, so the GA selects for temperaments that do well given what the
shared brain has learned. The brain is created once and persists across
generations, so over a run the bodies and the behaviour co-adapt.

### Background/Importance

The blue shark (Prionace glauca) is one of the most widely and highly fished sharks on Earth and is the only living member of its genus, so its persistence is entirely based on how well this  species can adapt to changing oceans [1]. In nature, adaptation and evolution run on two interleaved timescales; evolution slowly reshapes the body across generations, while learning rapidly reshapes behavior within a single life. In blue sharks, live young leave their mothers after birth and do not inherit instructions, but we wanted to approach this question with what if families could pass down data the way humans pass down stories through generations? 

### Sample Outputs
Sample output after the command python -m src.main --watch, which shows you the GUI of the sharks hunting and their related metrics as they continue to reproduce. 

<img width="918" height="803" alt="Screenshot 2026-06-12 at 8 29 42 AM" src="https://github.com/user-attachments/assets/178e2723-9d06-4259-b476-8e415ca20c36" />
<img width="919" height="810" alt="Screenshot 2026-06-12 at 8 29 52 AM" src="https://github.com/user-attachments/assets/bb990f74-cf48-489b-80bb-dba07db2f4d4" />


Unless `--no-plots` is passed, matplotlib also shows each evolvable trait's
trajectory and a best-vs-population-range fitness curve over the generations.

**Watch GUI** (`--watch`) animates the colony foraging in a shared ocean:
the sharks compete for one depleting, self-replenishing fish pool, so food is a
real constraint. **Space** to pause, and **M** for the live
metrics panel. **Play GUI** (`--play`) lets you steer a single randomly-statted
shark yourself. The standalone RL trainer also saves a learning curve to
`src/algorithms/rl_algo/training_plot.png` and per-episode numbers to
`training_log.csv`.

---

## Using this Repo

### Directory Structure

```
plenty-of-fish/
├── Makefile                     # housekeeping: make clean / help
├── README.md
├── experiments
└── src/
    ├── main.py                  # entry point — wires GA + RL together (CLI below)
    ├── simulation.py            
    ├── algorithms/
    │   ├── genetic_algo.py      # GA: population, selection, crossover, mutation, plots
    │   └── rl_algo/             # reinforcement-learning "family brain"
    │       ├── actions.py       
    │       ├── state.py         
    │       ├── q_table.py       
    │       ├── family_rl.py     # epsilon-greedy tabular Q-learning agent (shared table)
    │       ├── toy_env.py       
    │       ├── train_headless.py
    │       └── README.md        # notes on the RL side
    ├── environment/             # ocean environment built out 
    │   ├── config.py            # EnvConfig — every Ocean/GUI knob in one place
    │   └── ocean.py             
    ├── gui/                     # pygame rendering (only needed for --watch / --play)
    │   ├── gui.py               # Play mode (control one shark)
    │   ├── watch_gui.py         # Watch mode (spectate the family evolve)
    │   ├── seabed.py            
    │   ├── fish_drawing.py      
    │   └── shark_drawing.py     
    └── shark/
        ├── traits.py            # the trait registry (ranges, defaults, evolvable flag)
        ├── genome.py            # SharkGenome — trait values + GA operators
        └── shark.py             # a living shark: age, breeding, lifecycle
```

### Setting up Environment

Requires Python 3.10, but we used 3.14. Create a virtual environment and install the two
runtime dependencies (matplotlib for plots, pygame-ce for the GUIs):

```bash
# from the repo root
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install matplotlib pygame-ce
```

The headless GA+RL run needs only matplotlib; `pygame-ce` is required for the
`--watch` and `--play` GUIs.

### Run the project

Run everything as a module from the repo root (so the `src` package resolves):

```bash
python -m src.main --watch               # GUI: watch the family evolve (grid of mini-oceans)
python -m src.main --play                # GUI: control one shark in its own ocean
python -m src.main                       # headless: evolve + learn, save best genome
python -m src.main --generations 50 --no-plots
```

Useful flags (see `python -m src.main --help`): `--population`, `--generations`,
`--max-steps`, `--lives`, `--mutation-rate`, `--seed`, `--no-plots`.

Housekeeping via the Makefile (never touches `.venv/` or `.git/`):

```bash
make help            # list targets
make clean           # remove caches / bytecode / OS junk
```

---
<sup><sub>AI Acknowledgement: AI tools were used to build this project's code base. </sub></sup>
## Made by
Zoheb Akhtar, Abdurrahman Assaf, Zara Ceraj

