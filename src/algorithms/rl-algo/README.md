# RL algo (`rl-algo`)

Hey guys this folder is the **reinforcement learning** side of the project. Right now it trains a shark on a small practice grid (`toy_env.py`). Later we plug the same brain into the real environment we will build.

**Idea in one sentence:** each shark picks moves using a shared **Q-table** (what worked before in similar situations), and the table updates when sharks get rewards or die (especially from poison).

---

## Quick start

```bash
conda activate ai-class
cd src/algorithms/rl-algo
python train_headless.py
```

That trains for 2000 episodes and writes:

- `training_plot.png` — learning curve (reward, steps, epsilon)
- `training_log.csv` — raw numbers per episode

Other quick tests:

```bash
python toy_env.py    # run the mini world with a few hardcoded moves
python state.py      # see how observations become discrete "states"
python q_table.py    # tiny Q-table smoke test
```

---

## Files (what each one does)

| File | What it is |
|------|------------|
| **`actions.py`** | The six things a shark can do: up, down, left, right, attack, rest. |
| **`toy_env.py`** | Practice ocean: 10×10 grid, sharks, fish (safe + poison), energy, rewards. Stand-in until real `GridWorld` exists. |
| **`state.py`** | Turns the env's observation dict into a simple tuple like `('high', 'safe', 'close', 'small')` so the Q-table can use it as a key. |
| **`q_table.py`** | Stores Q-values: "how good is each action in each state?" |
| **`family_rl.py`** | The "brain": pick actions (explore vs exploit), update Q after each step. One `FamilyRL` = one Q-table for a whole family. |
| **`train_headless.py`** | Training loop: run many episodes, print progress, save plot + CSV. No pygame. |

Generated when you train (safe to delete / regenerate):

- `training_plot.png`, `training_log.csv`

---

## How one training step works

```text
1. env gives observation (energy + nearest fish info)
2. state.py  →  bucket that into a discrete state tuple
3. family_rl →  pick action (usually best Q, sometimes random while learning)
4. toy_env   →  shark moves / attacks / rests, gets reward, maybe dies
5. family_rl →  update Q(state, action) from reward + what happened next
```

Repeat until the shark dies or hits 200 steps. That's one **episode**. Then epsilon decays a bit (less random exploring) and we start a new episode.

---

## Observations (what the RL "sees")

The env returns something like:

```python
{
  "energy": 0.72,
  "nearest_fish": {
    "exists": True,
    "type": "safe",       # or "poisonous"
    "distance": 3,
    "size": "small",
  },
}
```

We **don't** feed raw x/y into the Q-table — only these buckets (low/med/high energy, close/med/far, etc.). That's in `state.py`.

**Important:** the shark is told fish *type* in the observation (safe vs poison). Learning is about **what to do** (attack vs run away), not guessing species from scratch.

---

## Rewards (toy env rules)

| What happens | Rough reward |
|--------------|--------------|
| Each step | -0.1 |
| Eat safe fish (adjacent + attack) | +25, energy up |
| Attack poisonous fish | -100, shark dies |
| Energy hits 0 | -50, shark dies |
| Rest | small energy gain |

Poison **should** kill — that's correct. Episode reward is often negative even when the shark is learning fine.

---

## Family / shared Q-table (design for later)

- **`FamilyRL`** holds one **`QTable`**.
- All sharks in the same family will use the same instance.
- When one shark dies from poison, the Q-update still writes to the shared table → other sharks benefit in similar states.

Right now training uses **one shark** in `toy_env`; multi-shark loop comes when the real env is ready.

---

## Hooking up the real environment

We need the same contract `toy_env` already has:

```python
obs = env.reset()                    # or get_observation(shark_id) for multi-shark
obs, reward, done = env.step(action) # or step(shark_id, action)
```

Same `obs` shape as above. Then in `train_headless.py` swap `ToyOcean()` for `GridWorld()` (or whatever we name it). **No changes needed** to `state.py` / `q_table.py` / `family_rl.py` if the observation dict matches.

---

## Genetic algo (not in this folder yet)

GA will live in `genetic_algo.py` and handle breeding, genomes (aggression, caution, etc.). RL handles **within-life** decisions. GA picks **who reproduces**. We'll bias action choice with genome later — not inside the Q-update itself.

---

## What's done vs what's next

**Done**

- Tabular Q-learning end-to-end on toy grid  
- Training script + plot + CSV  
- Multi-fish toy world (safe + poison lists)

**Next**

- Real 20×20 env + fish spawning / garbage  
- Multiple sharks per family in the step loop  
- Wire GA + reproduction timer  
- Pygame shows the same state (Zara) — training can stay headless  

