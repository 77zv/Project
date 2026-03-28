"""
CISC 474 – Coverage Tournament  ·  Training Script
====================================================
Implements:
  • 2 observation spaces   (Obs1 = flat simplified grid, Obs2 = local view + globals)
  • 3 reward functions     (R1 sparse, R2 shaped, R3 dense progress)
  • PPO from Stable Baselines 3
  • Experiment loop with plot generation
  • Final best-model training on random maps + all predefined maps

Run:   python3 train.py
Output:
  • plots/  – one plot per experiment + comparison plot
  • best_model.zip  – final trained model for the tournament
"""

import os
import numpy as np
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")          # headless backend (no display needed)
import matplotlib.pyplot as plt

# Register the custom environments
import coverage_gridworld        # noqa: F401  (side-effect: registers envs)

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GRID_SIZE      = 10
LOCAL_VIEW     = 5
HALF           = LOCAL_VIEW // 2

EXPERIMENT_STEPS = 300_000   # timesteps per experiment run
FINAL_STEPS      = 2_000_000 # timesteps for the final best model

os.makedirs("plots", exist_ok=True)

# ---------------------------------------------------------------------------
# Color → integer ID helpers  (keep in sync with custom.py)
# ---------------------------------------------------------------------------
_BLACK     = (0,   0,   0)
_WHITE     = (255, 255, 255)
_BROWN     = (101, 67,  33)
_GREY      = (160, 161, 161)
_GREEN     = (31,  198, 0)
_RED       = (255, 0,   0)
_LIGHT_RED = (255, 127, 127)

_COLOR_MAP = {
    _BLACK:     0,
    _WHITE:     1,
    _BROWN:     2,
    _GREY:      3,
    _GREEN:     4,
    _RED:       5,
    _LIGHT_RED: 6,
}

def _rgb_to_id(rgb):
    return _COLOR_MAP.get((int(rgb[0]), int(rgb[1]), int(rgb[2])), 0)

def _to_id_grid(grid):
    id_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            id_grid[r, c] = _rgb_to_id(grid[r, c])
    return id_grid

def _find_agent(id_grid):
    pos = np.argwhere(id_grid == 3)
    if len(pos):
        return int(pos[0][0]), int(pos[0][1])
    return 0, 0


# ===========================================================================
# OBSERVATION SPACE 1 – Flat simplified grid
# ===========================================================================
# Maps each cell of the 10×10 grid to an integer ID (0-6).
# Shape: (100,)  Values: 0.0–6.0
# Rationale: Complete global map view; simple baseline for comparison.
# Downside: Large input space, may not generalise to unseen maps as well.

def obs1_space():
    return gym.spaces.Box(low=0.0, high=6.0, shape=(100,), dtype=np.float32)

def obs1_fn(grid):
    """Return the full 10×10 grid as 100 integer IDs."""
    id_grid = _to_id_grid(grid)
    return id_grid.flatten().astype(np.float32)


# ===========================================================================
# OBSERVATION SPACE 2 – Local 5×5 view + global scalars  (BEST)
# ===========================================================================
# A compact 27-element vector:
#   25 floats – 5×5 local view centred on the agent (cell IDs 0-6)
#    1 float  – map coverage ratio ∈ [0, 1]
#    1 float  – binary enemy-threat flag in local area
# Rationale: Compact, translation-invariant; generalises to unseen maps.

def obs2_space():
    n = LOCAL_VIEW * LOCAL_VIEW + 2
    low  = np.zeros(n, dtype=np.float32)
    high = np.concatenate([np.full(LOCAL_VIEW * LOCAL_VIEW, 6.0),
                           np.array([1.0, 1.0])]).astype(np.float32)
    return gym.spaces.Box(low=low, high=high, dtype=np.float32)

def obs2_fn(grid):
    """Return 5×5 local view + coverage ratio + enemy threat flag."""
    id_grid = _to_id_grid(grid)
    ar, ac  = _find_agent(id_grid)

    # 5×5 local view; out-of-bounds cells default to wall (ID = 2)
    local = np.full((LOCAL_VIEW, LOCAL_VIEW), 2.0, dtype=np.float32)
    for di in range(-HALF, HALF + 1):
        for dj in range(-HALF, HALF + 1):
            r, c = ar + di, ac + dj
            if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
                local[di + HALF, dj + HALF] = float(id_grid[r, c])

    explored = int(np.sum((id_grid == 1) | (id_grid == 3) | (id_grid == 6)))
    total    = GRID_SIZE * GRID_SIZE - int(np.sum(id_grid == 2)) - int(np.sum(id_grid == 4))
    coverage = float(explored) / max(total, 1)
    enemy_threat = float(np.any((local == 5) | (local == 6)))

    return np.concatenate([local.flatten(), [coverage, enemy_threat]]).astype(np.float32)


# ===========================================================================
# REWARD FUNCTION 1 – Sparse
# ===========================================================================
# Only rewards covering new cells and penalises death.
# Rationale: Simplest possible signal; hard credit-assignment problem.

def reward1(info):
    r = 0.0
    if info["new_cell_covered"]:
        r += 1.0
    if info["game_over"]:
        r -= 50.0
    return r


# ===========================================================================
# REWARD FUNCTION 2 – Shaped with completion bonus + time penalty
# ===========================================================================
# Adds:  +50 for winning, −0.01/step time penalty.
# Rationale: Winning bonus encourages finishing; time penalty rewards speed.

def reward2(info):
    r = 0.0
    if info["new_cell_covered"]:
        r += 1.0
    if info["cells_remaining"] == 0:
        r += 50.0
    if info["game_over"]:
        r -= 50.0
    r -= 0.01
    return r


# ===========================================================================
# REWARD FUNCTION 3 – Dense with progress bonus  (BEST)
# ===========================================================================
# Adds a scaled bonus that grows as coverage increases, making later cells
# more valuable. Also gives a larger completion bonus.
# Rationale: Denser signal eases training; progress bonus fights plateau.

def reward3(info):
    r = 0.0
    if info["new_cell_covered"]:
        progress = info["total_covered_cells"] / max(info["coverable_cells"], 1)
        r += 1.0 + progress
    if info["cells_remaining"] == 0:
        r += 100.0
    if info["game_over"]:
        r -= 50.0
    r -= 0.01
    return r


# ===========================================================================
# Generic Wrapper – plugs any (obs_fn, obs_space, reward_fn) into the env
# ===========================================================================

class ExperimentWrapper(gym.Wrapper):
    """
    Wraps the base CoverageGridworld environment to use a custom observation
    function and reward function, bypassing custom.py's defaults.
    """
    def __init__(self, env, obs_fn, obs_space, reward_fn):
        super().__init__(env)
        self._obs_fn     = obs_fn
        self._reward_fn  = reward_fn
        self.observation_space = obs_space

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        obs = self._obs_fn(self.env.unwrapped.grid)
        return obs, info

    def step(self, action):
        _, _, terminated, truncated, info = self.env.step(action)
        obs = self._obs_fn(self.env.unwrapped.grid)
        rew = self._reward_fn(info)
        return obs, rew, terminated, truncated, info


# ===========================================================================
# Callback – tracks per-episode stats during training
# ===========================================================================

class TrainingCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards  = []
        self.episode_lengths  = []
        self.coverage_ratios  = []
        self._ep_reward = 0.0
        self._ep_len    = 0
        self._last_info = {}

    def _on_step(self):
        self._ep_reward += float(self.locals["rewards"][0])
        self._ep_len    += 1
        if "infos" in self.locals and self.locals["infos"]:
            self._last_info = self.locals["infos"][0]

        if self.locals["dones"][0]:
            self.episode_rewards.append(self._ep_reward)
            self.episode_lengths.append(self._ep_len)
            if self._last_info:
                covered   = self._last_info.get("total_covered_cells", 1)
                coverable = self._last_info.get("coverable_cells", 1)
                self.coverage_ratios.append(covered / max(coverable, 1))
            self._ep_reward = 0.0
            self._ep_len    = 0
        return True


# ===========================================================================
# Plotting helpers
# ===========================================================================

def _smooth(values, window=20):
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")

def plot_experiment(cb, label, filename):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    rewards  = cb.episode_rewards
    coverage = cb.coverage_ratios

    axes[0].plot(_smooth(rewards), label=f"{label} (smoothed)")
    axes[0].set_title(f"{label} – Episode Reward")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total Reward")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(_smooth(coverage), label=f"{label} (smoothed)", color="orange")
    axes[1].set_title(f"{label} – Coverage Ratio at Episode End")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Coverage (fraction)")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(f"plots/{filename}.png", dpi=120)
    plt.close()
    print(f"  [plot saved] plots/{filename}.png")


def plot_comparison(results):
    """Plot all experiments on one figure for easy comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for label, cb in results.items():
        axes[0].plot(_smooth(cb.episode_rewards, window=30), label=label)
        axes[1].plot(_smooth(cb.coverage_ratios, window=30), label=label)

    axes[0].set_title("Episode Reward – All Experiments")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total Reward (smoothed)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True)

    axes[1].set_title("Coverage Ratio – All Experiments")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Coverage fraction (smoothed)")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(fontsize=8)
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig("plots/comparison.png", dpi=120)
    plt.close()
    print("  [plot saved] plots/comparison.png")


# ===========================================================================
# Predefined maps for training diversity
# ===========================================================================

PREDEFINED_MAPS = [
    # Map 0 – open field (no enemies)
    [
        [3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ],
    # Map 1 – safe (walls, no enemies)
    [
        [3, 0, 0, 2, 0, 2, 0, 0, 0, 0],
        [0, 2, 0, 0, 0, 2, 0, 0, 2, 0],
        [0, 2, 0, 2, 2, 2, 2, 2, 2, 0],
        [0, 2, 0, 0, 0, 2, 0, 0, 0, 0],
        [0, 2, 0, 2, 0, 2, 0, 0, 2, 0],
        [0, 2, 0, 0, 0, 0, 0, 2, 0, 0],
        [0, 2, 2, 2, 0, 0, 0, 2, 0, 0],
        [0, 0, 0, 0, 0, 2, 0, 0, 2, 0],
        [0, 2, 0, 2, 0, 2, 2, 0, 0, 0],
        [0, 0, 0, 0, 0, 2, 0, 0, 0, 0],
    ],
    # Map 2 – maze (2 enemies)
    [
        [3, 2, 0, 0, 0, 0, 2, 0, 0, 0],
        [0, 2, 0, 2, 2, 0, 2, 0, 2, 2],
        [0, 2, 0, 2, 0, 0, 2, 0, 0, 0],
        [0, 2, 0, 2, 0, 2, 2, 2, 2, 0],
        [0, 2, 0, 2, 0, 0, 2, 0, 0, 0],
        [0, 2, 0, 2, 2, 0, 2, 0, 2, 2],
        [0, 2, 0, 2, 0, 0, 2, 0, 0, 0],
        [0, 2, 0, 2, 0, 2, 2, 2, 2, 0],
        [0, 2, 0, 2, 0, 4, 2, 4, 0, 0],
        [0, 0, 0, 2, 0, 0, 0, 0, 0, 0],
    ],
    # Map 3 – chokepoint (3 enemies)
    [
        [3, 0, 2, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 2, 0, 0, 0, 0, 0, 0, 4],
        [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
        [0, 4, 2, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 0, 0, 4, 0, 4, 2, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
    ],
    # Map 4 – sneaky enemies (5 enemies)
    [
        [3, 0, 0, 0, 0, 0, 0, 4, 0, 0],
        [0, 2, 0, 2, 0, 0, 2, 0, 2, 0],
        [0, 0, 0, 0, 4, 0, 0, 0, 0, 0],
        [0, 2, 0, 2, 0, 0, 2, 0, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 2, 0, 2, 0, 0, 2, 0, 2, 0],
        [4, 0, 0, 0, 0, 0, 0, 0, 0, 4],
        [0, 2, 0, 2, 0, 0, 2, 0, 2, 0],
        [0, 0, 0, 0, 0, 4, 0, 0, 0, 0],
        [0, 2, 0, 2, 0, 0, 2, 0, 2, 0],
    ],
]


def make_env(obs_fn, obs_space_fn, reward_fn, use_random_maps=False):
    """Create a wrapped env with the given obs/reward configuration."""
    if use_random_maps:
        base = gym.make("standard", render_mode=None)
    else:
        base = gym.make("standard", render_mode=None,
                        predefined_map_list=PREDEFINED_MAPS)
    return ExperimentWrapper(base, obs_fn, obs_space_fn(), reward_fn)


# ===========================================================================
# Main – run experiments then train final model
# ===========================================================================

if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Experiment matrix:  2 obs spaces × 3 reward functions = 6 runs
    # ------------------------------------------------------------------
    experiments = [
        ("Obs1+R1 (Flat/Sparse)",      obs1_fn, obs1_space, reward1),
        ("Obs1+R2 (Flat/Shaped)",      obs1_fn, obs1_space, reward2),
        ("Obs1+R3 (Flat/Dense)",       obs1_fn, obs1_space, reward3),
        ("Obs2+R1 (Local/Sparse)",     obs2_fn, obs2_space, reward1),
        ("Obs2+R2 (Local/Shaped)",     obs2_fn, obs2_space, reward2),
        ("Obs2+R3 (Local/Dense-BEST)", obs2_fn, obs2_space, reward3),
    ]

    all_results = {}

    for label, obs_fn, obs_sp_fn, rew_fn in experiments:
        print(f"\n{'='*60}")
        print(f"  Experiment: {label}")
        print(f"  Timesteps:  {EXPERIMENT_STEPS:,}")
        print(f"{'='*60}")

        env = make_env(obs_fn, obs_sp_fn, rew_fn, use_random_maps=True)
        cb  = TrainingCallback()

        model = PPO(
            "MlpPolicy",
            env,
            learning_rate = 3e-4,
            n_steps       = 2048,
            batch_size    = 64,
            n_epochs      = 10,
            gamma         = 0.99,
            ent_coef      = 0.01,
            verbose       = 0,
        )
        model.learn(total_timesteps=EXPERIMENT_STEPS, callback=cb)
        env.close()

        fname = label.replace(" ", "_").replace("/", "_").replace("+", "_")
        plot_experiment(cb, label, fname)
        all_results[label] = cb

        if cb.coverage_ratios:
            last50 = cb.coverage_ratios[-50:]
            print(f"  Avg coverage (last 50 eps): {np.mean(last50):.3f}")

    plot_comparison(all_results)

    # ------------------------------------------------------------------
    # Final best model: Obs2 + Reward3, two-phase training
    # Phase 1 (70%): random maps  → broad generalisation
    # Phase 2 (30%): predefined   → refinement on known layouts
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  FINAL TRAINING  (Obs2 + R3, {FINAL_STEPS:,} timesteps)")
    print(f"{'='*60}")

    env_random = make_env(obs2_fn, obs2_space, reward3, use_random_maps=True)
    final_cb   = TrainingCallback()

    final_model = PPO(
        "MlpPolicy",
        env_random,
        learning_rate = 3e-4,
        n_steps       = 2048,
        batch_size    = 64,
        n_epochs      = 10,
        gamma         = 0.99,
        ent_coef      = 0.01,
        verbose       = 1,
    )

    phase1_steps = int(FINAL_STEPS * 0.7)
    final_model.learn(total_timesteps=phase1_steps, callback=final_cb)
    env_random.close()

    env_predefined = make_env(obs2_fn, obs2_space, reward3, use_random_maps=False)
    final_cb2      = TrainingCallback()
    phase2_steps   = FINAL_STEPS - phase1_steps
    final_model.set_env(env_predefined)
    final_model.learn(total_timesteps=phase2_steps, callback=final_cb2,
                      reset_num_timesteps=False)
    env_predefined.close()

    final_model.save("best_model")
    print("\n  [saved] best_model.zip")

    # Combined training curve plot
    combined          = TrainingCallback()
    combined.episode_rewards = final_cb.episode_rewards + final_cb2.episode_rewards
    combined.coverage_ratios = final_cb.coverage_ratios + final_cb2.coverage_ratios
    plot_experiment(combined, "Final Model (Obs2+R3, 2M steps)", "final_model")

    if combined.coverage_ratios:
        last100 = combined.coverage_ratios[-100:]
        print(f"\n  Final avg coverage (last 100 eps): {np.mean(last100):.3f}")

    print("\nDone! Check plots/ for all figures and best_model.zip for submission.")
