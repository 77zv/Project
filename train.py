"""
CISC 474 – Coverage Tournament  ·  Training Script
====================================================
Implements:
  • 2 CNN observation spaces  (Obs1 = single-channel ID grid, Obs2 = 5-channel semantic)
  • 3 reward functions        (R1 sparse, R2 shaped, R3 dense progress)
  • PPO + custom small CNN feature extractor (Stable Baselines 3)
  • Experiment loop with plot generation
  • Final best-model training (two-phase: random maps → predefined maps)

Run:   python3 train.py
Output:
  • plots/  – one plot per experiment + comparison plot
  • best_model.zip  – final trained model for the tournament
"""

import os
import numpy as np
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch as th
import torch.nn as nn

import coverage_gridworld  # noqa: F401  registers environments

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GRID_SIZE        = 10
EXPERIMENT_STEPS = 300_000
FINAL_STEPS      = 2_000_000

os.makedirs("plots", exist_ok=True)

# ---------------------------------------------------------------------------
# Color → ID helpers
# ---------------------------------------------------------------------------
_BLACK     = (0,   0,   0)
_WHITE     = (255, 255, 255)
_BROWN     = (101, 67,  33)
_GREY      = (160, 161, 161)
_GREEN     = (31,  198, 0)
_RED       = (255, 0,   0)
_LIGHT_RED = (255, 127, 127)

_COLOR_MAP = {
    _BLACK: 0, _WHITE: 1, _BROWN: 2, _GREY: 3,
    _GREEN: 4, _RED: 5, _LIGHT_RED: 6,
}

def _rgb_to_id(rgb):
    return _COLOR_MAP.get((int(rgb[0]), int(rgb[1]), int(rgb[2])), 0)

def _to_id_grid(grid):
    ids = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            ids[r, c] = _rgb_to_id(grid[r, c])
    return ids


# ===========================================================================
# Custom CNN Feature Extractor
# ===========================================================================
# SB3's default NatureCNN is designed for 84×84 images and uses large kernels
# (8×8, 4×4) that are inappropriate for a 10×10 grid. This small CNN uses
# 3×3 kernels with padding to preserve spatial dimensions.

class SmallGridCNN(BaseFeaturesExtractor):
    """
    Two-layer CNN for 10×10 grid observations.
    Expected input shape: (C, H, W)  – channel-first, as required by PyTorch.
    """
    def __init__(self, observation_space, features_dim: int = 128):
        super().__init__(observation_space, features_dim)

        n_channels = observation_space.shape[0]   # (C, H, W) → C
        h          = observation_space.shape[1]
        w          = observation_space.shape[2]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # After two same-padding convolutions the spatial size is unchanged
        flat_size = 32 * h * w
        self.head = nn.Sequential(
            nn.Linear(flat_size, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs: th.Tensor) -> th.Tensor:
        return self.head(self.cnn(obs))


CNN_POLICY_KWARGS = dict(
    features_extractor_class=SmallGridCNN,
    features_extractor_kwargs=dict(features_dim=128),
)


# ===========================================================================
# OBSERVATION SPACE 1 – Single-channel ID grid  (CNN)
# ===========================================================================
# The full 10×10 grid encoded as a single float channel.
# Shape: (1, 10, 10),  values: 0.0–6.0
# Rationale: Complete global view; simple baseline for CNN comparison.

def obs1_space():
    return gym.spaces.Box(low=0.0, high=6.0,
                          shape=(1, GRID_SIZE, GRID_SIZE),
                          dtype=np.float32)

def obs1_fn(grid):
    """Full grid as a single-channel (1, 10, 10) float image."""
    return _to_id_grid(grid).reshape(1, GRID_SIZE, GRID_SIZE).astype(np.float32)


# ===========================================================================
# OBSERVATION SPACE 2 – 5-channel semantic grid  (CNN, BEST)
# ===========================================================================
# Each channel is a binary mask for one semantic category:
#   0 – agent position
#   1 – explored cells
#   2 – unexplored cells
#   3 – enemy FOV (danger)
#   4 – obstacles (walls + enemies)
# Shape: (5, 10, 10),  values: 0.0 or 1.0
# Rationale: Clean per-channel signals let the CNN learn specialized filters.
# Translation-invariant and generalises to unseen maps.

def obs2_space():
    return gym.spaces.Box(low=0.0, high=1.0,
                          shape=(5, GRID_SIZE, GRID_SIZE),
                          dtype=np.float32)

def obs2_fn(grid):
    """5-channel binary (5, 10, 10) semantic image."""
    ids = _to_id_grid(grid)
    ch = np.stack([
        (ids == 3).astype(np.float32),                  # agent
        ((ids == 1) | (ids == 6)).astype(np.float32),   # explored
        (ids == 0).astype(np.float32),                  # unexplored
        ((ids == 5) | (ids == 6)).astype(np.float32),   # enemy FOV
        ((ids == 2) | (ids == 4)).astype(np.float32),   # obstacles
    ], axis=0)
    return ch


# ===========================================================================
# REWARD FUNCTION 1 – Sparse
# ===========================================================================
def reward1(info):
    r = 0.0
    if info["new_cell_covered"]:
        r += 1.0
    if info["game_over"]:
        r -= 50.0
    return r


# ===========================================================================
# REWARD FUNCTION 2 – Shaped (completion bonus + time penalty)
# ===========================================================================
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
# Experiment Wrapper
# ===========================================================================

class ExperimentWrapper(gym.Wrapper):
    def __init__(self, env, obs_fn, obs_space, reward_fn):
        super().__init__(env)
        self._obs_fn    = obs_fn
        self._reward_fn = reward_fn
        self.observation_space = obs_space

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        return self._obs_fn(self.env.unwrapped.grid), info

    def step(self, action):
        _, _, terminated, truncated, info = self.env.step(action)
        obs = self._obs_fn(self.env.unwrapped.grid)
        rew = self._reward_fn(info)
        return obs, rew, terminated, truncated, info


# ===========================================================================
# Training callback
# ===========================================================================

class TrainingCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.coverage_ratios = []
        self._ep_reward = 0.0
        self._last_info = {}

    def _on_step(self):
        self._ep_reward += float(self.locals["rewards"][0])
        if self.locals.get("infos"):
            self._last_info = self.locals["infos"][0]
        if self.locals["dones"][0]:
            self.episode_rewards.append(self._ep_reward)
            if self._last_info:
                c = self._last_info.get("total_covered_cells", 1)
                t = self._last_info.get("coverable_cells", 1)
                self.coverage_ratios.append(c / max(t, 1))
            self._ep_reward = 0.0
        return True


# ===========================================================================
# Plotting
# ===========================================================================

def _smooth(v, w=20):
    if len(v) < w:
        return v
    return np.convolve(v, np.ones(w) / w, mode="valid")

def plot_experiment(cb, label, filename):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    axes[0].plot(_smooth(cb.episode_rewards), label=label)
    axes[0].set_title(f"{label} – Episode Reward")
    axes[0].set_xlabel("Episode"); axes[0].set_ylabel("Total Reward")
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(_smooth(cb.coverage_ratios), color="orange", label=label)
    axes[1].set_title(f"{label} – Coverage Ratio")
    axes[1].set_xlabel("Episode"); axes[1].set_ylabel("Coverage fraction")
    axes[1].set_ylim(0, 1.05); axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(f"plots/{filename}.png", dpi=120)
    plt.close()
    print(f"  [plot saved] plots/{filename}.png")

def plot_comparison(results):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for label, cb in results.items():
        axes[0].plot(_smooth(cb.episode_rewards, 30), label=label)
        axes[1].plot(_smooth(cb.coverage_ratios, 30), label=label)
    axes[0].set_title("Episode Reward – All Experiments")
    axes[0].set_xlabel("Episode"); axes[0].set_ylabel("Reward (smoothed)")
    axes[0].legend(fontsize=8); axes[0].grid(True)
    axes[1].set_title("Coverage Ratio – All Experiments")
    axes[1].set_xlabel("Episode"); axes[1].set_ylabel("Coverage (smoothed)")
    axes[1].set_ylim(0, 1.05); axes[1].legend(fontsize=8); axes[1].grid(True)
    plt.tight_layout()
    plt.savefig("plots/comparison.png", dpi=120)
    plt.close()
    print("  [plot saved] plots/comparison.png")


# ===========================================================================
# Maps
# ===========================================================================

PREDEFINED_MAPS = [
    [[3,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0,0,0]],
    [[3,0,0,2,0,2,0,0,0,0],[0,2,0,0,0,2,0,0,2,0],[0,2,0,2,2,2,2,2,2,0],
     [0,2,0,0,0,2,0,0,0,0],[0,2,0,2,0,2,0,0,2,0],[0,2,0,0,0,0,0,2,0,0],
     [0,2,2,2,0,0,0,2,0,0],[0,0,0,0,0,2,0,0,2,0],[0,2,0,2,0,2,2,0,0,0],
     [0,0,0,0,0,2,0,0,0,0]],
    [[3,2,0,0,0,0,2,0,0,0],[0,2,0,2,2,0,2,0,2,2],[0,2,0,2,0,0,2,0,0,0],
     [0,2,0,2,0,2,2,2,2,0],[0,2,0,2,0,0,2,0,0,0],[0,2,0,2,2,0,2,0,2,2],
     [0,2,0,2,0,0,2,0,0,0],[0,2,0,2,0,2,2,2,2,0],[0,2,0,2,0,4,2,4,0,0],
     [0,0,0,2,0,0,0,0,0,0]],
    [[3,0,2,0,0,0,0,2,0,0],[0,0,2,0,0,0,0,0,0,4],[0,0,2,0,0,0,0,2,0,0],
     [0,0,2,0,0,0,0,2,0,0],[0,4,2,0,0,0,0,2,0,0],[0,0,2,0,0,0,0,2,0,0],
     [0,0,2,0,0,0,0,2,0,0],[0,0,2,0,0,0,0,2,0,0],[0,0,0,0,4,0,4,2,0,0],
     [0,0,0,0,0,0,0,2,0,0]],
    [[3,0,0,0,0,0,0,4,0,0],[0,2,0,2,0,0,2,0,2,0],[0,0,0,0,4,0,0,0,0,0],
     [0,2,0,2,0,0,2,0,2,0],[0,0,0,0,0,0,0,0,0,0],[0,2,0,2,0,0,2,0,2,0],
     [4,0,0,0,0,0,0,0,0,4],[0,2,0,2,0,0,2,0,2,0],[0,0,0,0,0,4,0,0,0,0],
     [0,2,0,2,0,0,2,0,2,0]],
]


def make_env(obs_fn, obs_space_fn, reward_fn, use_random_maps=False):
    if use_random_maps:
        base = gym.make("standard", render_mode=None)
    else:
        base = gym.make("standard", render_mode=None,
                        predefined_map_list=PREDEFINED_MAPS)
    return ExperimentWrapper(base, obs_fn, obs_space_fn(), reward_fn)


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    experiments = [
        ("Obs1+R1 (ID-grid/Sparse)",      obs1_fn, obs1_space, reward1),
        ("Obs1+R2 (ID-grid/Shaped)",      obs1_fn, obs1_space, reward2),
        ("Obs1+R3 (ID-grid/Dense)",       obs1_fn, obs1_space, reward3),
        ("Obs2+R1 (Semantic/Sparse)",     obs2_fn, obs2_space, reward1),
        ("Obs2+R2 (Semantic/Shaped)",     obs2_fn, obs2_space, reward2),
        ("Obs2+R3 (Semantic/Dense-BEST)", obs2_fn, obs2_space, reward3),
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
            "CnnPolicy",
            env,
            policy_kwargs   = CNN_POLICY_KWARGS,
            learning_rate   = 3e-4,
            n_steps         = 2048,
            batch_size      = 64,
            n_epochs        = 10,
            gamma           = 0.99,
            ent_coef        = 0.01,
            verbose         = 0,
        )
        model.learn(total_timesteps=EXPERIMENT_STEPS, callback=cb)
        env.close()

        fname = label.replace(" ", "_").replace("/", "_").replace("+", "_")
        plot_experiment(cb, label, fname)
        all_results[label] = cb

        if cb.coverage_ratios:
            print(f"  Avg coverage (last 50 eps): {np.mean(cb.coverage_ratios[-50:]):.3f}")

    plot_comparison(all_results)

    # ------------------------------------------------------------------
    # Final model: Obs2 + R3 + CNN, two-phase training
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  FINAL TRAINING  (Obs2-CNN + R3, {FINAL_STEPS:,} timesteps)")
    print(f"{'='*60}")

    env_random = make_env(obs2_fn, obs2_space, reward3, use_random_maps=True)
    final_cb   = TrainingCallback()

    final_model = PPO(
        "CnnPolicy",
        env_random,
        policy_kwargs   = CNN_POLICY_KWARGS,
        learning_rate   = 3e-4,
        n_steps         = 2048,
        batch_size      = 64,
        n_epochs        = 10,
        gamma           = 0.99,
        ent_coef        = 0.01,
        verbose         = 1,
    )

    # Phase 1 (70%): random maps → generalisation
    final_model.learn(total_timesteps=int(FINAL_STEPS * 0.7), callback=final_cb)
    env_random.close()

    # Phase 2 (30%): predefined maps → refinement
    env_pre   = make_env(obs2_fn, obs2_space, reward3, use_random_maps=False)
    final_cb2 = TrainingCallback()
    final_model.set_env(env_pre)
    final_model.learn(total_timesteps=FINAL_STEPS - int(FINAL_STEPS * 0.7),
                      callback=final_cb2, reset_num_timesteps=False)
    env_pre.close()

    final_model.save("best_model")
    print("\n  [saved] best_model.zip")

    combined = TrainingCallback()
    combined.episode_rewards = final_cb.episode_rewards + final_cb2.episode_rewards
    combined.coverage_ratios = final_cb.coverage_ratios + final_cb2.coverage_ratios
    plot_experiment(combined, "Final Model (Obs2-CNN + R3, 2M steps)", "final_model")

    if combined.coverage_ratios:
        print(f"\n  Final avg coverage (last 100 eps): {np.mean(combined.coverage_ratios[-100:]):.3f}")

    print("\nDone! Check plots/ for figures and best_model.zip for submission.")
