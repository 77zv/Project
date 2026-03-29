"""
CISC 474 – Coverage Tournament  ·  Training Script
====================================================
Implements:
    • 4 CNN observation spaces  (Obs1 = single-channel ID grid, Obs2 = 5-channel semantic, Obs3 = timing-aware, Obs4 = 2-step forecast)
    • 7 reward functions        (R1 sparse, R2 shaped, R3 dense progress, R4 anti-stall, R5 timing-aware, R6 breakthrough, R7 aggressive explore)
  • PPO + custom small CNN feature extractor (Stable Baselines 3)
  • Experiment loop with plot generation
  • Final best-model training (two-phase: random maps → predefined maps)

Run:   python3 train.py
Output:
  • plots/  – one plot per experiment + comparison plot
  • best_model.zip  – final trained model for the tournament
"""

import os
import argparse
import inspect
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
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GRID_SIZE        = 10
EXPERIMENT_STEPS = 300_000
FINAL_STEPS      = 2_000_000
QUICK_STEPS      = 20_000

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
# OBSERVATION SPACE 3 – 6-channel semantic + next-step danger prediction
# ===========================================================================
# Channels:
#   0 – agent position
#   1 – explored cells
#   2 – unexplored cells
#   3 – current enemy FOV (danger now)
#   4 – obstacles (walls + enemies)
#   5 – predicted enemy FOV after next rotation step (danger next)

def obs3_space():
    return gym.spaces.Box(low=0.0, high=1.0,
                          shape=(6, GRID_SIZE, GRID_SIZE),
                          dtype=np.float32)


def _project_enemy_fov(ids: np.ndarray, enemy_row: int, enemy_col: int, orientation: int, distance: int):
    cells = []
    for i in range(1, distance + 1):
        if orientation == 0:      # LEFT
            rr, cc = enemy_row, enemy_col - i
        elif orientation == 1:    # DOWN
            rr, cc = enemy_row + i, enemy_col
        elif orientation == 2:    # RIGHT
            rr, cc = enemy_row, enemy_col + i
        else:                     # UP
            rr, cc = enemy_row - i, enemy_col

        if rr < 0 or cc < 0 or rr >= GRID_SIZE or cc >= GRID_SIZE:
            break
        if ids[rr, cc] in (2, 4):
            break

        cells.append((rr, cc))
    return cells


def obs3_fn(grid, env):
    """6-channel semantic image with predicted danger on the next step."""
    ids = _to_id_grid(grid)
    ch_agent = (ids == 3).astype(np.float32)
    ch_explored = ((ids == 1) | (ids == 6)).astype(np.float32)
    ch_unexplored = (ids == 0).astype(np.float32)
    ch_danger_now = ((ids == 5) | (ids == 6)).astype(np.float32)
    ch_obstacle = ((ids == 2) | (ids == 4)).astype(np.float32)

    ch_danger_next = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    enemy_fov_distance = int(getattr(env, "enemy_fov_distance", 4))
    for enemy in getattr(env, "enemy_list", []):
        next_orientation = (int(enemy.orientation) + 1) % 4
        for rr, cc in _project_enemy_fov(ids, int(enemy.y), int(enemy.x), next_orientation, enemy_fov_distance):
            ch_danger_next[rr, cc] = 1.0

    return np.stack([
        ch_agent,
        ch_explored,
        ch_unexplored,
        ch_danger_now,
        ch_obstacle,
        ch_danger_next,
    ], axis=0)


# ===========================================================================
# OBSERVATION SPACE 4 – 7-channel semantic + 2-step danger forecast
# ===========================================================================
# Channels:
#   0..5 -> Obs3 channels
#   6    -> predicted danger after two rotations (danger in t+2)


# ===========================================================================
# OBSERVATION SPACE 5 – 9-channel: semantic + FULL 4-phase danger cycle
# ===========================================================================
# Because enemies rotate on a 4-step cycle, showing all 4 danger maps gives
# the agent complete timing information without needing recurrence.
# Channels:
#   0 – agent position
#   1 – explored cells
#   2 – unexplored cells
#   3 – obstacles (walls + enemies)
#   4 – danger at t+0  (now)
#   5 – danger at t+1
#   6 – danger at t+2
#   7 – danger at t+3  (completes the 4-step rotation cycle)
#   8 – always-unsafe mask (dangerous in ALL 4 phases → never enter)

def obs4_space():
    return gym.spaces.Box(low=0.0, high=1.0,
                          shape=(7, GRID_SIZE, GRID_SIZE),
                          dtype=np.float32)


def obs4_fn(grid, env):
    ids = _to_id_grid(grid)
    ch_agent = (ids == 3).astype(np.float32)
    ch_explored = ((ids == 1) | (ids == 6)).astype(np.float32)
    ch_unexplored = (ids == 0).astype(np.float32)
    ch_danger_now = ((ids == 5) | (ids == 6)).astype(np.float32)
    ch_obstacle = ((ids == 2) | (ids == 4)).astype(np.float32)

    ch_danger_next = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    ch_danger_next2 = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    enemy_fov_distance = int(getattr(env, "enemy_fov_distance", 4))

    for enemy in getattr(env, "enemy_list", []):
        ori = int(enemy.orientation)
        next_ori = (ori + 1) % 4
        next2_ori = (ori + 2) % 4

        for rr, cc in _project_enemy_fov(ids, int(enemy.y), int(enemy.x), next_ori, enemy_fov_distance):
            ch_danger_next[rr, cc] = 1.0
        for rr, cc in _project_enemy_fov(ids, int(enemy.y), int(enemy.x), next2_ori, enemy_fov_distance):
            ch_danger_next2[rr, cc] = 1.0

    return np.stack([
        ch_agent,
        ch_explored,
        ch_unexplored,
        ch_danger_now,
        ch_obstacle,
        ch_danger_next,
        ch_danger_next2,
    ], axis=0)


def obs5_space():
    return gym.spaces.Box(low=0.0, high=1.0,
                          shape=(9, GRID_SIZE, GRID_SIZE),
                          dtype=np.float32)


def obs5_fn(grid, env):
    """9-channel observation with the complete 4-phase enemy danger cycle.

    By showing danger at t, t+1, t+2, and t+3 simultaneously the agent can
    read off exactly which steps are safe for any given cell and learn to
    time its crossings without needing a recurrent policy.
    """
    ids = _to_id_grid(grid)
    ch_agent    = (ids == 3).astype(np.float32)
    ch_explored = ((ids == 1) | (ids == 6)).astype(np.float32)
    ch_unexplored = (ids == 0).astype(np.float32)
    ch_obstacle = ((ids == 2) | (ids == 4)).astype(np.float32)

    # Build danger map for each of the 4 relative phases
    phase_danger = [np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32) for _ in range(4)]
    enemy_fov_distance = int(getattr(env, "enemy_fov_distance", 4))
    for enemy in getattr(env, "enemy_list", []):
        ori = int(enemy.orientation)
        for phase in range(4):
            phase_ori = (ori + phase) % 4
            for rr, cc in _project_enemy_fov(ids, int(enemy.y), int(enemy.x),
                                             phase_ori, enemy_fov_distance):
                phase_danger[phase][rr, cc] = 1.0

    # Always-unsafe: cells dangerous in every phase (agent should never enter)
    always_unsafe = np.ones((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    for pd in phase_danger:
        always_unsafe *= pd  # product is 1 only where all 4 phases are 1

    return np.stack([
        ch_agent,
        ch_explored,
        ch_unexplored,
        ch_obstacle,
        phase_danger[0],   # danger now
        phase_danger[1],   # danger in 1 step
        phase_danger[2],   # danger in 2 steps
        phase_danger[3],   # danger in 3 steps  (full cycle)
        always_unsafe,
    ], axis=0)


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
# REWARD FUNCTION 4 – Dense + anti-stall (for enemy-heavy maps)
# ===========================================================================
def reward4(info):
    r = 0.0
    if info["new_cell_covered"]:
        progress = info["total_covered_cells"] / max(info["coverable_cells"], 1)
        r += 1.0 + progress
    else:
        # Stronger penalty for unproductive steps to avoid "hover in safe zone" local optima.
        r -= 0.08

    if info["cells_remaining"] == 0:
        r += 120.0
    if info["game_over"]:
        # Keep failure costly but not so dominant that agent avoids all risk forever.
        r -= 35.0
    return r


# ===========================================================================
# REWARD FUNCTION 5 – Timing-aware danger crossing
# ===========================================================================
def reward5(info):
    r = 0.0

    if info["new_cell_covered"]:
        progress = info["total_covered_cells"] / max(info["coverable_cells"], 1)
        r += 1.0 + progress
    else:
        # Slightly stronger anti-idle pressure than R3.
        r -= 0.03

    # Reward correct timing behavior: entering currently-dangerous cells that become safe after rotation.
    if info.get("successful_danger_crossing", False):
        r += 0.35

    if info["cells_remaining"] == 0:
        r += 110.0

    if info["game_over"]:
        r -= 35.0

    return r


# ===========================================================================
# REWARD FUNCTION 6 – Breakthrough-focused (anti-freeze + milestone bonuses)
# ===========================================================================
def reward6(info):
    r = 0.0
    coverable = max(info["coverable_cells"], 1)
    covered = info["total_covered_cells"]
    progress = covered / coverable

    if info["new_cell_covered"]:
        r += 1.1 + progress
    else:
        r -= 0.04

    if info.get("successful_danger_crossing", False):
        r += 0.6

    if info.get("stayed", False):
        r -= 0.10
    if info.get("blocked_move", False):
        r -= 0.08

    # One-time bonuses per episode when crossing key coverage milestones.
    if info["new_cell_covered"]:
        milestone_fracs = (0.62, 0.70, 0.78, 0.86, 0.94)
        for frac in milestone_fracs:
            threshold_cells = int(np.ceil(frac * coverable))
            if covered == threshold_cells:
                r += 12.0

    if info["cells_remaining"] == 0:
        r += 140.0
    if info["game_over"]:
        r -= 30.0
    return r


# ===========================================================================
# REWARD FUNCTION 7 – Aggressive exploration (simple baseline)
# ===========================================================================
def reward7(info):
    """Aggressive-explore: fixed to remove danger-crossing exploit."""
    r = 0.0

    if info["new_cell_covered"]:
        r += 3.0
    else:
        r -= 0.02

    if info.get("stayed", False):
        r -= 0.05

    if info["cells_remaining"] == 0:
        r += 120.0

    if info["game_over"]:
        r -= 10.0

    return r


# ===========================================================================
# REWARD FUNCTION 8 – Ultra-simple exploration-first (no time pressure)
# ===========================================================================
def reward8(info):
    """Simplest possible reward: big carrot (new cells), tiny stick (death).
    Expected full-coverage return: (cells*5)+250. Death=-3.
    Agent must explore ~2 new cells before a single death becomes worthwhile."""
    r = 0.0
    if info["new_cell_covered"]:
        r += 5.0
        # Progress-shaped bonus on newly discovered cells only:
        # bonus = (1 - cells_remaining / coverable_cells) * scale
        # This increases reward as the map gets closer to completion.
        coverable = max(float(info.get("coverable_cells", 1)), 1.0)
        remaining = max(float(info.get("cells_remaining", 0)), 0.0)
        progress = 1.0 - (remaining / coverable)
        r += 4.0 * progress
    if info["cells_remaining"] == 0:
        # Strong terminal bonus to favor converting near-perfect runs into full clears.
        r += 250.0
    if info["game_over"]:
        r -= 3.0
    return r


# ===========================================================================
# Experiment Wrapper
# ===========================================================================

class ExperimentWrapper(gym.Wrapper):
    def __init__(self, env, obs_fn, obs_space, reward_fn):
        super().__init__(env)
        self._obs_fn    = obs_fn
        self._reward_fn = reward_fn
        self._obs_accepts_env = len(inspect.signature(obs_fn).parameters) >= 2
        self.observation_space = obs_space

    def _build_obs(self):
        if self._obs_accepts_env:
            return self._obs_fn(self.env.unwrapped.grid, self.env.unwrapped)
        return self._obs_fn(self.env.unwrapped.grid)

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        return self._build_obs(), info

    def step(self, action):
        pre_grid = self.env.unwrapped.grid.copy()
        pre_agent_pos = int(self.env.unwrapped.agent_pos)
        grid_size = int(self.env.unwrapped.grid_size)

        ax = pre_agent_pos % grid_size
        ay = pre_agent_pos // grid_size
        movement = [(0, -1), (1, 0), (0, 1), (-1, 0)]

        attempted_move = action != 4
        attempted_danger_crossing = False
        if attempted_move and 0 <= action <= 3:
            ny = ay + movement[action][0]
            nx = ax + movement[action][1]
            if 0 <= nx < grid_size and 0 <= ny < grid_size:
                cell = tuple(int(v) for v in pre_grid[ny, nx])
                # RED or LIGHT_RED in the pre-step frame.
                if cell == _RED or cell == _LIGHT_RED:
                    attempted_danger_crossing = True

        _, _, terminated, truncated, info = self.env.step(action)

        post_agent_pos = int(self.env.unwrapped.agent_pos)
        moved = post_agent_pos != pre_agent_pos
        blocked_move = attempted_move and (not moved)
        stayed = action == 4

        successful_danger_crossing = attempted_danger_crossing and (not info.get("game_over", False))
        info["attempted_danger_crossing"] = attempted_danger_crossing
        info["successful_danger_crossing"] = successful_danger_crossing
        info["moved"] = moved
        info["blocked_move"] = blocked_move
        info["stayed"] = stayed

        obs = self._build_obs()
        rew = self._reward_fn(info)
        return obs, rew, terminated, truncated, info


# ===========================================================================
# Training callback
# ===========================================================================

class TrainingCallback(BaseCallback):
    def __init__(self, total_steps=None, print_every=2048, converge_threshold=None,
                 best_model_path=None, stop_on_first_full_clear=False):
        super().__init__()
        self.episode_rewards = []
        self.coverage_ratios = []
        self._ep_reward = 0.0
        self._last_info = {}
        self._total_steps = total_steps
        self._print_every = print_every
        self._last_print = 0
        self._start_step = 0          # set in _on_training_start
        self._converge_threshold = converge_threshold
        self._converge_streak = 0     # consecutive windows above threshold
        self._best_model_path = best_model_path   # if set, save best checkpoint here
        self._best_avg_cov = -1.0
        self._stop_on_first_full_clear = stop_on_first_full_clear

    def _on_training_start(self):
        self._start_step = self.num_timesteps
        self._last_print = self.num_timesteps

    def _on_step(self):
        self._ep_reward += float(self.locals["rewards"][0])
        if self.locals.get("infos"):
            self._last_info = self.locals["infos"][0]
        if self.locals["dones"][0]:
            self.episode_rewards.append(self._ep_reward)
            if self._last_info:
                c = self._last_info.get("total_covered_cells", 1)
                t = self._last_info.get("coverable_cells", 1)
                cov = c / max(t, 1)
                self.coverage_ratios.append(cov)

                # Optional strict stop: as soon as one 100% clear is observed.
                if self._stop_on_first_full_clear and c >= t:
                    print("  [early stop] first 100% episode reached")
                    return False
            self._ep_reward = 0.0

        # Print progress every N steps
        if self.num_timesteps - self._last_print >= self._print_every:
            self._last_print = self.num_timesteps
            stage_step = self.num_timesteps - self._start_step
            n_ep = len(self.episode_rewards)
            avg_cov_val = np.mean(self.coverage_ratios[-20:]) if self.coverage_ratios else None
            avg_cov50_val = np.mean(self.coverage_ratios[-50:]) if len(self.coverage_ratios) >= 50 else None
            avg_rew_val = np.mean(self.episode_rewards[-20:]) if self.episode_rewards else None
            avg_cov = f"{avg_cov_val:.2f}" if avg_cov_val is not None else "n/a"
            avg_rew = f"{avg_rew_val:.1f}" if avg_rew_val is not None else "n/a"
            progress = (
                f" ({100*stage_step/self._total_steps:.0f}%)"
                if self._total_steps else ""
            )
            print(
                f"  step {stage_step:>7,}{progress}"
                f"  |  ep: {n_ep:>4}"
                f"  |  avg_cov: {avg_cov}"
                f"  |  avg_rew: {avg_rew}"
            )

            # Save best model checkpoint if this window is a new best
            if self._best_model_path and avg_cov_val is not None and avg_cov_val > self._best_avg_cov:
                self._best_avg_cov = avg_cov_val
                self.model.save(self._best_model_path)

            # Early stopping if converged
            if self._converge_threshold is not None and len(self.coverage_ratios) >= 50:
                if avg_cov50_val is not None and avg_cov50_val >= self._converge_threshold:
                    self._converge_streak += 1
                    if self._converge_streak >= 3:
                        print(f"  [early stop] avg_cov(last50) {avg_cov50_val:.2f} >= {self._converge_threshold} for 3 checks")
                        return False
                else:
                    self._converge_streak = 0

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


def make_env(obs_fn, obs_space_fn, reward_fn, use_random_maps=False, named_map=None, env_kwargs=None):
    kwargs = env_kwargs or {}
    if named_map is not None:
        base = gym.make(named_map, render_mode=None, **kwargs)
    elif use_random_maps:
        base = gym.make("standard", render_mode=None, **kwargs)
    else:
        base = gym.make("standard", render_mode=None,
                        predefined_map_list=PREDEFINED_MAPS, **kwargs)
    return ExperimentWrapper(base, obs_fn, obs_space_fn(), reward_fn)


def make_train_env(obs_fn, obs_space_fn, reward_fn,
                   use_random_maps=False, named_map=None, env_kwargs=None,
                   frame_stack=1):
    env = make_env(
        obs_fn,
        obs_space_fn,
        reward_fn,
        use_random_maps=use_random_maps,
        named_map=named_map,
        env_kwargs=env_kwargs,
    )
    if frame_stack <= 1:
        return env

    # Stack recent observations along channel dimension so PPO can infer motion/phase.
    vec_env = DummyVecEnv([lambda e=env: e])
    return VecFrameStack(vec_env, n_stack=frame_stack, channels_order="first")


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO on coverage-gridworld")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a single obs/reward pair for fast iteration (skips final training phase).",
    )
    parser.add_argument(
        "--obs",
        choices=["1", "2", "3", "4", "5"],
        default="2",
        help="Observation function: 1=Obs1, 2=Obs2, 3=Obs3(next-step), 4=Obs4(2-step), 5=Obs5(full 4-phase cycle).",
    )
    parser.add_argument(
        "--reward",
        choices=["1", "2", "3", "4", "5", "6", "7", "8"],
        default="3",
        help="Reward function: 1-6=previous, 7=R7(aggressive-explore fixed), 8=R8(ultra-simple exploration).",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Override experiment timesteps. Defaults: quick=50k, full=300k.",
    )
    parser.add_argument(
        "--final-steps",
        type=int,
        default=None,
        help="Override final training timesteps in full mode (default: 2,000,000).",
    )
    parser.add_argument(
        "--skip-final",
        action="store_true",
        help="Skip final two-phase training in full mode.",
    )
    parser.add_argument(
        "--map",
        choices=["random", "just_go", "safe", "maze", "maze_snippet", "chokepoint", "sneaky_enemies"],
        default="random",
        help="Map to use in quick mode. Default: random. Includes 'maze_snippet' for focused 6x4 timing-gate practice.",
    )
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="Run staged training in one process (just_go -> safe -> maze -> random) while keeping the same model.",
    )
    parser.add_argument(
        "--stack",
        type=int,
        default=1,
        help="Frame stack depth for temporal context (default: 1). Try 4 for timing-heavy maps.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="PPO learning rate (default: 3e-4).",
    )
    parser.add_argument(
        "--ent",
        type=float,
        default=0.01,
        help="PPO entropy coefficient (default: 0.01). Increase to encourage exploration.",
    )
    return parser.parse_args()


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    args = parse_args()

    obs_lookup = {
        "1": ("Obs1", obs1_fn, obs1_space),
        "2": ("Obs2", obs2_fn, obs2_space),
        "3": ("Obs3", obs3_fn, obs3_space),
        "4": ("Obs4", obs4_fn, obs4_space),
        "5": ("Obs5", obs5_fn, obs5_space),
    }
    rew_lookup = {
        "1": ("R1", reward1),
        "2": ("R2", reward2),
        "3": ("R3", reward3),
        "4": ("R4", reward4),
        "5": ("R5", reward5),
        "6": ("R6", reward6),
        "7": ("R7", reward7),
        "8": ("R8", reward8),
    }

    if args.curriculum:
        obs_name, obs_fn, obs_sp_fn = obs_lookup[args.obs]
        rew_name, rew_fn = rew_lookup[args.reward]
        stage_steps = args.timesteps if args.timesteps is not None else QUICK_STEPS
        # (stage_name, named_map, use_random, env_kwargs, converge_threshold)
        # converge_threshold: stop early if avg_cov (last 50 eps) >= this for 3 checks; None = never stop early
        stages = [
            ("just_go",   "just_go", False, {},                                                              0.92),
            ("safe",      "safe",    False, {},                                                              0.90),
            ("maze",      "maze",    False, {},                                                              0.75),
            ("random_e1", None,      True,  {"num_enemies": 1, "num_walls": 8,  "enemy_fov_distance": 2},  None),
            ("random_e2", None,      True,  {"num_enemies": 2, "num_walls": 10, "enemy_fov_distance": 3},  None),
            ("random_e3", None,      True,  {"num_enemies": 3, "num_walls": 12, "enemy_fov_distance": 3},  None),
            ("random_e4", None,      True,  {"num_enemies": 4, "num_walls": 12, "enemy_fov_distance": 4},  None),
            ("random_e5", None,      True,  {"num_enemies": 5, "num_walls": 12, "enemy_fov_distance": 4},  None),
        ]

        print(f"\n{'='*60}")
        print(f"  CURRICULUM MODE: {obs_name}+{rew_name}")
        print(f"  Stages: {', '.join([s[0] for s in stages])}")
        print(f"  Timesteps per stage: {stage_steps:,}")
        print(f"{'='*60}")

        all_results = {}
        model = None

        for i, (stage_name, named_map, use_random, stage_kwargs, converge_thr) in enumerate(stages):
            label = f"{obs_name}+{rew_name} (Curriculum/{stage_name})"
            conv_str = f"{converge_thr}" if converge_thr is not None else "none"
            print(f"\n{'='*60}")
            print(f"  Stage: {label}")
            print(f"  Max timesteps: {stage_steps:,}  |  early-stop threshold: {conv_str}")
            print(f"{'='*60}")

            env = make_train_env(
                obs_fn,
                obs_sp_fn,
                rew_fn,
                use_random_maps=use_random,
                named_map=named_map,
                env_kwargs=stage_kwargs,
                frame_stack=args.stack,
            )
            cb = TrainingCallback(total_steps=stage_steps, converge_threshold=converge_thr)

            if model is None:
                model = PPO(
                    "CnnPolicy",
                    env,
                    policy_kwargs=CNN_POLICY_KWARGS,
                    learning_rate=args.lr,
                    n_steps=2048,
                    batch_size=64,
                    n_epochs=10,
                    gamma=0.99,
                    ent_coef=args.ent,
                    verbose=0,
                )
            else:
                model.set_env(env)

            model.learn(total_timesteps=stage_steps, callback=cb, reset_num_timesteps=(i == 0))
            env.close()

            fname = label.replace(" ", "_").replace("/", "_").replace("+", "_")
            plot_experiment(cb, label, fname)
            all_results[label] = cb

            if cb.coverage_ratios:
                print(f"  Avg coverage (last 50 eps): {np.mean(cb.coverage_ratios[-50:]):.3f}")

        plot_comparison(all_results)
        model.save("curriculum_model")
        print("\n  [saved] curriculum_model.zip")
        print("\nDone! Curriculum run complete. Check plots/ and curriculum_model.zip.")
        raise SystemExit(0)

    if args.quick:
        obs_name, obs_fn, obs_sp_fn = obs_lookup[args.obs]
        rew_name, rew_fn = rew_lookup[args.reward]
        map_tag = f"/{args.map}" if args.map != "random" else ""
        experiments = [
            (f"{obs_name}+{rew_name} (Quick{map_tag})", obs_fn, obs_sp_fn, rew_fn),
        ]
        experiment_steps = args.timesteps if args.timesteps is not None else QUICK_STEPS
        quick_map = None if args.map == "random" else args.map
    else:
        quick_map = None
        experiments = [
            ("Obs1+R1 (ID-grid/Sparse)",      obs1_fn, obs1_space, reward1),
            ("Obs1+R2 (ID-grid/Shaped)",      obs1_fn, obs1_space, reward2),
            ("Obs1+R3 (ID-grid/Dense)",       obs1_fn, obs1_space, reward3),
            ("Obs2+R1 (Semantic/Sparse)",     obs2_fn, obs2_space, reward1),
            ("Obs2+R2 (Semantic/Shaped)",     obs2_fn, obs2_space, reward2),
            ("Obs2+R3 (Semantic/Dense-BEST)", obs2_fn, obs2_space, reward3),
        ]
        experiment_steps = args.timesteps if args.timesteps is not None else EXPERIMENT_STEPS

    run_final_training = (not args.quick) and (not args.skip_final)
    final_steps = args.final_steps if args.final_steps is not None else FINAL_STEPS

    all_results = {}

    for label, obs_fn, obs_sp_fn, rew_fn in experiments:
        print(f"\n{'='*60}")
        print(f"  Experiment: {label}")
        print(f"  Timesteps:  {experiment_steps:,}")
        print(f"{'='*60}")

        _use_random = not args.quick or args.map == "random"
        _named_map  = quick_map if args.quick else None
        env = make_train_env(
            obs_fn,
            obs_sp_fn,
            rew_fn,
            use_random_maps=_use_random,
            named_map=_named_map,
            frame_stack=args.stack,
        )

        # In quick mode: stop only when sustained perfection is reached over the last 50 episodes,
        # and checkpoint the best policy so
        # late-training degradation (PPO overcorrecting a converged policy) is ignored.
        _best_ckpt = "best_model_ckpt" if args.quick else None
        _conv_thr  = 1.00 if args.quick else None
        cb  = TrainingCallback(total_steps=experiment_steps,
                               converge_threshold=_conv_thr,
                       best_model_path=_best_ckpt,
                       stop_on_first_full_clear=False)

        model = PPO(
            "CnnPolicy",
            env,
            policy_kwargs   = CNN_POLICY_KWARGS,
            learning_rate   = args.lr,
            n_steps         = 2048,
            batch_size      = 64,
            n_epochs        = 10,
            gamma           = 0.99,
            ent_coef        = args.ent,
            verbose         = 0,
        )
        model.learn(total_timesteps=experiment_steps, callback=cb)

        # Restore the best checkpoint from training (guards against end-of-run degradation)
        import os
        if _best_ckpt and os.path.exists(_best_ckpt + ".zip"):
            model = PPO.load(_best_ckpt, env=env)
            print(f"  [checkpoint] Restored best model (avg_cov: {cb._best_avg_cov:.3f})")

        env.close()

        fname = label.replace(" ", "_").replace("/", "_").replace("+", "_")
        plot_experiment(cb, label, fname)
        all_results[label] = cb

        if cb.coverage_ratios:
            print(f"  Avg coverage (last 50 eps): {np.mean(cb.coverage_ratios[-50:]):.3f}")

    plot_comparison(all_results)

    if not run_final_training:
        print("\nDone! Quick/full experiments complete. Check plots/ for figures.")
        raise SystemExit(0)

    # ------------------------------------------------------------------
    # Final model: Obs2 + R3 + CNN, two-phase training
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  FINAL TRAINING  (Obs2-CNN + R3, {final_steps:,} timesteps)")
    print(f"{'='*60}")

    env_random = make_train_env(obs2_fn, obs2_space, reward3, use_random_maps=True, frame_stack=args.stack)
    final_cb   = TrainingCallback(total_steps=int(final_steps * 0.7))

    final_model = PPO(
        "CnnPolicy",
        env_random,
        policy_kwargs   = CNN_POLICY_KWARGS,
        learning_rate   = args.lr,
        n_steps         = 2048,
        batch_size      = 64,
        n_epochs        = 10,
        gamma           = 0.99,
        ent_coef        = args.ent,
        verbose         = 1,
    )

    # Phase 1 (70%): random maps -> generalisation
    final_model.learn(total_timesteps=int(final_steps * 0.7), callback=final_cb)
    env_random.close()

    # Phase 2 (30%): predefined maps → refinement
    env_pre   = make_train_env(obs2_fn, obs2_space, reward3, use_random_maps=False, frame_stack=args.stack)
    final_cb2 = TrainingCallback(total_steps=final_steps - int(final_steps * 0.7))
    final_model.set_env(env_pre)
    final_model.learn(total_timesteps=final_steps - int(final_steps * 0.7),
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
