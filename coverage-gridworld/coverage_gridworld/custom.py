import numpy as np
import gymnasium as gym

"""
Feel free to modify the functions below and experiment with different environment configurations.
"""

# Color constants matching env.py
_BLACK = (0, 0, 0)
_WHITE = (255, 255, 255)
_BROWN = (101, 67, 33)
_GREY = (160, 161, 161)
_GREEN = (31, 198, 0)
_RED = (255, 0, 0)
_LIGHT_RED = (255, 127, 127)

_GRID_SIZE = 10
_LOCAL_VIEW = 5
_HALF = _LOCAL_VIEW // 2

# Maps each RGB tuple to a compact integer ID (0-6)
_COLOR_MAP = {
    _BLACK:     0,  # unexplored
    _WHITE:     1,  # explored
    _BROWN:     2,  # wall
    _GREY:      3,  # agent
    _GREEN:     4,  # enemy
    _RED:       5,  # unexplored + enemy FOV
    _LIGHT_RED: 6,  # explored + enemy FOV
}


def _rgb_to_id(rgb):
    """Convert an RGB cell to its integer ID."""
    return _COLOR_MAP.get((int(rgb[0]), int(rgb[1]), int(rgb[2])), 0)


def _to_id_grid(grid):
    """Vectorised RGB → ID conversion for the full 10×10 grid."""
    id_grid = np.zeros((_GRID_SIZE, _GRID_SIZE), dtype=np.int8)
    for r in range(_GRID_SIZE):
        for c in range(_GRID_SIZE):
            id_grid[r, c] = _rgb_to_id(grid[r, c])
    return id_grid


def _find_agent(id_grid):
    """Return (row, col) of the agent (GREY cell, ID = 3)."""
    pos = np.argwhere(id_grid == 3)
    if len(pos):
        return int(pos[0][0]), int(pos[0][1])
    return 0, 0


# ---------------------------------------------------------------------------
# Observation Space 2 (BEST – used for the tournament submission)
# ---------------------------------------------------------------------------
# A compact 27-element vector:
#   - 25 floats: 5×5 local view centred on the agent (cell IDs 0-6)
#   - 1 float:   map coverage ratio ∈ [0, 1]
#   - 1 float:   binary flag – is any cell in the local view under enemy FOV?
# This local-relative design generalises well to unseen maps.
# ---------------------------------------------------------------------------

def observation_space(env: gym.Env) -> gym.spaces.Space:
    """
    Observation Space 2: 5×5 local view around the agent + 2 global scalars.
    """
    n = _LOCAL_VIEW * _LOCAL_VIEW + 2
    low = np.zeros(n, dtype=np.float32)
    high = np.concatenate([
        np.full(_LOCAL_VIEW * _LOCAL_VIEW, 6.0, dtype=np.float32),
        np.array([1.0, 1.0], dtype=np.float32),
    ])
    return gym.spaces.Box(low=low, high=high, dtype=np.float32)


def observation(grid: np.ndarray):
    """
    Returns a 27-element float32 vector:
      [local_5x5 (25 values)] + [coverage_ratio, enemy_threat_flag]
    """
    id_grid = _to_id_grid(grid)
    ar, ac = _find_agent(id_grid)

    # 5×5 local view; out-of-bounds cells default to wall (ID = 2)
    local = np.full((_LOCAL_VIEW, _LOCAL_VIEW), 2.0, dtype=np.float32)
    for di in range(-_HALF, _HALF + 1):
        for dj in range(-_HALF, _HALF + 1):
            r, c = ar + di, ac + dj
            if 0 <= r < _GRID_SIZE and 0 <= c < _GRID_SIZE:
                local[di + _HALF, dj + _HALF] = float(id_grid[r, c])

    # Coverage ratio (visited cells / all coverable cells)
    explored = int(np.sum((id_grid == 1) | (id_grid == 3) | (id_grid == 6)))
    total = _GRID_SIZE * _GRID_SIZE - int(np.sum(id_grid == 2)) - int(np.sum(id_grid == 4))
    coverage = float(explored) / max(total, 1)

    # Binary flag: is any cell in the local view under enemy surveillance?
    enemy_threat = float(np.any((local == 5) | (local == 6)))

    return np.concatenate([local.flatten(), [coverage, enemy_threat]]).astype(np.float32)


# ---------------------------------------------------------------------------
# Reward Function 3 (BEST – used for the tournament submission)
# ---------------------------------------------------------------------------
# Components:
#   +1 base reward for each newly covered cell, scaled by a progress bonus
#   +100 bonus for completing the map
#   -50 penalty for being caught by an enemy
#   -0.01 small per-step penalty (encourages speed, ties in tournament)
# ---------------------------------------------------------------------------

def reward(info: dict) -> float:
    """
    Reward Function 3: shaped reward with progress bonus.
    """
    new_cell_covered   = info["new_cell_covered"]
    game_over          = info["game_over"]
    cells_remaining    = info["cells_remaining"]
    coverable_cells    = info["coverable_cells"]
    total_covered_cells = info["total_covered_cells"]

    r = 0.0

    if new_cell_covered:
        # Scale reward by coverage progress so the agent is motivated to
        # keep exploring even when most of the map is already covered.
        progress = total_covered_cells / max(coverable_cells, 1)
        r += 1.0 + progress

    if cells_remaining == 0:
        r += 100.0          # completion bonus

    if game_over:
        r -= 50.0           # caught by enemy

    r -= 0.01               # per-step time penalty

    return float(r)
