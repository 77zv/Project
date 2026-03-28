import numpy as np
import gymnasium as gym

"""
Feel free to modify the functions below and experiment with different environment configurations.
"""

# Color constants matching env.py
_BLACK     = (0,   0,   0)
_WHITE     = (255, 255, 255)
_BROWN     = (101, 67,  33)
_GREY      = (160, 161, 161)
_GREEN     = (31,  198, 0)
_RED       = (255, 0,   0)
_LIGHT_RED = (255, 127, 127)

_GRID_SIZE = 10

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
    return _COLOR_MAP.get((int(rgb[0]), int(rgb[1]), int(rgb[2])), 0)


def _to_id_grid(grid):
    id_grid = np.zeros((_GRID_SIZE, _GRID_SIZE), dtype=np.int8)
    for r in range(_GRID_SIZE):
        for c in range(_GRID_SIZE):
            id_grid[r, c] = _rgb_to_id(grid[r, c])
    return id_grid


# ---------------------------------------------------------------------------
# Observation Space 2 – CNN (BEST, used for tournament submission)
# ---------------------------------------------------------------------------
# 5-channel binary image of shape (5, 10, 10):
#   Channel 0 – agent position          (1 where agent is)
#   Channel 1 – explored cells          (1 where visited)
#   Channel 2 – unexplored safe cells   (1 where not yet visited and not dangerous)
#   Channel 3 – enemy FOV / danger      (1 where RED or LIGHT_RED)
#   Channel 4 – obstacles               (1 where wall or enemy)
#
# Rationale: Each channel gives the CNN a clean binary signal to learn from.
# The spatial structure is preserved so the network can detect local patterns
# (e.g. "enemy FOV ahead, wall to the left"). Generalises well to unseen maps
# because the representation is relative to the current grid state.
# ---------------------------------------------------------------------------

def observation_space(env: gym.Env) -> gym.spaces.Space:
    """
    Observation Space 2 CNN: 5-channel 10×10 binary image, shape (5, 10, 10).
    """
    return gym.spaces.Box(low=0.0, high=1.0,
                          shape=(5, _GRID_SIZE, _GRID_SIZE),
                          dtype=np.float32)


def observation(grid: np.ndarray):
    """
    Returns a (5, 10, 10) float32 array with one binary channel per cell type.
    """
    ids = _to_id_grid(grid)

    ch_agent     = (ids == 3).astype(np.float32)
    ch_explored  = ((ids == 1) | (ids == 6)).astype(np.float32)
    ch_unexplored = (ids == 0).astype(np.float32)
    ch_danger    = ((ids == 5) | (ids == 6)).astype(np.float32)
    ch_obstacle  = ((ids == 2) | (ids == 4)).astype(np.float32)

    return np.stack([ch_agent, ch_explored, ch_unexplored,
                     ch_danger, ch_obstacle], axis=0)


# ---------------------------------------------------------------------------
# Reward Function 3 – Dense with progress bonus (BEST)
# ---------------------------------------------------------------------------

def reward(info: dict) -> float:
    new_cell_covered    = info["new_cell_covered"]
    game_over           = info["game_over"]
    cells_remaining     = info["cells_remaining"]
    coverable_cells     = info["coverable_cells"]
    total_covered_cells = info["total_covered_cells"]

    r = 0.0

    if new_cell_covered:
        progress = total_covered_cells / max(coverable_cells, 1)
        r += 1.0 + progress     # later cells are worth more

    if cells_remaining == 0:
        r += 100.0              # completion bonus

    if game_over:
        r -= 50.0               # caught by enemy

    r -= 0.01                   # per-step time penalty (rewards speed)

    return float(r)
