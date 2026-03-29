import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Predefined maps from coverage_gridworld registration (5 named maps)
# 0=empty, 2=wall, 3=start, 4=enemy
MAPS = {
    "just_go": np.array([
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
    ], dtype=int),
    "safe": np.array([
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
    ], dtype=int),
    "maze": np.array([
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
    ], dtype=int),
    "chokepoint": np.array([
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
    ], dtype=int),
    "sneaky_enemies": np.array([
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
    ], dtype=int),
}

FOV_DISTANCE = 4
ORIENTATION_NAMES = ["LEFT", "DOWN", "RIGHT", "UP"]


def _enemy_positions(map_data: np.ndarray):
    return list(zip(*np.where(map_data == 4)))


def _is_visible_cell(map_data: np.ndarray, r: int, c: int) -> bool:
    rows, cols = map_data.shape
    if r < 0 or c < 0 or r >= rows or c >= cols:
        return False
    # FOV ray stops at walls and other enemies, matching env.py behavior.
    return map_data[r, c] not in (2, 4)


def _fov_cells_for_enemy(map_data: np.ndarray, enemy_r: int, enemy_c: int, orientation: int, fov_distance: int):
    cells = []
    for i in range(1, fov_distance + 1):
        if orientation == 0:      # LEFT
            rr, cc = enemy_r, enemy_c - i
        elif orientation == 1:    # DOWN
            rr, cc = enemy_r + i, enemy_c
        elif orientation == 2:    # RIGHT
            rr, cc = enemy_r, enemy_c + i
        else:                     # UP
            rr, cc = enemy_r - i, enemy_c

        if _is_visible_cell(map_data, rr, cc):
            cells.append((rr, cc))
        else:
            break
    return cells


def _phase_fov_union(map_data: np.ndarray, phase_orientation: int, fov_distance: int):
    fov_union = set()
    for er, ec in _enemy_positions(map_data):
        fov_union.update(_fov_cells_for_enemy(map_data, er, ec, phase_orientation, fov_distance))
    return fov_union


def _all_orientation_union(map_data: np.ndarray, fov_distance: int):
    fov_union = set()
    for ori in range(4):
        fov_union.update(_phase_fov_union(map_data, ori, fov_distance))
    return fov_union


def _base_color_grid(map_data: np.ndarray):
    rows, cols = map_data.shape
    color_grid = np.zeros((rows, cols, 3), dtype=float)
    color_grid[map_data == 0] = [0.95, 0.95, 0.95]  # empty
    color_grid[map_data == 2] = [0.45, 0.29, 0.14]  # wall
    color_grid[map_data == 3] = [0.55, 0.55, 0.55]  # start
    color_grid[map_data == 4] = [0.12, 0.75, 0.12]  # enemy
    return color_grid


def _style_axis(ax, rows, cols):
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=0.8)
    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))
    ax.set_xticklabels([f"c{i+1}" for i in range(cols)], fontsize=8)
    ax.set_yticklabels([f"r{i+1}" for i in range(rows)], fontsize=8)


def _annotate_special_cells(ax, map_data: np.ndarray):
    rows, cols = map_data.shape
    for r in range(rows):
        for c in range(cols):
            if map_data[r, c] == 3:
                ax.text(c, r, "S", ha="center", va="center", fontsize=10, weight="bold")
            elif map_data[r, c] == 4:
                ax.text(c, r, "E", ha="center", va="center", fontsize=10, weight="bold")


def draw_maze(map_data: np.ndarray, out_path: str = "plots/maze_map.png", title: str = "Coverage Gridworld - Map Layout") -> None:
    rows, cols = map_data.shape
    color_grid = _base_color_grid(map_data)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(color_grid, origin="upper")

    _style_axis(ax, rows, cols)
    _annotate_special_cells(ax, map_data)

    legend_patches = [
        mpatches.Patch(color=[0.95, 0.95, 0.95], label="Empty"),
        mpatches.Patch(color=[0.45, 0.29, 0.14], label="Wall"),
        mpatches.Patch(color=[0.55, 0.55, 0.55], label="Start (S)"),
        mpatches.Patch(color=[0.12, 0.75, 0.12], label="Enemy (E)"),
    ]
    ax.legend(handles=legend_patches, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)

    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def draw_maze_with_fov_phases(
    map_data: np.ndarray,
    out_path: str = "plots/maze_map_fov_phases.png",
    fov_distance: int = FOV_DISTANCE,
) -> None:
    rows, cols = map_data.shape
    color_grid = _base_color_grid(map_data)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()

    for ori in range(4):
        ax = axes[ori]
        ax.imshow(color_grid, origin="upper")
        _style_axis(ax, rows, cols)

        fov_cells = _phase_fov_union(map_data, ori, fov_distance)
        for r, c in fov_cells:
            rect = plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color=(1.0, 0.0, 0.0, 0.35))
            ax.add_patch(rect)

        _annotate_special_cells(ax, map_data)
        ax.set_title(f"FOV Phase: {ORIENTATION_NAMES[ori]}")

    legend_patches = [
        mpatches.Patch(color=[0.95, 0.95, 0.95], label="Empty"),
        mpatches.Patch(color=[0.45, 0.29, 0.14], label="Wall"),
        mpatches.Patch(color=[0.55, 0.55, 0.55], label="Start (S)"),
        mpatches.Patch(color=[0.12, 0.75, 0.12], label="Enemy (E)"),
        mpatches.Patch(color=[1.0, 0.4, 0.4], label="Enemy FOV cells"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.02))
    fig.suptitle("Coverage Gridworld - Maze with Enemy FOV by Orientation Phase", fontsize=14)
    plt.tight_layout(rect=[0, 0.06, 1, 0.97])
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def draw_maze_fov_union(
    map_data: np.ndarray,
    out_path: str = "plots/maze_map_fov_union.png",
    fov_distance: int = FOV_DISTANCE,
) -> None:
    rows, cols = map_data.shape
    color_grid = _base_color_grid(map_data)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(color_grid, origin="upper")
    _style_axis(ax, rows, cols)

    union_cells = _all_orientation_union(map_data, fov_distance)
    for r, c in union_cells:
        rect = plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color=(1.0, 0.0, 0.0, 0.35))
        ax.add_patch(rect)

    _annotate_special_cells(ax, map_data)
    ax.set_title("Coverage Gridworld - Maze with FOV Union (All 4 Orientations)")

    legend_patches = [
        mpatches.Patch(color=[0.95, 0.95, 0.95], label="Empty"),
        mpatches.Patch(color=[0.45, 0.29, 0.14], label="Wall"),
        mpatches.Patch(color=[0.55, 0.55, 0.55], label="Start (S)"),
        mpatches.Patch(color=[0.12, 0.75, 0.12], label="Enemy (E)"),
        mpatches.Patch(color=[1.0, 0.4, 0.4], label="Seen in at least one orientation"),
    ]
    ax.legend(handles=legend_patches, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    for map_name, map_data in MAPS.items():
        out = f"plots/{map_name}_map.png"
        draw_maze(map_data, out_path=out, title=f"Coverage Gridworld - {map_name} layout")
        print(f"Saved: {out}")
