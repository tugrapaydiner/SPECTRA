"""ARC-style abstract grid reasoning task (BLUEPRINT section 24.3).

Full ARC is explicitly *not* a first build target (section 32). This is the
learnable "ARC-style" variant: a single grid transformation rule (reflect, rotate,
transpose, or recolor) applied to random small grids placed on a padded canvas.
The model learns to apply the rule; the ``arc_soft_score`` verifier (section 9.4)
checks structural plausibility. Difficulty scales with grid size and palette.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

# Transform library: each maps an HxW color grid to its transformed grid.
TRANSFORMS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "identity": lambda g: g.copy(),
    "flip_h": lambda g: np.fliplr(g).copy(),
    "flip_v": lambda g: np.flipud(g).copy(),
    "rotate180": lambda g: np.rot90(g, 2).copy(),
    "transpose": lambda g: g.T.copy(),
}


def random_grid(h: int, w: int, n_colors: int, rng: np.random.Generator) -> np.ndarray:
    """A random ``h x w`` grid with colors in ``1..n_colors`` (0 reserved for bg)."""
    return rng.integers(1, n_colors + 1, size=(h, w)).astype(np.int64)


def recolor(grid: np.ndarray, rng: np.random.Generator, n_colors: int) -> np.ndarray:
    """Apply a random color permutation (a learnable recolor rule)."""
    perm = rng.permutation(n_colors) + 1
    out = grid.copy()
    for c in range(1, n_colors + 1):
        out[grid == c] = perm[c - 1]
    return out


def pad_to_canvas(grid: np.ndarray, canvas_h: int, canvas_w: int, pad_token: int) -> np.ndarray:
    """Place ``grid`` at the top-left of a ``canvas_h x canvas_w`` padded canvas."""
    canvas = np.full((canvas_h, canvas_w), pad_token, dtype=np.int64)
    h, w = grid.shape
    canvas[:h, :w] = grid
    return canvas


def generate_pair(
    transform: str,
    rng: np.random.Generator,
    canvas_h: int = 10,
    canvas_w: int = 10,
    max_h: int = 6,
    max_w: int = 6,
    n_colors: int = 5,
    pad_token: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a flattened ``(input_canvas, output_canvas)`` ARC-style pair."""
    h = int(rng.integers(2, max_h + 1))
    w = int(rng.integers(2, max_w + 1))
    grid = random_grid(h, w, n_colors, rng)
    if transform == "recolor":
        out = recolor(grid, rng, n_colors)
    else:
        out = TRANSFORMS[transform](grid)
    x = pad_to_canvas(grid, canvas_h, canvas_w, pad_token)
    y = pad_to_canvas(out, canvas_h, canvas_w, pad_token)
    return x.reshape(-1), y.reshape(-1)
