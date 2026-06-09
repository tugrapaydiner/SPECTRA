#!/usr/bin/env python3
"""Exact memory-footprint chart for the SPECTRA recursive core (NO synthetic data).

Footprints are computed by the production packer (deploy/pack_ternary.packed_size_bytes);
cache ceilings are read from this host's real topology (/sys/.../cache).
    python scripts/render_footprint.py   ->   assets/01_core_memory_footprint.png
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from deploy.pack_ternary import packed_size_bytes

TERNARY_PARAMS = 5_600_000          # ternary core re-applied every recursion step
MB = 1024 * 1024

BG = "#0B0E14"; TEXT = "#E6EDF3"; MUTED = "#8B949E"; FAINT = "#6E7681"; GRID = "#1B2230"
TEAL = "#2BD4A7"; TEAL_GLOW = "#3DF2C0"; AMBER = "#E8B25A"; BLUE = "#5FB7F0"; RED = "#F2776E"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": TEXT, "axes.labelcolor": TEXT, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": GRID, "legend.frameon": False, "figure.dpi": 200,
})


def host_caches():
    base = Path("/sys/devices/system/cpu/cpu0/cache")
    out = {}
    try:
        for idx in base.glob("index*"):
            lvl = int((idx / "level").read_text())
            typ = (idx / "type").read_text().strip()
            s = (idx / "size").read_text().strip()
            kb = float(s.rstrip("K")) if s.endswith("K") else float(s.rstrip("M")) * 1024
            if lvl in (2, 3) and typ in ("Unified", "Data"):
                out[lvl] = kb / 1024.0
    except OSError:
        pass
    return out.get(2), out.get(3)


def render(out="assets/01_core_memory_footprint.png"):
    foot = [("FP32 core", TERNARY_PARAMS * 4 / MB, RED),
            ("FP16 core", TERNARY_PARAMS * 2 / MB, AMBER),
            ("INT8 core", TERNARY_PARAMS * 1 / MB, BLUE),
            ("W1.58 packed core", packed_size_bytes(TERNARY_PARAMS) / MB, TEAL_GLOW)]
    names = [f[0] for f in foot]
    vals = [f[1] for f in foot]
    cols = [f[2] for f in foot]
    y = list(range(len(foot)))[::-1]
    l2, l3 = host_caches()
    fp32, packed = vals[0], vals[-1]

    fig, ax = plt.subplots(figsize=(11.0, 5.9))
    if l2:
        ax.axvspan(0.7, l2, color=TEAL, alpha=0.06, zorder=0)

    for yi, (name, v, c) in zip(y, foot):
        for w, alp in [(15, 0.05), (10, 0.09)]:
            ax.barh(yi, v, height=0.018 * w, color=c, alpha=alp, zorder=2)
        ax.barh(yi, v, height=0.44, color=c, edgecolor=BG, linewidth=0.8, zorder=3)
        if name.startswith("W1.58"):
            ax.text(v * 1.10, yi + 0.12, f"{v:.2f} MB", va="center", color=c,
                    fontsize=12, fontweight="bold")
            ax.text(v * 1.10, yi - 0.16, f"{fp32/packed:.0f}x smaller than FP32",
                    va="center", color=TEAL, fontsize=10, fontweight="bold")
        elif name == "FP32 core":   # longest bar: label inside so the L3 line never crosses it
            ax.text(v * 0.95, yi, f"{v:.2f} MB", va="center", ha="right", color=BG,
                    fontsize=12, fontweight="bold")
        else:
            ax.text(v * 1.06, yi, f"{v:.2f} MB", va="center", color=c, fontsize=12, fontweight="bold")

    # real host cache ceilings + a legacy-target reference (so it isn't host-specific)
    if l2:
        ax.axvline(l2, color=TEAL, lw=1.5, alpha=0.9, zorder=4)
        ax.text(l2 * 1.04, len(foot) - 0.40, f"L2  {l2:.2f} MB", color=TEAL, fontsize=9.5,
                ha="left", va="bottom", fontweight="bold")
    if l3:
        ax.axvline(l3, color=BLUE, lw=1.5, ls=(0, (6, 4)), alpha=0.9, zorder=4)
        ax.text(l3, len(foot) - 0.40, f"L3  {l3:.0f} MB", color=BLUE, fontsize=9.5,
                ha="center", va="bottom", fontweight="bold")
    ax.axvline(8.0, color=FAINT, lw=1.1, ls=(0, (1, 3)), alpha=0.75, zorder=4)
    ax.text(8.0, len(foot) - 0.40, "legacy L3 ~8 MB", color=FAINT, fontsize=9, ha="center", va="bottom")

    ax.set_xscale("log")
    ax.set_xlim(0.7, fp32 * 3.0)
    ax.set_ylim(-0.7, len(foot) - 0.2)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=11.5)
    ax.set_xlabel("Recursive-core weight footprint   (MB, log scale)", fontsize=12)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(True, axis="x", color=GRID, alpha=0.5, linewidth=0.9)
    ax.tick_params(length=0)

    ax.set_title("1.58-bit packing shrinks the reasoning core 16x - into cache",
                 color="#FFFFFF", fontsize=16.5, fontweight="bold", loc="left", pad=30)
    ax.annotate("Recursive core = 5.6M ternary params - exact bytes from the production packer",
                (0, 1), xytext=(0, 10), xycoords="axes fraction", textcoords="offset points",
                color=MUTED, fontsize=10.5, va="bottom")
    l2s = f"{l2:.2f}" if l2 else "n/a"; l3s = f"{l3:.0f}" if l3 else "n/a"
    fig.text(0.012, 0.013,
             f"Exact: deploy/pack_ternary.py   |   host cache from /sys (L2 {l2s} MB, L3 {l3s} MB)   |   {date.today().isoformat()}",
             color=FAINT, fontsize=8, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(ROOT / out, bbox_inches="tight", pad_inches=0.28); plt.close(fig)
    print("[render] 01_core_memory_footprint.png")


if __name__ == "__main__":
    render()
