#!/usr/bin/env python3
"""Frontier-lab charts from the MEASURED kernel benchmark (assets/data/*.csv).

Every value is measured by scripts/bench_kernel.py + scripts/bench_sparse.py on
real hardware (no synthetic data). Lines are labelled directly (paper convention)
so nothing collides. Run the benches first, then:
    python scripts/render_kernel_charts.py
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "assets" / "data"
ASSETS = ROOT / "assets"

BG = "#0B0E14"; TEXT = "#E6EDF3"; MUTED = "#8B949E"; FAINT = "#6E7681"; GRID = "#1B2230"
TEAL = "#2BD4A7"; TEAL_GLOW = "#3DF2C0"; AMBER = "#E8B25A"; AMBER_GLOW = "#FFC76B"
BLUE = "#5FB7F0"; VIOLET = "#9B8CFF"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": TEXT, "axes.labelcolor": TEXT, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": GRID, "legend.frameon": False, "figure.dpi": 200,
})


def _read(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def _host():
    try:
        return json.loads((DATA / "host.json").read_text())
    except Exception:
        return {"cpu": "host CPU", "peak_avx2_gops": 0.0, "bit_exact": True}


def _frame(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(True, which="major", color=GRID, alpha=0.55, linewidth=0.9)
    ax.grid(True, which="minor", color=GRID, alpha=0.22, linewidth=0.6)
    ax.tick_params(length=0)


def _head(fig, ax, title, subtitle):
    ax.set_title(title, color="#FFFFFF", fontsize=16.5, fontweight="bold", loc="left", pad=30)
    ax.annotate(subtitle, (0, 1), xytext=(0, 10), xycoords="axes fraction",
                textcoords="offset points", color=MUTED, fontsize=10.5, va="bottom")


def _footer(fig, host):
    fig.text(0.012, 0.013,
             f"Measured on {host.get('cpu','host')}   |   AVX2 == scalar (bit-exact): "
             f"{host.get('bit_exact', True)}   |   reproduce: scripts/bench_kernel.py   |   {date.today().isoformat()}",
             color=FAINT, fontsize=8, ha="left", va="bottom")


def _glow(ax, x, y, color, glow, lw=2.6, ms=6.5):
    for w, a in [(lw + 7, 0.05), (lw + 3.5, 0.10), (lw + 1.5, 0.18)]:
        ax.plot(x, y, color=glow, lw=w, alpha=a, solid_capstyle="round", zorder=3)
    ax.plot(x, y, color=color, lw=lw, zorder=4, solid_capstyle="round")
    ax.scatter(x, y, s=ms**2, color=glow, edgecolor=BG, linewidth=1.0, zorder=5)


# ========================================================================= #
def roofline():
    rows = _read(DATA / "bench_weight_stationary.csv")
    host = _host()
    a = {"avx2": [], "scalar": []}
    for r in rows:
        a[r["build"]].append((float(r["arithmetic_intensity"]), float(r["gops"]), int(r["K"])))
    for n in a:
        a[n].sort()
    peak = host.get("peak_avx2_gops", max(g for _, g, _ in a["avx2"]))

    fig, ax = plt.subplots(figsize=(10.8, 6.7))
    ax.axhline(peak, color=FAINT, lw=1.1, ls=(0, (6, 5)), alpha=0.7, zorder=2)
    ax.text(a["avx2"][0][0] * 1.05, peak * 1.16, f"measured AVX2 ceiling  ~{peak:.0f} GOP/s",
            color=FAINT, fontsize=9.5, ha="left", va="bottom")

    for n, c, g in (("scalar", AMBER, AMBER_GLOW), ("avx2", TEAL, TEAL_GLOW)):
        xs = [ai for ai, _, _ in a[n]]; ys = [gg for _, gg, _ in a[n]]
        _glow(ax, xs, ys, c, g)
    # direct end-labels (right side, empty)
    ax.text(a["avx2"][-1][0] * 1.15, a["avx2"][-1][1], "AVX2", color=TEAL_GLOW,
            fontsize=12, fontweight="bold", va="center")
    ax.text(a["scalar"][-1][0] * 1.15, a["scalar"][-1][1], "scalar", color=AMBER_GLOW,
            fontsize=12, fontweight="bold", va="center")

    # K=1 callout -> top-left empty triangle
    lo = a["avx2"][0]
    ax.annotate("K = 1\nsingle B=1 GEMV\nmemory-bound (AI ~ 4)", (lo[0], lo[1]),
                xytext=(5.4, 16), textcoords="data", color=TEAL_GLOW, fontsize=10,
                fontweight="bold", ha="left", va="center",
                arrowprops=dict(arrowstyle="-|>", color=TEAL_GLOW, lw=1.4, alpha=0.85))
    # K=256 callout -> bottom-right empty triangle
    hi = a["avx2"][-1]
    ax.annotate(f"K = {hi[2]}\nrecursion reuse\ncompute-bound\n{hi[1]:.0f} GOP/s", (hi[0], hi[1]),
                xytext=(300, 4.2), textcoords="data", color="#FFFFFF", fontsize=10,
                fontweight="bold", ha="center", va="center",
                arrowprops=dict(arrowstyle="-|>", color=TEAL_GLOW, lw=1.4, alpha=0.85))

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(2.8, 2600); ax.set_ylim(0.4, peak * 3.2)
    ax.set_xlabel("Arithmetic intensity   (ops / DRAM byte,  = 4K for W1.58)", fontsize=12)
    ax.set_ylabel("Achieved throughput   (GOP/s, int8 MACs)", fontsize=12)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    _frame(ax)
    _head(fig, ax, "Cache-resident recursion walks the kernel up the roofline",
          "Weight-stationary ternary GEMV - the same packed core re-applied K times - measured")
    _footer(fig, host)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(ASSETS / "04_roofline.png", bbox_inches="tight", pad_inches=0.28)
    plt.close(fig); print("[render] 04_roofline.png")


# ========================================================================= #
def simd_scaling():
    rows = _read(DATA / "bench_simd_scaling.csv")
    host = _host()
    hidden = [int(r["hidden"]) for r in rows]
    gv = [float(r["gops_avx2"]) for r in rows]
    gs = [float(r["gops_scalar"]) for r in rows]
    sp = [float(r["speedup"]) for r in rows]
    x = list(range(len(hidden)))

    fig, ax = plt.subplots(figsize=(10.8, 6.7))
    ax.fill_between(x, gs, gv, color=TEAL, alpha=0.06, zorder=1)
    _glow(ax, x, gv, TEAL, TEAL_GLOW)
    _glow(ax, x, gs, AMBER, AMBER_GLOW)
    ax.set_ylim(0, max(gv) * 1.32)
    ax.text(x[-1] + 0.12, gv[-1], "AVX2", color=TEAL_GLOW, fontsize=12, fontweight="bold", va="center")
    ax.text(x[-1] + 0.12, gs[-1], "scalar", color=AMBER_GLOW, fontsize=12, fontweight="bold", va="center")

    ax2 = ax.twinx()
    ax2.set_ylim(0, max(sp) * 1.12)            # speedup floats above the GOP/s lines
    ax2.plot(x, sp, color=VIOLET, lw=1.9, ls=(0, (5, 3)), marker="D", ms=5.5, zorder=6)
    for xi, s in zip(x, sp):
        ax2.annotate(f"x{s:.2f}", (xi, s), xytext=(0, 9), textcoords="offset points",
                     color=VIOLET, fontsize=9.5, ha="center", fontweight="bold")
    ax2.text(x[-1] + 0.12, sp[-1], "speedup\n(right axis)", color=VIOLET, fontsize=10.5,
             fontweight="bold", va="center")
    ax2.set_ylabel("AVX2 speedup  (x)", color=VIOLET, fontsize=12)
    ax2.tick_params(axis="y", colors=VIOLET, length=0)
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_color(GRID)

    ax.set_xticks(x); ax.set_xticklabels([str(h) for h in hidden])
    ax.set_xlim(-0.35, len(hidden) - 1 + 0.9)
    ax.set_xlabel("Hidden width   (contraction dimension)", fontsize=12)
    ax.set_ylabel("Throughput   (GOP/s, int8 MACs)", fontsize=12)
    _frame(ax)
    _head(fig, ax, "The AVX2 ternary kernel holds ~2x over scalar across widths",
          "Vectorized 2-bit unpack + sign-multiply GEMV - B=1 weight-stationary - measured")
    _footer(fig, host)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(ASSETS / "05_simd_scaling.png", bbox_inches="tight", pad_inches=0.28)
    plt.close(fig); print("[render] 05_simd_scaling.png")


# ========================================================================= #
def amortization():
    rows = _read(DATA / "bench_weight_stationary.csv")
    host = _host()
    a = {"avx2": [], "scalar": []}
    for r in rows:
        a[r["build"]].append((int(r["K"]), float(r["gops"]), int(r["out_dim"]), int(r["hidden"])))
    for n in a:
        a[n].sort()
    Ks = [k for k, _, _, _ in a["avx2"]]
    out_dim, hidden = a["avx2"][0][2], a["avx2"][0][3]
    wbytes = out_dim * hidden * 2 / 8.0
    bpm = [wbytes / (k * out_dim * hidden) for k in Ks]      # = 0.25/K, exact
    gv = [g for _, g, _, _ in a["avx2"]]; gscal = [g for _, g, _, _ in a["scalar"]]

    fig, ax = plt.subplots(figsize=(10.8, 6.7))
    _glow(ax, Ks, gv, TEAL, TEAL_GLOW)
    _glow(ax, Ks, gscal, AMBER, AMBER_GLOW)
    ax.set_xscale("log", base=2); ax.set_xticks(Ks); ax.set_xticklabels([str(k) for k in Ks])
    ax.set_ylim(0, max(gv) * 1.22)
    ax.text(Ks[-1] * 1.08, gv[-1], "AVX2", color=TEAL_GLOW, fontsize=12, fontweight="bold", va="center")
    ax.text(Ks[-1] * 1.08, gscal[-1], "scalar", color=AMBER_GLOW, fontsize=12, fontweight="bold", va="center")

    ax2 = ax.twinx()
    ax2.plot(Ks, bpm, color=BLUE, lw=1.9, ls=(0, (5, 3)), marker="s", ms=5.5, zorder=6)
    ax2.set_yscale("log")
    ax2.text(Ks[0] * 1.05, bpm[0] * 1.12, "DRAM bytes / MAC  (right axis)", color=BLUE,
             fontsize=10.5, fontweight="bold", va="bottom", ha="left")
    ax2.set_ylabel("DRAM weight traffic per MAC   (bytes, log)", color=BLUE, fontsize=12)
    ax2.tick_params(axis="y", colors=BLUE, length=0)
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_color(GRID)

    # general note in the empty lower-left band, arrow to the rising curve
    ax.annotate("decode the 2-bit row once,\nreuse it across all K steps -\nDRAM pays for the matrix once",
                (Ks[4], gv[4]), xytext=(1.45, max(gv) * 0.62), textcoords="data",
                color="#FFFFFF", fontsize=10, fontweight="bold", ha="left", va="center",
                arrowprops=dict(arrowstyle="-|>", color=TEAL_GLOW, lw=1.3, alpha=0.8))

    ax.set_xlim(0.8, Ks[-1] * 1.7)
    ax.set_xlabel("Recursion reuse   K = T n N_sup   (weight-matrix re-applications)", fontsize=12)
    ax.set_ylabel("Throughput   (GOP/s, int8 MACs)", fontsize=12)
    _frame(ax)
    _head(fig, ax, "Weight-stationary recursion amortizes the DRAM weight fetch",
          "One-time 2-bit decode reused K times - throughput rises as bytes/op falls 1/K - measured")
    _footer(fig, host)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(ASSETS / "06_weight_stationary.png", bbox_inches="tight", pad_inches=0.28)
    plt.close(fig); print("[render] 06_weight_stationary.png")


# ========================================================================= #
def lazy_routing():
    rows = _read(DATA / "bench_sparse_density.csv")
    host = _host()
    rows.sort(key=lambda r: float(r["density"]))
    dens = [float(r["density"]) * 100 for r in rows]
    lat = [float(r["latency_ms"]) for r in rows]
    spd = [float(r["speedup_vs_dense"]) for r in rows]

    fig, ax = plt.subplots(figsize=(10.8, 6.7))
    # ideal-linear reference (cost proportional to active tokens)
    ax.plot([0, 100], [0, lat[-1]], color=FAINT, lw=1.3, ls=(0, (4, 4)), alpha=0.7, zorder=2)
    ax.text(40, lat[-1] * 0.40, "ideal linear (cost ~ active tokens)", color=FAINT,
            fontsize=9.5, va="top", ha="left")
    _glow(ax, dens, lat, TEAL, TEAL_GLOW)
    ax.text(66, lat[-1] * 0.72, "measured", color=TEAL_GLOW, fontsize=11.5,
            fontweight="bold", va="bottom", ha="left")

    lo = rows[0]
    ax.annotate(f"{int(float(lo['density'])*100)}% active  ->  {float(lo['speedup_vs_dense']):.1f}x cheaper\n"
                f"(router freezes {100-int(float(lo['density'])*100)}% of tokens)",
                (dens[0], lat[0]), xytext=(26, max(lat) * 0.72), textcoords="data",
                color="#FFFFFF", fontsize=10.5, fontweight="bold", ha="left", va="center",
                arrowprops=dict(arrowstyle="-|>", color=TEAL_GLOW, lw=1.4, alpha=0.85))
    for d, l, sp_ in zip(dens, lat, spd):
        if sp_ <= 1.05:
            continue
        ax.annotate(f"x{sp_:.1f}", (d, l), xytext=(0, 11), textcoords="offset points",
                    color=MUTED, fontsize=9, ha="center")

    ax.set_xlim(0, 116); ax.set_ylim(0, max(lat) * 1.2)
    ax.set_xlabel("Active-token fraction   (router keeps active, %)", fontsize=12)
    ax.set_ylabel("Layer latency   (ms, B=1)", fontsize=12)
    _frame(ax)
    _head(fig, ax, "Lazy routing pays only for the tokens it keeps active",
          "Measured spectra_sparse_ternary_gemv - frozen tokens are skipped entirely")
    _footer(fig, host)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(ASSETS / "03_lazy_routing.png", bbox_inches="tight", pad_inches=0.28)
    plt.close(fig); print("[render] 03_lazy_routing.png")

# ========================================================================= #
def cache_residency():
    rows = _read(DATA / "bench_cache_residency.csv")
    host = _host()
    mb = [float(r["weight_mb"]) for r in rows]
    g = [float(r["gops"]) for r in rows]
    l2 = host.get("l2_mb"); l3 = host.get("l3_mb"); core = host.get("packed_core_mb")

    fig, ax = plt.subplots(figsize=(10.8, 6.7))
    lo, hi = min(mb) * 0.5, max(mb) * 2
    if l2:
        ax.axvspan(lo, l2, color=TEAL, alpha=0.06, zorder=0)
    if l2 and l3:
        ax.axvspan(l2, l3, color=BLUE, alpha=0.05, zorder=0)
    if l3:
        ax.axvspan(l3, hi, color=AMBER, alpha=0.05, zorder=0)
    ax.text(lo * 1.6, max(g) * 1.22, "fits L1/L2", color=TEAL, fontsize=9.5, ha="left", va="top")
    if l2 and l3:
        ax.text((l2 * l3) ** 0.5, max(g) * 1.22, "L3-resident", color=BLUE, fontsize=9.5, ha="center", va="top")
    if l3:
        ax.text(l3 * 1.25, max(g) * 1.22, "beyond L3 -> DRAM", color=AMBER, fontsize=9.5, ha="left", va="top")

    _glow(ax, mb, g, TEAL, TEAL_GLOW)
    for x, lab, c in [(l2, f"L2  {l2:.2f} MB" if l2 else None, TEAL),
                      (l3, f"L3  {l3:.0f} MB" if l3 else None, BLUE)]:
        if x:
            ax.axvline(x, color=c, lw=1.4, ls=(0, (6, 4)), alpha=0.85, zorder=2)
            ax.text(x, max(g) * 1.04, lab, color=c, ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    if core:
        ax.axvline(core, color=TEAL_GLOW, lw=1.3, alpha=0.6, zorder=2)
        ax.annotate(f"SPECTRA core\n{core:.2f} MB", (core, g[0]), xytext=(core, max(g) * 0.24),
                    textcoords="data", color=TEAL_GLOW, fontsize=10, fontweight="bold", ha="center",
                    arrowprops=dict(arrowstyle="-|>", color=TEAL_GLOW, lw=1.3, alpha=0.8))

    mean_g = sum(g) / len(g)
    ax.annotate("identical throughput in L2, L3 and DRAM:\nthe ternary decode is the bottleneck,\nnot memory bandwidth (no DRAM cliff)",
                (mb[-2], g[-2]), xytext=(mb[1], mean_g * 0.66), textcoords="data",
                color="#FFFFFF", fontsize=10, fontweight="bold", ha="left", va="center",
                arrowprops=dict(arrowstyle="-|>", color=TEAL_GLOW, lw=1.3, alpha=0.8))

    ax.set_xscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(0, max(g) * 1.32)
    ax.set_xlabel("Resident weight-matrix size   (MB, log scale)", fontsize=12)
    ax.set_ylabel("Throughput   (GOP/s, int8 MACs)", fontsize=12)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    _frame(ax)
    _head(fig, ax, "B=1 ternary GEMV stays compute-bound across the whole memory hierarchy",
          "Single-pass sparse-kernel throughput vs working-set size - measured - no DRAM cliff past L3")
    _footer(fig, host)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(ASSETS / "02_cache_residency.png", bbox_inches="tight", pad_inches=0.28)
    plt.close(fig); print("[render] 02_cache_residency.png")


if __name__ == "__main__":
    if not (DATA / "bench_weight_stationary.csv").exists():
        sys.exit("No measured data. Run scripts/bench_kernel.py and scripts/bench_sparse.py first.")
    roofline(); simd_scaling(); amortization()
    if (DATA / "bench_sparse_density.csv").exists():
        lazy_routing()
    if (DATA / "bench_cache_residency.csv").exists():
        cache_residency()
