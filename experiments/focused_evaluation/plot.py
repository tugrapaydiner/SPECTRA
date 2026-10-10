"""Render the retained summary with matplotlib; never run a solver."""
from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", type=Path)
    args = parser.parse_args()
    report = json.loads((HERE/"SUMMARY.json").read_text())
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.fonttype": "none", "svg.hashsalt": "spectra-focused-20261005"})
    colors = {"indexed": "#6d7680", "poly": "#b4966c", "minbreak": "#9a7496",
              "novelty_break": "#187f79", "glucose4": "#405abb"}
    labels = {"indexed": "Indexed", "poly": "Dense polynomial", "minbreak": "Minbreak",
              "novelty_break": "Break / age", "glucose4": "Native Glucose4"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1, 1.35]})
    fig.patch.set_facecolor("#faf9f6")
    for ax in axes:
        ax.set_facecolor("#faf9f6")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#c6c8c6")
        ax.grid(axis="y", color="#dedfdb", linewidth=.6)
        ax.set_axisbelow(True)
    for arm, group in report["overall"].items():
        axes[0].scatter(group["wall_mean_ms"], group["sat_verified"], s=75,
                        c=colors[arm], label=labels[arm], zorder=3)
    axes[0].set(xlabel="Complete mean wall time (ms)", ylabel="Verified SAT pairs / 192",
                ylim=(0, 192), title="Quality and total solve cost")
    axes[0].legend(frameon=False, fontsize=9, loc="upper right")
    cells = list(report["cells"])
    for i, arm in enumerate(("indexed", "novelty_break", "glucose4")):
        axes[1].bar([j+(i-1)*.25 for j in range(len(cells))],
                    [report["cells"][c][arm]["sat_verified"] for c in cells], width=.23,
                    color=colors[arm], label=labels[arm])
    axes[1].set_xticks(range(len(cells)), [c.replace("planted3:", "Planted\n").replace("uniform3:", "Uniform\n") for c in cells])
    axes[1].set(ylabel="Verified SAT pairs / 32", ylim=(0, 36), title="Every declared family / size cell")
    axes[1].legend(frameon=False, fontsize=9, ncol=3, loc="upper right")
    fig.suptitle("Focused search: fresh, frozen CPU evaluation", x=.07, ha="left", fontsize=17, fontweight="bold")
    fig.text(.07, .035, "96 synthetic formulas · 2 search seeds · 3 timing rounds · setup and checking included\n"
             "Glucose4 uses an unequal conflict budget. Native UNSAT reports are excluded from SAT counts. One CPU host.",
             fontsize=9, color="#525b64")
    fig.subplots_adjust(left=.07, right=.98, bottom=.22, top=.82, wspace=.28)
    fig.savefig(HERE/"quality-cost.svg", metadata={"Date": None})
    svg = HERE/"quality-cost.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines())+"\n")
    if args.preview:
        fig.savefig(args.preview, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
