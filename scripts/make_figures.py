"""Figures for the README (reads results/summary.json and outputs/main/**/results.json)."""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#8a8984", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "figure.dpi": 150})

# numbers printed in the paper (Tables I and II)
PAPER = {
    "full": {"ddi": 0.0341, "ja": 0.5541, "prauc": 0.7676, "f1": 0.7052},
    "no_moo": {"ddi": 0.0562, "ja": 0.4284, "prauc": 0.6429, "f1": 0.5217},
    "no_cl": {"ddi": 0.0511, "ja": 0.4724, "prauc": 0.6793, "f1": 0.5045},
    "official_SafeDrug": {"ddi": 0.0682, "ja": 0.4839, "prauc": 0.7452, "f1": 0.6418},
    "official_GAMENet": {"ddi": 0.0874, "ja": 0.4672, "prauc": 0.7162, "f1": 0.6383},
    "official_Retain": {"ddi": 0.0835, "ja": 0.4134, "prauc": 0.7256, "f1": 0.6017},
    "official_Leap": {"ddi": 0.0631, "ja": 0.4426, "prauc": 0.7031, "f1": 0.5941},
}
LABEL = {"official_Leap": "LEAP", "official_Retain": "RETAIN", "official_GAMENet": "GAMENet",
         "official_SafeDrug": "SafeDrug", "base": "Mamba + L_base", "no_cssm": "w/o cond. SSM",
         "no_cl": "w/o CL", "no_moo": "w/o MOO", "full": "MambaGuard-MOO"}
METRICS = [("ddi", "DDI rate ↓"), ("ja", "Jaccard ↑"), ("prauc", "PRAUC ↑"), ("f1", "F1 ↑")]


def comparison(summary):
    rows = [k for k in LABEL if k in summary]
    fig, axes = plt.subplots(1, 4, figsize=(12, 0.42 * len(rows) + 1.3), sharey=True)
    y = np.arange(len(rows))
    for ax, (m, title) in zip(axes, METRICS):
        for i, k in enumerate(rows):
            mean, std = summary[k][m]
            c = BLUE if k == "full" else (INK if k in ("no_moo", "no_cl", "no_cssm", "base") else GRAY)
            ax.errorbar(mean, i, xerr=std, fmt="o", ms=7, color=c, ecolor=c, elinewidth=2, capsize=0, zorder=3)
            if k in PAPER:
                ax.plot(PAPER[k][m], i, marker="D", ms=7, mfc="none", mec=ORANGE, mew=1.6, ls="none", zorder=2)
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [LABEL[k] for k in rows])
    axes[0].invert_yaxis()
    h = [plt.Line2D([], [], marker="o", color=INK, ls="none", ms=7, label="reproduced (this repo, test set; blue = full model)"),
         plt.Line2D([], [], marker="D", mfc="none", mec=ORANGE, mew=1.6, ls="none", ms=7,
                    label="number printed in the paper")]
    fig.legend(handles=h, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig("assets/comparison.png", bbox_inches="tight")


def tradeoff(summary):
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for k in LABEL:
        if k not in summary:
            continue
        (d, ds), (j, js) = summary[k]["ddi"], summary[k]["ja"]
        c = BLUE if k == "full" else (INK if k in ("no_moo", "no_cl", "no_cssm", "base") else GRAY)
        ax.errorbar(d, j, xerr=ds, yerr=js, fmt="o", ms=8, color=c, mec="white", mew=1.5, zorder=3)
        off = {"full": (8, -13), "no_cssm": (8, 5)}.get(k, (6, 4))
        ax.annotate(LABEL[k], (d, j), xytext=off, textcoords="offset points", fontsize=8.5,
                    color=BLUE if k == "full" else MUTED)
    if "gt_test_ddi" in summary:
        ax.axvline(summary["gt_test_ddi"], color=GRAY, ls="--", lw=1)
        ax.annotate("DDI rate of real prescriptions", (summary["gt_test_ddi"], ax.get_ylim()[0]),
                    xytext=(4, 6), textcoords="offset points", fontsize=8, color=MUTED, rotation=90, va="bottom")
    ax.set_xlabel("DDI rate (lower is safer)")
    ax.set_ylabel("Jaccard (higher is more accurate)")
    ax.set_title("Safety–accuracy trade-off on the MIMIC-III test set", loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig("assets/tradeoff.png")


def curves():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for name, c in (("base", GRAY), ("full", BLUE)):
        hs = [json.load(open(f))["history"] for f in glob.glob(f"outputs/main/{name}/seed*/results.json")]
        if not hs:
            continue
        n = min(len(h) for h in hs)
        for ax, key in zip(axes, ("val_ja", "val_ddi")):
            v = np.array([[h[e][key] for e in range(n)] for h in hs])
            ax.plot(np.arange(1, n + 1), v.mean(0), color=c, lw=2, label=LABEL[name])
            ax.fill_between(np.arange(1, n + 1), v.min(0), v.max(0), color=c, alpha=0.15, lw=0)
    axes[0].set_title("Validation Jaccard", loc="left", color=INK, fontsize=11)
    axes[1].set_title("Validation DDI rate", loc="left", color=INK, fontsize=11)
    for ax in axes:
        ax.set_xlabel("epoch")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig("assets/curves.png")


if __name__ == "__main__":
    os.makedirs("assets", exist_ok=True)
    s = json.load(open("results/summary.json"))
    comparison(s)
    tradeoff(s)
    curves()
    print("wrote assets/comparison.png, assets/tradeoff.png, assets/curves.png")
