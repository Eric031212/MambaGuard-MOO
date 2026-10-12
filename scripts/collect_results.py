"""Aggregates outputs/**/results.json and the official-baseline test logs into results/*.md|json.

  python scripts/collect_results.py sweep     # validation table of the hyper-parameter sweep
  python scripts/collect_results.py main      # test table (mean +- std over seeds) + baselines
"""
import glob
import json
import os
import re
import shutil
import sys

import numpy as np

KEYS = ("ddi", "ja", "prauc", "f1", "avg_med")
HEAD = "| Method | DDI Rate ↓ | Jaccard ↑ | PRAUC ↑ | F1 ↑ | Avg. #Med |\n|---|---|---|---|---|---|"
PAPER = {
    "SafeDrug (paper Table I)": ("0.0682 ± 0.004", "0.4839 ± 0.002", "0.7452 ± 0.006", "0.6418 ± 0.004"),
    "MambaGuard-MOO (paper Table I)": ("0.0341 ± 0.001", "0.5541 ± 0.002", "0.7676 ± 0.002", "0.7052 ± 0.003"),
    "w/o MOO (paper Table II)": ("0.0562 ± 0.004", "0.4284 ± 0.003", "0.6429 ± 0.005", "0.5217 ± 0.005"),
    "w/o contrastive learning (paper Table II)": ("0.0511 ± 0.005", "0.4724 ± 0.004", "0.6793 ± 0.004", "0.5045 ± 0.003"),
}
DISPLAY = {"Retain": "RETAIN", "Leap": "LEAP"}
NAMES = {"full": "MambaGuard-MOO (full)", "no_moo": "w/o MOO", "no_cl": "w/o contrastive learning",
         "no_cssm": "w/o DDI/EHR-conditioned SSM", "base": "Mamba backbone + L_base only"}


def load(pattern):
    out = {}
    for f in sorted(glob.glob(pattern)):
        name = f.split(os.sep)[-3]
        out.setdefault(name, []).append(json.load(open(f)))
    return out


def sweep():
    rows = []
    for name, runs in load("outputs/sweep/*/seed*/results.json").items():
        r = runs[0]
        v = r["best_val"]
        rows.append((name, r["best_epoch"], r["best_feasible"], v))
    print("| Config | best epoch | DDI≤τ | val DDI | val Jaccard | val PRAUC | val F1 | val #Med |")
    print("|---|---|---|---|---|---|---|---|")
    for name, ep, feas, v in sorted(rows, key=lambda x: -x[3]["ja"]):
        print(f"| {name} | {ep} | {feas} | {v['ddi']:.4f} | {v['ja']:.4f} | {v['prauc']:.4f} | {v['f1']:.4f} "
              f"| {v['avg_med']:.2f} |")


def official_baselines():
    rows = {}
    for f in sorted(glob.glob("outputs/baselines/*/test.log")):
        m = re.findall(r"([0-9.]+) \$\\pm\$ ([0-9.]+)", open(f).read())
        if len(m) >= 5:
            vals = [(float(a), float(b)) for a, b in m[:5]]
            # official scripts print: ddi, ja, f1, prauc, avg_med
            rows[f.split(os.sep)[-2]] = dict(zip(("ddi", "ja", "f1", "prauc", "avg_med"), vals))
    return rows


def main_table():
    res, summary = load("outputs/main/*/seed*/results.json"), {}
    lines = [HEAD]
    official = official_baselines()
    for name in [m for m in ("Retain", "Leap", "GAMENet", "SafeDrug") if m in official]:
        runs = official[name]
        lines.append(f"| {DISPLAY.get(name, name)} (official code, 1 run, bootstrap std) | "
                     + " | ".join(f"{runs[k][0]:.4f} ± {runs[k][1]:.4f}" for k in KEYS) + " |")
        summary[f"official_{name}"] = runs
    for name in ("base", "no_cssm", "no_cl", "no_moo", "full"):
        if name not in res:
            continue
        runs = res[name]
        # each seed's test score is the mean of its 10 bootstrap rounds; report mean +- std over seeds
        vals = {k: np.array([r["test"][k][0] for r in runs]) for k in KEYS}
        summary[name] = {k: (float(v.mean()), float(v.std())) for k, v in vals.items()}
        summary[name]["n_seeds"] = len(runs)
        summary[name]["n_params"] = runs[0]["n_params"]
        summary[name]["train_minutes"] = float(np.mean([r["train_seconds"] for r in runs]) / 60)
        lines.append(f"| {NAMES[name]} ({len(runs)} seeds) | "
                     + " | ".join(f"{v.mean():.4f} ± {v.std():.4f}" for v in vals.values()) + " |")
    gt = [r["gt_test_ddi"] for runs in res.values() for r in runs]
    if gt:
        lines.append(f"\nGround-truth DDI rate of the test prescriptions: {gt[0]:.4f}")
        summary["gt_test_ddi"] = gt[0]
    lines += ["", "Numbers printed in the paper, for reference:", "", HEAD.replace(" Avg. #Med |", "").replace("|---|---|---|---|---|---|", "|---|---|---|---|---|")]
    for name, row in PAPER.items():
        lines.append(f"| {name} | " + " | ".join(row) + " |")
    os.makedirs("results", exist_ok=True)
    open("results/main_table.md", "w").write("\n".join(lines) + "\n")
    if os.path.exists("README.md"):
        readme = open("README.md").read()
        block = "<!--MAIN_TABLE-->\n" + "\n".join(lines) + "\n<!--/MAIN_TABLE-->"
        readme = re.sub(r"<!--MAIN_TABLE-->.*?<!--/MAIN_TABLE-->|<!--MAIN_TABLE-->", lambda _: block, readme, flags=re.S)
        open("README.md", "w").write(readme)
    json.dump(summary, open("results/summary.json", "w"), indent=1)
    # keep the small per-run artefacts (results.json, baseline logs) under version control
    for f in glob.glob("outputs/*/*/seed*/results.json") + glob.glob("outputs/baselines/*/*.log") \
            + glob.glob("outputs/baselines/*/best.txt"):
        dst = os.path.join("results/runs", os.path.relpath(f, "outputs"))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(f, dst)
    print("\n".join(lines))


if __name__ == "__main__":
    {"sweep": sweep, "main": main_table}[sys.argv[1] if len(sys.argv) > 1 else "main"]()
