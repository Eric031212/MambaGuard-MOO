"""Writes jobs/<group>.txt command lists for scripts/array_job.sh."""
import os

os.makedirs("jobs", exist_ok=True)
SEEDS = range(5)
VARIANTS = {
    "full": "",
    "no_moo": "--no_moo",
    "no_cl": "--no_cl",
    "no_cssm": "--no_cssm",
    "base": "--no_moo --no_cl --no_cssm",
}
# validation-only hyper-parameter sweeps (seed 0); the selected setting is now the default in train.py.
# Sweep 1 used the original 40-epoch schedule, sweep 2 a 15-epoch schedule with stronger regularisation.
OLD = "--dim 64 --dropout 0.3 --w_safe 0.5"
S1 = OLD + " --epochs 40 --warmup_epochs 3 --tau_ramp_epochs 3 --stage3_epoch 15"
S2 = OLD + " --epochs 15 --warmup_epochs 1 --tau_ramp_epochs 1 --stage3_epoch 6"
SWEEP2 = {
    "e15": "", "e15_do5": "--dropout 0.5", "e15_wd": "--weight_decay 0.05",
    "e15_do5_wd": "--dropout 0.5 --weight_decay 0.05", "e15_d128_do5": "--dim 128 --dropout 0.5",
    "e15_d128_do5_wd": "--dim 128 --dropout 0.5 --weight_decay 0.05", "e15_safe1": "--w_safe 1.0",
    "e15_safe1_d128_do5": "--w_safe 1.0 --dim 128 --dropout 0.5",  # selected (Eq. 31: best feasible val score)
    "e15_base_d128_do5": "--no_moo --no_cl --no_cssm --dim 128 --dropout 0.5",
}
SWEEP = {
    "default": "",
    "safe2": "--w_safe 2",
    "target04": "--ddi_target 0.04 --tau_ddi 0.05",
    "cl05": "--lambda_cl 0.5",
    "lr5e-4": "--lr 5e-4 --epochs 60",
    "rank0": "--w_rank 0 --w_pr 0",
    "eff0": "--w_eff 0",
    "dim128": "--dim 128",
}


def write(name, lines):
    with open(f"jobs/{name}.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"jobs/{name}.txt: {len(lines)} jobs")


write("sweep", [f"python train.py --out outputs/sweep/{k}/seed0 --seed 0 {S1} {v}".strip() for k, v in SWEEP.items()]
      + [f"python train.py --out outputs/sweep/base/seed0 --seed 0 {S1} {VARIANTS['base']}"])
write("sweep2", [f"python train.py --out outputs/sweep/{k}/seed0 --seed 0 {S2} {v}".strip() for k, v in SWEEP2.items()])
write("main", [f"python train.py --out outputs/main/{k}/seed{s} --seed {s} {v}".strip()
               for k, v in VARIANTS.items() for s in SEEDS])
