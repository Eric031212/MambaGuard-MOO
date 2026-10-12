"""Train / evaluate MambaGuard-MOO on MIMIC-III (SafeDrug preprocessing).

  python train.py --out outputs/main/full/seed0 --seed 0                  # full model
  python train.py --out outputs/ablation/no_moo/seed0 --no_moo            # w/o MOO
  python train.py --out outputs/ablation/no_cl/seed0 --no_cl              # w/o contrastive learning
  python train.py --out outputs/ablation/no_cssm/seed0 --no_cssm          # vanilla Mamba transitions
  python train.py --out outputs/ablation/base/seed0 --no_moo --no_cl --no_cssm   # Mamba + L_base only
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch

from mambaguard.data import load_mimic, split_safedrug, train_statistics, PatientBatcher
from mambaguard.losses import base_loss, MOOLoss, ContrastiveLoss
from mambaguard.metrics import patient_stats, aggregate, bootstrap
from mambaguard.model import MambaGuard


def get_args(argv=None):
    a = argparse.ArgumentParser()
    a.add_argument("--data_dir", default="data/raw")
    a.add_argument("--out", required=True)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # model
    a.add_argument("--dim", type=int, default=128)
    a.add_argument("--n_layers", type=int, default=2)
    a.add_argument("--d_state", type=int, default=16)
    a.add_argument("--n_heads", type=int, default=4)
    a.add_argument("--dropout", type=float, default=0.5)
    # optimisation
    a.add_argument("--epochs", type=int, default=15)
    a.add_argument("--batch_size", type=int, default=16)
    a.add_argument("--lr", type=float, default=1e-3)
    a.add_argument("--weight_decay", type=float, default=1e-4)
    # variants
    a.add_argument("--no_moo", action="store_true")
    a.add_argument("--no_cl", action="store_true")
    a.add_argument("--no_cssm", action="store_true", help="vanilla Mamba transitions (no DDI/EHR in A)")
    # base loss
    a.add_argument("--alpha", type=float, default=0.95)
    # MOO
    a.add_argument("--lambda_moo", type=float, default=1.0)
    a.add_argument("--w_eff", type=float, default=0.5)
    a.add_argument("--w_safe", type=float, default=1.0)
    a.add_argument("--w_rank", type=float, default=0.5)
    a.add_argument("--w_pr", type=float, default=0.1)
    a.add_argument("--ehr_weight", type=float, default=0.1)
    a.add_argument("--tau_eff", type=float, default=0.9)
    a.add_argument("--tau_eff_neg", type=float, default=0.05)
    a.add_argument("--gamma", type=float, default=0.1)
    a.add_argument("--pr_alpha", type=float, default=0.5)
    a.add_argument("--ddi_target", type=float, default=0.06)
    a.add_argument("--rho_safe", type=float, default=20.0)
    a.add_argument("--rho", type=float, default=5.0)
    # contrastive
    a.add_argument("--lambda_cl", type=float, default=0.1)
    a.add_argument("--tau", type=float, default=0.1)
    a.add_argument("--mu", type=float, default=0.0)
    a.add_argument("--lambda_hard", type=float, default=1.0)
    a.add_argument("--lambda_enh", type=float, default=1.0)
    a.add_argument("--min_count", type=int, default=5)
    a.add_argument("--n_pos_sample", type=int, default=512)
    a.add_argument("--n_enh_sample", type=int, default=256)
    # training stages (Eq. 27-29)
    a.add_argument("--warmup_epochs", type=float, default=1)
    a.add_argument("--tau_ramp_epochs", type=float, default=1)
    a.add_argument("--stage3_epoch", type=int, default=6)
    a.add_argument("--xi", type=float, default=0.5)
    # model selection (Eq. 30-31)
    a.add_argument("--omega", type=float, nargs=3, default=[1.0, 1.0, 0.5])
    a.add_argument("--tau_ddi", type=float, default=0.06)
    a.add_argument("--boot_rounds", type=int, default=10)
    return a.parse_args(argv)


@torch.no_grad()
def predict(model, patients, batcher, bs=64):
    model.eval()
    out = []
    for i in range(0, len(patients), bs):
        chunk = patients[i:i + bs]
        d, p, m, mask = batcher(chunk)
        prob = torch.sigmoid(model(d, p)[0]).cpu().numpy()
        m = m.cpu().numpy()
        for b, pat in enumerate(chunk):
            out.append((m[b, :len(pat)], prob[b, :len(pat)]))
    return out


def evaluate(model, patients, batcher, ddi_np):
    return [patient_stats(gt, prob, ddi_np) for gt, prob in predict(model, patients, batcher)]


def main(args):
    os.makedirs(args.out, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dev = torch.device(args.device)

    records, voc, ddi_np, sizes, med_names = load_mimic(args.data_dir)
    train, val, test = split_safedrug(records)
    st = train_statistics(train, sizes)
    n_train_visits = sum(len(p) for p in train)
    print(f"patients train/val/test = {len(train)}/{len(val)}/{len(test)}; |D|,|P|,|M| = {sizes}")

    ddi_A = torch.as_tensor(ddi_np, device=dev)
    med_med = torch.as_tensor(st["med_med"], device=dev)
    model = MambaGuard(sizes, st["E_dm"], ddi_np, dim=args.dim, n_layers=args.n_layers, d_state=args.d_state,
                       n_heads=args.n_heads, dropout=args.dropout, constrained_ssm=not args.no_cssm,
                       efficacy_prior=not args.no_moo).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    batcher = PatientBatcher(sizes, dev)
    moo = None if args.no_moo else MOOLoss(ddi_A, med_med, args)
    cl = None if args.no_cl else ContrastiveLoss(ddi_A, med_med, torch.as_tensor(st["med_freq"]), med_names,
                                                 n_train_visits, args, dev)
    steps_per_epoch = math.ceil(len(train) / args.batch_size)
    t_warm = args.warmup_epochs * steps_per_epoch
    t_ramp = args.tau_ramp_epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs * steps_per_epoch)

    best = {"score": -1e9, "feasible": False}
    history, prev_val = [], None
    step, t0 = 0, time.time()
    for epoch in range(args.epochs):
        model.train()
        order = np.random.permutation(len(train))
        agg = {}
        for i in range(0, len(order), args.batch_size):
            d, p, m, mask = batcher([train[j] for j in order[i:i + args.batch_size]])
            logits, rep = model(d, p)
            logits, rep, y, dv = logits[mask], rep[mask], m[mask], d[mask]
            loss = base_loss(logits, y, args.alpha)
            logs = {"base": loss.item()}
            if moo is not None:
                lam_moo = args.lambda_moo * min(1.0, step / max(t_warm, 1))                       # Eq. (27)
                l_moo, lg = moo(model, logits, y, dv)
                loss = loss + lam_moo * l_moo
                logs.update(lg)
            if cl is not None and step >= t_warm:
                lam_cl = args.lambda_cl * (1 - math.exp(-(step - t_warm) / max(t_ramp, 1)))       # Eq. (28)
                l_cl, lg = cl(model, rep, y)
                loss = loss + lam_cl * l_cl
                logs.update(lg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            step += 1
            for k, v in logs.items():
                agg[k] = agg.get(k, 0.0) + v
        n_b = steps_per_epoch
        val_m = aggregate(evaluate(model, val, batcher, ddi_np))
        # Eq. (29): stage-3 performance-based re-weighting (objectives whose metric degraded get more weight)
        if moo is not None and epoch + 1 >= args.stage3_epoch and prev_val is not None:
            delta = {"efficacy": 100 * (prev_val["ja"] - val_m["ja"]),
                     "safety": 100 * (val_m["ddi"] - prev_val["ddi"]),
                     "ranking": 100 * (prev_val["prauc"] - val_m["prauc"]),
                     "pr": 100 * (prev_val["f1"] - val_m["f1"])}
            moo.stage3_mult = {k: 1 + args.xi / (1 + math.exp(-v)) for k, v in delta.items()}
        prev_val = val_m
        # Eq. (30)-(31)
        w1, w2, w3 = args.omega
        score = w1 * val_m["ja"] + w2 * val_m["prauc"] + w3 * (1 - val_m["ddi"])
        feasible = val_m["ddi"] <= args.tau_ddi
        improved = (feasible and (not best["feasible"] or score > best["score"])) or \
                   (not feasible and not best["feasible"] and score > best["score"])
        if improved:
            best = {"score": score, "feasible": feasible, "epoch": epoch, "val": val_m}
            torch.save(model.state_dict(), os.path.join(args.out, "best.pt"))
        rec = {"epoch": epoch, "time": time.time() - t0, **{k: v / n_b for k, v in agg.items()},
               **{f"val_{k}": v for k, v in val_m.items()}, "score": score}
        history.append(rec)
        print(f"ep {epoch:2d} | loss {agg['base'] / n_b:.4f} | val ddi {val_m['ddi']:.4f} ja {val_m['ja']:.4f} "
              f"prauc {val_m['prauc']:.4f} f1 {val_m['f1']:.4f} med {val_m['avg_med']:.1f}"
              f"{' *' if improved else ''} | {time.time() - t0:.0f}s", flush=True)

    model.load_state_dict(torch.load(os.path.join(args.out, "best.pt"), map_location=dev, weights_only=True))
    test_stats = evaluate(model, test, batcher, ddi_np)
    res = {
        "args": vars(args), "n_params": n_params, "best_epoch": best["epoch"], "best_val": best["val"],
        "best_feasible": best["feasible"], "test": bootstrap(test_stats, args.boot_rounds, seed=args.seed),
        "test_full": aggregate(test_stats), "train_seconds": time.time() - t0, "history": history,
        "contrastive_pairs": cl.stats if cl is not None else None,
        "gt_test_ddi": aggregate([patient_stats(gt, gt, ddi_np) for gt, _ in predict(model, test, batcher)])["ddi"],
    }
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(res, f, indent=1)
    t = res["test"]
    print("TEST " + "  ".join(f"{k} {t[k][0]:.4f}±{t[k][1]:.4f}" for k in ("ddi", "ja", "prauc", "f1", "avg_med")))


if __name__ == "__main__":
    main(get_args())
