"""Evaluation identical to SafeDrug (Yang et al., IJCAI'21) `multi_label_metric` / `ddi_rate_score`.

Metrics are averaged over visits within a patient, then over patients. The test protocol draws 10
bootstrap samples of 80% of the test patients (with replacement) and reports mean +- std. Since the
bootstrap only resamples patients, per-patient statistics are computed once and re-aggregated.
"""
import numpy as np
from sklearn.metrics import average_precision_score


def patient_stats(y_gt, y_prob, ddi_A, thr=0.5):
    """y_gt, y_prob: [T, M] arrays for one patient."""
    y_pred = y_prob >= thr
    ja, prauc, p, r, f1 = [], [], [], [], []
    dd_cnt = all_cnt = 0
    for gt, prob, pred in zip(y_gt, y_prob, y_pred):
        tgt, out = set(np.where(gt == 1)[0]), np.where(pred)[0]
        inter = len(tgt & set(out))
        union = len(tgt | set(out))
        ja.append(0 if union == 0 else inter / union)
        prc = 0 if len(out) == 0 else inter / len(out)
        rec = 0 if len(tgt) == 0 else inter / len(tgt)
        p.append(prc)
        r.append(rec)
        f1.append(0 if prc + rec == 0 else 2 * prc * rec / (prc + rec))
        prauc.append(average_precision_score(gt, prob, average="macro"))
        sub = ddi_A[np.ix_(out, out)]
        k = len(out)
        all_cnt += k * (k - 1) // 2
        dd_cnt += int(np.triu(sub + sub.T > 0, 1).sum())
    return {"ja": np.mean(ja), "prauc": np.mean(prauc), "p": np.mean(p), "r": np.mean(r), "f1": np.mean(f1),
            "dd": dd_cnt, "all": all_cnt, "med": int(y_pred.sum()), "visits": len(y_gt)}


def aggregate(stats):
    return {
        "ddi": sum(s["dd"] for s in stats) / max(sum(s["all"] for s in stats), 1),
        "ja": float(np.mean([s["ja"] for s in stats])),
        "prauc": float(np.mean([s["prauc"] for s in stats])),
        "p": float(np.mean([s["p"] for s in stats])),
        "r": float(np.mean([s["r"] for s in stats])),
        "f1": float(np.mean([s["f1"] for s in stats])),
        "avg_med": sum(s["med"] for s in stats) / max(sum(s["visits"] for s in stats), 1),
    }


def bootstrap(stats, rounds=10, frac=0.8, seed=0):
    rng = np.random.RandomState(seed)
    runs = [aggregate([stats[i] for i in rng.choice(len(stats), round(len(stats) * frac), replace=True)])
            for _ in range(rounds)]
    return {k: (float(np.mean([r[k] for r in runs])), float(np.std([r[k] for r in runs]))) for k in runs[0]}
