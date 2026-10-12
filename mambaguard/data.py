"""MIMIC-III data in the SafeDrug preprocessing format.

records_final.pkl : list[patient]; patient = list[visit]; visit = [diag_ids, proc_ids, med_ids]
voc_final.pkl     : {'diag_voc', 'pro_voc', 'med_voc'} (dill-pickled Voc objects with idx2word / word2idx)
ddi_A_final.pkl   : |M| x |M| binary DDI adjacency (TWOSIDES, top-40 severe side effects)

Split follows SafeDrug exactly: first 2/3 of patients train, then test, then validation.
"""
import os
import dill
import numpy as np
import torch


def load_mimic(data_dir):
    def _load(name):
        with open(os.path.join(data_dir, name), "rb") as f:
            return dill.load(f)

    records = _load("records_final.pkl")
    voc = _load("voc_final.pkl")
    ddi_A = np.asarray(_load("ddi_A_final.pkl"), dtype=np.float32)
    sizes = (len(voc["diag_voc"].idx2word), len(voc["pro_voc"].idx2word), len(voc["med_voc"].idx2word))
    med_names = [voc["med_voc"].idx2word[i] for i in range(sizes[2])]
    return records, voc, ddi_A, sizes, med_names


def split_safedrug(records):
    split_point = int(len(records) * 2 / 3)
    train = records[:split_point]
    eval_len = int(len(records[split_point:]) / 2)
    test = records[split_point:split_point + eval_len]
    val = records[split_point + eval_len:]
    return train, val, test


def train_statistics(train, sizes):
    """Co-occurrence statistics computed on the training split only (no test leakage)."""
    n_diag, _, n_med = sizes
    diag_med = np.zeros((n_diag, n_med), dtype=np.float64)
    diag_cnt = np.zeros(n_diag, dtype=np.float64)
    med_med = np.zeros((n_med, n_med), dtype=np.float64)
    med_cnt = np.zeros(n_med, dtype=np.float64)
    for patient in train:
        for diag, _, med in patient:
            diag, med = np.asarray(diag), np.asarray(med)
            diag_med[np.ix_(diag, med)] += 1
            diag_cnt[diag] += 1
            med_med[np.ix_(med, med)] += 1
            med_cnt[med] += 1
    np.fill_diagonal(med_med, 0)
    # E in the paper: P(med | diag), the EHR diagnosis -> medication association matrix (|D| x |M|)
    E_dm = diag_med / np.maximum(diag_cnt[:, None], 1)
    return {
        "E_dm": E_dm.astype(np.float32),
        "med_med": med_med.astype(np.float32),
        "med_freq": (med_cnt / max(sum(len(p) for p in train), 1)).astype(np.float32),
    }


class PatientBatcher:
    """Turns lists of patients into padded multi-hot tensors [B, T, |X|]."""

    def __init__(self, sizes, device):
        self.n_diag, self.n_proc, self.n_med = sizes
        self.device = device

    def __call__(self, patients):
        B = len(patients)
        T = max(len(p) for p in patients)
        d = torch.zeros(B, T, self.n_diag)
        p = torch.zeros(B, T, self.n_proc)
        m = torch.zeros(B, T, self.n_med)
        mask = torch.zeros(B, T, dtype=torch.bool)
        for b, patient in enumerate(patients):
            for t, (diag, proc, med) in enumerate(patient):
                d[b, t, diag] = 1
                p[b, t, proc] = 1
                m[b, t, med] = 1
                mask[b, t] = True
        to = lambda x: x.to(self.device, non_blocking=True)
        return to(d), to(p), to(m), to(mask)
