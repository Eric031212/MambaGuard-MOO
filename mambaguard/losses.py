"""Loss functions of MambaGuard-MOO (Sec. IV-C, IV-D, IV-E)."""
import numpy as np
import torch
import torch.nn.functional as F


def pairwise_hinge(prob, y, margin):
    """sum_{i: y_i=1, j: y_j=0} max(0, margin - (p_i - p_j)) per visit; returns [N] sums and [N] pair counts."""
    diff = F.relu(margin - (prob.unsqueeze(-1) - prob.unsqueeze(-2)))  # [N,M,M], row = positive, col = negative
    pair = y.unsqueeze(-1) * (1 - y).unsqueeze(-2)
    return (diff * pair).sum((-1, -2)), pair.sum((-1, -2))


def base_loss(logits, y, alpha=0.95):
    """Eq. (20)-(22): alpha * BCE + (1 - alpha) * multi-label margin. logits, y: [N, |M|]."""
    bce = F.binary_cross_entropy_with_logits(logits, y)
    hinge, _ = pairwise_hinge(torch.sigmoid(logits), y, 1.0)
    multi = (hinge / y.shape[-1]).mean()
    return alpha * bce + (1 - alpha) * multi


def soft_pair_rate(prob, M):
    """p^T M p normalised by the total soft pair mass p^T (11^T - I) p (a differentiable pair rate)."""
    num = ((prob @ M) * prob).sum(-1)
    den = prob.sum(-1) ** 2 - (prob ** 2).sum(-1)
    return num / den.clamp(min=1e-6)


def hard_ddi_rate(prob, ddi_A, thr=0.5):
    pred = (prob > thr).float()
    num = ((pred @ ddi_A) * pred).sum()
    den = (pred.sum(-1) ** 2 - pred.sum(-1)).sum()
    return (num / den.clamp(min=1)).item()


class MOOLoss:
    """MOO loss calculator (Eq. 9-15, 23-25, 29).

    Objectives: efficacy (Eq. 10), safety = DDI (Eq. 11) + EHR (Eq. 12), ranking (Eq. 13), PR (Eq. 14).
    Dosage and personalisation terms of Eq. (25) are not available: the MIMIC-III benchmark files contain
    no dose information (see README).
    """

    NAMES = ("efficacy", "safety", "ranking", "pr")

    def __init__(self, ddi_A, med_med, cfg):
        self.ddi_A = ddi_A
        # EHR safety relation E: drug pairs never co-prescribed in the training EHR (unsupported combinations)
        unsupported = (med_med == 0).float()
        unsupported.fill_diagonal_(0)
        self.ehr_U = unsupported
        self.cfg = cfg
        self.w0 = {"efficacy": cfg.w_eff, "safety": cfg.w_safe, "ranking": cfg.w_rank, "pr": cfg.w_pr}
        self.rho = {"efficacy": -cfg.rho, "safety": cfg.rho_safe, "ranking": -cfg.rho, "pr": -cfg.rho}
        self.stage3_mult = {k: 1.0 for k in self.NAMES}
        self.ddi_ema = None

    def adaptive_weights(self, ddi_current):
        # Eq. (24): w_k = w0_k * max(0, 1 + rho_k (DDI_current - DDI_target)), times stage-3 multipliers (Eq. 29)
        dev = ddi_current - self.cfg.ddi_target
        return {k: self.w0[k] * max(0.0, 1 + self.rho[k] * dev) * self.stage3_mult[k] for k in self.NAMES}

    def __call__(self, model, logits, y, d):
        cfg = self.cfg
        prob = torch.sigmoid(logits)
        # Eq. (10): sigma(W_eff[m_i, d_j]) -> tau for target drugs, -> tau_neg for the other drugs
        G = torch.sigmoid(model.W_eff)                                       # [M, D]
        n_d = d.sum(-1).clamp(min=1)
        q_pos = d @ ((G - cfg.tau_eff) ** 2).t()                             # [N, M]
        q_neg = d @ ((G - cfg.tau_eff_neg) ** 2).t()
        l_eff = ((y * q_pos + (1 - y) * q_neg).sum(-1) / (n_d * y.shape[-1])).mean()
        # Eq. (11)-(12)
        l_ddi = soft_pair_rate(prob, self.ddi_A).mean()
        l_ehr = soft_pair_rate(prob, self.ehr_U).mean()
        l_safe = l_ddi + cfg.ehr_weight * l_ehr
        # Eq. (13): pairwise ranking hinge with margin gamma (mean over pos/neg pairs)
        hinge, n_pairs = pairwise_hinge(prob, y, cfg.gamma)
        l_rank = (hinge / n_pairs.clamp(min=1)).mean()
        # Eq. (14): soft precision / recall
        tp = (prob * y).sum(-1)
        prec = tp / prob.sum(-1).clamp(min=1e-6)
        rec = tp / y.sum(-1).clamp(min=1)
        l_pr = (cfg.pr_alpha * (1 - prec) + (1 - cfg.pr_alpha) * (1 - rec)).mean()

        ddi_now = hard_ddi_rate(prob.detach(), self.ddi_A)
        self.ddi_ema = ddi_now if self.ddi_ema is None else 0.9 * self.ddi_ema + 0.1 * ddi_now
        w = self.adaptive_weights(self.ddi_ema)
        parts = {"efficacy": l_eff, "safety": l_safe, "ranking": l_rank, "pr": l_pr}
        total = sum(w[k] * parts[k] for k in self.NAMES)
        logs = {f"moo_{k}": v.item() for k, v in parts.items()}
        logs.update({f"w_{k}": v for k, v in w.items()})
        logs["ddi_ema"] = self.ddi_ema
        return total, logs


class ContrastiveLoss:
    """Drug-level contrastive learning (Eq. 16-18, 26).

    Positive pairs P: drug pairs co-prescribed more often than chance in the training EHR (lift > 1,
    count >= min_count) that have no DDI ("safe combinations").
    Hard negatives H: DDI pairs that are therapeutically similar (same ATC level-2 class, or co-prescribed
    with lift > 1).
    L_enhanced: patient-to-prescription InfoNCE that aligns the patient representation with the centroid of
    its ground-truth drug set in the contrastive space (in-batch negatives); this is the only contrastive
    term that reaches the sequence encoder.
    """

    def __init__(self, ddi_A, med_med, med_freq, med_names, n_visits, cfg, device):
        self.cfg = cfg
        ddi = ddi_A.cpu().numpy() > 0
        mm = med_med.cpu().numpy()
        f = med_freq.cpu().numpy()
        lift = (mm / max(n_visits, 1)) / np.maximum(np.outer(f, f), 1e-12)
        iu = np.triu_indices(len(f), k=1)
        pos = (lift[iu] > 1) & (mm[iu] >= cfg.min_count) & (~ddi[iu])
        atc2 = np.array([n[:3] for n in med_names])
        similar = (atc2[:, None] == atc2[None, :])[iu] | ((lift[iu] > 1) & (mm[iu] >= cfg.min_count))
        hard = ddi[iu] & similar
        pairs = np.stack(iu, 1)
        P = pairs[pos]
        self.pos = torch.as_tensor(np.concatenate([P, P[:, ::-1]]), device=device)  # both directions
        self.hard = torch.as_tensor(pairs[hard], device=device)
        self.stats = {"n_pos_pairs": int(pos.sum()), "n_hard_neg_pairs": int(hard.sum())}

    def __call__(self, model, rep, y):
        cfg = self.cfg
        z = model.drug_z()                                                   # [M, k]
        # Eq. (17): InfoNCE over sampled positive pairs, denominator over all k != i
        idx = torch.randint(len(self.pos), (min(cfg.n_pos_sample, len(self.pos)),), device=z.device)
        a, b = self.pos[idx, 0], self.pos[idx, 1]
        sim = z[a] @ z.t() / cfg.tau
        sim = sim.masked_fill(F.one_hot(a, z.shape[0]).bool(), float("-inf"))
        l_pos = F.cross_entropy(sim, b)
        # Eq. (18): hinge on hard negatives
        l_hard = F.relu(cfg.mu + (z[self.hard[:, 0]] * z[self.hard[:, 1]]).sum(-1)).mean() \
            if len(self.hard) else z.sum() * 0
        # L_enhanced: patient <-> prescription alignment
        if len(rep) > cfg.n_enh_sample:
            keep = torch.randperm(len(rep), device=rep.device)[:cfg.n_enh_sample]
            rep, y = rep[keep], y[keep]
        zp = model.patient_z(rep)
        zc = F.normalize(y @ z, dim=-1)
        l_enh = F.cross_entropy(zp @ zc.t() / cfg.tau, torch.arange(len(zp), device=zp.device))
        total = l_pos + cfg.lambda_hard * l_hard + cfg.lambda_enh * l_enh
        return total, {"cl_pos": l_pos.item(), "cl_hard": l_hard.item(), "cl_enh": l_enh.item()}
