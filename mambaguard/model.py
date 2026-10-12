"""MambaGuard-MOO model (Sec. IV-B and the prescription generation module of Fig. 1)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mamba import MambaLayer


class MambaGuard(nn.Module):
    def __init__(self, sizes, E_dm, ddi_A, dim=64, n_layers=2, d_state=16, n_heads=4, dropout=0.3,
                 constrained_ssm=True, efficacy_prior=True, proj_dim=64):
        super().__init__()
        n_diag, n_proc, n_med = sizes
        self.n_med = n_med
        self.register_buffer("E_dm", torch.as_tensor(E_dm))                 # |D| x |M|
        self.register_buffer("ddi_A", torch.as_tensor(ddi_A))               # |M| x |M|
        # Eq. (2)-(3): multi-hot codes times learnable embedding tables
        self.emb_d = nn.Parameter(torch.randn(n_diag, dim) * 0.02)
        self.emb_p = nn.Parameter(torch.randn(n_proc, dim) * 0.02)
        self.emb_drop = nn.Dropout(dropout)
        cond_dim = n_med if constrained_ssm else None
        self.constrained_ssm = constrained_ssm
        self.diag_layers = nn.ModuleList(
            [MambaLayer(dim, dropout, cond_dim=cond_dim, d_state=d_state) for _ in range(n_layers)])
        self.proc_layers = nn.ModuleList(
            [MambaLayer(dim, dropout, cond_dim=cond_dim, d_state=d_state) for _ in range(n_layers)])
        # Eq. (8) + query attention over the drug memory
        self.query = nn.Sequential(nn.Linear(2 * dim, dim), nn.LayerNorm(dim))
        self.E_drug = nn.Parameter(torch.randn(n_med, dim) * 0.1)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.attn_ln = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, dim)
        self.out_bias = nn.Parameter(torch.zeros(n_med))
        # W_efficacy (|M| x |D|) of the MOO calculator, also used as a diagnosis->drug efficacy prior
        self.efficacy_prior = efficacy_prior
        self.W_eff = nn.Parameter(torch.zeros(n_med, n_diag))
        self.eff_scale = nn.Parameter(torch.zeros(()))
        # contrastive projection head, Eq. (16): z = W_proj ReLU(W_1 e)
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, proj_dim))
        self.patient_proj = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, proj_dim))

    def constraint_signals(self, d):
        n = d.sum(-1, keepdim=True).clamp(min=1)
        c_ehr = (d @ self.E_dm) / n                                          # EHR-implied drug distribution
        c_ddi = (c_ehr @ self.ddi_A) / c_ehr.sum(-1, keepdim=True).clamp(min=1e-6)  # expected DDI exposure
        return c_ddi, c_ehr

    def forward(self, d, p):
        """d: [B,T,|D|], p: [B,T,|P|] multi-hot. Returns logits [B,T,|M|] and the patient representation."""
        cond = self.constraint_signals(d) if self.constrained_ssm else None
        hd = self.emb_drop(d @ self.emb_d)
        hp = self.emb_drop(p @ self.emb_p)
        for layer in self.diag_layers:
            hd = layer(hd, cond)
        for layer in self.proc_layers:
            hp = layer(hp, cond)
        B, T, _ = hd.shape
        q = self.query(torch.cat([hd, hp], dim=-1)).reshape(B * T, 1, -1)
        mem = self.E_drug.unsqueeze(0).expand(B * T, -1, -1)
        ctx, _ = self.attn(q, mem, mem, need_weights=False)
        rep = self.attn_ln(q + ctx).reshape(B, T, -1)
        logits = self.out(rep) @ self.E_drug.t() + self.out_bias
        if self.efficacy_prior:
            n = d.sum(-1, keepdim=True).clamp(min=1)
            logits = logits + self.eff_scale * (d @ self.W_eff.t()) / n
        return logits, rep

    def drug_z(self):
        return F.normalize(self.proj(self.E_drug), dim=-1)

    def patient_z(self, rep):
        return F.normalize(self.patient_proj(rep), dim=-1)
