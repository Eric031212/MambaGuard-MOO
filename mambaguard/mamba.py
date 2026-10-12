"""Selective state-space (Mamba) block with safety-conditioned state transitions, in plain PyTorch.

Visit sequences in MIMIC-III are short (<= 29 visits), so a sequential scan is cheap and avoids the
mamba-ssm CUDA extension. With `cond_dim=None` the block is a standard Mamba (S6) block.

Eq. (6) of the paper, A^(d,p) = A0 + W_DDI * D + W_EHR * E, is realised as a per-visit, per-channel
modulation of the (negative, diagonal) transition matrix:
    A_t = -exp(A_log + tanh(W_DDI c^ddi_t + W_EHR c^ehr_t))
where c^ehr_t = d_t E / |d_t| is the EHR-implied medication distribution of the visit and
c^ddi_t = c^ehr_t D its expected DDI exposure. W_DDI, W_EHR are zero-initialised, so training starts
from vanilla Mamba and A stays negative (stable) for any input.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveSSM(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, cond_dim=None):
        super().__init__()
        self.d_inner = d_inner = expand * d_model
        self.d_state = d_state
        self.dt_rank = math.ceil(d_model / 16)
        self.in_proj = nn.Linear(d_model, 2 * d_inner)
        self.conv = nn.Conv1d(d_inner, d_inner, d_conv, groups=d_inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_inner)
        # Mamba dt initialisation: dt ~ LogUniform(1e-3, 1e-1)
        dt = torch.exp(torch.rand(d_inner) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3))
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model)
        self.conditioned = cond_dim is not None
        if self.conditioned:
            self.w_ddi = nn.Linear(cond_dim, d_inner)
            self.w_ehr = nn.Linear(cond_dim, d_inner)
            for lin in (self.w_ddi, self.w_ehr):
                nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)

    def forward(self, x, cond=None):
        B, T, _ = x.shape
        x, z = self.in_proj(x).chunk(2, dim=-1)
        x = F.silu(self.conv(x.transpose(1, 2))[..., :T].transpose(1, 2))
        dt, Bm, Cm = self.x_proj(x).split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))                                   # [B,T,C]
        log_rate = self.A_log.expand(B, T, -1, -1)                         # [B,T,C,N]
        if self.conditioned and cond is not None:
            c_ddi, c_ehr = cond
            log_rate = log_rate + torch.tanh(self.w_ddi(c_ddi) + self.w_ehr(c_ehr)).unsqueeze(-1)
        dA = torch.exp(-dt.unsqueeze(-1) * torch.exp(log_rate))            # exp(dt * A_t), A_t < 0
        dBx = dt.unsqueeze(-1) * Bm.unsqueeze(2) * x.unsqueeze(-1)          # [B,T,C,N]
        h = x.new_zeros(B, self.d_inner, self.d_state)
        ys = []
        for t in range(T):
            h = dA[:, t] * h + dBx[:, t]
            ys.append((h * Cm[:, t].unsqueeze(1)).sum(-1))
        y = torch.stack(ys, dim=1) + x * self.D
        return self.out_proj(y * F.silu(z))


class MambaLayer(nn.Module):
    """Eq. (7): LN(Drop(Mamba(x)) + x), followed by a position-wise feed-forward sub-layer."""

    def __init__(self, d_model, dropout, cond_dim=None, **ssm_kw):
        super().__init__()
        self.ssm = SelectiveSSM(d_model, cond_dim=cond_dim, **ssm_kw)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU(), nn.Linear(2 * d_model, d_model))
        self.ln2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, cond=None):
        x = self.ln1(self.drop(self.ssm(x, cond)) + x)
        return self.ln2(self.drop(self.ffn(x)) + x)
