| Method | DDI Rate ↓ | Jaccard ↑ | PRAUC ↑ | F1 ↑ | Avg. #Med |
|---|---|---|---|---|---|
| Mamba backbone + L_base only (5 seeds) | 0.0832 ± 0.0035 | 0.5302 ± 0.0015 | 0.7754 ± 0.0020 | 0.6845 ± 0.0013 | 21.9575 ± 0.6269 |
| w/o DDI/EHR-conditioned SSM (5 seeds) | 0.0562 ± 0.0009 | 0.5259 ± 0.0015 | 0.7734 ± 0.0011 | 0.6807 ± 0.0013 | 21.9170 ± 0.1893 |
| w/o contrastive learning (5 seeds) | 0.0570 ± 0.0018 | 0.5274 ± 0.0015 | 0.7745 ± 0.0011 | 0.6821 ± 0.0013 | 21.5196 ± 0.1881 |
| w/o MOO (5 seeds) | 0.0829 ± 0.0018 | 0.5280 ± 0.0013 | 0.7729 ± 0.0018 | 0.6825 ± 0.0012 | 21.8446 ± 0.4934 |
| MambaGuard-MOO (full) (5 seeds) | 0.0560 ± 0.0015 | 0.5257 ± 0.0013 | 0.7731 ± 0.0010 | 0.6805 ± 0.0012 | 21.9194 ± 0.2032 |

Ground-truth DDI rate of the test prescriptions: 0.0758

Numbers printed in the paper, for reference:

| Method | DDI Rate ↓ | Jaccard ↑ | PRAUC ↑ | F1 ↑ |
|---|---|---|---|---|
| SafeDrug (paper Table I) | 0.0682 ± 0.004 | 0.4839 ± 0.002 | 0.7452 ± 0.006 | 0.6418 ± 0.004 |
| MambaGuard-MOO (paper Table I) | 0.0341 ± 0.001 | 0.5541 ± 0.002 | 0.7676 ± 0.002 | 0.7052 ± 0.003 |
| w/o MOO (paper Table II) | 0.0562 ± 0.004 | 0.4284 ± 0.003 | 0.6429 ± 0.005 | 0.5217 ± 0.005 |
| w/o contrastive learning (paper Table II) | 0.0511 ± 0.005 | 0.4724 ± 0.004 | 0.6793 ± 0.004 | 0.5045 ± 0.003 |
