"""Minimal compatibility patches for the official SafeDrug-repo scripts under numpy>=1.24 / torch 2.x.

np.random.choice() cannot sample from a ragged list of patients any more, so the bootstrap draws patient
indices instead (same distribution), and LEAP's ragged label lists are wrapped as object arrays.
Nothing else is changed.
"""
import re
import sys
from pathlib import Path

PAT = re.compile(r"np\.random\.choice\(\s*data_test,\s*round\(len\(data_test\) \* 0\.8\),\s*replace=True\s*\)")
NEW = "[data_test[i] for i in np.random.choice(len(data_test), round(len(data_test) * 0.8), replace=True)]"
for name in ("SafeDrug.py", "GAMENet.py", "Retain.py", "Leap.py"):
    f = Path(sys.argv[1]) / name
    src = f.read_text()
    out, n = PAT.subn(NEW, src)
    # LEAP: ragged predicted-label lists cannot become a regular ndarray
    out = out.replace("np.array(y_pred_label),", "np.array(y_pred_label, dtype=object),")
    if out != src:
        f.write_text(out)
    print(f"{name}: patched {n} bootstrap call(s)")
