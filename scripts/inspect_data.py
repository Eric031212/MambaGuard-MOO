"""Print basic statistics of the preprocessed MIMIC-III files."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from mambaguard.data import load_mimic, split_safedrug

records, voc, ddi, sizes, names = load_mimic(sys.argv[1] if len(sys.argv) > 1 else "data/raw")
tr, va, te = split_safedrug(records)
visits = [v for p in records for v in p]
print("patients", len(records), "visits", len(visits), "|D|,|P|,|M|", sizes)
print("train/val/test patients", len(tr), len(va), len(te))
print("max visits/patient", max(len(p) for p in records), "avg meds/visit", np.mean([len(v[2]) for v in visits]))
print("DDI pairs", int(ddi.sum() // 2), "example meds", names[:8])
