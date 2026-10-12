#!/usr/bin/env bash
# Fetches the preprocessed MIMIC-III benchmark files released with SafeDrug (Yang et al., IJCAI 2021),
# https://github.com/ycq091044/SafeDrug (commit 88ce5c3), into data/raw/.
# MIMIC-III itself is credentialed data (PhysioNet DUA); if you have access you can regenerate these files
# from the raw tables with SafeDrug's data/processing.py. The files are NOT redistributed in this repository.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw
BASE=https://raw.githubusercontent.com/ycq091044/SafeDrug/88ce5c377dcdc2aa01aaa88f5478dfa4373ba49a/data/output
for f in records_final.pkl voc_final.pkl ddi_A_final.pkl ehr_adj_final.pkl; do
  curl -fsSL -o "data/raw/$f" "$BASE/$f"
done
ls -l data/raw
