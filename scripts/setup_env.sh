#!/usr/bin/env bash
# Creates the conda env `mambaguard` (pure-PyTorch selective scan, so no CUDA extension build is needed).
set -euo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
if ! conda env list | grep -q '^mambaguard '; then
  conda create -y -n mambaguard python=3.10
fi
conda activate mambaguard
python -m pip install --no-cache-dir torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install --no-cache-dir "numpy<2" dill pandas scikit-learn matplotlib tqdm rdkit dnc
python -c "import torch, dill, sklearn; print('torch', torch.__version__)"
