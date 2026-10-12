#!/usr/bin/env bash
# Runs an official baseline from the SafeDrug repository (https://github.com/ycq091044/SafeDrug, commit 88ce5c3)
# on the very same preprocessed MIMIC-III files and split, then tests the epoch with the best validation
# Jaccard (the repository's own selection rule) with its own 10x bootstrap test.
#   sbatch -J bl_SafeDrug baselines/run_official.sh SafeDrug      (SafeDrug | GAMENet | Retain | Leap)
#SBATCH -p gpus
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
MODEL=$1
source ~/miniforge3/etc/profile.d/conda.sh
conda activate mambaguard
cd "${SLURM_SUBMIT_DIR:-.}"
ROOT=$PWD
SD=$ROOT/baselines/SafeDrug
if [ ! -d "$SD" ]; then
  git clone -q https://github.com/ycq091044/SafeDrug.git "$SD"
  git -C "$SD" checkout -q 88ce5c377dcdc2aa01aaa88f5478dfa4373ba49a
fi
python -I "$ROOT/baselines/patch_official.py" "$SD/src"
cd "$SD/src"
OUT=$ROOT/outputs/baselines/$MODEL
mkdir -p "$OUT" saved/$MODEL
echo "[$(date)] $(hostname) training $MODEL"
python -u $MODEL.py --cuda 0 2>&1 | tr '\r' '\n' | grep -v "step:" > "$OUT/train.log"
# best validation Jaccard, epoch != 0 (same rule as the original scripts)
BEST=$(ls saved/$MODEL/Epoch_*.model | grep -v "/Epoch_0_" | python -c "
import sys, re
fs = [l.strip() for l in sys.stdin]
print(max(fs, key=lambda f: float(re.search(r'JA_([0-9.]+?)_', f).group(1))))")
echo "best checkpoint: $BEST" | tee "$OUT/best.txt"
python -u $MODEL.py --cuda 0 --Test --resume_path "$BEST" 2>&1 | tr '\r' '\n' | grep -v "step:" > "$OUT/test.log"
tail -3 "$OUT/test.log"
