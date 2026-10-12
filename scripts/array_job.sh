#!/usr/bin/env bash
# sbatch --array=1-N%16 scripts/array_job.sh jobs/<group>.txt   (runs line $SLURM_ARRAY_TASK_ID of the file)
#SBATCH -p gpus
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mem=12G
#SBATCH --time=4:00:00
#SBATCH -o logs/%x-%A_%a.out
source ~/miniforge3/etc/profile.d/conda.sh
conda activate mambaguard
cd "${SLURM_SUBMIT_DIR:-.}"
CMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$1")
echo "[$(date)] $(hostname) GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader -i ${CUDA_VISIBLE_DEVICES:-0})"
echo "+ $CMD"
eval "$CMD"
