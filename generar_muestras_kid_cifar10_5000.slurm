#!/bin/bash
#SBATCH --job-name=kid5k_gen_cifar
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --array=0-3%1
#SBATCH --output=kid5k-gen-%A_%a.log
#SBATCH --error=kid5k-gen-%A_%a.err

set -euo pipefail
module load python/3.11
source ~/venvs/thesis/bin/activate
cd ~/tesis

echo "Array ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} | $(hostname) | $(date)"
python -u evaluar_kid_cifar10_5000.py --mode sample --model-index "${SLURM_ARRAY_TASK_ID}"
echo "Muestreo completo del índice ${SLURM_ARRAY_TASK_ID} | $(date)"
