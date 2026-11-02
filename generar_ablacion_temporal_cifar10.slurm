#!/bin/bash
#SBATCH --job-name=ablac_cifar_gen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --array=0-5%1
#SBATCH --output=ablac-gen-%A_%a.log
#SBATCH --error=ablac-gen-%A_%a.err

set -euo pipefail
module load python/3.11
source ~/venvs/thesis/bin/activate
cd ~/tesis

echo "Ablación CIFAR10 / híbrido ${SLURM_ARRAY_TASK_ID} / GPU $(hostname) / $(date)"
python -u ablacion_temporal_cifar10.py --mode sample --variant-index "${SLURM_ARRAY_TASK_ID}"
echo "Híbrido completado / $(date)"
