#!/bin/bash
#SBATCH --job-name=ddim_cifar_gen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --array=0-9%1
#SBATCH --output=ddim-gen-%A_%a.log
#SBATCH --error=ddim-gen-%A_%a.err
set -euo pipefail
module load python/3.11
source ~/venvs/thesis/bin/activate
cd ~/tesis
echo "Inicio DDIM variante ${SLURM_ARRAY_TASK_ID} / $(hostname) / $(date)"
python -u comparar_ddim_cifar10.py --mode sample --variant-index "${SLURM_ARRAY_TASK_ID}"
echo "Generacion DDIM terminada / $(date)"
