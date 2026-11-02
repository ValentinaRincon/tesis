#!/bin/bash
#SBATCH --job-name=CIFAR_REP
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --array=0-3%1
#SBATCH --output=rep-train-%A_%a.log
#SBATCH --error=rep-train-%A_%a.err
set -euo pipefail
module load python/3.11
source ~/venvs/thesis/bin/activate
cd ~/tesis
export CUBLAS_WORKSPACE_CONFIG=:4096:8
echo "Comienzo: $(date) nodo $(hostname) indice ${SLURM_ARRAY_TASK_ID}"
python -u replicar_entrenamiento_cifar10.py --mode train --run-index "${SLURM_ARRAY_TASK_ID}" --time-budget-seconds 13200
echo "Finalizacion de este tramo: $(date)"
