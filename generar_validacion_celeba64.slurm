#!/bin/bash
#SBATCH --job-name=CELEBA64_GEN
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=celeba64-val-gen-%j.log
#SBATCH --error=celeba64-val-gen-%j.err
set -euo pipefail
cd ~/tesis
source ~/venvs/thesis/bin/activate
export OMP_NUM_THREADS=1
python -u evaluar_celeba64.py --mode generate --time-budget-seconds 13200
