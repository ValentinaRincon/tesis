#!/bin/bash
#SBATCH --job-name=CELEBA64
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=24G
#SBATCH --time=1-00:00:00
#SBATCH --output=celeba64-%j.log
#SBATCH --error=celeba64-%j.err
set -euo pipefail
cd ~/tesis
source ~/venvs/thesis/bin/activate
export OMP_NUM_THREADS=1
python -u entrenar_celeba64.py --mode train --time-budget-seconds 248400
