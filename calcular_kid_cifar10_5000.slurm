#!/bin/bash
#SBATCH --job-name=kid5k_cifar10
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=kid5k-metrica-%j.log
#SBATCH --error=kid5k-metrica-%j.err

set -euo pipefail
module load python/3.11
source ~/venvs/thesis/bin/activate
cd ~/tesis

echo "KID ${SLURM_JOB_ID} | $(hostname) | $(date)"
python -u evaluar_kid_cifar10_5000.py --mode kid
echo "KID completo | $(date)"
