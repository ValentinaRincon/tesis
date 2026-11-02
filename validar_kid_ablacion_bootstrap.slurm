#!/bin/bash
#SBATCH --job-name=kid_ic_cifar10
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=kid-ic-%j.log
#SBATCH --error=kid-ic-%j.err

set -euo pipefail
module load python/3.11
source ~/venvs/thesis/bin/activate
cd ~/tesis

echo "Análisis IC KID iniciado | $(date) | $(hostname)"
python -u validar_kid_ablacion_bootstrap.py --mode run --reps 400
echo "Análisis IC KID completado | $(date)"
