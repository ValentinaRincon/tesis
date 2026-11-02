#!/bin/bash
#SBATCH --job-name=ablac_cifar_kid
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=ablac-kid-%j.log
#SBATCH --error=ablac-kid-%j.err

set -euo pipefail
module load python/3.11
source ~/venvs/thesis/bin/activate
cd ~/tesis

echo "KID ablación CIFAR10 / $(hostname) / $(date)"
python -u ablacion_temporal_cifar10.py --mode kid
echo "KID y figuras completadas / $(date)"
