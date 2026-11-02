#!/bin/bash
#SBATCH --job-name=score_cifar10
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=score-cifar10-%j.log
#SBATCH --error=score-cifar10-%j.err

set -euo pipefail
module load python/3.11
source ~/venvs/thesis/bin/activate
cd ~/tesis

echo "Score CIFAR-10 | Job ${SLURM_JOB_ID} | Nodo $(hostname) | $(date)"
python -u analizar_score_cifar10.py --mode all
echo "Análisis terminado | $(date)"
