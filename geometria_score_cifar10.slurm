#!/bin/bash
#SBATCH --job-name=geo_score_cifar10
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=geo-score-%j.log
#SBATCH --error=geo-score-%j.err

set -euo pipefail
module load python/3.11
source ~/venvs/thesis/bin/activate
cd ~/tesis

echo "Geometria del score | Job ${SLURM_JOB_ID} | Nodo $(hostname) | $(date)"
python -u geometria_score_cifar10.py --mode all --images 24 --probes 6 --times 0,50,250,500,750,999 --bootstrap 1200
echo "Analisis geometrico terminado | $(date)"
