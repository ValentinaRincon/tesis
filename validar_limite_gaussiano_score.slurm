#!/bin/bash
#SBATCH --job-name=gauss_score_cifar10
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=gauss-score-%j.log
#SBATCH --error=gauss-score-%j.err

set -euo pipefail
module load python/3.11
source ~/venvs/thesis/bin/activate
cd ~/tesis

echo "Limite gaussiano del score | job ${SLURM_JOB_ID} | $(hostname) | $(date)"
python -u validar_limite_gaussiano_score.py --mode all --images 24 --probes 16 --times 500,750,900,950,975,999 --bootstrap 1500
echo "Analisis gaussiano completado | $(date)"
