#!/bin/bash
#SBATCH --job-name=DDIM_KID_IC
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=ddim-ic-%j.log
#SBATCH --error=ddim-ic-%j.err

set -euo pipefail
module load python/3.11
source ~/venvs/thesis/bin/activate
cd ~/tesis

echo "VALIDACION DDIM KID iniciada: $(date) $(hostname)"
python -u validar_kid_ddim_bootstrap.py --mode run --reps 400
echo "VALIDACION DDIM KID completada: $(date)"
