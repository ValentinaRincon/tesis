#!/bin/bash
#SBATCH --job-name=CELEBA128_FINE
#SBATCH --partition=gpu
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --output=celeba128-fine-%j.log
#SBATCH --error=celeba128-fine-%j.err
set -euo pipefail
cd /hpcfs/home/fisica/v.rincono/tesis
source ~/venvs/thesis/bin/activate
export OMP_NUM_THREADS=3
python -u refinar_celeba128.py --mode train --time-budget-seconds 27600
