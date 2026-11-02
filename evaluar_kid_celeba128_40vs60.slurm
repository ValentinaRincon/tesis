#!/bin/bash
#SBATCH --job-name=C128_KID_EVAL
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=celeba128-kid-eval-%j.log
#SBATCH --error=celeba128-kid-eval-%j.err
set -euo pipefail
cd /hpcfs/home/fisica/v.rincono/tesis
source ~/venvs/thesis/bin/activate
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
python -u evaluar_celeba128_40vs60.py --mode evaluate
