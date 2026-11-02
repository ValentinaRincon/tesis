#!/bin/bash
#SBATCH --job-name=C128_FINE80
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=celeba128-calidad-train-%j.log
#SBATCH --error=celeba128-calidad-train-%j.err
set -euo pipefail
cd /hpcfs/home/fisica/v.rincono/tesis
source ~/venvs/thesis/bin/activate
export OMP_NUM_THREADS=3
python -u refinar_celeba128_80.py --mode train --budget 27300
