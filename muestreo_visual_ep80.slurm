#!/bin/bash
#SBATCH --job-name=C128_SAMPLER_80
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=celeba128-calidad-vis80-%j.log
#SBATCH --error=celeba128-calidad-vis80-%j.err
set -euo pipefail
cd /hpcfs/home/fisica/v.rincono/tesis
source ~/venvs/thesis/bin/activate
export OMP_NUM_THREADS=3
python -u calidad_celeba128.py --mode preview --epoch 80 --sampler all
