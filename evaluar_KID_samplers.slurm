#!/bin/bash
#SBATCH --job-name=C128_SAMPLER_KID
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=celeba128-calidad-kid-%j.log
#SBATCH --error=celeba128-calidad-kid-%j.err
set -euo pipefail
cd /hpcfs/home/fisica/v.rincono/tesis
source ~/venvs/thesis/bin/activate
export OMP_NUM_THREADS=4
: "${CASES:=60:ddim250,60:ddim500,60:ddpm1000}"
python -u calidad_celeba128.py --mode evaluate --cases "$CASES"
