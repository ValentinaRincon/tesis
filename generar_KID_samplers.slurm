#!/bin/bash
#SBATCH --job-name=C128_SAMPLER_KGEN
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=celeba128-calidad-gen-%j.log
#SBATCH --error=celeba128-calidad-gen-%j.err
set -euo pipefail
cd /hpcfs/home/fisica/v.rincono/tesis
source ~/venvs/thesis/bin/activate
export OMP_NUM_THREADS=3
: "${EPOCH:=60}"
: "${SAMPLER:=ddim250}"
python -u calidad_celeba128.py --mode generate --epoch "$EPOCH" --sampler "$SAMPLER" --budget 27300
