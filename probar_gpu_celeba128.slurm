#!/bin/bash
#SBATCH --job-name=CELEBA128_TEST
#SBATCH --partition=gpu
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --output=celeba128-test-%j.log
#SBATCH --error=celeba128-test-%j.err
set -euo pipefail
cd /hpcfs/home/fisica/v.rincono/tesis
source ~/venvs/thesis/bin/activate
export OMP_NUM_THREADS=3
python -u entrenar_celeba128.py --mode gpu-test
