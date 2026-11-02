#!/bin/bash
#SBATCH --job-name=C128_APPENDIX_TRAJ
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=celeba128-apendice-%j.log
#SBATCH --error=celeba128-apendice-%j.err
set -euo pipefail
cd /hpcfs/home/fisica/v.rincono/tesis
source ~/venvs/thesis/bin/activate
export OMP_NUM_THREADS=3
export MKL_NUM_THREADS=3
python -u apendice_trayectorias_celeba128.py --mode run --sampler all
