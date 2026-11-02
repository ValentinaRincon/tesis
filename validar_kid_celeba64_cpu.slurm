#!/bin/bash
#SBATCH --job-name=CELEBA64_IC_CPU
#SBATCH --partition=express
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=03:00:00
#SBATCH --output=celeba64-ic-cpu-%j.log
#SBATCH --error=celeba64-ic-cpu-%j.err
set -euo pipefail
cd ~/tesis
source ~/venvs/thesis/bin/activate
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
python -u validar_kid_celeba64_cpu.py --mode run
