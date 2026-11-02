#!/bin/bash
#SBATCH --job-name=CIFAR_D_EFF
#SBATCH --partition=express
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=dim-eff-%j.log
#SBATCH --error=dim-eff-%j.err
set -euo pipefail
cd /hpcfs/home/fisica/v.rincono/tesis
source /hpcfs/home/fisica/v.rincono/venvs/thesis/bin/activate
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
python -u dimensionalidad_efectiva_cifar10.py --mode run
