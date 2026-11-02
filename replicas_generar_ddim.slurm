#!/bin/bash
#SBATCH --job-name=REP_DDIM_GEN
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --array=0-3%1
#SBATCH --output=rep-gen-%A_%a.log
#SBATCH --error=rep-gen-%A_%a.err
set -euo pipefail
module load python/3.11
source ~/venvs/thesis/bin/activate
cd ~/tesis
python -u evaluar_replicas_ddim250.py --mode sample --run-index "${SLURM_ARRAY_TASK_ID}"
