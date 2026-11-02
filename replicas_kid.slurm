#!/bin/bash
#SBATCH --job-name=REP_KID
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=rep-kid-%j.log
#SBATCH --error=rep-kid-%j.err
set -euo pipefail
module load python/3.11
source ~/venvs/thesis/bin/activate
cd ~/tesis
python -u evaluar_replicas_ddim250.py --mode kid
