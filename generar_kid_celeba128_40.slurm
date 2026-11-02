#!/bin/bash
#SBATCH --job-name=C128_KID_40
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=celeba128-kid40-%j.log
#SBATCH --error=celeba128-kid40-%j.err
set -euo pipefail
cd /hpcfs/home/fisica/v.rincono/tesis
source ~/venvs/thesis/bin/activate
export OMP_NUM_THREADS=3
python -u evaluar_celeba128_40vs60.py --mode generate --epoch 40 --time-budget-seconds 20400
