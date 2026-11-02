#!/bin/bash
#SBATCH --job-name=CELEBA64_SAMPLES
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=celeba64-samples-%j.log
#SBATCH --error=celeba64-samples-%j.err
set -euo pipefail
cd ~/tesis
source ~/venvs/thesis/bin/activate
python -u entrenar_celeba64.py --mode sample
