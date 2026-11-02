#!/bin/bash
#SBATCH --job-name=CELEBA_PREP
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=celeba-prep-%j.log
#SBATCH --error=celeba-prep-%j.err
set -euo pipefail
cd ~/tesis
source ~/venvs/thesis/bin/activate
python -u entrenar_celeba64.py --mode prepare
