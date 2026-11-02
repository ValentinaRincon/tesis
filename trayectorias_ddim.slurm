#!/bin/bash
#SBATCH --job-name=DDIM_TRAY
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=ddim-tray-%j.log
#SBATCH --error=ddim-tray-%j.err
set -euo pipefail
module load python/3.11
source ~/venvs/thesis/bin/activate
cd ~/tesis
python -u trayectorias_ddim_cifar10.py --mode run
