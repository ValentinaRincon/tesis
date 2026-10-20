#!/bin/bash
#SBATCH --job-name="eval_cifar10"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output="evaluacion-cifar10-%j.log"
#SBATCH --error="evaluacion-cifar10-%j.err"

set -u

echo "========================================"
echo "Evaluación común DDPM CIFAR-10"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodo: $(hostname)"
echo "Inicio: $(date)"
echo "========================================"

module load python/3.11
source ~/venvs/thesis/bin/activate

cd ~/tesis || exit 1

echo
echo "Python:"
which python

echo
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv

echo

python -u evaluar_modelos_cifar10.py --require-resunet
STATUS=$?

echo
echo "========================================"
echo "Fin: $(date)"
echo "Código de salida: ${STATUS}"
echo "========================================"

exit ${STATUS}
