#!/bin/bash

#SBATCH --output="cifar10-%j.log"
#SBATCH --error="cifar10-%j.err"
#SBATCH --job-name="cifar10_ddpm"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00


echo "========================================"
echo "Entrenamiento DDPM CIFAR-10"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodo: $(hostname)"
echo "Inicio: $(date)"
echo "========================================"


# Cargar el mismo Python usado en Hypatia
module load python/3.11


# Activar tu entorno
source ~/venvs/thesis/bin/activate


# Ir a tu proyecto
cd ~/tesis


echo ""
echo "Python utilizado:"
which python

echo ""
echo "PyTorch:"
python -c "import torch; print(torch.__version__)"

echo ""
echo "CUDA disponible:"
python -c "import torch; print(torch.cuda.is_available())"

echo ""
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv


echo ""
echo "========================================"
echo "Comenzando entrenamiento"
echo "========================================"


python -u Prueba_con_CIFAR10_seguro.py


echo ""
echo "========================================"
echo "Fin del job: $(date)"
echo "========================================"
