#!/bin/bash

#SBATCH --job-name="cifar10_resatt"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=04:00:00

#SBATCH --output="cifar10-resunet-att-ema-%j.log"
#SBATCH --error="cifar10-resunet-att-ema-%j.err"


echo "========================================"
echo "DDPM CIFAR-10 + ResUNet + Self-Attention + EMA"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodo: $(hostname)"
echo "Inicio: $(date)"
echo "========================================"


module load python/3.11

source ~/venvs/thesis/bin/activate

cd ~/tesis


echo "Python:"
which python

echo ""
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv

echo ""


# El Python se detiene de forma limpia después de ~3 h 30 min
# si aún no ha terminado. Deja margen frente al walltime de 4 h.
export MAX_RUNTIME_MINUTES=210


python -u CIFAR10_ResUNet_Attention_EMA_job.py

STATUS=$?


echo ""
echo "Python terminó con código: $STATUS"
echo "Hora: $(date)"


# Si hubo un error real, NO crear más jobs automáticamente.
if [ "$STATUS" -ne 0 ]; then

    echo "El programa terminó con error."
    echo "Revisa cifar10-resunet-att-ema-${SLURM_JOB_ID}.err"

    exit "$STATUS"
fi


COMPLETE_FLAG="$HOME/tesis/modelos/cifar10_resunet_attention_ema_COMPLETE.flag"


# Si terminó las 200 épocas, no enviar otro job.
if [ -f "$COMPLETE_FLAG" ]; then

    echo "========================================"
    echo "Entrenamiento completo."
    echo "No se requiere otro job."
    echo "Fin: $(date)"
    echo "========================================"

    exit 0
fi


# Si Python salió limpiamente por el límite interno de tiempo,
# mandar automáticamente el siguiente job.
echo "========================================"
echo "Entrenamiento incompleto pero checkpoint guardado."
echo "Enviando continuación automáticamente..."
echo "========================================"

NEW_JOB_ID=$(sbatch --parsable "$0")

if [ $? -eq 0 ]; then

    echo "Nuevo job enviado: $NEW_JOB_ID"

else

    echo "ERROR: no se pudo enviar automáticamente la continuación."
    exit 1

fi
