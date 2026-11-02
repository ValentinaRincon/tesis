# Experimento 64x64 RGB en Hypatia — CelebA / ResUNet + Attention + EMA

El objetivo es una **extensión exploratoria** de la investigación de CIFAR-10 32x32 a CelebA 64x64 (12.288 componentes RGB). No altera ni reutiliza los pesos de las redes 32x32: entrenamos una red nueva desde cero. El dataset también cambia, así que **no compare numéricamente el KID obtenido sobre CelebA con el KID de CIFAR-10** como si fueran mediciones en la misma distribución.

## Contenido del paquete

- `entrenar_celeba64.py`: arquitectura, validaciones, entrenamiento reanudable, muestreo y salidas.
- `descargar_celeba64.slurm`: descarga opcional desde una copia pública de CelebA en Hugging Face, ejecutada en nodo de cómputo CPU, NO en nodo login. Requiere paquete Python `huggingface_hub` y conectividad saliente desde el nodo. Si falla, usar la vía manual oficial.
- `preparar_celeba64.slurm`: extracción segura de imágenes del ZIP en un nodo CPU.
- `entrenar_celeba64.slurm`: entrenamiento en una GPU, máximo 4 horas por envío, con guardado al final de cada época y reanudación.
- `muestras_celeba64.slurm`: generador de una cuadrícula de 16 imágenes con DDIM-250 desde el último checkpoint EMA.

## Protocolo explícito

- CelebA `img_align_celeba.zip`, imágenes alineadas 178x218, `CenterCrop(160)` + `Resize((64,64))`, normalización `[-1,1]`.
- Usa solo la **partición oficial de entrenamiento**, `list_eval_partition.txt`, columna 0. Subconjunto de **50.000** imágenes (semilla fija `20261029`) para limitar el coste. No se usan imágenes del test para entrenar.
- Arquitectura ResUNet de ~6,03 M de parámetros, encoder 64→32→16→8 y atención SOLO en resoluciones 16x16/8x8, para contener consumo de VRAM. No es exactamente la misma arquitectura que 32x32: incluye un nivel adicional de resolución.
- Objetivo DDPM ε-MSE, beta lineal 0.0001→0.02 y T=1000. Adam 1e-4, batch 16, EMA 0.999 tras cada paso válido del optimizador, clipping de gradientes a 1.0, mixed precision FP16.
- Hasta 60 épocas; snapshots completos al final de cada época; previews de 16 imágenes DDIM 50 pasos cada 5 épocas. Se puede interrumpir y reanudar; no hay estimación garantizada de calidad ni duración.
- Se fija la semilla, pero el entrenamiento en GPU/AMP no asegura reproducibilidad bit a bit entre ejecuciones o máquinas.

## Paso 0: comprobar recursos (login Hypatia)

```bash
sinfo -p gpu -N -o '%N %t %C %G'
squeue -p gpu -t R -o '%.15i %.15u %.12M %.20R'
```

## Paso 1: subir el paquete desde Windows PowerShell

```powershell
scp "$env:USERPROFILE\Downloads\celeba64_inicio_Hypatia.zip" "v.rincono@hypatia.uniandes.edu.co:/hpcfs/home/fisica/v.rincono/tesis/"
ssh hypatia
```

En Hypatia:

```bash
cd ~/tesis
unzip -t celeba64_inicio_Hypatia.zip
unzip -o celeba64_inicio_Hypatia.zip
source ~/venvs/thesis/bin/activate
python entrenar_celeba64.py --mode selftest
python entrenar_celeba64.py --mode check
```

`selftest` debe completarse. **`check` fallará si aún falta CelebA**: ese es un bloqueo real, no un fallo de arquitectura.

## Paso 2: disponer de los datos

El código necesita, en estas rutas:

```text
~/tesis/data/celeba/img_align_celeba.zip       (para descomprimir, si hace falta)
~/tesis/data/celeba/list_eval_partition.txt    (particiones oficiales)
~/tesis/data/celeba/img_align_celeba/*.jpg     (imágenes extraídas)
```

La fuente primaria es la página académica **CelebA, MMLAB CUHK**: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html. **CelebA tiene una licencia de uso no comercial para investigación**; respete sus condiciones. El repositorio espejo https://huggingface.co/datasets/Yuehao/celeba documenta el archivo comprimido y la descarga, pero su disponibilidad puede variar.

**Opción automatizada, si hay conectividad desde nodos CPU y `huggingface_hub` instalado:**

```bash
python -c 'import huggingface_hub; print(huggingface_hub.__version__)'
sbatch descargar_celeba64.slurm
```

Si falta la biblioteca, instalarla en el entorno virtual según políticas de Hypatia (`python -m pip install huggingface_hub`), o usar la descarga manual. **No mande el job de preparación hasta que la descarga haya finalizado**. Si el espejo no tiene `list_eval_partition.txt`, descargarlo desde la fuente oficial. La copia pública de imágenes no cambia la licencia original.

**Opción manual:** descargar el archivo oficial `img_align_celeba.zip` y `list_eval_partition.txt` en Windows. Desde PowerShell:

```powershell
ssh hypatia 'mkdir -p ~/tesis/data/celeba'
scp "$env:USERPROFILE\Downloads\img_align_celeba.zip" "$env:USERPROFILE\Downloads\list_eval_partition.txt" "v.rincono@hypatia.uniandes.edu.co:/hpcfs/home/fisica/v.rincono/tesis/data/celeba/"
```

Después de tener ambos archivos, en Hypatia:

```bash
cd ~/tesis
sbatch preparar_celeba64.slurm
```

Este paso extrae más de 200.000 archivos JPEG y puede ocupar espacio en disco y tiempo de CPU. NO ejecutarlo en el nodo login. Al finalizar:

```bash
python entrenar_celeba64.py --mode check
```

Si marca `CHECK OK`, el dataset está listo. En Hypatia deben existir espacio suficiente y cuota para el ZIP, los JPEG extraídos y checkpoints.

## Paso 3: entrenar con GPU

```bash
cd ~/tesis
source ~/venvs/thesis/bin/activate
python entrenar_celeba64.py --mode check
JOB=$(sbatch --parsable entrenar_celeba64.slurm)
echo "Entrenamiento 64x64: $JOB"
squeue -j "$JOB"
```

El entrenamiento se guarda en `~/tesis/CelebA 64x64 ResUNet/`. Durante el trabajo se almacenan `checkpoint.pt`, `perdidas.csv`, `estado.json` y `previews/epoca_005.png`, etc. En Hypatia login:

```bash
python entrenar_celeba64.py --mode status
sacct -j "$JOB" --format=JobID,State,ExitCode,Elapsed
```

El script pausa de forma controlada **al final de una época** si se aproxima al límite de tiempo. Cuando el job finalice, comprobar `status`: si no ha alcanzado 60/60, repetir `sbatch entrenar_celeba64.slurm` para continuar desde checkpoint. **No lanzar jobs duplicados en paralelo sobre la misma carpeta**. Si el job termina por error u OOM, inspeccionar `celeba64-<jobid>.err` y no borrar el checkpoint.

Si necesitamos una muestra DDIM-250 adicional en una GPU:

```bash
sbatch muestras_celeba64.slurm
```

Las muestras de progreso anteriores tienen 50 pasos DDIM; la cuadrícula final `muestras_ddim250.png` utiliza 250 pasos. Ambas son imágenes de 64x64 genuinas (transformadas desde fotografías de mayor resolución), no CIFAR-10 reescalado.

## Limitaciones

- Se trata de **un modelo de exploración con 50.000 imágenes**, no una evaluación exhaustiva sobre CelebA. El resultado puede necesitar más épocas o datos, según las imágenes obtenidas.
- CelebA contiene rostros de personas reales, así que el conjunto de entrenamiento implica consideraciones de licencia, consentimiento y posibles sesgos; las imágenes sintéticas no deben utilizarse para identificar ni atribuir identidades.
- Antes de afirmar calidad cuantitativa sobre CelebA, hará falta generar un conjunto suficientemente grande y evaluar KID con imágenes reales de **CelebA test** usando el mismo preprocesado (no reutilizar CIFAR-10 test).
- No se garantiza tiempo de asignación de GPU aunque haya recursos físicos disponibles: depende de SLURM.
