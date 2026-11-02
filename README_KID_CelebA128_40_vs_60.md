# CelebA 128×128: evaluación controlada de las épocas 40 y 60

Este paquete **NO entrena** ni modifica ningún checkpoint. Compara modelos EMA de las épocas 40 y 60 sobre exactamente las mismas condiciones de generación, para determinar si el refinamiento redujo KID.

## Qué produce

- **5.000 imágenes** nuevas para cada checkpoint (10.000 en total), con **DDIM determinista, 250 pasos** y las mismas semillas de ruido por lote.
- Características **Inception-v3/2048** para las imágenes generadas y **10.000 imágenes del split oficial TEST** de CelebA, usando `CenterCrop(160)→Resize(128×128)`.
- **KID U no sesgado** sobre todos los pares de las muestras, con kernel polinomial de grado 3, reportado multiplicado por 1.000.
- **400 réplicas bootstrap pareadas** (mismos índices de imágenes generadas y de test en ambos checkpoints), con intervalos básicos exploratorios del 95 % para KID 40, KID 60 y diferencia **época 60 menos época 40**.
- Control adicional de referencia: 5.000 fotos de entrenamiento vs 10.000 fotos de test (no es un umbral de calidad).
- Vecinos cercanos mediante similitud coseno de Inception: 5.000 imágenes por modelo comparadas con el subconjunto exacto de **50.000** fotos utilizadas para entrenar; control de 1.000 fotografías de test vs train y mosaicos de casos con mayor similitud.

**Interpretación:** para KID, más pequeño es mejor; **Δ = KID₆₀ − KID₄₀ negativo favorece la época 60**. Los IC son exploratorios y condicionados a estos checkpoints, datos y semillas: no incluyen variación entre entrenamientos. La similitud con vecinos no demuestra ni descarta memorización. No compare directamente valores KID de distintas resoluciones o preprocesamientos.

## Entradas requeridas en Hypatia

En `/hpcfs/home/fisica/v.rincono/tesis` deben existir:

```text
entrenar_celeba64.py
entrenar_celeba128.py
refinar_celeba128.py
CelebA 128x128 ResUNet/checkpoint.pt             # época 40
CelebA 128x128 ResUNet/estado.json
CelebA 128x128 Refinamiento/checkpoint.pt         # época 60
CelebA 128x128 Refinamiento/estado.json
data/celeba/list_eval_partition.txt
data/celeba/img_align_celeba/*.jpg
~/venvs/thesis/  # PyTorch, torchvision, torchmetrics, torch-fidelity, NumPy, Matplotlib, PIL
```

## 1. Subir el ZIP desde PowerShell de Windows

Descarga `evaluacion_KID_CelebA128_40_vs_60_Hypatia.zip` y ejecuta fuera de SSH:

```powershell
scp "$env:USERPROFILE\Downloads\evaluacion_KID_CelebA128_40_vs_60_Hypatia.zip" "v.rincono@hypatia.uniandes.edu.co:/hpcfs/home/fisica/v.rincono/tesis/"
ssh hypatia
```

En **Hypatia**:

```bash
cd ~/tesis
unzip -t evaluacion_KID_CelebA128_40_vs_60_Hypatia.zip
unzip -o evaluacion_KID_CelebA128_40_vs_60_Hypatia.zip
source ~/venvs/thesis/bin/activate
python evaluar_celeba128_40vs60.py --mode selftest
python evaluar_celeba128_40vs60.py --mode check
```

**No ejecute `--mode generate` ni `--mode evaluate` en el nodo de acceso**. `check`, `status` y `selftest` son ligeros.

## 2. Generar 5.000 muestras por checkpoint

Puedes enviar los dos trabajos al mismo tiempo; SLURM los asignará cuando haya GPU disponible:

```bash
JOB40=$(sbatch --parsable generar_kid_celeba128_40.slurm)
JOB60=$(sbatch --parsable generar_kid_celeba128_60.slurm)
echo "Generación época 40: $JOB40; época 60: $JOB60"
squeue -u v.rincono
```

Para ver los registros **en la misma sesión de SSH**:

```bash
tail -n 25 "celeba128-kid40-${JOB40}.log"
tail -n 25 "celeba128-kid60-${JOB60}.log"
```

Si se cerró SSH, el trabajo no se cancela; usa `ls -lt celeba128-kid*.log` y los JobID concretos. Al reenviar un script, los lotes existentes se reutilizan. No envíes **dos jobs simultáneos del mismo checkpoint**, porque producirían archivos concurrentes en la misma carpeta.

**Un estado `COMPLETED` en SLURM no siempre implica que ya existan las 5.000 imágenes:** el script puede detenerse de forma segura antes del límite de tiempo. Consulta:

```bash
python evaluar_celeba128_40vs60.py --mode status
```

Solo cuando **ambas épocas indiquen 5000/5000**, continúa. Si una está incompleta, repite **únicamente** su `sbatch`; reanudará automáticamente.

## 3. Evaluación KID, bootstrap y vecinos

```bash
EVAL=$(sbatch --parsable evaluar_kid_celeba128_40vs60.slurm)
echo "Evaluación KID: $EVAL"
squeue -j "$EVAL"
```

Al terminar (o con el JobID concreto tras reconectar):

```bash
sacct -j "$EVAL" --format=JobID,State,ExitCode,Elapsed,MaxRSS
tail -n 40 "celeba128-kid-eval-${EVAL}.log"
python evaluar_celeba128_40vs60.py --mode status
```

La evaluación guarda las features Inception por conjunto. Si el job se interrumpe, puedes reenviarlo: las features ya completadas se recuperan desde la caché, con comprobación del protocolo y del checkpoint.

## 4. Resultados

Carpeta `~/tesis/Evaluacion CelebA128 ep40 vs ep60/`:

| Archivo | Contenido |
| --- | --- |
| `kid_comparacion.csv` | KID 40, KID 60, diferencia pareada, control real vs real e IC básicos 95 % |
| `bootstrap_pareado.csv` | 400 réplicas para reproducibilidad |
| `kid_ep40_vs_ep60_IC95.png` | Gráfico de KID e intervalos |
| `diferencia_kid_60_menos_40_IC95.png` | Gráfico de cambio por refinamiento |
| `vecinos_resumen.csv` | Estadísticas de similitud coseno, por época |
| `vecinos_detalle.csv` | Vecinos más cercanos de las 10.000 muestras generadas |
| `vecinos_mayor_similitud_40.png` | Mosaico de muestras con similitud alta en época 40 |
| `vecinos_mayor_similitud_60.png` | Ídem en época 60 |
| `protocolo.json`, `resumen.json` | Metadatos y resumen |
| `features/*.pt` | Caché Inception para reproducibilidad (no es necesario descargar) |
| `chunks/ep40` y `chunks/ep60` | Muestras generadas en bloques de 16 |

Para descargar **solo los resultados ligeros** desde PowerShell de Windows, no desde SSH:

```powershell
scp "v.rincono@hypatia.uniandes.edu.co:/hpcfs/home/fisica/v.rincono/tesis/Evaluacion CelebA128 ep40 vs ep60/*.csv" "$env:USERPROFILE\Downloads\"
scp "v.rincono@hypatia.uniandes.edu.co:/hpcfs/home/fisica/v.rincono/tesis/Evaluacion CelebA128 ep40 vs ep60/*.png" "$env:USERPROFILE\Downloads\"
```

**Nota técnica:** las incertidumbres son aproximadas mediante bootstrap básico y pueden ser delicadas cerca de KID = 0; no son pruebas de equivalencia ni sustituyen múltiples semillas de entrenamiento. Las imágenes se generan por lotes de semillas pareadas, con igual tamaño y resolución, de modo que la diferencia compara los checkpoints EMA en las mismas condiciones.
