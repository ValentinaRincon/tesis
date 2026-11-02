# Experimento CelebA 128×128 — Hypatia

## Alcance científico

- **Resolución:** RGB 128×128 = 49.152 componentes por imagen.
- **Datos:** CelebA alineado, split oficial de entrenamiento, **las mismas 50.000 identidades de archivo** seleccionadas en el experimento 64×64 (semilla `20261029`). Recorte central de 160 px, luego resize a 128×128 y normalización `[-1,1]`. No se utiliza CIFAR-10 redimensionado ni se mezclan los splits oficiales.
- **Modelo:** ResUNet128 (aprox. 6,15 millones de parámetros), con escala adicional `128→64→32→16→8` y atención exclusivamente en `16×16` y `8×8`.
- **Inicialización:** **transferencia de pesos** desde los componentes compatibles del checkpoint EMA del modelo CelebA64 previamente entrenado. Los bloques nuevos de la escala 128×128 se inicializan aleatoriamente. **No describir este experimento como entrenamiento desde cero en 128×128.**
- **Objetivo:** predicción de ruido DDPM (`1000` timesteps, betas lineales 0.0001 a 0.02), con EMA de `0.999`. Vistas previas DDIM determinista de 50 pasos. La configuración es **exploratoria**; 40 épocas no garantizan calidad visual.
- **Optimización:** Adam; learning rate de `5e-5` para parámetros transferidos y `1e-4` para nuevos parámetros; batch de 4, AMP fp16, grad clip de 1.0.

## 0. Requisitos previos

Trabajar desde `~/tesis` y mantener **sin cambios** el archivo `entrenar_celeba64.py`, la carpeta `CelebA 64x64 ResUNet/` y `data/celeba/`.

No se requiere descargar ni extraer CelebA otra vez. No se modifica ni sobrescribe ningún checkpoint de 64×64.

## 1. Subir y descomprimir

En **Windows PowerShell** (la ruta presupone que el ZIP está en Descargas):

```powershell
scp "$env:USERPROFILE\Downloads\celeba128_inicio_Hypatia.zip" "v.rincono@hypatia.uniandes.edu.co:/hpcfs/home/fisica/v.rincono/tesis/"
ssh hypatia
```

En **Hypatia**:

```bash
cd ~/tesis
unzip -t celeba128_inicio_Hypatia.zip
unzip -o celeba128_inicio_Hypatia.zip
source ~/venvs/thesis/bin/activate
python entrenar_celeba128.py --mode selftest
python entrenar_celeba128.py --mode check
```

El `check` usa poca memoria y **no carga** el checkpoint completo. Si muestra `CHECK OK`, continuar.

## 2. GPU preflight — primero ejecutar esto

Antes de un entrenamiento prolongado, comprobar con **una actualización de gradiente en GPU** la forma del tensor, el `forward`, el `backward`, la carga del checkpoint 64×64 y la memoria pico:

```bash
TEST=$(sbatch --parsable probar_gpu_celeba128.slurm)
echo "GPU test: $TEST"
squeue -j "$TEST"
```

Al desaparecer de la cola:

```bash
sacct -j "$TEST" --format=JobID,State,ExitCode,Elapsed,MaxRSS
cat "celeba128-test-${TEST}.log"
cat "celeba128-test-${TEST}.err"
```

**Solo seguir si el job terminó `COMPLETED` y aparece `GPU TEST OK`**. Si aparece `CUDA out of memory`, reducir `batch_size` en `CONFIG` **antes** del primer entrenamiento; no modificarlo a mitad de un experimento reanudable. El script no modifica la carpeta de salida durante esta prueba.

## 3. Entrenamiento reanudable

La configuración actual solicita **12 horas de GPU** y un presupuesto interno de 11 h 40 min. Es un máximo, no una predicción de duración. El planificador puede retrasar solicitudes largas.

```bash
JOB=$(sbatch --parsable entrenar_celeba128.slurm)
echo "Entrenamiento 128x128: $JOB"
squeue -j "$JOB"
```

Para revisar épocas terminadas sin cargar tensores:

```bash
python entrenar_celeba128.py --mode status
tail -n 20 "celeba128-${JOB}.log"
```

El programa guarda `checkpoint.pt`, `perdidas.csv`, `estado.json` y `previews/epoca_NNN.png` en una carpeta **nueva**: `~/tesis/CelebA 128x128 ResUNet`.

Si el trabajo se detiene con una época incompleta, **reanuda** desde la última época guardada reenviando exactamente el mismo script:

```bash
JOB=$(sbatch --parsable entrenar_celeba128.slurm)
echo "$JOB"
```

No enviar dos entrenamientos 128×128 simultáneos: sobrescribirían el mismo checkpoint. Verificar que el anterior no sigue ejecutándose antes de reenviar.

El modelo final, si llega a la época 40, se guarda en `ema_final.pt`. Antes de eso, el checkpoint más reciente también contiene pesos EMA.

## 4. Vistas previas y generación al final

```bash
ls -lh "CelebA 128x128 ResUNet/previews"
```

Descargar las vistas previas desde PowerShell:

```powershell
scp -r "v.rincono@hypatia.uniandes.edu.co:/hpcfs/home/fisica/v.rincono/tesis/CelebA 128x128 ResUNet/previews" "$env:USERPROFILE\Downloads\"
```

Para generar nuevas muestras con una semilla distinta de la utilizada en las vistas previas:

```bash
SAMPLE=$(sbatch --parsable muestras_celeba128.slurm)
echo "$SAMPLE"
```

Archivo final previsto: `CelebA 128x128 ResUNet/muestras_ddim250.png`.

## 5. Lectura e interpretación

- El primer objetivo es observar generación coherente a 128×128. **No atribuir mejor calidad únicamente al mayor número de píxeles**.
- Este entrenamiento utiliza **warm-start desde 64×64**; no es comparación controlada de entrenamiento desde cero entre resoluciones.
- Las vistas previas provienen de ruido gaussiano con semilla fija para facilitar la comparación entre épocas. Las muestras finales emplean otra semilla.
- La evaluación con KID a 128×128 requerirá generar miles de muestras y compararlas con el split oficial de prueba, usando el mismo recorte y redimensionamiento.
- Si llega el plazo de redacción antes del entrenamiento, los resultados validados de 32×32 y 64×64 ya están completos; la extensión 128×128 queda exploratoria.
