# Refinamiento controlado de CelebA 128x128 (40 -> 60 épocas)

**No reemplaza** `entrenar_celeba128.py`, `entrenar_celeba128.slurm`, el checkpoint original, ni la carpeta `CelebA 128x128 ResUNet`. Genera resultados únicamente en `CelebA 128x128 Refinamiento/`.

## Configuración

- Inicio desde el checkpoint COMPLETO de la época 40: modelo, EMA, Adam, GradScaler y estados RNG.
- Meta: **60 épocas totales** (20 adicionales).
- El entrenamiento original utilizaba dos grupos Adam: capas heredadas a `5e-5` y bloques nuevos a `1e-4`. Ambos se reducen a la mitad: **compartidas 2.5e-5, nuevas 5e-5**.
- Se conservan exactamente las 50.000 imágenes CelebA, la arquitectura y demás hiperparámetros del código original.
- Se guardan checkpoints al final de cada época y previews a partir de la 45, cada 5 épocas.
- Se compara **época 40 vs época 60** con 16 condiciones iniciales idénticas y DDIM determinista de 250 y 500 pasos; las imágenes son útiles para evaluación visual, no proporcionan por sí mismas una mejora estadística de KID.
- La GPU es obligatoria para `train` y `compare`, pero **`--mode check/status` son ligeros y no importan PyTorch**.

## Instalación

Desde PowerShell en Windows:

```powershell
scp "$env:USERPROFILE\Downloads\refinamiento_celeba128_Hypatia.zip" "v.rincono@hypatia.uniandes.edu.co:/hpcfs/home/fisica/v.rincono/tesis/"
ssh hypatia
```

En Hypatia:

```bash
cd ~/tesis
unzip -t refinamiento_celeba128_Hypatia.zip
unzip -o refinamiento_celeba128_Hypatia.zip
source ~/venvs/thesis/bin/activate
python refinar_celeba128.py --mode check
```

Antes de entrenar, opcionalmente generar referencias del checkpoint de 40 épocas con 250 y 500 pasos (para comprobar si el muestreo por sí solo ya mejora las previews rápidas):

```bash
BASE=$(sbatch --parsable muestrear_checkpoint_40_celeba128.slurm)
echo "Muestreo base: $BASE"
```

Iniciar refinamiento:

```bash
JOB=$(sbatch --parsable refinar_celeba128.slurm)
echo "Refinamiento: $JOB"
squeue -j "$JOB"
```

Comprobar avance:

```bash
python refinar_celeba128.py --mode status
sacct -j "$JOB" --format=JobID,State,ExitCode,Elapsed
```

Al completar 60/60:

```bash
CMP=$(sbatch --parsable comparar_celeba128.slurm)
echo "Comparación: $CMP"
```

O encadenar automáticamente **en el mismo envío** si se desea: `CMP=$(sbatch --parsable --dependency=afterok:${JOB} comparar_celeba128.slurm)`. Si el job de entrenamiento se detiene de forma segura antes de la época 60 (aunque termine con código cero), el comparador fallará; en ese caso reenviar el entrenamiento hasta completar las 60 épocas.

Después de completar la comparación, desde PowerShell:

```powershell
scp "v.rincono@hypatia.uniandes.edu.co:/hpcfs/home/fisica/v.rincono/tesis/CelebA 128x128 Refinamiento/*.png" "$env:USERPROFILE\Downloads\"
scp "v.rincono@hypatia.uniandes.edu.co:/hpcfs/home/fisica/v.rincono/tesis/CelebA 128x128 Refinamiento/comparacion.json" "$env:USERPROFILE\Downloads\"
```

## Seguridad de los experimentos

- No modifiques el código original de 128; su hash y el checkpoint original forman parte del protocolo del refinamiento.
- Evita trabajos simultáneos que escriban en la misma carpeta `CelebA 128x128 Refinamiento/`.
- Asegúrate de que no siga ejecutándose otro job de entrenamiento 128 antes de iniciar este refinamiento.
- Si alguna época falla por GPU, revisa el `.err` y `.log`, no borres el checkpoint; el script reanuda desde la última época completa.
- Las imágenes de referencia son de CelebA y deben tratarse según las condiciones de uso del dataset.
