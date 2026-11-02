# CelebA 64×64 — KID y auditoría de vecinos cercanos

## Propósito

Evaluar el **checkpoint EMA final de 60 épocas** entrenado con `entrenar_celeba64.py`, sin reentrenar y sin copiar ni modificar el código original. Se generan 5.000 muestras nuevas DDIM determinista (250 pasos, semillas diferentes a las vistas previas). Se compara la distribución de las muestras con 10.000 imágenes extraídas exclusivamente del **split oficial test de CelebA** mediante **Kernel Inception Distance (KID)**. Se auditan vecinos más cercanos de las muestras generadas entre las **50.000 imágenes efectivamente utilizadas para entrenar**.

Se reutiliza la clase `ResUNet64`, el algoritmo `sample_ddim` y el preprocesamiento `FaceDataset` del archivo original del proyecto. **No sobrescribir `entrenar_celeba64.py`**.

## Contenido del paquete

- `evaluar_celeba64.py` — programa principal.
- `generar_validacion_celeba64.slurm` — generación desde ruido en GPU, reanudable (lotes atómicos).
- `evaluar_kid_vecinos_celeba64.slurm` — características Inception, KID y vecinos, en GPU.
- Este README.

## Requisitos

En `~/tesis` deben existir:

- `entrenar_celeba64.py` (el mismo programa usado en el entrenamiento).
- `CelebA 64x64 ResUNet/ema_final.pt` y `estado.json` mostrando 60/60 épocas completas.
- `data/celeba/img_align_celeba/` y `data/celeba/list_eval_partition.txt`.
- Entorno `~/venvs/thesis`, con PyTorch, torchvision, torchmetrics, torch-fidelity y PIL (ya se usaron en los experimentos anteriores).

Si el checkpoint original no corresponde al entrenamiento completo, el script detiene la evaluación. El protocolo contiene la huella SHA-256 del checkpoint y del archivo de particiones; no mezcla resultados de modelos distintos.

## Instalación: PowerShell Windows

Primero descargar el ZIP de esta conversación a `Downloads`. Luego:

```powershell
scp "$env:USERPROFILE\Downloads\validacion_celeba64_KID_vecinos_Hypatia.zip" "v.rincono@hypatia.uniandes.edu.co:/hpcfs/home/fisica/v.rincono/tesis/"
ssh hypatia
```

En Hypatia:

```bash
cd ~/tesis
unzip -t validacion_celeba64_KID_vecinos_Hypatia.zip
unzip -o validacion_celeba64_KID_vecinos_Hypatia.zip
source ~/venvs/thesis/bin/activate
python evaluar_celeba64.py --mode selftest
python evaluar_celeba64.py --mode check
```

## Generación y evaluación

Primero generar las imágenes nuevas:

```bash
GEN=$(sbatch --parsable generar_validacion_celeba64.slurm)
echo "Generacion: $GEN"
squeue -j "$GEN"
```

El generador guarda **313 lotes**, 312 de 16 imágenes y uno de 8, como `torch.uint8`, cada uno en un archivo `.pt`. La salida está en `~/tesis/Evaluacion CelebA64 KID Vecinos/chunks_ddim250/`. El `status` es ligero y no carga esos tensores.

Al concluir SLURM, verificar:

```bash
sacct -j "$GEN" --format=JobID,State,ExitCode,Elapsed
python evaluar_celeba64.py --mode status
```

El estado debe ser `COMPLETED` y `313/313` lotes. Si alcanzó el límite, **reenviar el mismo job de generación** y aprovechará los lotes completos. No enviar la evaluación antes de que la generación finalice correctamente.

Después:

```bash
KID=$(sbatch --parsable evaluar_kid_vecinos_celeba64.slurm)
echo "Evaluacion: $KID"
```

Se guardan las características Inception (2048 dimensiones) de los 50k rostros realmente usados en entrenamiento, 10k de test y 5k muestras. Se reutilizan las caches si se interrumpe la evaluación.

## Archivos de salida

`~/tesis/Evaluacion CelebA64 KID Vecinos/`:

- `kid_celeba64.csv`: KID U insesgado calculado con todos los vectores de características 10k+5k, más promedio/desviación entre 100 subconjuntos de tamaño 500. **La desviación entre subconjuntos NO es un IC del 95 %.**
- `vecinos_resumen.csv`: distribuciones de similitud coseno con las 50k imágenes de train, y control con 1k imágenes del split test.
- `vecinos_detalle.csv`: para cada una de las 5k muestras, nombre y similitud de los dos vecinos Inception más cercanos en entrenamiento, más MSE de píxel contra el primer vecino Inception.
- `vecinos_primeras16.png`: primeras dieciséis muestras nuevas y sus dos vecinos de entrenamiento.
- `vecinos_mayor_similitud.png`: dieciséis casos con mayores similitudes Inception a entrenamiento.
- `features/*.pt`: cachés Inception reproducibles.
- `chunks_ddim250/*.pt`: 5.000 muestras uint8; no hace falta descargarlas para analizar tablas y mosaicos.

**Interpretación**: la similitud de Inception alta por sí sola no demuestra memorización; la proximidad en píxeles contra el vecino encontrado también es exploratoria. Aun si no se hallan duplicados evidentes, NO se puede demostrar ausencia de memorización en todas sus formas. Los 50k vecinos buscados son el subconjunto realmente usado durante el entrenamiento, no todos los 162.770 rostros del split train oficial.

**KID de CelebA no es directamente comparable numéricamente con KID de CIFAR-10**, porque cambia la distribución de referencia y la resolución/dominio.

## Descargar en Windows

```powershell
scp "v.rincono@hypatia.uniandes.edu.co:/hpcfs/home/fisica/v.rincono/tesis/Evaluacion CelebA64 KID Vecinos/*.csv" "$env:USERPROFILE\Downloads\"
scp "v.rincono@hypatia.uniandes.edu.co:/hpcfs/home/fisica/v.rincono/tesis/Evaluacion CelebA64 KID Vecinos/*.png" "$env:USERPROFILE\Downloads\"
```

## Límites

- Los tiempos de SLURM son **máximos**, no estimaciones de duración.
- El extractor Inception estándar puede descargar pesos la primera vez; el entorno de Hypatia ya calculó KID anteriormente, pero conviene validar disponibilidad en `check`.
- Los vecinos son aproximaciones visuales/perceptuales, no una prueba concluyente de originalidad.
- El estudio no incluye intervalos de confianza bootstrap para el KID en este paso: calcula U completo y la variación descriptiva entre subconjuntos. Se puede añadir un bootstrap posterior sin volver a generar muestras.
