# CelebA 128×128 — mejora de calidad desde época 60

Paquete para **Hypatia**. Reutiliza la arquitectura, pesos y evaluación de CelebA128 disponibles en `~/tesis`; **no modifica ningún checkpoint anterior**. El orden importa: primero evaluar muestreo, y **solo si las muestras siguen insuficientes**, continuar hasta época 80.

## Requisitos existentes

En `~/tesis` deben estar `entrenar_celeba128.py`, `entrenar_celeba64.py`, `refinar_celeba128.py`, `CelebA 128x128 Refinamiento/checkpoint.pt` (época 60), `CelebA 128x128 Refinamiento/estado.json` y, para KID, la caché `Evaluacion CelebA128 ep40 vs ep60/features/test.pt` de las 10.000 fotografías oficiales de prueba. Entorno GPU: `~/venvs/thesis/bin/activate`.

### A. Subir desde Windows PowerShell

Descarga `mejorar_calidad_CelebA128_Hypatia.zip` a `Descargas`. En **PowerShell**, no dentro de Hypatia:

```powershell
scp "$env:USERPROFILE\Downloads\mejorar_calidad_CelebA128_Hypatia.zip" "v.rincono@hypatia.uniandes.edu.co:/hpcfs/home/fisica/v.rincono/tesis/"
ssh hypatia
```

En **Hypatia**, nodo de acceso:

```bash
cd ~/tesis
unzip -t mejorar_calidad_CelebA128_Hypatia.zip
unzip -o mejorar_calidad_CelebA128_Hypatia.zip
source ~/venvs/thesis/bin/activate
python calidad_celeba128.py --mode selftest
python calidad_celeba128.py --mode check
python refinar_celeba128_80.py --mode check
```

`check`, `status` y `selftest` no cargan modelos en GPU y son seguros en el nodo de acceso. Otros modos solo vía SLURM.

## Fase 1: comparar tres muestreadores con **época 60**

```bash
VIS=$(sbatch --parsable muestreo_visual_ep60.slurm)
echo "Job de comparación visual: $VIS"
squeue -j "$VIS"
```

Cuando el job ya no aparezca en `squeue`, **verifica su estado** (no deduzcas que terminó correctamente solo porque desapareció):

```bash
sacct -j "$VIS" --format=JobID,State,ExitCode,Elapsed,MaxRSS
tail -n 30 "celeba128-calidad-vis-${VIS}.log"
python calidad_celeba128.py --mode status
```

Deben existir cuatro imágenes en:

```text
CelebA 128x128 Mejor Calidad/previas/
  ep060_ddim250.png
  ep060_ddim500.png
  ep060_ddpm1000.png
  comparacion_samplers_ep060.png
```

Todas utilizan el **mismo x_T por muestra** y el modelo **EMA** de la época 60. DDIM es determinista, con `η=0`. DDPM utiliza innovaciones aleatorias adicionales y puede producir diferencias por ese motivo; no se promete que 1.000 pasos sean mejores.

Desde **PowerShell** descarga el mosaico:

```powershell
scp "v.rincono@hypatia.uniandes.edu.co:/hpcfs/home/fisica/v.rincono/tesis/CelebA 128x128 Mejor Calidad/previas/comparacion_samplers_ep060.png" "$env:USERPROFILE\Downloads\"
```

**Regla inicial:** si DDIM250 y DDIM500 se ven prácticamente iguales, no se justifican más pasos sin evidencia cuantitativa. Esta grilla de 16 imágenes es exploratoria, no una evaluación estadística.

## Fase 1B opcional: KID de muestreadores

Para decidir por KID entre los tres métodos, generar **1024 imágenes por muestreador**. Las imágenes corresponden a las mismas semillas y ruido inicial por índice; la parte aleatoria de DDPM no es idéntica a DDIM. Las muestras quedan guardadas por lotes y **pueden reanudarse** tras un corte.

```bash
J250=$(sbatch --parsable --export=ALL,EPOCH=60,SAMPLER=ddim250 generar_KID_samplers.slurm)
J500=$(sbatch --parsable --export=ALL,EPOCH=60,SAMPLER=ddim500 generar_KID_samplers.slurm)
JDDPM=$(sbatch --parsable --export=ALL,EPOCH=60,SAMPLER=ddpm1000 generar_KID_samplers.slurm)
echo "KID: DDIM250=$J250 DDIM500=$J500 DDPM1000=$JDDPM"
```

Consulta:

```bash
python calidad_celeba128.py --mode status
squeue -u v.rincono
```

**Importante:** cada muestreador debe mostrar `1024/1024 imágenes` antes de evaluar. Si uno quedó incompleto, **reenvía solo ese mismo script**; reutiliza los lotes previos. No envíes dos trabajos concurrentes para el mismo muestreador y época.

Cuando estén los tres:

```bash
KID=$(sbatch --parsable evaluar_KID_samplers.slurm)
echo "Job KID=$KID"
```

Verificar:

```bash
sacct -j "$KID" --format=JobID,State,ExitCode,Elapsed
tail -n 35 "celeba128-calidad-kid-${KID}.log"
cat "CelebA 128x128 Mejor Calidad/KID_ep60_samplers.csv"
```

La evaluación **reutiliza** las mismas características Inception de las 10.000 imágenes reales de TEST usadas en la comparación época 40 vs 60 y el mismo kernel de KID U. El KID de 1024 muestras **no es directamente equiparable como estimador de igual precisión** al KID anterior calculado con 5000 muestras (tienen incertidumbres distintas). Usa los intervalos bootstrap del 95% y tiempo de muestreo para decidir: una diferencia pequeña e incierta no justifica duplicar el tiempo de generación.

## Fase 2: continuación **opcional** época 60 → 80

Si las imágenes del mejor muestreador aún tienen defectos importantes, prolonga entrenamiento desde el checkpoint completo **de época 60**, con Adam/EMA/GradScaler/RNG restaurados:

- Parámetros heredados: LR pasa de `2.5e-5` a `1.25e-5`.
- Parámetros nuevos de la etapa 128: LR pasa de `5e-5` a `2.5e-5`.
- Batch 4, datos fijos 50.000 imágenes, EMA 0.999, hasta época 80.
- Checkpoints **nuevos** en `CelebA 128x128 Refinamiento80/`; los originales no se sobrescriben.
- Vistas previas de entrenamiento cada 5 épocas con DDIM50 (para evaluar calidad usar el muestreador elegido).

En Hypatia:

```bash
TR=$(sbatch --parsable refinar_ep60_a_80.slurm)
echo "Refinamiento a 80: $TR"
```

Consultar:

```bash
sacct -j "$TR" --format=JobID,State,ExitCode,Elapsed,MaxRSS
python refinar_celeba128_80.py --mode status
tail -n 40 "celeba128-calidad-train-${TR}.log"
```

El trabajo guarda un checkpoint **después de cada época**. Si hace pausa segura antes de época 80, reenviar `sbatch refinar_ep60_a_80.slurm` **una vez** y reanudará.

Tras `epoch: 80` y `complete: true`:

```bash
VIS80=$(sbatch --parsable muestreo_visual_ep80.slurm)
echo "Muestras finales época80: $VIS80"
```

Esto genera vistas con las mismas condiciones iniciales de la época60 y además `comparacion_ep060_vs_ep080_ddim250.png`, etc. Comparar visualmente y **cuantitativamente**; más épocas no garantizan mejores imágenes.

### KID opcional época 60 vs 80, mismo muestreador

Por ejemplo, si ganó DDIM250 (requiere haber generado los 1024 casos época60 DDIM250 de la fase 1B):

```bash
J80=$(sbatch --parsable --export=ALL,EPOCH=80,SAMPLER=ddim250 generar_KID_samplers.slurm)
echo "Muestras época80=$J80"
```

Tras confirmar `1024/1024` época80:

```bash
export CASES="60:ddim250,80:ddim250"
KID80=$(sbatch --parsable evaluar_KID_samplers.slurm)
unset CASES
echo "Comparación KID 60 vs80=$KID80"
```

Resultado:

```bash
cat "CelebA 128x128 Mejor Calidad/KID_ep60_vs_80.csv"
cat "CelebA 128x128 Mejor Calidad/KID_ep60_vs_80_diferencia.csv"
```

Si ganó otro muestreador, sustituir `ddim250` en **ambos** puestos y en `SAMPLER`. No confundir muestras `n=1024` con los KID anteriores de `n=5000`.

## Descargar figuras

Desde **PowerShell**:

```powershell
scp 'v.rincono@hypatia.uniandes.edu.co:/hpcfs/home/fisica/v.rincono/tesis/CelebA 128x128 Mejor Calidad/previas/*.png' "$env:USERPROFILE\Downloads\"
scp 'v.rincono@hypatia.uniandes.edu.co:/hpcfs/home/fisica/v.rincono/tesis/CelebA 128x128 Mejor Calidad/*.csv' "$env:USERPROFILE\Downloads\"
```

## Precauciones científicas y técnicas

- Se estudia un checkpoint concreto; los IC no incluyen repetición de entrenamientos con otras semillas. Son IC básicos bootstrap **exploratorios**, no pruebas formales de equivalencia.
- El KID no evalúa directamente nitidez, identidad facial ni ausencia de artefactos: revisar también mosaicos.
- El estimador depende del conjunto real de prueba, del preprocesamiento, del extractor Inception y del tamaño de muestra. No comparar KID entre datasets distintos.
- DDPM1000 utiliza la varianza **posterior** `beta_tilde`, mientras DDIM250/500 usa `η=0`; los resultados son comparaciones empíricas de los samplers aquí implementados.
- No ejecutar modos GPU en el nodo de acceso. `sbatch` protege el trabajo si se cierra la conexión SSH. Después de reconectar, el shell **pierde las variables `VIS`, `TR`, etc.** y hay que utilizar el JobID que se imprimió.
