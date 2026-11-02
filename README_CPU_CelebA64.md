# CelebA 64×64 — validación estadística del KID en CPU

## Qué hace

Ejecuta **exclusivamente en CPU** con las características Inception ya creadas por `evaluar_celeba64.py`:

1. KID U completo para **5.000 imágenes generadas vs 10.000 fotografías oficiales de test**.
2. Control real-versus-real: **5.000 fotografías extraídas del subconjunto de entrenamiento realmente utilizado** frente a las **mismas 10.000 fotografías de test**. Se utiliza un muestreo aleatorio reproducible, no todas las 50.000 imágenes a la vez.
3. **400 réplicas bootstrap**, con remuestreo independiente dentro de cada conjunto y referencia test compartida entre los dos contrastes. Se obtienen IC percentiles marginales del 95 % para cada KID y la diferencia.

La comparación *real-versus-real* da una escala de referencia para interpretar el KID, pero **no demuestra ausencia de memorización** ni constituye un umbral universal de KID. Los intervalos son condicionales a los checkpoints, características y muestras guardadas. El bootstrap estándar de la U-estadística puede no tener calibración correcta si el KID poblacional es cero (estadística degenerada); evita interpretarlo como prueba de significancia en ese caso. No se han repetido entrenamientos del modelo CelebA64.

No genera imágenes, no calcula nuevas features y no toca los checkpoints. El resultado existente de KID (`kid_celeba64.csv`) se utiliza para un control numérico de consistencia.

## Entradas necesarias en `~/tesis`

```text
Evaluacion CelebA64 KID Vecinos/
├── protocolo.json
├── kid_celeba64.csv
└── features/
    ├── fake.pt
    ├── test.pt
    └── train.pt
```

Son las cachés creadas por la evaluación CelebA64 anterior. El programa revisa la cantidad de features y las huellas del checkpoint y del split.

## Paso 1 — Subir desde PowerShell (Windows)

Descarga `validacion_KID_CelebA64_CPU_Hypatia.zip` y ejecuta **en PowerShell de Windows**, no dentro de SSH:

```powershell
scp "$env:USERPROFILE\Downloads\validacion_KID_CelebA64_CPU_Hypatia.zip" "v.rincono@hypatia.uniandes.edu.co:/hpcfs/home/fisica/v.rincono/tesis/"
ssh hypatia
```

## Paso 2 — Comprobar en login (ligero)

```bash
cd ~/tesis
unzip -t validacion_KID_CelebA64_CPU_Hypatia.zip
unzip -o validacion_KID_CelebA64_CPU_Hypatia.zip
source ~/venvs/thesis/bin/activate
python validar_kid_celeba64_cpu.py --mode selftest
python validar_kid_celeba64_cpu.py --mode check
```

El modo `check` **no importa PyTorch ni carga decenas de miles de features**, para evitar `Killed` por memoria en el nodo de acceso.

## Paso 3 — Enviar job CPU sin afectar la fase B en GPU

```bash
CPU_JOB=$(sbatch --parsable validar_kid_celeba64_cpu.slurm)
echo "Job de KID CelebA64 en CPU: $CPU_JOB"
squeue -j "$CPU_JOB"
```

El SLURM pide `express`, **4 CPU, 20 GB de RAM, 0 GPU**, máximo 3 horas. Si el clúster no permite esa combinación por políticas de QoS, consulta el error y adapta `--partition` o `--time` (no ejecutes `--mode run` en el login). `express` era la partición CPU con mayor disponibilidad en la última consulta, pero eso puede cambiar.

Ver log:

```bash
tail -n 35 "celeba64-ic-cpu-${CPU_JOB}.log"
```

Ver estado de resultados sin cargar tensores:

```bash
python validar_kid_celeba64_cpu.py --mode status
```

Confirmar fin:

```bash
sacct -j "$CPU_JOB" --format=JobID,State,ExitCode,Elapsed,MaxRSS
```

## Salidas

Carpeta `~/tesis/Incertidumbre KID CelebA64 CPU/`:

- `resumen_kid_celeba64_cpu.csv`: KID original y referencia real-versus-real, diferencia, IC percentil aproximado de cada uno.
- `bootstrap_replicas_kid_celeba64_cpu.csv`: 400 réplicas para auditoría.
- `kid_celeba64_IC95.png`: comparación KID generado vs referencia real.
- `diferencia_generado_vs_real_IC95.png`: diferencia en KID con IC.
- `indices_train_control.npy`: índices de las 5.000 fotografías reales de entrenamiento seleccionadas.
- `protocolo.json`: configuración y huellas para reproducibilidad.

Para descargar los CSV desde PowerShell de Windows:

```powershell
scp "v.rincono@hypatia.uniandes.edu.co:/hpcfs/home/fisica/v.rincono/tesis/Incertidumbre KID CelebA64 CPU/*.csv" "$env:USERPROFILE\Downloads\"
scp "v.rincono@hypatia.uniandes.edu.co:/hpcfs/home/fisica/v.rincono/tesis/Incertidumbre KID CelebA64 CPU/*.png" "$env:USERPROFILE\Downloads\"
```

**No interpretes el IC como incertidumbre entre entrenamientos independientes** ni como validación formal de ausencia de memorización.
