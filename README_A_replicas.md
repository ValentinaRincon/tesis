# FASE A — Robustez entre entrenamientos independientes en CIFAR-10

**Objetivo:** entrenar desde cero dos semillas nuevas por arquitectura (cuatro entrenamientos), evaluar cada EMA con DDIM determinista de 250 pasos y 5.000 imágenes, y comparar el KID sobre las mismas 10.000 imágenes de CIFAR-10 test. Esta fase debe realizarse **antes** de la fase B.

## Advertencia sobre equivalencia con entrenamientos anteriores

Las **clases de las arquitecturas y la discretización DDPM** se importan directamente del script anterior `evaluar_kid_cifar10_5000.py`, pero el registro disponible del entrenamiento original de 200 épocas **no verifica todos sus hiperparámetros**, especialmente el coeficiente EMA y el criterio de actualización EMA. El código nuevo registra explícitamente un protocolo controlado con estas opciones:

- 200 épocas por entrenamiento, 50.000 imágenes de entrenamiento CIFAR-10.
- Adam (no AdamW), LR = 1e-4, weight_decay = 0.
- batch = 64; `num_workers=0`; orden de imágenes barajado con semilla independiente; normalización a `[-1,1]`.
- Objetivo MSE de predicción de ruido, `T=1000`, beta lineal entre 1e-4 y 0.02.
- EMA decay = **0.999**, EMA inicializada desde los pesos iniciales y actualizada **después de cada optimizer.step**.
- Sin scheduler, sin AMP ni grad clipping.
- Semillas nuevas `20261101` y `20261102`; arquitectura para cada semilla en índices 0..3.

Si el entrenamiento original usó otro `ema_decay`, scheduler, batch o regla de actualización, **modifica `DEFAULT` en `replicar_entrenamiento_cifar10.py` ANTES del primer entrenamiento**. Tras comenzar una réplica no edites sus opciones: el protocolo impide mezclar pesos. Si se desconoce el protocolo original, las nuevas ejecuciones son un **estudio de robustez bajo protocolo común**, no réplicas idénticas del entrenamiento histórico.

Cada arquitectura se entrena de manera independiente desde cero y ninguna toca `~/tesis/modelos/`.

## Instalación desde Windows PowerShell

```powershell
scp "$env:USERPROFILE\Downloads\fase_A_replicas_cifar10_Hypatia.zip" "v.rincono@hypatia.uniandes.edu.co:/hpcfs/home/fisica/v.rincono/tesis/"
ssh hypatia
```

Dentro de Hypatia:

```bash
cd ~/tesis
ls -lh fase_A_replicas_cifar10_Hypatia.zip
unzip -t fase_A_replicas_cifar10_Hypatia.zip
unzip -o fase_A_replicas_cifar10_Hypatia.zip
source ~/venvs/thesis/bin/activate
python replicar_entrenamiento_cifar10.py --mode selftest
python replicar_entrenamiento_cifar10.py --mode check
```

`check` confirma el dataset existente; **no descarga** datos en el login. Confirma que `DEFAULT` coincide con el protocolo que quieres reproducir antes de enviar los jobs.

## 1. Entrenar cuatro réplicas

```bash
JOB=$(sbatch --parsable replicas_entrenamiento.slurm)
echo "Entrenamiento array: $JOB"
squeue -j "$JOB"
```

Se usan **una GPU a la vez** (`--array=0-3%1`), con tope SLURM de 4 horas **por tarea**. Cada réplica guarda `checkpoint.pt`, `perdidas_entrenamiento.csv`, `estado.json` al final de **cada época** y `ema_final.pt` y `raw_final.pt` solo al completar las 200 épocas. Se puede reanudar exactamente el entrenamiento al término de la última época guardada, con RNG, optimizador, orden de datos y EMA incluidos en checkpoint.

Para consultar (no entrena):

```bash
python replicar_entrenamiento_cifar10.py --mode status
sacct -j "$JOB" --format=JobID,State,ExitCode,Elapsed
```

El job se **pausa de forma controlada** tras una época si se acerca a 4 horas. Si alguna réplica aparece `menos de 200/200`, vuelve a ejecutar **el mismo** `sbatch replicas_entrenamiento.slurm` cuando el array previo haya terminado; el programa retoma los checkpoints y omite las réplicas completas. Repite hasta `200/200` en los cuatro casos y verifica que existan sus cuatro `ema_final.pt`. Es normal que este experimento cueste bastante más GPU que los anteriores.

**No envíes la generación de imágenes hasta que las cuatro réplicas estén completas.**

## 2. Evaluar cada entrenamiento con DDIM-250

En login:

```bash
python evaluar_replicas_ddim250.py --mode selftest
python evaluar_replicas_ddim250.py --mode check
python evaluar_replicas_ddim250.py --mode status
```

Cuando existan los cuatro `ema_final.pt`, ejecuta:

```bash
GEN=$(sbatch --parsable replicas_generar_ddim.slurm)
echo "Generación: $GEN"
KID=$(sbatch --parsable --dependency=afterok:${GEN} replicas_kid.slurm)
echo "KID: $KID"
```

La generación se realiza en lotes reanudables de 32 imágenes usando las mismas semillas de ruido inicial en todas las configuraciones. El análisis de KID replica el procedimiento anterior de 100 subconjuntos de 500 (KID × 1000), no el estimador U global usado en los análisis bootstrap posteriores. El job KID depende de éxito total del array de generación.

Para revisar:

```bash
python evaluar_replicas_ddim250.py --mode status
squeue -u v.rincono
```

## Salidas

Carpetas separadas:

```text
~/tesis/Replicas Entrenamiento CIFAR10/
    attention_seed20261101/{checkpoint.pt,ema_final.pt,estado.json,perdidas_entrenamiento.csv,...}
    resunet_seed20261101/{...}
    attention_seed20261102/{...}
    resunet_seed20261102/{...}
    Evaluacion DDIM250/
        kid_replicas.csv
        diferencias_por_semilla.csv
        resumen_replicas.csv
        chunks/
```

Los dos pares de semillas sirven para explorar robustez, **no para justificar por sí solos una conclusión con IC del 95 % sobre la población de entrenamientos**. Idealmente deben compararse con las ejecuciones históricas solo si el protocolo original coincide. La caché separada preserva los resultados previos y la identidad de los pesos está protegida por huellas SHA256.

Para descargar CSV desde PowerShell:

```powershell
scp "v.rincono@hypatia.uniandes.edu.co:/hpcfs/home/fisica/v.rincono/tesis/Replicas Entrenamiento CIFAR10/Evaluacion DDIM250/*.csv" "$env:USERPROFILE\Downloads\"
```

## Errores comunes

- `python: command not found`: activar `source ~/venvs/thesis/bin/activate`.
- `FileNotFoundError ... ema_final.pt`: aún falta completar el entrenamiento correspondiente.
- `CIFAR10 train ... FALTA`: comprobar dónde están los cinco lotes de entrenamiento del dataset original (no descargar en nodo login).
- Salida SLURM fallida: `tail -n 30 rep-train-<JOB>_<TASK>.err`.
- **No usar `--mode train`, `--mode sample` ni `--mode kid` en el nodo login.**
