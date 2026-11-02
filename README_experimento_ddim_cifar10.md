# Comparación DDIM por número de pasos — CIFAR-10 / Hypatia

## Pregunta

¿Cómo afecta el número de evaluaciones de la red durante el muestreo **DDIM determinista (η=0)** a la calidad generativa (KID) y al tiempo de inferencia de Attention + EMA y ResUNet + Attention + EMA?

**Importante:** nuestro control histórico DDPM de 1.000 pasos es un **muestreador ancestral estocástico**. El resultado DDIM con 1.000 pasos NO es el mismo algoritmo. La comparación entre DDIM 1.000 y DDIM 50 aísla mejor el efecto del número de pasos; la comparación de DDPM 1.000 y DDIM 1.000 modifica el algoritmo de muestreo. Se incluyen los puntos del DDPM histórico solo como **referencia descriptiva** (sin tiempos de ejecución históricos).

## Métodos

* Red y checkpoints EMA originales, cargados desde `evaluar_kid_cifar10_5000.py`.
* Programación lineal de β, 1000 niveles de ruido, β inicial 1e-4 y final 0.02.
* Para K∈{1000, 500, 250, 100, 50}, seleccionar K tiempos enteros espaciados uniformemente entre t=0 y t=999, incluyendo ambos extremos.
* En cada paso del DDIM determinista (η=0), calcular:

  `x0_hat = (x_t - sqrt(1-ab_t)*eps_theta(x_t,t))/sqrt(ab_t)`

  `x_previous = sqrt(ab_previous)*x0_hat + sqrt(1-ab_previous)*eps_theta(x_t,t)`

  con `ab_previous=1` en el último paso, para obtener la estimación `x0_hat`. Sin clipping intermedio; solo clipping [-1,1] al convertir a uint8.
* **Mismo ruido inicial** para una misma posición de muestra en los 10 experimentos, semilla por lote `20261020 + 1009*indice` como en el estudio DDPM previo.
* 5000 muestras por condición; 10 condiciones → 50.000 imágenes. Cada lote de 32 se guarda separadamente y se puede reanudar. No se requiere entrenar.
* KID usando 10.000 imágenes reales de CIFAR10 test, Inception-2048, 100 subconjuntos aleatorios de 500, semilla 42, idéntico al cálculo previo. La desviación KID es **entre subconjuntos, NO intervalo de confianza**.
* Tiempo: se registra cada lote con sincronización GPU, exclusivamente inferencia y conversión a CPU de imágenes, sin carga del checkpoint ni cálculo KID. La primera iteración puede incluir calentamiento. Las diferencias de hardware/carga afectan la velocidad.

## Archivos necesarios en `~/tesis`

* `comparar_ddim_cifar10.py` (este nuevo script)
* `generar_ddim_cifar10.slurm`
* `calcular_kid_ddim_cifar10.slurm`
* `evaluar_kid_cifar10_5000.py` (script ORIGINAL)
* `modelos/cifar10_attention_ema_model.pt`
* `modelos/cifar10_resunet_attention_ema_model.pt`
* `data/cifar-10-batches-py/test_batch`
* entorno `~/venvs/thesis/` con PyTorch, torchmetrics, torch-fidelity, torchvision, matplotlib.

## Pasos

**PowerShell de Windows** (una vez descargados los tres archivos):

```powershell
scp "$env:USERPROFILE\Downloads\comparar_ddim_cifar10.py" "$env:USERPROFILE\Downloads\generar_ddim_cifar10.slurm" "$env:USERPROFILE\Downloads\calcular_kid_ddim_cifar10.slurm" "v.rincono@hypatia.uniandes.edu.co:/hpcfs/home/fisica/v.rincono/tesis/"
ssh hypatia
```

**Shell de Hypatia**:

```bash
cd ~/tesis
source ~/venvs/thesis/bin/activate
python comparar_ddim_cifar10.py --mode selftest
python comparar_ddim_cifar10.py --mode check
```

Si ambas pruebas pasan, enviar las tareas. **Una sola GPU simultánea** para no saturar el recurso:

```bash
GEN_JOB=$(sbatch --parsable generar_ddim_cifar10.slurm)
echo "Generación: $GEN_JOB"
KID_JOB=$(sbatch --parsable --dependency=afterok:${GEN_JOB} calcular_kid_ddim_cifar10.slurm)
echo "KID: $KID_JOB"
squeue -u v.rincono
```

**NOTA SLURM:** el array tiene 10 tareas en serie (`0-9%1`) por defecto; las condiciones con 1000 pasos pueden ser costosas. Si alguna tarea agota las 4 horas, revisa el error/log y reenvía solo esos índices con un tiempo de pared autorizado por el clúster. El código reconoce lotes válidos y retoma. Si falla cualquier tarea, la dependencia `afterok` no dispara el KID; se debe volver a solicitar tras completar TODAS las tareas.

Estado ligero en nodo de acceso (no abre archivos tensores):

```bash
python comparar_ddim_cifar10.py --mode status
sacct -j "$GEN_JOB,$KID_JOB" --format=JobID,State,ExitCode,Elapsed
```

Para un log específico (ejemplo con índice 0):

```bash
tail -n 30 "ddim-gen-${GEN_JOB}_0.log"
```

## Resultados

Carpeta separada `~/tesis/DDIM CIFAR10 EMA/`:

* `protocolo.json` — hashes de checkpoints y script original, semillas, número de pasos y secuencias t.
* `chunks/ddim_*/batch_*.pt` — imágenes uint8 por lote.
* `tiempos/ddim_*/batch_*.json` — segundos de inferencia por lote.
* `kid_ddim.csv` — KID y métricas de tiempo por condición.
* `kid_vs_pasos.png` — gráfico KID vs pasos, opcionalmente con control DDPM histórico.
* `velocidad_vs_pasos.png` — imágenes/s medido por lote (no tiempo end-to-end).
* `muestras_pareadas_ddim.png` — primeras cuatro semillas compartidas.

## Límites y cautelas

* Los KID con 100 subconjuntos NO son equivalentes a intervalos de confianza. Si el patrón es interesante, podemos hacer bootstrap pareado reutilizando las muestras.
* Solo hay un checkpoint por arquitectura: no se cuantifica variabilidad por entrenamiento.
* A 50 pasos no se espera automáticamente mejor ni peor resultado: los datos deben decidirlo.
* Los errores de la aproximación temporal dependen del muestreador; eta=0 describe un procedimiento determinista concreto, no necesariamente la solución numérica única de la reverse SDE.
* El error en la distribución puede depender del schedule, selección de tiempos y calidad de los predictores. No atribuirlo exclusivamente a la discretización.
