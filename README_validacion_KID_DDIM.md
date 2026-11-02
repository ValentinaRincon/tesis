# Intervalos de confianza pareados para DDIM — CIFAR-10 / Hypatia

## Objetivo

Evaluar con intervalos bootstrap básicos del 95 % el efecto de usar DDIM determinista (η=0) con **1000, 500, 250, 100 o 50 pasos** para los dos checkpoints EMA ya entrenados. **No se vuelve a entrenar ni a generar imágenes**: el programa extrae características Inception de los 50.000 ejemplos ya almacenados y de las 10.000 imágenes CIFAR-10 test.

## Qué archivos se necesitan en Hypatia

Este paquete incluye:

- `validar_kid_ddim_bootstrap.py` — validación, extracción/cache y estadísticas;
- `validar_kid_ddim_bootstrap.slurm` — ejecución en 1 GPU;
- `README_validacion_KID_DDIM.md` — esta guía.

En `~/tesis` deben seguir existiendo los archivos de tu estudio original:

- `comparar_ddim_cifar10.py` y `evaluar_kid_cifar10_5000.py`;
- `DDIM CIFAR10 EMA/protocolo.json`;
- `DDIM CIFAR10 EMA/kid_ddim.csv`;
- `DDIM CIFAR10 EMA/chunks/ddim_*/batch_*.pt` (157 lotes por cada una de las 10 condiciones, 5000 imágenes cada una);
- `data/cifar-10-batches-py/test_batch`;
- entorno virtual `~/venvs/thesis/` con torch, torchvision, torchmetrics y torch-fidelity.

## 1. Subir desde Windows (PowerShell)

Descarga `validacion_kid_ddim_Hypatia.zip` a `Downloads` y ejecuta:

```powershell
scp "$env:USERPROFILE\Downloads\validacion_kid_ddim_Hypatia.zip" "v.rincono@hypatia.uniandes.edu.co:/hpcfs/home/fisica/v.rincono/tesis/"
ssh hypatia
```

## 2. Extraer y comprobar desde la terminal de Hypatia

```bash
cd ~/tesis
ls -lh validacion_kid_ddim_Hypatia.zip
unzip -t validacion_kid_ddim_Hypatia.zip
unzip -o validacion_kid_ddim_Hypatia.zip
source ~/venvs/thesis/bin/activate
python validar_kid_ddim_bootstrap.py --mode selftest
python validar_kid_ddim_bootstrap.py --mode check
```

`selftest` prueba matemáticamente el estimador U, el remuestreo con observaciones repetidas, el término cruzado y el cálculo del intervalo básico. `check` hace una inspección de rutas **sin cargar tensores**, apropiada en el nodo de acceso de Hypatia. Ante mensajes de archivos faltantes, comprobar el nombre de la carpeta original antes de enviar el job.

## 3. Enviar el trabajo a SLURM

```bash
JOB=$(sbatch --parsable validar_kid_ddim_bootstrap.slurm)
echo "Job DDIM-IC: $JOB"
squeue -j "$JOB"
```

Para revisar la ejecución:

```bash
sacct -j "$JOB" --format=JobID,State,ExitCode,Elapsed
tail -n 30 "ddim-ic-${JOB}.log"
cat "ddim-ic-${JOB}.err"
```

El programa guarda un archivo `.pt` por configuración en una caché de características; si se interrumpe y se reenvía el job, reutilizará los archivos completados, siempre que coincidan rutas, huellas del inventario, protocolo y versiones del extractor.

Para consultar el progreso de extracción, una vez iniciado:

```bash
python validar_kid_ddim_bootstrap.py --mode status
```

No ejecutes `--mode run`, `--mode extract` o `--mode bootstrap` directamente desde el nodo de acceso: necesitan GPU.

## 4. Resultados

En `~/tesis/Incertidumbre KID DDIM CIFAR10/`:

- `kid_ddim_intervalos.csv`: KID completo sobre las 5.000/10.000 imágenes, IC 95 % y referencia histórica del KID de subconjuntos;
- `kid_ddim_diferencias_pareadas.csv`: **13** diferencias: 4×2 variantes de pasos vs. 1000 y cinco comparaciones Attention − ResUNet a igual número de pasos;
- `ddim_diferencias_pasos_IC95.png`: aumento de KID al reducir pasos, por modelo;
- `ddim_diferencias_arquitecturas_IC95.png`: diferencia de KID entre arquitecturas a iguales pasos;
- `bootstrap_distribuciones_ddim.npz`: réplicas completas, para análisis posterior;
- `config_analisis.json`: metodología, versiones y huella del protocolo;
- `features/*.pt`: características Inception en caché (no es necesario descargarlas).

Para descargar a Windows:

```powershell
scp "v.rincono@hypatia.uniandes.edu.co:/hpcfs/home/fisica/v.rincono/tesis/Incertidumbre KID DDIM CIFAR10/*.csv" "$env:USERPROFILE\Downloads\"
scp "v.rincono@hypatia.uniandes.edu.co:/hpcfs/home/fisica/v.rincono/tesis/Incertidumbre KID DDIM CIFAR10/*.png" "$env:USERPROFILE\Downloads\"
```

## Método estadístico

El KID anterior es una media sobre 100 subconjuntos aleatorios de 500. Este nuevo análisis calcula **el estimador U de MMD² en el conjunto completo**, con kernel polinomial `((u·v)/2048+1)^3`. Los valores puntuales pueden variar por utilizar estimadores muestrales distintos.

En cada réplica bootstrap:

1. Remuestrear con reemplazo 5000 índices y aplicar **los mismos índices** a las diez condiciones generadas, aprovechando sus semillas iniciales pareadas.
2. Remuestrear con reemplazo 10000 imágenes reales y utilizar exactamente los mismos índices reales en las diez condiciones.
3. Calcular KID en cada muestra bootstrap y las **diferencias dentro de la misma réplica**.
4. Formar un intervalo básico nominal 95 % mediante `2*estimacion − [cuantil97.5%, cuantil2.5%]`.

Los índices repetidos **sí se contabilizan** para el estadístico U de la muestra bootstrap.

### Interpretación y límites

- La pregunta principal es si las diferencias KID de 250/500 frente a 1000 pasos son detectables con las imágenes y modelos disponibles. El análisis no prueba equivalencia por un IC que incluya cero.
- Los intervalos son **exploratorios y no ajustados por comparaciones múltiples**. La inferencia se condiciona a los dos checkpoints entrenados; no incluye variación entre entrenamientos independientes.
- Estas incertidumbres capturan la variación al remuestrear las imágenes observadas, manteniendo fijas las semillas de generación y los modelos. No representan variación a través de distintos datasets, entrenamientos ni hardware.
- La velocidad procede del registro previo de inferencia en GPU; no se estiman intervalos de confianza sobre tiempos de ejecución.
- La comparación histórica con DDPM 1000 pasos **no es un contraste pareado incluido**: no confundir diferencias entre algoritmos y diferencias de cantidad de pasos DDIM.
- El bootstrap no paramétrico de MMD² puede tener comportamiento no regular cerca de la hipótesis nula de igualdad exacta de distribuciones; las inferencias son aproximadas.
