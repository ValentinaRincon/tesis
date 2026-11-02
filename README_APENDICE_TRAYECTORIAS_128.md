# Apéndice: trayectorias de generación CelebA 128×128, EMA época 60

## Qué hace

Utiliza el **modelo ya entrenado** `CelebA 128x128 Refinamiento/checkpoint.pt` y
reutiliza la arquitectura, normalización y ecuaciones de `calidad_celeba128.py`.
No modifica ningún checkpoint ni los resultados de KID previos.

Para cada muestreador (DDIM 250, DDIM 500, DDPM 1000) exporta:

1. `trayectoria_<metodo>_xt.png` — Estados **ruidosos reales** `x_t` del proceso de generación.
2. `trayectoria_<metodo>_x0_estimado.png` — Estimaciones de imagen limpia
   `xhat_0(x_t,t) = (x_t - sqrt(1-alphabar_t)*epsilon_theta(x_t,t))/sqrt(alphabar_t)`.
   **Estas estimaciones no son estados de la trayectoria física**; son diagnósticos de qué
   imagen predice el modelo desde cada tiempo. En tiempos ruidosos pueden saturarse al mostrarlas.
3. `trayectoria_<metodo>_x0_detalle.png` — Vista ampliada en la región final de la difusión (`t<=500`).
4. `protocolo_<metodo>.json` — Parámetros, tiempos discretos realmente capturados y huellas SHA256.

La figura incluye **cuatro caras (filas)** y **13 capturas temporales más el resultado final (columnas)**.
Los tiempos objetivo de difusión son `t=999,900,800,700,600,500,400,300,200,150,100,50,10`.
Para DDIM se elige el índice realmente usado más cercano y la figura etiqueta ese `t` efectivo.
Los **250/500/1000** son pasos del algoritmo; `t` **no es el número de paso del algoritmo**.

Se utiliza el mismo `x_T` inicial (`seed=20261120`) para los tres algoritmos.
DDPM añade ruido adicional en cada paso y, por tanto, no tiene por qué generar
la misma cara que DDIM a partir del mismo `x_T`.

## Windows PowerShell — subir

Descarga `apendice_trayectorias_CelebA128_Hypatia.zip` en Descargas:

```powershell
scp "$env:USERPROFILE\Downloads\apendice_trayectorias_CelebA128_Hypatia.zip" "v.rincono@hypatia.uniandes.edu.co:/hpcfs/home/fisica/v.rincono/tesis/"
ssh hypatia
```

## Hypatia — comprobar y lanzar

```bash
cd ~/tesis
unzip -o apendice_trayectorias_CelebA128_Hypatia.zip
source ~/venvs/thesis/bin/activate
python apendice_trayectorias_celeba128.py --mode selftest
python apendice_trayectorias_celeba128.py --mode check
JOB=$(sbatch --parsable apendice_trayectorias_celeba128.slurm)
echo "Job trayectorias: $JOB"
squeue -j "$JOB"
```

Cuando desaparezca de `squeue`, usa el número de job (p. ej. reemplaza `788000`):

```bash
sacct -j 788000 --format=JobID,State,ExitCode,Elapsed,MaxRSS
tail -n 50 celeba128-apendice-788000.log
python apendice_trayectorias_celeba128.py --mode status
```

El archivo final `~/tesis/CelebA128_apendice_trayectorias_ep60.zip` agrupa
las 9 figuras PNG y los metadatos. Se genera solamente cuando los tres
muestreadores estén completos.

## Windows PowerShell — descargar figuras

```powershell
scp 'v.rincono@hypatia.uniandes.edu.co:/hpcfs/home/fisica/v.rincono/tesis/CelebA128_apendice_trayectorias_ep60.zip' "$env:USERPROFILE\Downloads\"
```

## Sugerencia de redacción para la tesis

**Pie de figura (`x_t`):** «Trayectorias inversas de generación con el modelo
EMA de CelebA 128×128 en la época 60. Cada fila corresponde a una condición
inicial de ruido; las columnas muestran estados a distintos tiempos de difusión,
con la imagen final a la derecha. Los métodos DDIM y DDPM comparten el estado
inicial pero difieren en su discretización y, para DDPM, en el ruido añadido
por paso». 

**Pie de figura (`xhat_0`):** «Evolución de las estimaciones de imagen limpia
`\hat{x}_0(x_t,t)` durante la dinámica inversa. Cada panel muestra predicciones
instantáneas de la red y no constituye una trayectoria física de estados
`x_t`. Las predicciones se recortan a `[-1,1]` únicamente para visualización».

En LaTeX, si las 14 columnas se ven demasiado pequeñas, usa las figuras
`*_x0_detalle.png` o colócalas en una página horizontal con `pdflscape`.

## Notas metodológicas

- El modelo evalúa en `torch.autocast('cuda', float16)`, igual que la comparación previa.
- En la propagación inversa **no se recorta** `x0` ni `x_t`: el recorte se aplica
  solo al convertir cada estado a imagen.
- Las estimaciones `x0` en `t≈999` pueden lucir muy saturadas: no son imágenes
  generadas completas ni deben evaluarse como calidad final.
- El checkpoint y la configuración de ruido son verificables en JSON.
- No son métricas KID ni evidencia de superioridad estadística de un sampler.
