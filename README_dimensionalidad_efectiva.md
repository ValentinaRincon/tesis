# Dimensionalidad efectiva de CIFAR-10 — CPU Hypatia

## Objetivo

Calcular el espectro de la **covarianza centrada de CIFAR-10 test** (10.000 imágenes × 3.072 canales/píxeles) y estudiar cómo cambia la dimensión efectiva **de la covarianza esperada** durante el *forward DDPM*. No se necesita PyTorch, checkpoints ni GPU. No detiene ni modifica el job array de entrenamiento A (`785091`).

### Fórmulas

Las imágenes originales se normalizan a `[-1,1]`, como en la tesis. Se forma

\[
C_0 = \frac{1}{N-1}\sum_i (x_i-\overline x)(x_i-\overline x)^\top.
\]

Sea el forward DDPM \(x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon\), con \(\epsilon\sim \mathcal N(0,I)\). Si entendemos la distribución empírica de CIFAR-10 como distribución de referencia y consideramos el ruido **en esperanza**, la covarianza teórica es

\[
C_t=\bar\alpha_t C_0+(1-\bar\alpha_t)I.
\]

Por eso, basta diagonalizar **una vez** \(C_0\): los autovalores evolucionan como

\[
\lambda_k(t)=\bar\alpha_t\lambda_k(0)+1-\bar\alpha_t.
\]

Calculamos la dimensión de participación

\[
d_{\rm PR}(t)=\frac{[\sum_k\lambda_k(t)]^2}{\sum_k\lambda_k(t)^2},
\]

el rango entrópico \(\exp[-\sum_k p_k\log p_k]\), y cuántas componentes explican 90 %, 95 % y 99 % de la varianza. La dimensión de referencia isotrópica es \(3072\).

**Interpretación importante:** el análisis describe únicamente el **espectro de segundo orden (covarianza)**, **no** la dimensión intrínseca de la variedad de imágenes, ni que las imágenes originales tengan únicamente esa cantidad de grados de libertad. Las curvas para \(t>0\) son **predicciones analíticas del forward con ruido gaussiano en esperanza**, no mediciones de nuevas imágenes ruidosas simuladas. Evitamos así sesgos fuertes de covarianza muestral en alta dimensión al usar un número finito de realizaciones del ruido. No hay intervalos de confianza ni pruebas de significación en esta versión.

## 1. Instalar desde Windows PowerShell

Descarga `dimension_efectiva_cifar10_CPU_Hypatia.zip` en tu carpeta Descargas y ejecuta:

```powershell
scp "$env:USERPROFILE\Downloads\dimension_efectiva_cifar10_CPU_Hypatia.zip" "v.rincono@hypatia.uniandes.edu.co:/hpcfs/home/fisica/v.rincono/tesis/"
ssh hypatia
```

Dentro de Hypatia:

```bash
cd ~/tesis
unzip -t dimension_efectiva_cifar10_CPU_Hypatia.zip
unzip -o dimension_efectiva_cifar10_CPU_Hypatia.zip
source ~/venvs/thesis/bin/activate
python dimensionalidad_efectiva_cifar10.py --mode selftest
python dimensionalidad_efectiva_cifar10.py --mode check
```

`check` verifica que ya esté `~/tesis/data/cifar-10-batches-py/test_batch`; **no descarga** nada ni carga toda CIFAR. Si falta, comprueba dónde se descargó en los experimentos anteriores y proporciona `--data-file /ruta/al/test_batch`; para SLURM modifica el último comando del archivo `.slurm` con la ruta correspondiente.

## 2. Enviar a la partición express (solo CPU)

```bash
CPU_JOB=$(sbatch --parsable dimensionalidad_efectiva_cifar10.slurm)
echo "Trabajo CPU: $CPU_JOB"
squeue -j "$CPU_JOB"
```

El trabajo solicita 4 CPU, 12 GB de RAM y 2 horas como límite **máximo** (no como duración estimada). `express` podría tener restricciones institucionales adicionales. Si SLURM rechaza la solicitud, no alteres la fase A: informa el mensaje exacto antes de cambiar la partición. El cálculo más exigente es la diagonalización simétrica de una matriz de 3072 × 3072. No lo ejecutes en el nodo de acceso.

## 3. Estado y logs

```bash
python dimensionalidad_efectiva_cifar10.py --mode status
squeue -j "$CPU_JOB"
sacct -j "$CPU_JOB" --format=JobID,State,ExitCode,Elapsed
ls -t dim-eff-*.log | head
```

Consulta el registro (sustituye `JOBID`):

```bash
tail -n 25 dim-eff-JOBID.log
```

`--mode status` solo mira archivos; no procesa imágenes ni usa grandes cantidades de RAM.

## 4. Resultados

Carpeta: `~/tesis/Dimensionalidad Efectiva CIFAR10/`

- `resumen_dimensionalidad.csv`: \(d_{\rm PR}\), rango entrópico, componentes necesarias y programación de ruido.
- `espectro_covarianza.csv`: 3.072 autovalores y espectros teóricos en distintos tiempos.
- `protocolo.json`: origen y hash de los datos, normalización y fórmulas.
- `figuras/dimension_efectiva_vs_t.png`
- `figuras/componentes_95_vs_t.png`
- `figuras/espectros_covarianza.png`
- `figuras/varianza_acumulada.png`

Desde **PowerShell de Windows**, para traer los resultados:

```powershell
scp -r "v.rincono@hypatia.uniandes.edu.co:/hpcfs/home/fisica/v.rincono/tesis/Dimensionalidad Efectiva CIFAR10" "$env:USERPROFILE\Downloads\"
```

Comparte `resumen_dimensionalidad.csv` y las cuatro figuras para una interpretación conjunta con la tesis.
