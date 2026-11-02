# Verificación del límite gaussiano — tesis CIFAR-10

## Pregunta física

¿Al aumentar el tiempo de difusión se acercan los predictores **Attention + EMA** y **ResUNet + Attention + EMA** a la predicción del score para la distribución normal estándar, tanto en valor como en derivadas?

En el DDPM VP discreto, `x_t = sqrt(alpha_bar_t)*x_0 + sqrt(1-alpha_bar_t)*epsilon`, con `epsilon ~ N(0,I)` y `x_0` normalizado a `[-1,1]`. Cuando `alpha_bar_t` es muy pequeña, `p_t` se aproxima a `N(0,I)`; por tanto `s_t(x) ≈ -x` y `J_s(x) ≈ -I`. Para `D=3072`, `||J_s||_F ≈ sqrt(3072)`. Que solo coincida la **norma** no basta para determinar la **matriz**.

## Métricas directas

`E_J(t) = ||J_s + I||_F / sqrt(D)`. Se estima sin construir el jacobiano D×D: para sondas Rademacher `v` con `E[vv^T] = I`, `E_v ||J_s^T v + v||² = ||J_s + I||_F²`. En PyTorch un VJP se calcula con `torch.autograd.grad((s * v).sum(), x)`. El estimador global promedia cuadrados sobre imágenes y sondas *antes* de tomar la raíz.

`E_score(t) = sqrt( sum ||s_theta(x_t,t)+x_t||² / sum ||x_t||² )`, medida complementaria que no requiere derivadas. También se registra `||J||_F` y traza estimados mediante las mismas sondas.

**Interpretación:** la magnitud **exacta** `E_J=0` equivale a `J_s=-I` en los puntos evaluados; un estimador Monte Carlo cercano a cero no lo certifica. Es una medida local de la derivada, no una certificación de que el score coincida globalmente con el verdadero ni de que la red genere correctamente. `E_score=0` representa concordancia en valor en los puntos evaluados. `p_999` es *aproximadamente* gaussiana, no exactamente; no se exige error exactamente cero.

## Protocolo

- Reutiliza `analizar_score_cifar10.py` y checkpoints EMA del directorio `~/tesis/modelos/`.
- `24` imágenes CIFAR-10 test seleccionadas mediante la misma secuencia determinista que el experimento anterior.
- Tiempos `500, 750, 900, 950, 975, 999` (énfasis en ruido elevado).
- `16` direcciones Rademacher por imagen y tiempo. Mismos `x_t` y mismas sondas para los dos modelos.
- `1.500` réplicas bootstrap por **imagen**, compartiendo índices entre modelos (intervalos de confianza descriptivos del 95 %, condicionados a checkpoints y sondas fijos).
- Parámetros congelados, `model.eval()`, backpropagación únicamente respecto a la imagen; precisión FP32 sin TF32.
- Sin entrenamiento, sin generación de nuevas imágenes, sin modificar experimentos previos. Datos y resultados en carpeta distinta.
- `protocolo.json` incluye hash SHA256 de las arquitecturas y pesos; se rechaza mezclar mediciones si el protocolo cambia. `mediciones_gaussianas.csv` se guarda por imagen para reanudar si SLURM interrumpe.

## Pasos

Desde **PowerShell Windows**, tras descargar ambos archivos en Descargas:

```powershell
scp "$env:USERPROFILE\Downloads\validar_limite_gaussiano_score.py" "$env:USERPROFILE\Downloads\validar_limite_gaussiano_score.slurm" "v.rincono@hypatia.uniandes.edu.co:/hpcfs/home/fisica/v.rincono/tesis/"
ssh hypatia
```

En la consola bash de **Hypatia**:

```bash
cd ~/tesis
source ~/venvs/thesis/bin/activate
python validar_limite_gaussiano_score.py --mode selftest
python validar_limite_gaussiano_score.py --mode check
sbatch validar_limite_gaussiano_score.slurm
squeue -u v.rincono
```

`--mode selftest` es una prueba matemática local (J=-I, J=-2I, J=0 y matriz conocida), sin GPU. `--mode check` no descarga datos ni calcula gradientes. **Nunca lanzar `--mode all` en el nodo login**; reservar GPU con `sbatch`.

Los logs siguen `gauss-score-JOBID.log` y `gauss-score-JOBID.err`. Se puede consultar `python validar_limite_gaussiano_score.py --mode status` sin reservar GPU, porque únicamente lee los CSV guardados.

## Archivos generados

```
~/tesis/Limite Gaussiano Score CIFAR10/
├── protocolo.json
├── mediciones_gaussianas.csv
├── resumen_limite_gaussiano.csv
├── diferencias_pareadas_gaussiano.csv
└── figuras/
    ├── error_jacobiano_vs_t.png
    ├── diferencia_error_IC95.png
    ├── error_score_gaussiano_vs_t.png
    └── norma_jacobiano_con_referencia.png
```

### Precauciones de interpretación

- Un IC bootstrap que cruza cero para la diferencia entre arquitecturas no prueba igualdad.
- Las sondas finitas aportan error Monte Carlo que no está completamente representado por el bootstrap condicional por imagen.
- No hay variación entre entrenamientos independientes.
- El error de jacobiano es local en los estados ruidosos evaluados y no una métrica directa de calidad generativa.
- La normalización del dato y el schedule afectan el límite gaussiano; conservamos el procesamiento del proyecto.
