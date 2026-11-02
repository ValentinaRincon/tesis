# FASE B — Trayectorias deterministas DDIM en CIFAR-10

**Orden de trabajo:** ejecutar **después** de la fase A y de revisar `kid_replicas.csv`. La fase B utiliza los **checkpoints EMA originales** (no los cuatro pesos nuevos), porque su propósito es profundizar las trayectorias de los modelos que ya comparamos en score/KID/ablación.

## Preguntas

1. ¿Cómo se separan en el tiempo las trayectorias DDIM de Attention + EMA y ResUNet + Attention + EMA si se parte del mismo `x_T`?
2. ¿Qué distancia terminal aparece al comparar DDIM de 1.000, 250 y 50 pasos para la **misma arquitectura y la misma condición inicial**?

Usamos DDIM `η=0`, 16 muestras iniciales compartidas, pasos `[1000,250,50]`, betas y arquitectura importados del código ya evaluado. El análisis es exploratorio sobre 16 condiciones iniciales.

**Precisión metodológica:** los grids temporales de 1.000, 250 y 50 pasos son diferentes. Los estados intermedios entre distintas discretizaciones no coinciden exactamente en tiempo; por eso la comparación entre pasos se realiza fundamentalmente sobre el estado terminal `x0`. La distancia entre arquitecturas sí se calcula para el **mismo t dentro de cada discretización**. La similitud de predicciones ε se refiere a las predicciones sobre **estados distintos** de las dos trayectorias y no debe confundirse con semejanza de campos evaluados en el mismo punto.

## Subir

PowerShell:

```powershell
scp "$env:USERPROFILE\Downloads\fase_B_trayectorias_ddim_Hypatia.zip" "v.rincono@hypatia.uniandes.edu.co:/hpcfs/home/fisica/v.rincono/tesis/"
ssh hypatia
```

Hypatia:

```bash
cd ~/tesis
unzip -t fase_B_trayectorias_ddim_Hypatia.zip
unzip -o fase_B_trayectorias_ddim_Hypatia.zip
source ~/venvs/thesis/bin/activate
python trayectorias_ddim_cifar10.py --mode selftest
python trayectorias_ddim_cifar10.py --mode check
JOB=$(sbatch --parsable trayectorias_ddim.slurm)
echo "Trayectorias: $JOB"
squeue -j "$JOB"
```

Después:

```bash
sacct -j "$JOB" --format=JobID,State,ExitCode,Elapsed
python trayectorias_ddim_cifar10.py --mode status
```

## Salidas

```text
~/tesis/Trayectorias DDIM CIFAR10/
    protocolo.json
    trayectorias_todas.csv
    snapshots_aproximados.csv
    distancias_finales.csv
    estados_finales.pt
    separacion_trayectorias_vs_t.png
    comparacion_terminal_pareada.png
```

El protocolo bloquea reutilizar la carpeta si los pesos originales o scripts cambian. El trabajo se ejecuta en GPU SLURM, no en el nodo login.

## Descargar a Windows

```powershell
scp "v.rincono@hypatia.uniandes.edu.co:/hpcfs/home/fisica/v.rincono/tesis/Trayectorias DDIM CIFAR10/*.csv" "$env:USERPROFILE\Downloads\"
scp "v.rincono@hypatia.uniandes.edu.co:/hpcfs/home/fisica/v.rincono/tesis/Trayectorias DDIM CIFAR10/*.png" "$env:USERPROFILE\Downloads\"
```
