#!/bin/bash
#SBATCH --job-name=CELEBA_DOWNLOAD
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=05:00:00
#SBATCH --output=celeba-download-%j.log
#SBATCH --error=celeba-download-%j.err
set -euo pipefail
cd ~/tesis
source ~/venvs/thesis/bin/activate
python -u - <<'PY'
from pathlib import Path
from zipfile import is_zipfile
try:
    from huggingface_hub import hf_hub_download
except ImportError as exc:
    raise RuntimeError('Falta huggingface_hub en el venv. Instalar con python -m pip install huggingface_hub o usar descarga manual.') from exc
root=Path.home()/'tesis'/'data'/'celeba'
root.mkdir(parents=True,exist_ok=True)
for name in ['list_eval_partition.txt', 'img_align_celeba.zip']:
    path=root/name
    if path.exists() and path.stat().st_size>1_000_000 and name.endswith('.zip') and is_zipfile(path):
        print('Ya existe ZIP valido:',path,flush=True);continue
    if path.exists() and name.endswith('.txt') and path.stat().st_size>100_000:
        print('Ya existe particion:',path,flush=True);continue
    print('Descargando:',name,flush=True)
    saved=hf_hub_download(repo_id='Yuehao/celeba',repo_type='dataset',filename=name,local_dir=str(root))
    print('Descargado:',saved,flush=True)
print('Descarga finalizada; ejecutar job de preparacion/extraccion.',flush=True)
PY
