#!/usr/bin/env python3
"""Bootstrap PAREADO de diferencias de KID en la ablación temporal de CIFAR-10.

Diseñado para ~/tesis en Hypatia. No entrena ni vuelve a generar imágenes.
Usa las mismas características InceptionV3 de torchmetrics/torch-fidelity
que la evaluación KID previa (feature=2048, normalize=False).

Motivación estadística:
  * El KID previo promedia 100 subconjuntos de tamaño 500. Su std entre
    subconjuntos NO es IC para diferencias entre modelos.
  * Aquí calculamos KID completo (estimador U no sesgado de MMD^2) usando
    las 5000 imágenes generadas y las 10000 reales.
  * Bootstrap no paramétrico de imágenes: cada réplica remuestrea CON
    reemplazo los 5000 índices generativos CON LOS MISMOS ÍNDICES entre
    las ocho variantes (semillas pareadas) y los 10000 índices reales.
  * Recalcula el KID U en la muestra bootstrap y emplea IC básico centrado
    para diferencias pareadas; NO concluye equivalencia o causalidad.

Modos:
  --mode check     inventario ligero sin GPU ni torch.load repetidos
  --mode run       extrae features (cache) y calcula IC con GPU SLURM
  --mode extract   solo extrae features (cache) con GPU SLURM
  --mode bootstrap solo estadística desde cache con GPU SLURM
  --mode selftest  prueba del estimador U y de las diferencias (CPU)

En GPU:
  python -u validar_kid_ablacion_bootstrap.py --mode run --reps 400
"""

import argparse
import csv
import hashlib
import json
import math
import os
import platform
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

import numpy as np
import torch

import ablacion_temporal_cifar10 as a

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'Incertidumbre KID CIFAR10'
FEATURES_DIR = OUT / 'features'
N_REAL = 10000
N_FAKE = a.N_SAMPLES
KID_SCALE = 1000.0
SEED = 20261025
FEATURE_DIM = 2048


def pkg_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return 'not-installed'


def list_sources():
    """Enumerar referencias existentes SIN cargar las imágenes en el login."""
    real_path = a.DATA_DIR / 'cifar-10-batches-py' / 'test_batch'
    if not real_path.is_file():
        raise FileNotFoundError(real_path)
    expected = math.ceil(a.N_SAMPLES / a.BATCH_SIZE)
    sources = {'real': [real_path]}
    for v in a.all_variants():
        paths = [a.source_chunk(v, i) for i in range(expected)]
        missing = [p for p in paths if not p.is_file()]
        if missing:
            raise FileNotFoundError(f'{v["key"]}: faltan {len(missing)} archivos; primero: {missing[0]}')
        sources[v['key']] = paths
    return sources


def fingerprint(paths):
    """Huella rápida de inventario; detecta reemplazos por tamaño/mtime."""
    sha = hashlib.sha256()
    for path in paths:
        st = path.stat()
        sha.update(f'{path}|{st.st_size}|{st.st_mtime_ns}\n'.encode())
    return sha.hexdigest()


def metadata_for(key, paths):
    protocol = a.OUT / 'protocol.json'
    if not protocol.is_file():
        raise FileNotFoundError(protocol)
    return {
        'key': key,
        'file_fingerprint': fingerprint(paths),
        'protocol_sha256': hashlib.sha256(protocol.read_bytes()).hexdigest(),
        'torchmetrics_version': pkg_version('torchmetrics'),
        'torch_fidelity_version': pkg_version('torch-fidelity'),
        'feature_dim': FEATURE_DIM,
        'image_dtype': 'uint8',
        'normalize': False,
    }


def check():
    print('Chequeo SIN GPU: IC pareado para KID, ocho configuraciones')
    print('Destino:', OUT)
    print('Modelo referencia B_all, y comparaciones frente a A_all.')
    print('Num. imágenes: generado=', N_FAKE, 'real=', N_REAL)
    print('Feature extractor: TorchMetrics KID / torch-fidelity Inception-v3, 2048.')
    print('Paquetes:', {p:pkg_version(p) for p in ['torch', 'torchmetrics', 'torch-fidelity', 'torchvision']})
    if (a.OUT / 'protocol.json').is_file():
        print('Protocolo de ablación: OK')
    else:
        raise FileNotFoundError(a.OUT / 'protocol.json')
    sources = list_sources()
    for key, paths in sources.items():
        print(f'  {key}: {len(paths)} archivos localizados (no cargados)')
    print('CHECK OK. Ejecutar --mode run solamente dentro de un job de GPU.')


def cache_path(key):
    return FEATURES_DIR / f'{key}.pt'


def load_cached(key, sources):
    path = cache_path(key)
    if not path.is_file():
        return None
    saved = torch.load(path, map_location='cpu', weights_only=True)
    expected = metadata_for(key, sources[key])
    if saved['meta'] != expected:
        raise RuntimeError(f'Cache no coincide para {key}; no mezclar experimentos. {path}')
    feats = saved['features']
    n = N_REAL if key == 'real' else N_FAKE
    if feats.shape != (n, FEATURE_DIM) or not torch.isfinite(feats).all():
        raise RuntimeError(f'Cache corrupta para {key}; no sobrescribir automáticamente.')
    return feats


def extract():
    if not torch.cuda.is_available():
        raise RuntimeError('La extracción Inception requiere GPU asignada por SLURM.')
    from torchmetrics.image.kid import KernelInceptionDistance
    from torchvision.datasets import CIFAR10

    sources = list_sources()
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda:0')
    metric = KernelInceptionDistance(feature=FEATURE_DIM, subsets=2,
                                     subset_size=500, normalize=False).to(device)
    metric.eval()
    inception = metric.inception
    print('GPU:', torch.cuda.get_device_name(0), flush=True)
    print('Extractor:', inception.__class__.__name__, flush=True)
    with torch.inference_mode():
        for key in sources:
            cached = load_cached(key, sources)
            if cached is not None:
                print('CACHE OK:', key, tuple(cached.shape), flush=True)
                continue
            print('Extrayendo características:', key, flush=True)
            extracted = []
            if key == 'real':
                ds = CIFAR10(root=str(a.DATA_DIR), train=False, download=False)
                if len(ds.data) != N_REAL:
                    raise RuntimeError(f'Se esperaban {N_REAL} CIFAR10 test; hay {len(ds.data)}')
                for start in range(0, N_REAL, 64):
                    img = torch.from_numpy(ds.data[start:start+64].copy()).permute(0,3,1,2).contiguous()
                    extracted.append(inception(img.to(device)).detach().float().cpu())
                    if (start // 64) % 40 == 0:
                        print('  real:', start+len(img), '/', N_REAL, flush=True)
            else:
                for i, path in enumerate(sources[key]):
                    img = torch.load(path, map_location='cpu', weights_only=True)
                    required = min(a.BATCH_SIZE, N_FAKE - i*a.BATCH_SIZE)
                    if tuple(img.shape) != (required,3,32,32) or img.dtype != torch.uint8:
                        raise RuntimeError(f'{path}: lote incompatible: {tuple(img.shape)}, {img.dtype}')
                    extracted.append(inception(img.to(device)).detach().float().cpu())
                    if i % 40 == 0:
                        print(' ', key, i+1, '/', len(sources[key]), flush=True)
            features = torch.cat(extracted, dim=0).contiguous()
            required = N_REAL if key == 'real' else N_FAKE
            if features.shape != (required, FEATURE_DIM):
                raise RuntimeError(f'Forma incorrecta: {key}: {features.shape}')
            if not torch.isfinite(features).all():
                raise RuntimeError(f'Features no finitas: {key}')
            target = cache_path(key)
            tmp = target.with_suffix('.tmp')
            torch.save({'meta': metadata_for(key, sources[key]), 'features': features}, tmp)
            tmp.replace(target)
            print('Guardado cache:', target, flush=True)
    print('Extracción completa. No se regeneraron imágenes.', flush=True)


def poly_kernel(x, y):
    return ((x @ y.T) / x.shape[1] + 1.0).pow_(3)


def weighted_u(k, counts):
    """U-statistic sobre bootstrap de n observaciones, incluso con duplicados."""
    n = counts.shape[1]
    numerator = ((counts @ k) * counts).sum(dim=1)
    numerator -= (counts * torch.diagonal(k)[None, :]).sum(dim=1)
    return numerator / (n * (n - 1))


def weighted_cross(k_real_fake, real_counts, fake_counts):
    return ((real_counts @ k_real_fake) * fake_counts).sum(dim=1) / (
        real_counts.shape[1] * fake_counts.shape[1]
    )


def counts_bootstrap(n, reps, seed, device):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    idx = torch.randint(n, (reps, n), device=device, generator=g)
    count = torch.zeros((reps, n), device=device, dtype=torch.float32)
    count.scatter_add_(1, idx, torch.ones_like(idx, dtype=torch.float32))
    return count


def basic_interval(theta, draws):
    low, high = np.quantile(draws, [0.025, 0.975])
    return 2*theta-high, 2*theta-low


def write_csv(path, rows):
    if not rows: return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)


def calculate(reps=400, seed=SEED):
    if reps < 100:
        raise ValueError('Usar al menos 100 réplicas bootstrap; se recomiendan 400 o más.')
    if not torch.cuda.is_available():
        raise RuntimeError('El bootstrap se ejecuta en GPU dentro de SLURM.')
    sources = list_sources()
    device = torch.device('cuda:0')
    # Fuerza precisión float32 en matmul, evitando TF32 si la GPU lo admite.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_grad_enabled(False)
    features = {}
    for key in sources:
        features[key] = load_cached(key, sources)
        if features[key] is None:
            raise RuntimeError(f'Faltan features de {key}; ejecutar --mode run o extract.')
    real = features['real'].to(device)
    fake_counts = counts_bootstrap(N_FAKE, reps, seed, device)
    real_counts = counts_bootstrap(N_REAL, reps, seed+1, device)
    one_real = torch.ones((1,N_REAL), dtype=torch.float32, device=device)
    one_fake = torch.ones((1,N_FAKE), dtype=torch.float32, device=device)
    print('Gram de imágenes reales: matriz 10000 × 10000', flush=True)
    krr = poly_kernel(real, real)
    real_u_boot = weighted_u(krr, real_counts)
    real_u_point = weighted_u(krr, one_real)[0]
    del krr
    torch.cuda.empty_cache()
    kid_boot = {}
    kid_full = {}
    for v in a.all_variants():
        key = v['key']
        fake = features[key].to(device)
        print('Calculando KID U completo y bootstrap:', key, flush=True)
        kff = poly_kernel(fake, fake)
        krf = poly_kernel(real, fake)
        fake_u_boot = weighted_u(kff, fake_counts)
        cross_boot = weighted_cross(krf, real_counts, fake_counts)
        fake_u_full = weighted_u(kff, one_fake)[0]
        cross_full = weighted_cross(krf, one_real, one_fake)[0]
        full = real_u_point + fake_u_full - 2*cross_full
        draws = real_u_boot + fake_u_boot - 2*cross_boot
        kid_full[key] = float(full.cpu())
        kid_boot[key] = draws.cpu().numpy().astype(np.float64)
        print(f'  KID U completo ×1000: {kid_full[key]*1000:.5f}', flush=True)
        del kff, krf, fake, draws, fake_u_boot, cross_boot
        torch.cuda.empty_cache()

    old_rows = {}
    with (a.OUT / 'kid_ablation.csv').open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            old_rows[row['key']] = row

    rows = []
    comparisons = []
    keys = [v['key'] for v in a.all_variants()]
    for key in keys:
        low, high = basic_interval(kid_full[key], kid_boot[key])
        rows.append({
            'key': key,
            'KID_subconjuntos_x1000_previo': float(old_rows[key]['KID_x1000']),
            'KID_U_completo_x1000': kid_full[key]*KID_SCALE,
            'CI95_basico_KID_U_inferior_x1000': low*KID_SCALE,
            'CI95_basico_KID_U_superior_x1000': high*KID_SCALE,
            'bootstrap_reps': reps,
            'n_generadas': N_FAKE,
            'n_reales': N_REAL,
        })
        for reference in ['A_all','B_all']:
            if key == reference: continue
            delta = kid_full[key] - kid_full[reference]
            paired = kid_boot[key] - kid_boot[reference]
            lo,hi = basic_interval(delta, paired)
            comparisons.append({
                'model_minus_reference': f'{key} - {reference}',
                'model': key,
                'reference': reference,
                'delta_KID_U_x1000': delta*KID_SCALE,
                'CI95_basico_delta_inferior_x1000': lo*KID_SCALE,
                'CI95_basico_delta_superior_x1000': hi*KID_SCALE,
                'bootstrap_sd_delta_x1000': np.std(paired, ddof=1)*KID_SCALE,
                'IC_contiene_0': bool(lo<=0<=hi),
            })
    write_csv(OUT/'kid_puntos_intervalos.csv',rows)
    write_csv(OUT/'kid_diferencias_pareadas.csv',comparisons)
    np.savez_compressed(OUT/'bootstrap_distribuciones.npz',
                        keys=np.array(keys), means=np.array([kid_full[k] for k in keys]),
                        bootstrap=np.stack([kid_boot[k] for k in keys]),seed=seed,reps=reps)
    with (OUT/'analisis_config.json').open('w',encoding='utf-8') as f:
        json.dump({
            'seed':seed,'bootstrap_replicates':reps,
            'n_generated_each':N_FAKE,'n_real':N_REAL,
            'kernel':'((u·v)/2048 + 1)^3',
            'estimator':'unbiased full-sample U-statistic MMD2, not subset KID mean',
            'bootstrap':'paired multinomial resampling of 5000 generator indices across all 8 models; independent but shared real resampling',
            'ci':'basic centered 2*point-[q975,q025], approximate, conditional on fitted checkpoints',
            'torch':pkg_version('torch'),'torchmetrics':pkg_version('torchmetrics'),
            'torch_fidelity':pkg_version('torch-fidelity'),
            'python':platform.python_version(),
            'gpu':torch.cuda.get_device_name(0),
        },f,indent=2,ensure_ascii=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected = {r['model']:r for r in comparisons if r['reference']=='B_all'}
    fig, ax = plt.subplots(figsize=(10,5))
    vals = [selected[k]['delta_KID_U_x1000'] for k in keys if k!='B_all']
    lower = [selected[k]['CI95_basico_delta_inferior_x1000'] for k in keys if k!='B_all']
    upper = [selected[k]['CI95_basico_delta_superior_x1000'] for k in keys if k!='B_all']
    labels = [k for k in keys if k!='B_all']
    yy = np.arange(len(labels))
    # Dibujar IC basic incluso cuando el estimador puntual no cae dentro
    # de su propio IC debido al sesgo/corrección bootstrap.
    for x, y, lo, hi in zip(vals, yy, lower, upper):
        ax.hlines(y, lo, hi, color='tab:blue', lw=2)
        ax.vlines([lo,hi], y-0.14, y+0.14, color='tab:blue', lw=1.5)
        ax.plot(x,y,'o',color='tab:blue')
    ax.set_yticks(yy, labels)
    ax.axvline(0,color='black',ls='--',linewidth=1)
    ax.invert_yaxis()
    ax.set_xlabel('ΔKID U completo × 1000 (configuración − ResUNet completo)')
    ax.set_title('Diferencias pareadas de KID e IC bootstrap básico del 95 %')
    ax.grid(axis='x',alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT/'kid_diferencias_IC95.png',dpi=180)
    plt.close(fig)
    print('RESULTADOS:', OUT, flush=True)
    for row in comparisons:
        if row['reference']=='B_all':
            print(row['model_minus_reference'],
                  f"Δ×1000={row['delta_KID_U_x1000']:.3f}",
                  f"IC95=({row['CI95_basico_delta_inferior_x1000']:.3f},"
                  f"{row['CI95_basico_delta_superior_x1000']:.3f})", flush=True)


def selftest():
    import numpy as np
    torch.manual_seed(77)
    X = torch.randn(9,5)
    Y = torch.randn(6,5)
    k = poly_kernel(X,X)
    one = torch.ones(1,9)
    expected = (k.sum()-k.diagonal().sum())/(9*8)
    assert torch.allclose(weighted_u(k,one)[0], expected,atol=1e-7)
    g = torch.Generator().manual_seed(7)
    ids = torch.randint(9,(1,9),generator=g)
    counts = torch.bincount(ids.flatten(),minlength=9).float()[None,:]
    repeated = X[ids[0]]
    krep = poly_kernel(repeated,repeated)
    fullrep = (krep.sum()-krep.diagonal().sum())/(9*8)
    assert torch.allclose(weighted_u(k,counts)[0],fullrep,atol=2e-6)
    ky = poly_kernel(Y,X)
    assert torch.allclose(weighted_cross(ky,torch.ones(1,6),one)[0],ky.mean(),atol=1e-7)
    print('SELFTEST OK: estimador U con/sin repeticiones y término cruzado.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['check','extract','bootstrap','run','selftest'],required=True)
    parser.add_argument('--reps',type=int,default=400)
    parser.add_argument('--seed',type=int,default=SEED)
    args = parser.parse_args()
    if args.mode == 'check': check()
    elif args.mode == 'selftest': selftest()
    else:
        if args.mode in ('extract','run'):
            extract()
        if args.mode in ('bootstrap','run'):
            calculate(reps=args.reps,seed=args.seed)


if __name__ == '__main__':
    main()
