#!/usr/bin/env python3
"""PCA y dimensión efectiva de CIFAR-10 y su evolución bajo el forward DDPM.

No carga modelos ni necesita PyTorch/GPU.  Para ruido gaussiano isotrópico,
Cov(x_t)=alpha_bar_t Cov(x_0)+(1-alpha_bar_t) I en el conjunto poblacional
con la distribución empírica de CIFAR-10 test como referencia x_0.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "Dimensionalidad Efectiva CIFAR10"
DATA = ROOT / "data" / "cifar-10-batches-py" / "test_batch"
D = 3 * 32 * 32
T = 1000
TIMES = (0, 50, 250, 500, 750, 900, 950, 975, 999)


def alpha_bars():
    return np.cumprod(np.linspace(1e-4, 0.02, T, dtype=np.float64).ravel().__rsub__(1.0))


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fp:
        for b in iter(lambda: fp.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def participation(lams):
    lams = np.asarray(lams, dtype=np.float64)
    tr = float(np.sum(lams))
    tr2 = float(np.dot(lams, lams))
    return tr * tr / tr2


def entropy_rank(lams):
    p = lams / np.sum(lams)
    p = p[p > 0]
    return float(np.exp(-np.dot(p, np.log(p))))


def k_var(lams, fraction):
    s = np.sort(np.asarray(lams))[::-1]
    return int(np.searchsorted(np.cumsum(s), fraction * np.sum(s), side='left') + 1)


def check_data(path):
    if not path.is_file():
        raise FileNotFoundError(
            f"No encuentro {path}. Debe existir el CIFAR-10 test original "
            "en ~/tesis/data/cifar-10-batches-py/test_batch; "
            "este script NUNCA descarga datos."
        )
    if path.stat().st_size < 100000:
        raise ValueError(f'Archivo CIFAR demasiado pequeño ({path.stat().st_size} bytes): {path}')


def load_cifar(path, limit):
    print(f"Cargando CIFAR-10 test desde {path}", flush=True)
    with path.open('rb') as fp:
        obj = pickle.load(fp, encoding='bytes')
    raw = obj.get(b'data', obj.get('data'))
    if raw is None or raw.ndim != 2 or raw.shape[1] != D or raw.dtype != np.uint8:
        raise RuntimeError(f'Datos CIFAR inesperados; forma {getattr(raw,"shape",None)}, dtype {getattr(raw,"dtype",None)}')
    if raw.shape[0] != 10000:
        raise RuntimeError(f'Esperaba 10000 imágenes de CIFAR-10 test, hay {raw.shape[0]}')
    if limit is not None:
        if limit < 2 or limit > 10000:
            raise ValueError('--limit debe estar entre 2 y 10000')
        raw = raw[:limit]
    # Mismo escalado que torchvision.transforms.ToTensor() + Normalize([.5]*3,[.5]*3).
    # CIFAR serializado almacena canales antes de las coordenadas espaciales.
    x = np.asarray(raw, dtype=np.float64)
    x *= 1.0 / 127.5
    x -= 1.0
    return x


def save_csv(path, fieldnames, rows):
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', newline='', encoding='utf-8') as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    tmp.replace(path)


def prepare_plots(rows, arrays, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figures = out / 'figuras'
    figures.mkdir(exist_ok=True)
    timed = [r for r in rows if r['t'] != 'clean']
    t = np.asarray([int(r['t']) for r in timed])
    deff = np.asarray([float(r['d_participacion']) for r in timed])
    erank = np.asarray([float(r['rango_entropico']) for r in timed])
    k95 = np.asarray([int(r['k95']) for r in timed])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t, deff, 'o-', label='Dimensión de participación')
    ax.plot(t, erank, 's--', label='Rango entrópico')
    ax.axhline(D, ls=':', c='gray', label=f'Isotrópico: {D}')
    ax.set(xlabel='Paso de difusión t', ylabel='Dimensión efectiva', title='Dimensionalidad efectiva de la covarianza esperada en DDPM')
    ax.legend()
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(figures / 'dimension_efectiva_vs_t.png', dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t, k95, 'o-')
    ax.axhline(int(np.ceil(.95 * D)), ls=':', color='gray', label='95 % isotrópico')
    ax.set(xlabel='Paso de difusión t', ylabel='Componentes para explicar 95 % de varianza', title='Distribución espectral bajo ruido gaussiano')
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / 'componentes_95_vs_t.png', dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for label in ('clean', '250', '500', '750', '999'):
        z = np.sort(arrays[label])[::-1]
        ax.semilogy(np.arange(1, D + 1), np.maximum(z, 1e-12), label=f'{label if label=="clean" else "t="+label}')
    ax.set(xlabel='Índice de autovalor (orden descendente)', ylabel='Autovalor de covarianza', title='Espectro esperado del forward DDPM')
    ax.legend()
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(figures / 'espectros_covarianza.png', dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for label in ('clean', '250', '500', '750', '999'):
        z = np.sort(arrays[label])[::-1]
        frac = np.cumsum(z) / np.sum(z)
        ax.plot(np.arange(1, D + 1), frac, label=f'{label if label=="clean" else "t="+label}')
    ax.axhline(.95, color='gray', ls=':')
    ax.set(xlabel='Número de componentes principales', ylabel='Fracción de varianza acumulada', title='Varianza explicada por PCA')
    ax.legend()
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(figures / 'varianza_acumulada.png', dpi=180)
    plt.close(fig)


def selftest():
    # Identidad isotrópica: participación y rango entropía exactamente igual a d.
    assert abs(participation(np.ones(12)) - 12) < 1e-12
    assert abs(entropy_rank(np.ones(12)) - 12) < 1e-12
    assert k_var(np.ones(12), .95) == 12
    # Covarianza con dos ejes, rango de participación exacto 100/82.
    assert abs(participation([9, 1]) - 100 / 82) < 1e-12
    # Covarianza de x_t = sqrt(a) x + sqrt(1-a) eps.
    base = np.array([6., 2., .25])
    a = .3
    mixed = a * base + (1-a)
    assert np.allclose(mixed, np.array([2.5, 1.3, .775]))
    assert np.isclose(mixed.sum(), a*base.sum() + (1-a)*3)
    # La dimension efectiva de la covarianza isotrópica es d.
    assert abs(participation(np.ones(3)) - 3) < 1e-12
    ab = alpha_bars()
    assert len(ab) == 1000 and np.all(np.diff(ab) < 0)
    assert np.isclose(ab[-1], 4.035829765375676e-5, rtol=1e-5)
    print('SELFTEST OK: PCA, dimensiones, mezcla de covarianza, schedule DDPM.')


def run(path, out, limit):
    check_data(path)
    out.mkdir(parents=True, exist_ok=True)
    x = load_cifar(path, limit)
    n, d = x.shape
    assert d == D
    print(f'Datos: {n} imagenes x {d} dimensiones; rango aproximado {float(x.min()):.2f}..{float(x.max()):.2f}', flush=True)
    # El centro es importante: covarianza, no momento no centrado.
    mean = np.mean(x, axis=0)
    x -= mean
    print('Calculando la covarianza muestral C0 = Xc.T Xc/(n-1)...', flush=True)
    C0 = (x.T @ x) / (n - 1)
    del x
    print('Diagonalizando C0 una sola vez (dimension 3072)...', flush=True)
    l0 = np.linalg.eigvalsh(C0)
    maxv = max(1.0, float(l0[-1]))
    if float(l0[0]) < -1e-8 * maxv:
        raise RuntimeError(f'Autovalor negativo no esperado: {l0[0]}')
    l0 = np.maximum(l0, 0.0)
    l0 = l0[::-1]  # mayor primero
    print(f'Eigenspectrum listo. rango: [{l0[-1]:.5g}, {l0[0]:.5g}], traza={l0.sum():.5g}', flush=True)
    ab = alpha_bars()
    arrays = {'clean': l0.copy()}
    for t in TIMES:
        a = float(ab[t])
        arrays[str(t)] = a * l0 + (1.0-a)

    rows = []
    for key, eig in arrays.items():
        a = 1.0 if key == 'clean' else float(ab[int(key)])
        tr = float(eig.sum())
        tr2 = float(np.dot(eig,eig))
        rows.append({
            't': key, 'alpha_bar_t': a, 'varianza_media_por_dimension': tr/D,
            'traza_covarianza': tr, 'traza_covarianza_cuadrado': tr2,
            'd_participacion': tr*tr/tr2,
            'rango_entropico': entropy_rank(eig),
            'k90': k_var(eig,.90), 'k95': k_var(eig,.95), 'k99': k_var(eig,.99),
            'numero_dimensiones': D, 'numero_imagenes': n,
        })
        print(f'{key:>5} : α_bar={a:.7f}, d_PR={rows[-1]["d_participacion"]:.2f}, k95={rows[-1]["k95"]}', flush=True)

    print('Guardando CSV y figuras...', flush=True)
    cols = list(rows[0].keys())
    save_csv(out / 'resumen_dimensionalidad.csv', cols, rows)
    cols_s = ['componente', 'autovalor_clean'] + [f'autovalor_t{t}' for t in TIMES]
    save_csv(out / 'espectro_covarianza.csv', cols_s, [
        {'componente': i+1, 'autovalor_clean': float(l0[i]),
         **{f'autovalor_t{t}': float(arrays[str(t)][i]) for t in TIMES}}
        for i in range(D)])
    prepare_plots(rows, arrays, out)
    protocol = {
        'analysis': 'pca_cifar10_forward_ddpm_expected_covariance_v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'dataset_file': str(path), 'dataset_sha256': sha256(path),
        'n_images': n, 'features': D,
        'dataset_normalization': 'float64(raw_uint8/127.5-1), no data augmentation',
        'sample_covariance': 'centered X.T @ X/(n-1)',
        'times': list(TIMES), 'T': T, 'beta_start':1e-4, 'beta_end':0.02,
        'expected_forward_cov': 'alpha_bar_t*C0 + (1-alpha_bar_t)*I',
        'participation_dimension': '(trace(C))**2/trace(C@C)',
        'entropy_rank':'exp(-sum_k p_k log(p_k)); p_k = lambda_k/trace(C)',
        'important': ('Curves are analytic for the expected forward covariance from the '
                      'empirical CIFAR10 test distribution, not covariance estimates of '
                      'a newly sampled finite dataset at each t. No bootstrap here.'),
        'numpy': np.__version__,
    }
    (out / 'protocolo.json').write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding='utf-8')
    print('FINALIZADO:', out, flush=True)


def status(out):
    target = [
        'resumen_dimensionalidad.csv', 'espectro_covarianza.csv', 'protocolo.json',
        'figuras/dimension_efectiva_vs_t.png', 'figuras/componentes_95_vs_t.png',
        'figuras/espectros_covarianza.png','figuras/varianza_acumulada.png'
    ]
    for name in target:
        p = out / name
        print(('OK  ' if p.is_file() and p.stat().st_size > 0 else 'FALTA ') + str(p))
    print('Completo:', all((out / s).is_file() and (out / s).stat().st_size > 0 for s in target))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['selftest','check','run','status'], default='status')
    p.add_argument('--data-file', type=Path, default=DATA)
    p.add_argument('--out', type=Path, default=OUT)
    p.add_argument('--limit', type=int, default=None, help='Opcional: primeras n imagenes test (para diagnostico).')
    args = p.parse_args()
    if args.mode == 'selftest':
        selftest()
    elif args.mode == 'check':
        check_data(args.data_file)
        import matplotlib
        print('CHECK OK:', args.data_file, '|', args.data_file.stat().st_size, 'bytes')
        print('numpy', np.__version__, 'matplotlib', matplotlib.__version__, '| sin GPU ni torch')
    elif args.mode == 'run':
        run(args.data_file, args.out, args.limit)
    else:
        status(args.out)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        sys.exit(1)
