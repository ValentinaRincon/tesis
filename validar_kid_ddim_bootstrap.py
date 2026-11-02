#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validacion KID DDIM con bootstrap pareado; no entrena ni genera muestras.

Usa las mismas imagenes uint8 de ~/tesis/DDIM CIFAR10 EMA/chunks,
CIFAR-10 test y el extractor Inception de TorchMetrics KID 2048.

Las 10 condiciones comparten el indice de semilla de ruido inicial. Se
remuestrean los mismos 5000 indices generados y los mismos 10000 indices
reales en todas las condiciones; esto permite evaluar diferencias PAREADAS.

El estimador puntual es KID U sobre el conjunto COMPLETO; no es el promedio
historico sobre 100 subconjuntos de 500. IC bootstrap basico del 95%.

Modos: --mode selftest, check, extract, bootstrap, run, status.
Solamente extract/bootstrap/run se ejecutan en GPU mediante SLURM.
"""
import argparse
import csv
import hashlib
import json
import math
import platform
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

import numpy as np
import torch
import comparar_ddim_cifar10 as ddim

ROOT = Path(__file__).resolve().parent
SOURCE = ddim.OUT
OUT = ROOT / 'Incertidumbre KID DDIM CIFAR10'
FEATURES = OUT / 'features'
REAL_N = 10000
FAKE_N = ddim.N
FEATURE_DIM = 2048
SCALE = 1000.0
DEFAULT_SEED = 20261026


def pkg_version(name):
    try: return version(name)
    except PackageNotFoundError: return 'not-installed'


def real_file():
    return ROOT / 'data' / 'cifar-10-batches-py' / 'test_batch'


def inventory(require_complete=True):
    """Solo operaciones stat; nunca carga imagenes ni checkpoints."""
    items = {'real': [real_file()]}
    for v in ddim.VARIANTS:
        items[v['key']] = [ddim.chunk_path(v, i) for i in range(ddim.nchunks())]
    if require_complete:
        for key, paths in items.items():
            missing = [p for p in paths if not p.is_file() or p.stat().st_size <= 0]
            if missing:
                raise FileNotFoundError(f'{key}: {len(missing)} archivos ausentes/vacios; primero {missing[0]}')
        if not (SOURCE / 'protocolo.json').is_file():
            raise FileNotFoundError(SOURCE / 'protocolo.json')
        if not (SOURCE / 'kid_ddim.csv').is_file():
            raise FileNotFoundError(SOURCE / 'kid_ddim.csv')
    return items


def fingerprint(paths):
    h = hashlib.sha256()
    for p in paths:
        stat = p.stat()
        h.update(f'{p.resolve()}|{stat.st_size}|{stat.st_mtime_ns}\n'.encode())
    return h.hexdigest()


def metadata(key, paths):
    return dict(
        key=key,
        source_inventory_sha256=fingerprint(paths),
        source_protocol_sha256=hashlib.sha256((SOURCE / 'protocolo.json').read_bytes()).hexdigest(),
        torchmetrics_version=pkg_version('torchmetrics'),
        torch_fidelity_version=pkg_version('torch-fidelity'),
        feature_dim=FEATURE_DIM,
        normalize=False,
        image_dtype='uint8',
    )


def feature_file(key):
    return FEATURES / f'{key}.pt'


def read_features_cache(key, inv):
    p = feature_file(key)
    if not p.exists(): return None
    obj = torch.load(p, map_location='cpu', weights_only=True)
    if obj['meta'] != metadata(key, inv[key]):
        raise RuntimeError(f'Cache incompatible: {p}. No mezclar resultados de distintos protocolos.')
    features = obj['features']
    n = REAL_N if key == 'real' else FAKE_N
    if features.shape != (n, FEATURE_DIM) or not torch.isfinite(features).all():
        raise RuntimeError(f'Cache corrupta o incompleta: {p}')
    return features


def historical_rows():
    with (SOURCE / 'kid_ddim.csv').open(newline='', encoding='utf8') as file:
        rows = list(csv.DictReader(file))
    old = {r['variant']: r for r in rows}
    expected = {v['key'] for v in ddim.VARIANTS}
    if set(old) != expected or len(rows) != len(expected):
        raise RuntimeError('kid_ddim.csv no contiene exactamente las 10 variantes DDIM.')
    for key in expected:
        if int(old[key]['n_generated']) != FAKE_N or int(old[key]['n_real']) != REAL_N:
            raise RuntimeError(f'Tamano distinto del esperado en el KID historico: {key}')
    return old


def check():
    print('IC pareado DDIM | comprobacion de rutas (NO carga imagenes, NO usa GPU)')
    print('Directorio origen:', SOURCE)
    print('Directorio salida:', OUT)
    print('Configuraciones:',len(ddim.VARIANTS),'| generadas/config:',FAKE_N,'| reales:', REAL_N)
    print('Paquetes:', {n:pkg_version(n) for n in ('torch','torchmetrics','torch-fidelity','torchvision')})
    inv = inventory()
    historical_rows()
    for key, paths in inv.items():
        print(' OK:',key,'(',len(paths),'archivos )')
    protocol = json.loads((SOURCE/'protocolo.json').read_text(encoding='utf8'))
    assert protocol['step_counts'] == ddim.STEPS
    assert protocol['n_samples'] == FAKE_N and protocol['batch_size'] == ddim.BS
    assert float(protocol['ddim_eta']) == 0.0
    print('CHECK OK: imagenes y protocolo DDIM presentes; enviar job GPU SLURM.')


def status():
    inv = inventory(require_complete=False)
    print('Incertidumbre KID DDIM: caches de features')
    for key in inv:
        p = feature_file(key)
        print(f'  {key:29s}', 'OK' if p.exists() and p.stat().st_size else 'pendiente')
    for name in ('kid_ddim_intervalos.csv', 'kid_ddim_diferencias_pareadas.csv'):
        p = OUT / name
        print(' ', name, 'OK' if p.exists() else 'pendiente')


def extract():
    if not torch.cuda.is_available():
        raise RuntimeError('Extraccion solo bajo SLURM con GPU disponible.')
    from torchmetrics.image.kid import KernelInceptionDistance
    from torchvision.datasets import CIFAR10
    inv = inventory()
    historical_rows()
    FEATURES.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda:0')
    metric = KernelInceptionDistance(feature=FEATURE_DIM, subsets=2,
                                      subset_size=500, normalize=False).to(device)
    metric.eval()
    inception = metric.inception
    print('GPU:',torch.cuda.get_device_name(0),'| extractor:',inception.__class__.__name__,flush=True)
    with torch.inference_mode():
        for key, paths in inv.items():
            cached = read_features_cache(key, inv)
            if cached is not None:
                print('CACHE OK:', key, tuple(cached.shape), flush=True)
                continue
            print('Extraccion Inception:',key,flush=True)
            parts = []
            if key == 'real':
                ds = CIFAR10(root=str(ROOT / 'data'), train=False, download=False)
                if len(ds.data) != REAL_N:
                    raise RuntimeError(f'CIFAR10 test: se esperaban {REAL_N}, encontrados {len(ds.data)}')
                for i in range(0, REAL_N, 64):
                    batch = torch.from_numpy(ds.data[i:i+64].copy()).permute(0,3,1,2).contiguous()
                    parts.append(inception(batch.to(device)).detach().float().cpu())
                    if (i//64)%40 == 0: print('  reales',i+len(batch),'/', REAL_N,flush=True)
            else:
                for i, p in enumerate(paths):
                    batch = torch.load(p, map_location='cpu', weights_only=True)
                    n_expected = ddim.expected_size(i)
                    if not isinstance(batch, torch.Tensor) or batch.dtype != torch.uint8 or tuple(batch.shape)!=(n_expected,3,32,32):
                        raise RuntimeError(f'Lote distinto del esperado: {p} {getattr(batch, "shape", "?")}')
                    parts.append(inception(batch.to(device)).detach().float().cpu())
                    if i%40==0:print(' ',key,'lote',i+1,'/',len(paths),flush=True)
            feats = torch.cat(parts,dim=0).contiguous()
            expected_n = REAL_N if key=='real' else FAKE_N
            if feats.shape != (expected_n, FEATURE_DIM) or not torch.isfinite(feats).all():
                raise RuntimeError(f'Features con dimensiones/valores invalidos: {key} {tuple(feats.shape)}')
            target = feature_file(key)
            tmp = target.with_suffix('.tmp')
            torch.save({'meta':metadata(key, paths),'features':feats},tmp)
            tmp.replace(target)
            print('Guardado:',target,flush=True)
    print('EXTRACCION COMPLETA: no se generaron imagenes.',flush=True)


def kernel(x,y):
    return ((x @ y.T)/x.shape[1]+1.0).pow_(3)


def weighted_u(k,counts):
    """Estimador U correcto en muestras bootstrap con indices repetidos."""
    n = counts.shape[1]
    numerator = ((counts @ k)*counts).sum(dim=1)
    numerator -= (counts*torch.diagonal(k)[None,:]).sum(dim=1)
    return numerator/(n*(n-1))


def weighted_cross(krf,real_counts,fake_counts):
    return ((real_counts @ krf)*fake_counts).sum(dim=1) / (real_counts.shape[1]*fake_counts.shape[1])


def draw_counts(n,reps,seed,device):
    g = torch.Generator(device=device).manual_seed(seed)
    # Reps x n ints may require >100 MB at large reps; split into groups.
    out = torch.zeros((reps,n),device=device,dtype=torch.float32)
    for s in range(0,reps,50):
        q=min(reps,s+50)
        index=torch.randint(n,(q-s,n),generator=g,device=device)
        out[s:q].scatter_add_(1,index,torch.ones(index.shape,device=device,dtype=torch.float32))
    return out


def basic_ci(point,draws):
    q_lo,q_hi = np.quantile(draws,[0.025,0.975])
    return (2*point-q_hi, 2*point-q_lo)


def csv_write(p, rows):
    if not rows:raise RuntimeError('Sin filas para escribir CSV')
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp = p.with_suffix('.tmp')
    with tmp.open('w',encoding='utf8',newline='') as file:
        w=csv.DictWriter(file,fieldnames=rows[0].keys())
        w.writeheader();w.writerows(rows)
    tmp.replace(p)


def compute(reps=400, seed=DEFAULT_SEED):
    if reps < 100:raise ValueError('Se requieren al menos 100 repeticiones bootstrap.')
    if not torch.cuda.is_available():raise RuntimeError('Calcular solamente con GPU via SLURM.')
    inv = inventory()
    old = historical_rows()
    device = torch.device('cuda:0')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.set_grad_enabled(False)
    feats = {}
    for key in inv:
        feats[key] = read_features_cache(key,inv)
        if feats[key] is None:
            raise RuntimeError(f'Faltan features: {key}; ejecutar --mode extract o run.')
    real = feats['real'].to(device)
    real_cts = draw_counts(REAL_N,reps,seed+1,device)
    fake_cts = draw_counts(FAKE_N,reps,seed,device)
    one_real=torch.ones((1,REAL_N),device=device,dtype=torch.float32)
    one_fake=torch.ones((1,FAKE_N),device=device,dtype=torch.float32)
    print('Kernel K(real,real), matriz 10000 x 10000',flush=True)
    krr=kernel(real,real)
    rr_pt=weighted_u(krr,one_real)[0]
    rr_bs=weighted_u(krr,real_cts)
    del krr
    torch.cuda.empty_cache()
    points={}
    draws={}
    for v in ddim.VARIANTS:
        key=v['key']
        print('Bootstrap KID:',key,flush=True)
        fake=feats[key].to(device)
        kff=kernel(fake,fake)
        krf=kernel(real,fake)
        pt=rr_pt+weighted_u(kff,one_fake)[0]-2*weighted_cross(krf,one_real,one_fake)[0]
        bs=rr_bs+weighted_u(kff,fake_cts)-2*weighted_cross(krf,real_cts,fake_cts)
        points[key]=float(pt.cpu())
        draws[key]=bs.cpu().numpy().astype(np.float64)
        print('  KID U x 1000:',round(points[key]*1000,6),flush=True)
        del fake,kff,krf,pt,bs
        torch.cuda.empty_cache()

    overview=[]
    for v in ddim.VARIANTS:
        key=v['key']
        lo,hi=basic_ci(points[key],draws[key])
        overview.append(dict(
            variant=key, model_key=v['model'], pasos=v['steps'],
            KID_subconjuntos_previo_x1000=float(old[key]['KID_x1000']),
            KID_U_completo_x1000=points[key]*SCALE,
            IC95_basico_inferior_x1000=lo*SCALE,
            IC95_basico_superior_x1000=hi*SCALE,
            imagenes_generadas=FAKE_N, imagenes_reales=REAL_N, bootstrap_reps=reps,
            imagenes_por_segundo=float(old[key]['effective_images_per_second']),
        ))

    comparisons=[]
    def add_compare(kind, v, ref):
        key=v['key']; rkey=ref['key']
        value=points[key]-points[rkey]
        distribution=draws[key]-draws[rkey]
        lo,hi=basic_ci(value,distribution)
        comparisons.append(dict(
            contraste=kind, variante=key, referencia=rkey,
            arquitectura=v['model'], pasos=v['steps'],
            delta_KID_U_x1000=value*SCALE,
            IC95_basico_inferior_x1000=lo*SCALE,
            IC95_basico_superior_x1000=hi*SCALE,
            bootstrap_sd_delta_x1000=float(np.std(distribution,ddof=1))*SCALE,
            IC_contiene_cero=bool(lo<=0<=hi)
        ))
    for model,_,_,_ in ddim.MODEL_INFO:
        reference=next(v for v in ddim.VARIANTS if v['model']==model and v['steps']==1000)
        for v in ddim.VARIANTS:
            if v['model']==model and v['steps']!=1000:
                add_compare('pasos_vs_1000_misma_arquitectura',v,reference)
    for steps in ddim.STEPS:
        attention=next(v for v in ddim.VARIANTS if v['model']=='attention_ema' and v['steps']==steps)
        resunet=next(v for v in ddim.VARIANTS if v['model']=='resunet_ema' and v['steps']==steps)
        add_compare('attention_menos_resunet_mismos_pasos',attention,resunet)

    csv_write(OUT/'kid_ddim_intervalos.csv',overview)
    csv_write(OUT/'kid_ddim_diferencias_pareadas.csv',comparisons)
    np.savez_compressed(OUT/'bootstrap_distribuciones_ddim.npz',
                        keys=np.array([v['key'] for v in ddim.VARIANTS]),
                        points=np.array([points[v['key']] for v in ddim.VARIANTS]),
                        bootstrap=np.stack([draws[v['key']] for v in ddim.VARIANTS]),
                        reps=reps,seed=seed)
    config=dict(
        fecha='2026-10-26',
        analisis='DDIM eta=0 bootstrap pareado', semilla=seed, bootstrap_replicas=reps,
        n_reales=REAL_N,n_generadas_por_condicion=FAKE_N,
        sampler='DDIM eta=0; timesteps compartidos por modelo en indice de muestra',
        extractor='TorchMetrics/TorchFidelity Inception-v3 feature=2048 normalize=False',
        kernel='((x dot y)/2048 + 1)^3',
        estimador='KID MMD^2 estimador U usando TODAS las imagenes',
        bootstrap='multinomial pareado: mismos indices de muestras generadas para las 10 condiciones; mismos indices reales en cada replica',
        intervalo='IC bootstrap basico nominal 95%, 2*estimacion - quantiles 97.5% y 2.5%',
        disclaimer='exploratorio; sin correccion de comparaciones multiples; condicionado a pesos entrenados y conjuntos muestrales; no inferir equivalencia',
        torch=pkg_version('torch'), torchmetrics=pkg_version('torchmetrics'),
        torch_fidelity=pkg_version('torch-fidelity'),python=platform.python_version(),
        gpu=torch.cuda.get_device_name(0),
        ddim_protocol_sha256=hashlib.sha256((SOURCE/'protocolo.json').read_bytes()).hexdigest()
    )
    (OUT/'config_analisis.json').write_text(json.dumps(config,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    make_plots(comparisons)
    print('COMPLETADO. RESULTADOS:',OUT,flush=True)
    for r in comparisons:
        print(r['contraste'],r['variante'],
              f"Δ×1000={r['delta_KID_U_x1000']:.5f}",
              f"IC95=({r['IC95_basico_inferior_x1000']:.5f}, {r['IC95_basico_superior_x1000']:.5f})",flush=True)


def make_plots(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def errorbar(ax,r,y,color):
        lo=r['IC95_basico_inferior_x1000'];hi=r['IC95_basico_superior_x1000']
        pt=r['delta_KID_U_x1000']
        ax.hlines(y,lo,hi,color=color,linewidth=2)
        ax.vlines([lo,hi],y-.14,y+.14,color=color,linewidth=1.5)
        ax.plot(pt,y,'o',color=color)
    fig,ax=plt.subplots(figsize=(9.5,5.3))
    palette={'attention_ema':'tab:blue','resunet_ema':'tab:orange'}
    compare=[r for r in rows if r['contraste']=='pasos_vs_1000_misma_arquitectura']
    positions=[]; labels=[]
    for i,r in enumerate(compare):
        positions.append(i)
        labels.append(f"{r['arquitectura']} / {r['pasos']} pasos")
        errorbar(ax,r,i,palette[r['arquitectura']])
    ax.axvline(0,color='black',linestyle='--')
    ax.set_yticks(positions,labels)
    ax.invert_yaxis()
    ax.set_xlabel('Δ KID U completo × 1000 (pasos indicados − 1000 pasos)')
    ax.set_title('DDIM: costo de reducir pasos (IC bootstrap pareado 95 %)')
    ax.grid(axis='x',alpha=.22)
    fig.tight_layout();fig.savefig(OUT/'ddim_diferencias_pasos_IC95.png',dpi=180);plt.close(fig)

    fig,ax=plt.subplots(figsize=(9,4.8))
    compare=[r for r in rows if r['contraste']=='attention_menos_resunet_mismos_pasos']
    for i,r in enumerate(compare):
        errorbar(ax,r,i,'tab:blue')
    ax.axvline(0,color='black',linestyle='--')
    ax.set_yticks(range(len(compare)),[str(r['pasos'])+' pasos' for r in compare])
    ax.invert_yaxis()
    ax.set_xlabel('Δ KID U completo × 1000 (Attention − ResUNet)')
    ax.set_title('DDIM: comparación entre arquitecturas (IC pareado 95 %)')
    ax.grid(axis='x',alpha=.22)
    fig.tight_layout();fig.savefig(OUT/'ddim_diferencias_arquitecturas_IC95.png',dpi=180);plt.close(fig)


def selftest():
    torch.manual_seed(8)
    a=torch.randn(8,6)
    b=torch.randn(7,6)
    ka=kernel(a,a)
    assert torch.allclose(weighted_u(ka,torch.ones(1,8))[0],
                          (ka.sum()-ka.diagonal().sum())/(8*7),atol=1e-7)
    indices=torch.tensor([0,0,2,2,2,4,5,7])
    c=torch.bincount(indices,minlength=8).float().unsqueeze(0)
    ka_repeated=kernel(a[indices],a[indices])
    expect=(ka_repeated.sum()-ka_repeated.diagonal().sum())/(8*7)
    assert torch.allclose(weighted_u(ka,c)[0],expect,atol=2e-6)
    kb=kernel(b,a)
    assert torch.allclose(weighted_cross(kb,torch.ones(1,7),torch.ones(1,8))[0],kb.mean(),atol=1e-7)
    np.testing.assert_allclose(basic_ci(3.0,np.array([2.0,3.0,4.0]))[0],2.05,rtol=0,atol=1e-9)
    one=torch.ones((1,8))
    # diferencias entre estimadores identicos deben cancelarse exactamente
    delta=weighted_u(ka,c)-weighted_u(ka,c)
    assert torch.equal(delta,torch.zeros_like(delta))
    print('SELFTEST OK: kernel, U estadistico, duplicados bootstrap, pareamiento e IC basico.')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',required=True,choices=['selftest','check','status','extract','bootstrap','run'])
    p.add_argument('--reps',type=int,default=400)
    p.add_argument('--seed',type=int,default=DEFAULT_SEED)
    a=p.parse_args()
    if a.mode=='selftest':selftest()
    elif a.mode=='check':check()
    elif a.mode=='status':status()
    else:
        if a.mode in ('extract','run'):extract()
        if a.mode in ('bootstrap','run'):compute(reps=a.reps,seed=a.seed)


if __name__ == '__main__':main()
