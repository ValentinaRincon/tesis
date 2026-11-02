#!/usr/bin/env python3
"""DDIM determinista CIFAR10: Attention+EMA vs ResUNet+Attention+EMA.

Reutiliza EXACTAMENTE clases, pesos, betas y parametros KID de
'evaluar_kid_cifar10_5000.py' (debe coexistir en ~/tesis).

Modos aptos para LOGIN (ligeros): --mode check, --mode status, --mode selftest
Modos GPU exclusivamente bajo SLURM: --mode sample --variant-index 0..9,
                                  --mode kid

Cada variante usa el mismo ruido inicial x_T por (indice de lote, semilla),
no comparte ruido por paso porque DDIM eta=0 es determinista.
La referencia DDPM 1000 pasos es estocastica, NO es DDIM 1000 pasos.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import statistics
import time
from pathlib import Path

import torch
import evaluar_kid_cifar10_5000 as old

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'DDIM CIFAR10 EMA'
CHUNKS = OUT / 'chunks'
TIMES = OUT / 'tiempos'
MODEL_DIR = ROOT / 'modelos'
STEPS = [1000, 500, 250, 100, 50]
N = old.N_SAMPLES
BS = old.SAMPLE_BATCH_SIZE
T = old.T
SAMPLE_SEED = old.SAMPLE_SEED
KID_SEED = old.METRIC_SEED
MODEL_INFO = [
    ('attention_ema', 'Attention + EMA', old.SimpleUNetAttention,
     MODEL_DIR / 'cifar10_attention_ema_model.pt'),
    ('resunet_ema', 'ResUNet + Attention + EMA', old.ResUNetAttention,
     MODEL_DIR / 'cifar10_resunet_attention_ema_model.pt'),
]
VARIANTS = [dict(key=f'ddim_{key}_{steps:04d}', model=key, label=label, steps=steps,
                 constructor=constructor, weight=path)
            for key, label, constructor, path in MODEL_INFO for steps in STEPS]


def hash_file(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for c in iter(lambda: f.read(1024 * 1024), b''):
            h.update(c)
    return h.hexdigest()


def atomic_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n', encoding='utf8')
    temp.replace(path)


def timetable(num_steps, total=T):
    """Marcas equiespaciadas, extremos t=0 y t=T-1 INCLUIDOS.

    Redondeo entero determinista half-up, sin decimales imprecisos.
    """
    if not 2 <= num_steps <= total:
        raise ValueError('2 <= num_steps <= total requerido')
    q = num_steps - 1
    times = [(i * (total - 1) * 2 + q) // (2 * q) for i in range(num_steps)]
    assert times[0] == 0 and times[-1] == total-1
    assert all(u < v for u, v in zip(times, times[1:]))
    return times


def protocol():
    return {
        'experiment': 'CIFAR10 DDIM step ablation eta=0',
        'models': {key: hash_file(weight) for key, _, _, weight in MODEL_INFO},
        'reference_script': 'evaluar_kid_cifar10_5000.py',
        'reference_script_sha256': hash_file(ROOT / 'evaluar_kid_cifar10_5000.py'),
        'T': T, 'beta_start': old.BETA_START, 'beta_end': old.BETA_END,
        'n_samples': N, 'batch_size': BS, 'seed': SAMPLE_SEED,
        'step_counts': STEPS,
        'timestep_selection': 'rounded evenly spaced 0..999 inclusive',
        'schedule': {str(k): timetable(k) for k in STEPS},
        'ddim_eta': 0.0, 'x0_prediction_clipping': False,
        'final_image_clipping': True, 'dtype': 'uint8 RGB',
        'kid': {'feature': 2048, 'subsets': old.KID_SUBSETS,
                'subset_size': old.KID_SUBSET_SIZE, 'seed': KID_SEED},
    }


def ensure_protocol():
    p = OUT / 'protocolo.json'
    desired = protocol()
    if p.exists():
        if json.loads(p.read_text(encoding='utf8')) != desired:
            raise RuntimeError('Protocolo o pesos diferentes: no mezclar archivos de ejecuciones distintas.')
    else:
        atomic_json(p, desired)


def variant_manifest(v):
    return {'variant': v['key'], 'model': v['model'], 'steps': v['steps'],
            'protocol_sha256': hash_file(OUT / 'protocolo.json')}


def manifest_path(v):
    return CHUNKS / v['key'] / 'manifest.json'


def ensure_manifest(v):
    p = manifest_path(v)
    expected = variant_manifest(v)
    if p.exists():
        if json.loads(p.read_text(encoding='utf8')) != expected:
            raise RuntimeError(f'Manifiesto incompatible: {v["key"]}')
    else:
        atomic_json(p, expected)


def chunk_path(v, i):
    return CHUNKS / v['key'] / f'batch_{i:04d}.pt'


def timing_path(v, i):
    return TIMES / v['key'] / f'batch_{i:04d}.json'


def nchunks():
    return math.ceil(N / BS)


def expected_size(i):
    return min(BS, N - i*BS)


def simple_count(v):
    return sum(chunk_path(v, i).is_file() and chunk_path(v, i).stat().st_size>0
               for i in range(nchunks()))


def check():
    print('DDIM CIFAR10 ETA=0 |', len(VARIANTS), 'variantes |', N, 'imagenes/variante')
    print('Directorio de salida:', OUT)
    if (T, N, BS, old.BETA_START, old.BETA_END) != (1000, 5000, 32, 1e-4, 0.02):
        raise RuntimeError('La configuracion KID original ha cambiado, revisar compatibilidad.')
    for key, _, _, p in MODEL_INFO:
        print(('OK   ' if p.is_file() else 'FALTA'), key, p)
        if not p.is_file():
            raise FileNotFoundError(p)
    cifar = ROOT/'data'/'cifar-10-batches-py'/'test_batch'
    print(('OK   ' if cifar.is_file() else 'FALTA'), cifar)
    if not cifar.is_file():
        raise FileNotFoundError(cifar)
    for m in ('torchvision', 'torchmetrics', 'torch_fidelity', 'matplotlib'):
        if importlib.util.find_spec(m) is None:
            raise RuntimeError(f'Dependencia faltante: {m}')
    # Antes de invertir horas de GPU: comprobar semilla y controles previos.
    prev_file = old.OUTPUT_DIR/'kid_comparison.csv'
    print('Referencia DDPM previa:', prev_file if prev_file.is_file() else 'NO DISPONIBLE (opcional)')
    if (OUT/'protocolo.json').exists() and json.loads((OUT/'protocolo.json').read_text()) != protocol():
        raise RuntimeError('Hay datos previos incompatibles en la carpeta de salida.')
    for k in STEPS:
        t = timetable(k)
        print(f' DDIM {k:4d} pasos: [{t[-1]}, {t[-2]}, ..., {t[1]}, {t[0]}]')
    print('CHECK OK: ejecutar sample/kid UNICAMENTE en nodo GPU SLURM.')


def status():
    print('DDIM CIFAR10 (conteo ligero sin cargar tensores):')
    total = nchunks()
    for i, v in enumerate(VARIANTS):
        count = simple_count(v)
        print(f' {i:2d}: {v["key"]:28s} {count:3d}/{total}')
    print('CSV KID:', OUT/'kid_ddim.csv' if (OUT/'kid_ddim.csv').exists() else 'no disponible')


def ddim_update(x, eps, abar_t, abar_prev):
    """Paso DDIM eta=0 sin clipping intermedio; abar_prev=1 para t_prev=-1."""
    sqrt_ab = torch.sqrt(abar_t)
    x0hat = (x-torch.sqrt(1-abar_t)*eps)/sqrt_ab
    xprev = torch.sqrt(abar_prev)*x0hat + torch.sqrt(1-abar_prev)*eps
    return xprev


@torch.inference_mode()
def generate_batch(model, batch_size, generator, schedule, device, steps):
    _, _, abar = schedule
    x = torch.randn((batch_size, 3, 32, 32), generator=generator, device=device)
    selected = timetable(steps)
    for j in range(len(selected)-1, -1, -1):
        t = selected[j]
        previous = selected[j-1] if j>0 else -1
        ts = torch.full((batch_size,), t, dtype=torch.long, device=device)
        eps = model(x, ts)
        ab_prev = abar[previous] if previous>=0 else x.new_tensor(1.0)
        x = ddim_update(x, eps, abar[t], ab_prev)
    return ((x.clamp(-1,1)+1)*127.5).round().to(torch.uint8).cpu()


def selftest():
    # Analitica: si x = sqrt(ab_t)*x0 + sqrt(1-ab_t)*eps, DDIM eta0
    # transporta exactamente al t_prev para eps perfecto.
    for n in (2, 7, 50, 100, 250, 500, 1000):
        ts = timetable(n)
        assert len(ts)==n and len(set(ts))==n and ts[0]==0 and ts[-1]==999
    aa = torch.tensor(0.37)
    ap = torch.tensor(0.81)
    x0 = torch.tensor([[[[0.28, -0.48]]]])
    eps = torch.tensor([[[[0.91, -1.23]]]])
    xt = aa.sqrt()*x0+(1-aa).sqrt()*eps
    expect = ap.sqrt()*x0+(1-ap).sqrt()*eps
    got = ddim_update(xt, eps, aa, ap)
    assert torch.allclose(got, expect, atol=1e-6)
    final = ddim_update(xt, eps, aa, torch.tensor(1.0))
    assert torch.allclose(final, x0, atol=1e-6)
    class ZeroNoise(torch.nn.Module):
        def forward(self, x, t): return torch.zeros_like(x)
    device = torch.device('cpu')
    schedule = (None, None, torch.cumprod(1-torch.linspace(1e-4,.02,1000),0))
    g1=torch.Generator(device=device).manual_seed(314)
    g2=torch.Generator(device=device).manual_seed(314)
    a=generate_batch(ZeroNoise(),2,g1,schedule,device,5)
    b=generate_batch(ZeroNoise(),2,g2,schedule,device,5)
    assert torch.equal(a,b) and a.shape==(2,3,32,32) and a.dtype==torch.uint8
    print('SELFTEST OK: formula DDIM, timesteps, endpoint, semillas, uint8.')


def validate_chunk(p, sz):
    if not p.exists():
        return False
    try:
        data=torch.load(p,map_location='cpu',weights_only=True)
        return isinstance(data,torch.Tensor) and data.dtype==torch.uint8 and tuple(data.shape)==(sz,3,32,32)
    except Exception:
        return False


def sample(index):
    if not torch.cuda.is_available():
        raise RuntimeError('Se necesita GPU asignada por SLURM para generar.')
    v=VARIANTS[index]
    ensure_protocol()
    ensure_manifest(v)
    device=torch.device('cuda:0')
    print('Modelo:',v['key'],'GPU:',torch.cuda.get_device_name(0),'Pasos:',v['steps'],flush=True)
    missing=[i for i in range(nchunks()) if not validate_chunk(chunk_path(v,i),expected_size(i))]
    print(f'Existentes {nchunks()-len(missing)}/{nchunks()}; pendientes {len(missing)}',flush=True)
    if not missing: return
    model=v['constructor']().to(device)
    model.load_state_dict(old.load_state_dict(v['weight'],device),strict=True)
    model.eval()
    schedule=old.get_schedule(device)
    for i in missing:
        sz=expected_size(i)
        g=torch.Generator(device=device).manual_seed(SAMPLE_SEED+1009*i)
        torch.cuda.synchronize()
        start=time.perf_counter()
        img=generate_batch(model,sz,g,schedule,device,v['steps'])
        torch.cuda.synchronize()
        seconds=time.perf_counter()-start
        dest=chunk_path(v,i)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp=dest.with_suffix('.tmp')
        torch.save(img,tmp)
        tmp.replace(dest)
        atomic_json(timing_path(v,i),{'key':v['key'],'batch':i,'images':sz,
            'seconds':seconds,'seconds_per_image':seconds/sz,
            'steps':v['steps'],'device':torch.cuda.get_device_name(0)})
        if (i+1)%10==0 or i==0 or i==nchunks()-1:
            print(f'{v["key"]} {i+1}/{nchunks()} lotes | {seconds:.2f} s ultimo lote',flush=True)
    print('GENERACION COMPLETA:',v['key'],flush=True)


def verify_all():
    ensure_protocol()
    for v in VARIANTS:
        manifest=manifest_path(v)
        if not manifest.is_file() or json.loads(manifest.read_text())!=variant_manifest(v):
            raise RuntimeError(f'Manifiesto ausente o incompatible: {v["key"]}')
        for i in range(nchunks()):
            if not validate_chunk(chunk_path(v,i),expected_size(i)):
                raise RuntimeError(f'Falta lote integro: {v["key"]} #{i}')


def timing_summary(v):
    seconds=[]; n_images=0
    for i in range(nchunks()):
        p=timing_path(v,i)
        if p.is_file():
            data=json.loads(p.read_text())
            if data.get('key')==v['key'] and data.get('batch')==i and data.get('images')==expected_size(i):
                seconds.append((data['seconds'],data['images']))
                n_images+=data['images']
    if not seconds:
        return {'median_batch_seconds':'','total_sample_seconds':'','timed_images':0,
                'effective_images_per_second':''}
    total=sum(s for s,_ in seconds)
    return {'median_batch_seconds':statistics.median(s for s,_ in seconds),
            'total_sample_seconds':total,'timed_images':n_images,
            'effective_images_per_second':n_images/total}


def reference_ddpm_rows():
    path=old.OUTPUT_DIR/'kid_comparison.csv'
    if not path.exists(): return {}
    try:
        with path.open(newline='',encoding='utf8') as f:
            data=list(csv.DictReader(f))
        return {row['model_key']:float(row['KID_x1000']) for row in data
                if row.get('model_key') in ('attention_ema','resunet_ema')}
    except (KeyError,TypeError,ValueError):
        return {}


def write_csv(rows):
    path=OUT/'kid_ddim.csv'
    cols=['variant','model','model_key','sampler','steps','eta','KID_mean','KID_std_subsets',
          'KID_x1000','n_generated','n_real','subsets','subset_size','feature_dim',
          'median_batch_seconds','total_sample_seconds','timed_images','effective_images_per_second']
    tmp=path.with_suffix('.tmp')
    with tmp.open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=cols)
        writer.writeheader();writer.writerows(rows)
    tmp.replace(path)


def plot(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8.5,5.2))
    for key,label,_,_ in MODEL_INFO:
        rr=sorted((x for x in rows if x['model_key']==key),key=lambda r:r['steps'])
        ax.plot([r['steps'] for r in rr],[r['KID_x1000'] for r in rr],marker='o',label=f'{label} (DDIM η=0)')
    refs=reference_ddpm_rows()
    for key,label,_,_ in MODEL_INFO:
        if key in refs:
            ax.axhline(refs[key],linestyle='--',alpha=.55,label=f'{label} (DDPM 1000)')
    ax.set_xscale('log');ax.set_xticks(STEPS);ax.set_xticklabels([str(s) for s in STEPS]);ax.invert_xaxis()
    ax.set_xlabel('Numero de pasos (DDIM determinista)');ax.set_ylabel('KID × 1000 (menor es mejor)')
    ax.set_title('Calidad generativa frente al numero de pasos')
    ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'kid_vs_pasos.png',dpi=200);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.5,5.2))
    for key,label,_,_ in MODEL_INFO:
        rr=sorted((x for x in rows if x['model_key']==key and x['timed_images']==N), key=lambda r:r['steps'])
        if rr:
            ax.plot([r['steps'] for r in rr],[r['effective_images_per_second'] for r in rr],marker='o',label=label)
    ax.set_xscale('log');ax.set_xticks(STEPS);ax.set_xticklabels([str(s) for s in STEPS]);ax.invert_xaxis()
    ax.set_xlabel('Numero de pasos DDIM');ax.set_ylabel('Imagenes/s (muestreo de lotes; sin cargar modelos)')
    ax.set_title('Velocidad observada de muestreo (dependiente de GPU/carga)')
    ax.grid(alpha=.25);ax.legend(fontsize=9)
    fig.tight_layout();fig.savefig(OUT/'velocidad_vs_pasos.png',dpi=200);plt.close(fig)
    fig,axes=plt.subplots(2,len(STEPS),figsize=(15,6))
    for mi,(key,label,_,_) in enumerate(MODEL_INFO):
        for j,steps in enumerate(STEPS):
            v=next(x for x in VARIANTS if x['model']==key and x['steps']==steps)
            b=torch.load(chunk_path(v,0),map_location='cpu',weights_only=True)
            imgs=b[:4]
            # 2x5 celdas, en cada celda 2x2 mosaico de las mismas 4 semillas
            mosaic=torch.cat([torch.cat([imgs[0],imgs[1]],dim=2),
                              torch.cat([imgs[2],imgs[3]],dim=2)],dim=1)
            axes[mi,j].imshow(mosaic.permute(1,2,0).numpy(),interpolation='nearest')
            axes[mi,j].axis('off');axes[mi,j].set_title(f'{steps} pasos',fontsize=10)
        axes[mi,0].set_ylabel(label,fontsize=9)
    fig.suptitle('Primeras cuatro semillas pareadas por modelo y pasos DDIM η=0')
    fig.tight_layout();fig.savefig(OUT/'muestras_pareadas_ddim.png',dpi=200);plt.close(fig)


def kid():
    if not torch.cuda.is_available():
        raise RuntimeError('El KID debe calcularse en nodo GPU SLURM.')
    verify_all()
    from torchmetrics.image.kid import KernelInceptionDistance
    from torchvision.datasets import CIFAR10
    device=torch.device('cuda:0')
    metric=KernelInceptionDistance(feature=2048,subsets=old.KID_SUBSETS,
            subset_size=old.KID_SUBSET_SIZE,normalize=False,reset_real_features=False).to(device)
    cifar=CIFAR10(root=str(ROOT/'data'),train=False,download=False)
    real=cifar.data
    with torch.inference_mode():
        for i in range(0,len(real),old.KID_BATCH_SIZE):
            batch=torch.from_numpy(real[i:i+old.KID_BATCH_SIZE].copy()).permute(0,3,1,2).contiguous().to(device)
            metric.update(batch,real=True)
    rows=[]
    for v in VARIANTS:
        with torch.inference_mode():
            print('Evaluando KID:',v['key'],flush=True)
            for i in range(nchunks()):
                b=torch.load(chunk_path(v,i),map_location='cpu',weights_only=True).to(device)
                metric.update(b,real=False)
            torch.manual_seed(KID_SEED);torch.cuda.manual_seed_all(KID_SEED)
            mean,std=metric.compute()
        r={
            'variant':v['key'],'model':v['label'],'model_key':v['model'],
            'sampler':'DDIM deterministic','steps':v['steps'],'eta':0,
            'KID_mean':float(mean.cpu()),'KID_std_subsets':float(std.cpu()),
            'KID_x1000':float(mean.cpu())*1000,
            'n_generated':N,'n_real':len(real),'subsets':old.KID_SUBSETS,
            'subset_size':old.KID_SUBSET_SIZE,'feature_dim':2048,
            **timing_summary(v)
        }
        rows.append(r);write_csv(rows)
        print(f'KID x 1000 = {r["KID_x1000"]:.4f}',flush=True)
        metric.reset()  # reset_real_features=False conserva muestras reales
    plot(rows)
    print('KID DDIM COMPLETADO:',OUT,flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',required=True,choices=['check','selftest','status','sample','kid'])
    p.add_argument('--variant-index',type=int,choices=range(len(VARIANTS)))
    a=p.parse_args()
    if a.mode=='check':check()
    elif a.mode=='selftest':selftest()
    elif a.mode=='status':status()
    elif a.mode=='sample':
        if a.variant_index is None: p.error('--variant-index 0..9 obligatorio en sample')
        sample(a.variant_index)
    elif a.mode=='kid':kid()


if __name__=='__main__':
    main()
