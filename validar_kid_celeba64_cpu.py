#!/usr/bin/env python3
"""IC bootstrap KID y control real-vs-real CelebA64 sin GPU.

Usa features Inception precalculadas por evaluar_celeba64.py.
No importa torch ni numpy en --mode check/status (uso en login).
"""
import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'Evaluacion CelebA64 KID Vecinos'
FEATURES = SOURCE / 'features'
OUT = ROOT / 'Incertidumbre KID CelebA64 CPU'
N_TEST, N_FAKE, N_TRAIN_SUB = 10000, 5000, 5000
DIM = 2048
SEED = 20261031
REPS = 400
CI_LEVEL = 0.95


def atomic_json(path, data):
    path.parent.mkdir(exist_ok=True, parents=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def write_csv(path, rows):
    path.parent.mkdir(exist_ok=True, parents=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    os.replace(tmp, path)


def source_protocol():
    p = SOURCE / 'protocolo.json'
    if not p.is_file():
        raise FileNotFoundError('No existe protocolo original: ' + str(p))
    doc = json.loads(p.read_text(encoding='utf-8'))
    required = ('checkpoint_sha256', 'split_sha256', 'feature_dim')
    missing = [k for k in required if k not in doc]
    if missing:
        raise RuntimeError('Protocolo incompleto: ' + str(missing))
    if doc['feature_dim'] != DIM:
        raise RuntimeError('Dimensión Inception inesperada.')
    return doc


def protocol_doc(src):
    return {
        'experiment': 'celeba64_cpu_kid_bootstrap_and_real_baseline_v1',
        'model_checkpoint_sha256': src['checkpoint_sha256'],
        'split_sha256': src['split_sha256'],
        'source_experiment': src.get('experiment'),
        'source_fake_count': N_FAKE,
        'reference_test_count': N_TEST,
        'train_control_count': N_TRAIN_SUB,
        'train_subset_seed': SEED,
        'n_bootstrap': REPS,
        'confidence_level': CI_LEVEL,
        'inception_dim': DIM,
        'kernel': '(1 + x dot y / 2048)**3, estimador U sin sesgo',
        'bootstrap': 'remuestreo independiente por conjunto, misma muestra test en dos comparaciones',
        'scope': 'incertidumbre de muestreo condicional a checkpoint, Inception y features guardadas',
        'real_baseline': '5000 train aleatorias versus las mismas 10000 test (no mide ausencia de memorización)',
    }


def check():
    print('COMPROBACION LIGERA: no carga tensores ni PyTorch')
    try:
        p = source_protocol()
        print('Protocolo original:', p.get('experiment'))
        for name in ('fake', 'test', 'train'):
            f = FEATURES / (name + '.pt')
            print(f'{name:5s}:', 'SI' if f.is_file() else 'FALTA', f, f.stat().st_size if f.is_file() else '')
        original = SOURCE / 'kid_celeba64.csv'
        print('KID original:', 'SI' if original.is_file() else 'FALTA')
        if not all((FEATURES/(name+'.pt')).is_file() for name in ('fake','test','train')):
            raise FileNotFoundError('Faltan caches Inception. Completa evaluar_celeba64.py --mode evaluate primero.')
        print('CHECK OK: listo para enviar SLURM CPU')
    except Exception as e:
        print('CHECK ERROR:', repr(e))
        raise


def status():
    for p in ('protocolo.json','resumen_kid_celeba64_cpu.csv','bootstrap_replicas_kid_celeba64_cpu.csv',
              'kid_celeba64_IC95.png','diferencia_generado_vs_real_IC95.png'):
        f = OUT / p
        print(f'{p}:', 'SI' if f.is_file() else 'NO')


def load_features(kind, source, torch):
    fp = FEATURES / (kind + '.pt')
    data = torch.load(fp, map_location='cpu', weights_only=True)
    if not isinstance(data, dict) or 'features' not in data or 'meta' not in data:
        raise RuntimeError('Formato de cache inesperado: ' + str(fp))
    meta = data['meta']
    expected_n = {'fake': N_FAKE, 'test': N_TEST, 'train': 50000}[kind]
    if meta.get('kind') != kind or meta.get('n') != expected_n or meta.get('feature_dim') != DIM:
        raise RuntimeError(f'Cache {kind}: metadatos incompatibles: {meta}')
    if meta.get('checkpoint_sha256') != source['checkpoint_sha256'] or meta.get('split_sha256') != source['split_sha256']:
        raise RuntimeError('Caches de un modelo o split distinto, abortando sin mezclar resultados')
    if kind == 'fake' and not meta.get('fake_fingerprint'):
        raise RuntimeError('Falta huella de muestras generadas')
    features = data['features']
    if features.shape != (expected_n, DIM) or not torch.isfinite(features).all():
        raise RuntimeError(f'Features de {kind} corruptas')
    print('Cargado',kind,tuple(features.shape),flush=True)
    return features.cpu().numpy().astype('float32', copy=False)


def gram_poly(X, Y, torch, block=256):
    """Kernel polinómico grado 3, bloques para evitar temporales grandes en RAM."""
    import numpy as np
    A = torch.from_numpy(np.ascontiguousarray(X))
    B = torch.from_numpy(np.ascontiguousarray(Y)).T.contiguous()
    n, m = len(X), len(Y)
    K = np.empty((n, m), dtype=np.float32)
    tic = time.monotonic()
    for a in range(0, n, block):
        c = min(a+block,n)
        Q = (A[a:c] @ B).div_(DIM).add_(1.0).pow_(3.0)
        K[a:c, :] = Q.numpy()
        if (a // block) % 10 == 0:
            print(f'  Gram {c}/{n}, {m} columnas, {time.monotonic()-tic:.1f}s',flush=True)
    return K


def weight_gram(K, counts, block_reps=40):
    """Devuelve parte U por réplica. K precomputado en muestras originales;
    counts son multiplicidades bootstrap, no reemplazar diagonales de K a cero.
    """
    import numpy as np
    n = int(counts[0].sum())
    out = np.empty(counts.shape[0], dtype=np.float64)
    diag = np.asarray(K.diagonal(), dtype=np.float64)
    for i in range(0, len(counts), block_reps):
        j = min(i + block_reps, len(counts))
        w = np.ascontiguousarray(counts[i:j].T.astype(np.float32))
        Kw = K @ w
        numer = np.sum(w * Kw, axis=0, dtype=np.float64) - diag @ w.astype(np.float64)
        out[i:j] = numer / (n*(n-1))
    return out


def weight_cross(K, left_counts, right_counts, block_reps=40):
    import numpy as np
    n, m = int(left_counts[0].sum()), int(right_counts[0].sum())
    out = np.empty(left_counts.shape[0], dtype=np.float64)
    for i in range(0,len(out),block_reps):
        j = min(i + block_reps,len(out))
        wl = np.ascontiguousarray(left_counts[i:j].T.astype(np.float32))
        wr = np.ascontiguousarray(right_counts[i:j].T.astype(np.float32))
        out[i:j] = np.sum(wl * (K @ wr), axis=0, dtype=np.float64) / (n*m)
    return out


def kid_complete(Kxx,Kyy,Kxy):
    import numpy as np
    n,m = Kxx.shape[0],Kyy.shape[0]
    xx=(np.sum(Kxx,dtype=np.float64)-np.trace(Kxx,dtype=np.float64))/(n*(n-1))
    yy=(np.sum(Kyy,dtype=np.float64)-np.trace(Kyy,dtype=np.float64))/(m*(m-1))
    xy=np.sum(Kxy,dtype=np.float64)/(n*m)
    return float(xx+yy-2*xy)


def kid_bootstrap(Kxx,Kyy,Kxy,w_x,w_y,shared_x=None):
    xx=shared_x if shared_x is not None else weight_gram(Kxx,w_x)
    yy=weight_gram(Kyy,w_y)
    xy=weight_cross(Kxy,w_x,w_y)
    return xx+yy-2*xy


def percentile_ci(draws):
    import numpy as np
    lo,hi=np.quantile(draws,[0.025,0.975])
    return float(lo),float(hi)


def selftest():
    import numpy as np
    # Prueba exacta con kernel grado 3 (dimension pequeña). Verifica duplicados bootstrap.
    rng = np.random.default_rng(1985)
    x=rng.normal(size=(7,4))
    y=rng.normal(size=(6,4))
    # Para selftest usamos kernel/4, no la constante global DIM de Inception.
    fun=lambda a,b: (1 + (a@b.T)/4)**3
    xx,yy,xy=fun(x,x),fun(y,y),fun(x,y)
    full=kid_complete(xx,yy,xy)
    ix=np.array([0,1,1,3,5,5,6]); iy=np.array([0,0,2,3,3,5]);
    brute=kid_complete(xx[np.ix_(ix,ix)],yy[np.ix_(iy,iy)],xy[np.ix_(ix,iy)])
    wx=np.bincount(ix,minlength=len(x))[None,:]
    wy=np.bincount(iy,minlength=len(y))[None,:]
    estimate=kid_bootstrap(xx.astype('float32'), yy.astype('float32'), xy.astype('float32'),wx,wy)[0]
    assert abs(brute-estimate)<1e-5,(brute,estimate)
    assert np.isfinite(full)
    a=np.ones((4,3)); b=np.ones((4,3))
    assert abs(kid_complete(fun(a,a),fun(b,b),fun(a,b)))<1e-10
    print('SELFTEST OK: U-estadistico, duplicados bootstrap, termo cruzado, conjuntos identicos')


def run():
    import numpy as np
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    start=time.monotonic()
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','4')))
    source=source_protocol()
    proto=protocol_doc(source)
    prot_path=OUT/'protocolo.json'
    if prot_path.is_file() and json.loads(prot_path.read_text(encoding='utf-8')) != proto:
        raise RuntimeError('La carpeta de salida pertenece a otro protocolo. No se mezclan resultados.')
    OUT.mkdir(parents=True,exist_ok=True)
    atomic_json(prot_path,proto)
    x=load_features('test',source,torch)
    y=load_features('fake',source,torch)
    train_all=load_features('train',source,torch)
    rng=np.random.default_rng(SEED)
    selection=np.sort(rng.choice(len(train_all),size=N_TRAIN_SUB,replace=False))
    z=np.ascontiguousarray(train_all[selection])
    del train_all
    select_path=OUT/'indices_train_control.npy'
    if select_path.is_file():
        old=np.load(select_path)
        if not np.array_equal(old,selection):raise RuntimeError('Indice train distinto, no mezclar')
    else:
        np.save(select_path,selection)

    print('Calculando Gram test-test',flush=True)
    Kxx=gram_poly(x,x,torch)
    print('Calculando Gram fake-fake',flush=True)
    Kyy=gram_poly(y,y,torch)
    print('Calculando Gram train-train',flush=True)
    Kzz=gram_poly(z,z,torch)
    print('Calculando Gram test-fake',flush=True)
    Kxy=gram_poly(x,y,torch)
    print('Calculando Gram test-train',flush=True)
    Kxz=gram_poly(x,z,torch)
    del x,y,z
    kid_gen=kid_complete(Kxx,Kyy,Kxy)
    kid_control=kid_complete(Kxx,Kzz,Kxz)
    print(f'KID gen x1000={kid_gen*1000:.5f}, real-vs-real x1000={kid_control*1000:.5f}',flush=True)
    orig=SOURCE/'kid_celeba64.csv'
    if orig.is_file():
        with orig.open(newline='',encoding='utf-8') as f:
            old=float(next(csv.DictReader(f))['kid_u_complete'])
        diff=abs(kid_gen-old)
        print(f'Control numerico con KID anterior: delta x1000={diff*1000:.6f}',flush=True)
        if diff>0.0005:
            raise RuntimeError('KID CPU no coincide con KID original: posible problema de features/protocolo.')
    counts_x = rng.multinomial(N_TEST, np.full(N_TEST,1/N_TEST),size=REPS).astype(np.int16)
    counts_y = rng.multinomial(N_FAKE, np.full(N_FAKE,1/N_FAKE),size=REPS).astype(np.int16)
    counts_z = rng.multinomial(N_TRAIN_SUB, np.full(N_TRAIN_SUB,1/N_TRAIN_SUB),size=REPS).astype(np.int16)
    print('Bootstrap sobre Gram; muestra test compartida entre contrastes...',flush=True)
    boot_xx=weight_gram(Kxx,counts_x)
    boot_fake=kid_bootstrap(Kxx,Kyy,Kxy,counts_x,counts_y,shared_x=boot_xx)
    boot_control=kid_bootstrap(Kxx,Kzz,Kxz,counts_x,counts_z,shared_x=boot_xx)
    boot_delta=boot_fake-boot_control
    delta=kid_gen-kid_control
    for label,v in [('Generadas vs test',kid_gen),('Train vs test',kid_control),('Diferencia',delta)]:
        print(label,'KID x1000',round(v*1000,5),flush=True)
    summaries=[]
    for label,estimate,draw in [('Generadas vs test',kid_gen,boot_fake),
                                 ('Train real vs test real',kid_control,boot_control),
                                 ('Generadas menos real-versus-real',delta,boot_delta)]:
        lo,hi=percentile_ci(draw)
        summaries.append({'comparacion':label,'kid_u_x1000':estimate*1000,
                          'ic95_inf_x1000':lo*1000,'ic95_sup_x1000':hi*1000,
                          'replicas_bootstrap':REPS,'n_test':N_TEST,
                          'n_otro':N_FAKE if label=='Generadas vs test' else N_TRAIN_SUB,
                          'nota':'IC percentil condicional al checkpoint; baseline cerca de 0 requiere cautela bajo hipotesis nula'})
    write_csv(OUT/'resumen_kid_celeba64_cpu.csv', summaries)
    write_csv(OUT/'bootstrap_replicas_kid_celeba64_cpu.csv',[
        {'replica':i,'KID_generadas_x1000':float(boot_fake[i]*1000),
         'KID_train_vs_test_x1000':float(boot_control[i]*1000),
         'delta_x1000':float(boot_delta[i]*1000)} for i in range(REPS)])

    def fig_error_bar(names,vals,ranges,file,title):
        fig,ax=plt.subplots(figsize=(8.6,3.6))
        pos=np.arange(len(names))
        low=np.array([v-r[0] for v,r in zip(vals,ranges)])
        up=np.array([r[1]-v for v,r in zip(vals,ranges)])
        ax.errorbar(vals,pos,xerr=np.array([np.maximum(0,low),np.maximum(0,up)]),fmt='o',capsize=5)
        ax.axvline(0,color='gray',linestyle='--',linewidth=1)
        ax.set_yticks(pos,labels=names)
        ax.invert_yaxis()
        ax.set_xlabel('KID U completo × 1000 (intervalo bootstrap percentil 95 %)')
        ax.set_title(title)
        ax.grid(axis='x',alpha=.2)
        fig.tight_layout()
        fig.savefig(OUT/file,dpi=250)
        plt.close(fig)

    fig_error_bar(['Generadas vs TEST','TRAIN real vs TEST'],
                  [kid_gen*1000,kid_control*1000],
                  [(percentile_ci(boot_fake)[0]*1000,percentile_ci(boot_fake)[1]*1000),
                   (percentile_ci(boot_control)[0]*1000,percentile_ci(boot_control)[1]*1000)],
                  'kid_celeba64_IC95.png','Calidad generativa y referencia real-versus-real')
    fig_error_bar(['Generadas − real-versus-real'],[delta*1000],
                  [(percentile_ci(boot_delta)[0]*1000,percentile_ci(boot_delta)[1]*1000)],
                  'diferencia_generado_vs_real_IC95.png','Diferencia de KID respecto a referencia real-versus-real')
    print('FINALIZADO:',OUT,'tiempo_s=',round(time.monotonic()-start,1),flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',choices=['selftest','check','status','run'],required=True)
    a=ap.parse_args()
    {'selftest':selftest,'check':check,'status':status,'run':run}[a.mode]()

if __name__=='__main__':
    main()
