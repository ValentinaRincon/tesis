#!/usr/bin/env python3
"""Muestreo y KID DDIM eta=0 250 pasos para cuatro nuevos entrenamientos.
En login: --mode check|status|selftest. GPU: --mode sample --run-index 0..3, --mode kid.
"""
import argparse
import csv
import json
import math
import statistics
import time
import hashlib
from pathlib import Path

import torch
import comparar_ddim_cifar10 as ddim
import evaluar_kid_cifar10_5000 as base
import replicar_entrenamiento_cifar10 as rep

ROOT=Path(__file__).resolve().parent
OUT=rep.OUT/'Evaluacion DDIM250'
N=5000
BATCH=32
STEPS=250
SEED=base.SAMPLE_SEED


def pchunk(i,k):return OUT/'chunks'/rep.key(i)/f'batch_{k:04d}.pt'

def n_chunks():return math.ceil(N/BATCH)

def size(k):return min(BATCH,N-BATCH*k)

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''):h.update(c)
    return h.hexdigest()


def manifest(i):
    w=rep.folder(i)/'ema_final.pt'
    return {'key':rep.key(i),'model_sha256':sha(w),'reference_ddim_sha256':sha(ROOT/'comparar_ddim_cifar10.py'),
            'ddim_eta':0,'steps':STEPS,'seed':SEED,'images':N,'batch_size':BATCH,
            'timetable':ddim.timetable(STEPS)}


def ensure_manifest(i):
    target=OUT/'chunks'/rep.key(i)/'manifest.json'
    want=manifest(i)
    if target.exists() and json.loads(target.read_text())!=want:
        raise RuntimeError('Manifiesto incompatible. No mezcles generaciones con pesos distintos: '+str(target))
    if not target.exists():ddim.atomic_json(target,want)


def valid(p,n):
    if not p.is_file():return False
    try:
        x=torch.load(p,map_location='cpu',weights_only=True)
        return isinstance(x,torch.Tensor) and x.dtype==torch.uint8 and tuple(x.shape)==(n,3,32,32)
    except Exception:return False


def status():
    for i in range(4):
        n=sum(pchunk(i,j).is_file() and pchunk(i,j).stat().st_size>0 for j in range(n_chunks()))
        weight=rep.folder(i)/'ema_final.pt'
        print(f'{i} {rep.key(i)}: pesos={weight.is_file()} chunks={n}/{n_chunks()}')
    print('KID disponible:',(OUT/'kid_replicas.csv').is_file())


def check():
    if (not (ROOT/'comparar_ddim_cifar10.py').exists() or
            not (ROOT/'evaluar_kid_cifar10_5000.py').exists()):
        raise RuntimeError('Faltan scripts originales DDIM o arquitecturas.')
    test=ROOT/'data'/'cifar-10-batches-py'/'test_batch'
    if not test.exists():raise FileNotFoundError(test)
    print('CHECK OK | salida:',OUT)
    for i in range(4):print(i,rep.key(i),'pesos generados?',(rep.folder(i)/'ema_final.pt').is_file())


def selftest():
    assert ddim.timetable(250)[0]==0 and ddim.timetable(250)[-1]==999
    assert n_chunks()==157 and size(156)==8
    print('SELFTEST OK: malla DDIM, conteo de lotes y pares de semillas.')


@torch.inference_mode()
def sample(i):
    if not torch.cuda.is_available():raise RuntimeError('Requiere GPU mediante SLURM')
    weight=rep.folder(i)/'ema_final.pt'
    if not weight.exists():raise FileNotFoundError(f'Entrenamiento no ha terminado: {weight}')
    ensure_manifest(i)
    dev=torch.device('cuda')
    cls=dict(rep.ARCHS)[rep.RUNS[i][0]]
    model=cls().to(dev)
    model.load_state_dict(base.load_state_dict(weight,dev),strict=True)
    model.eval()
    schedule=base.get_schedule(dev)
    missing=[j for j in range(n_chunks()) if not valid(pchunk(i,j),size(j))]
    print(rep.key(i),'pendientes',len(missing),'de',n_chunks(),flush=True)
    for j in missing:
        gen=torch.Generator(device=dev).manual_seed(SEED+1009*j)
        torch.cuda.synchronize();start=time.perf_counter()
        x=ddim.generate_batch(model,size(j),gen,schedule,dev,STEPS)
        torch.cuda.synchronize();secs=time.perf_counter()-start
        path=pchunk(i,j);path.parent.mkdir(parents=True,exist_ok=True)
        tmp=path.with_suffix('.tmp');torch.save(x,tmp);tmp.replace(path)
        ddim.atomic_json(OUT/'times'/rep.key(i)/f'batch_{j:04d}.json',{'n':size(j),'seconds':secs})
        if j%20==0 or j==n_chunks()-1:print(rep.key(i),'lote',j+1,'/',n_chunks(),flush=True)
    print('SAMPLE COMPLETO:',rep.key(i),flush=True)


def table_csv(name,rows,cols):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp')
    with q.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)
    q.replace(p)


@torch.inference_mode()
def kid():
    if not torch.cuda.is_available():raise RuntimeError('KID requiere GPU SLURM')
    for i in range(4):
        if not (OUT/'chunks'/rep.key(i)/'manifest.json').exists():raise RuntimeError('Sin manifiesto '+rep.key(i))
        ensure_manifest(i)
        for j in range(n_chunks()):
            if not valid(pchunk(i,j),size(j)):raise RuntimeError(f'Falta lote {rep.key(i)} {j}')
    from torchmetrics.image.kid import KernelInceptionDistance
    from torchvision.datasets import CIFAR10
    dev=torch.device('cuda')
    metric=KernelInceptionDistance(feature=2048,subsets=100,subset_size=500,
                                  normalize=False,reset_real_features=False).to(dev)
    real=CIFAR10(root=str(ROOT/'data'),train=False,download=False).data
    for j in range(0,len(real),64):
        x=torch.from_numpy(real[j:j+64].copy()).permute(0,3,1,2).contiguous().to(dev)
        metric.update(x,real=True)
    records=[]
    for i in range(4):
        for j in range(n_chunks()):
            x=torch.load(pchunk(i,j),weights_only=True,map_location='cpu').to(dev)
            metric.update(x,real=False)
        torch.manual_seed(base.METRIC_SEED);torch.cuda.manual_seed_all(base.METRIC_SEED)
        mean,std=metric.compute()
        r={'key':rep.key(i),'architecture':rep.RUNS[i][0],'seed':rep.RUNS[i][1],
           'steps':STEPS,'eta':0,'n_generated':N,'n_real':len(real),
           'KID_x1000':float(mean.cpu())*1000,'KID_std_subsets_x1000':float(std.cpu())*1000,
           'n_subsets':100,'subset_size':500}
        records.append(r)
        table_csv('kid_replicas.csv',records,list(r))
        print('KID:',r,flush=True)
        metric.reset()
    by={(x['architecture'],x['seed']):x for x in records}
    diffs=[]
    for s in rep.SEEDS:
        a=by[('attention',s)]['KID_x1000'];b=by[('resunet',s)]['KID_x1000']
        diffs.append({'seed':s,'attention_KID_x1000':a,'resunet_KID_x1000':b,
                      'difference_attention_minus_resunet':a-b})
    table_csv('diferencias_por_semilla.csv',diffs,list(diffs[0]))
    summary=[]
    for arch in ('attention','resunet'):
        values=[by[(arch,s)]['KID_x1000'] for s in rep.SEEDS]
        summary.append({'group':arch,'n_trainings':len(values),'mean_KID_x1000':statistics.mean(values),
                        'sd_between_trainings_x1000':statistics.stdev(values)})
    values=[r['difference_attention_minus_resunet'] for r in diffs]
    summary.append({'group':'paired_attention_minus_resunet','n_trainings':2,
                    'mean_KID_x1000':statistics.mean(values),
                    'sd_between_trainings_x1000':statistics.stdev(values)})
    table_csv('resumen_replicas.csv',summary,list(summary[0]))
    print('KID COMPLETADO. n=2 pares; NO concluir significancia de variacion entre entrenamientos.',flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',required=True,choices=['check','selftest','status','sample','kid'])
    p.add_argument('--run-index',type=int,choices=range(4))
    args=p.parse_args()
    if args.mode=='check':check()
    elif args.mode=='selftest':selftest()
    elif args.mode=='status':status()
    elif args.mode=='sample':
        if args.run_index is None:p.error('--run-index obligatorio')
        sample(args.run_index)
    else:kid()

if __name__=='__main__':main()
