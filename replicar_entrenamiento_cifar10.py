#!/usr/bin/env python3
"""Entrenamientos independientes CIFAR10 con estado reanudable por epoca.
Necesita ~/tesis/evaluar_kid_cifar10_5000.py (clases originales).
En login: --mode check/selftest/status. En GPU: --mode train --run-index {0,1,2,3}.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import evaluar_kid_cifar10_5000 as original

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'Replicas Entrenamiento CIFAR10'
ARCHS = [('attention', original.SimpleUNetAttention), ('resunet', original.ResUNetAttention)]
SEEDS = (20261101, 20261102)
RUNS = [(arch, seed) for seed in SEEDS for arch, _ in ARCHS]
DEFAULT = {'epochs':200, 'batch_size':64, 'lr':1e-4, 'ema_decay':0.999,
           'optimizer':'Adam', 'weight_decay':0.0, 't_steps':1000,
           'beta_start':1e-4, 'beta_end':0.02, 'data':'CIFAR10 train [-1,1]',
           'ema_initialization':'copy_initial_weights', 'ema_update':'every_optimizer_step',
           'shuffle':'independent DataLoader generator; workers=0'}


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def key(i):
    arch,seed=RUNS[i]
    return f'{arch}_seed{seed}'


def folder(i): return OUT/key(i)


def atomic_json(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix('.json.tmp')
    temp.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    temp.replace(p)


def atomic_torch(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_name(p.name+'.tmp')
    torch.save(obj,temp)
    temp.replace(p)


def spec(i):
    d=dict(DEFAULT); d.update(architecture=RUNS[i][0],seed=RUNS[i][1],
                             reference_sha256=sha(ROOT/'evaluar_kid_cifar10_5000.py'))
    return d


def check():
    assert original.T==1000 and original.BETA_START==1e-4 and original.BETA_END==0.02
    print('Arquitecturas originales:',ROOT/'evaluar_kid_cifar10_5000.py')
    p=ROOT/'data'/'cifar-10-batches-py'/'data_batch_1'
    print('CIFAR10 train:', 'OK' if p.is_file() else 'FALTA',p)
    if not p.is_file(): raise FileNotFoundError(p)
    print('PROTOCOLO NUEVO (verificar contra trainer original de 200 epocas):',json.dumps(DEFAULT,indent=2))
    for i in range(4):
        r=folder(i); manifest=r/'protocolo.json'
        if manifest.exists() and json.loads(manifest.read_text())!=spec(i):
            raise RuntimeError('Existe protocolo incompatible: '+str(manifest))
        print(i,key(i),'salida',r)
    print('CHECK OK; comprobar parametros EMA y optimizador originales antes de interpretar réplicas como equivalentes.')


def status():
    for i in range(4):
        p=folder(i)/'estado.json'
        if p.exists():
            d=json.loads(p.read_text());print(f'{i} {key(i)}: {d.get("epochs_done",0)}/{DEFAULT["epochs"]} epocas, final={d.get("complete",False)}')
        else:print(f'{i} {key(i)}: sin comenzar')


def selftest():
    assert len(RUNS)==4 and len(set(RUNS))==4
    betas=torch.linspace(DEFAULT['beta_start'],DEFAULT['beta_end'],DEFAULT['t_steps'])
    ab=torch.cumprod(1-betas,0)
    assert 0<ab[-1]<ab[0]<1 and torch.isfinite(ab).all()
    # Verifica la regla EMA aritmetica sin cargar redes pesadas ni dataset
    a=torch.tensor([2.]);b=torch.tensor([4.]); decay=0.9
    a.mul_(decay).add_(b,alpha=1-decay)
    assert torch.allclose(a,torch.tensor([2.2]))
    print('SELFTEST OK: cronograma DDPM, configuraciones y regla EMA.')


def set_rng(seed):
    random.seed(seed); np.random.seed(seed%(2**32));torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)


def rng_state(loader_gen):
    return {'python':random.getstate(),'numpy':np.random.get_state(),
            'torch_cpu':torch.get_rng_state(),'torch_cuda':torch.cuda.get_rng_state_all(),
            'dataloader':loader_gen.get_state()}


def restore_rng(d,loader_gen):
    random.setstate(d['python']);np.random.set_state(d['numpy'])
    torch.set_rng_state(d['torch_cpu']);torch.cuda.set_rng_state_all(d['torch_cuda'])
    loader_gen.set_state(d['dataloader'])


def progress_csv(path,rows):
    tmp=path.with_suffix('.tmp')
    with tmp.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=['epoch','train_mse','steps','epoch_seconds'])
        w.writeheader();w.writerows(rows)
    tmp.replace(path)


def train(i,max_seconds):
    if not torch.cuda.is_available():raise RuntimeError('El entrenamiento requiere GPU asignada mediante SLURM')
    from torchvision import datasets,transforms
    arch,seed=RUNS[i]; rd=folder(i);rd.mkdir(parents=True,exist_ok=True)
    protocol=spec(i); manifest=rd/'protocolo.json'
    if manifest.exists() and json.loads(manifest.read_text())!=protocol:
        raise RuntimeError('CAMBIO DE PROTOCOLO: no continuar en carpeta de pesos existentes')
    atomic_json(manifest,protocol)
    if (rd/'ema_final.pt').exists():print('Ya completo:',key(i));return
    device=torch.device('cuda')
    set_rng(seed)
    model=dict(ARCHS)[arch]().to(device)
    ema=dict(ARCHS)[arch]().to(device);ema.load_state_dict(model.state_dict())
    ema.eval();ema.requires_grad_(False)
    optimizer=torch.optim.Adam(model.parameters(),lr=DEFAULT['lr'],weight_decay=DEFAULT['weight_decay'])
    loader_gen=torch.Generator(device='cpu').manual_seed(seed+67)
    tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize((.5,.5,.5),(.5,.5,.5))])
    data=datasets.CIFAR10(root=str(ROOT/'data'),train=True,download=False,transform=tf)
    loader=DataLoader(data,batch_size=DEFAULT['batch_size'],shuffle=True,num_workers=0,
                      pin_memory=True,generator=loader_gen,drop_last=False)
    betas=torch.linspace(DEFAULT['beta_start'],DEFAULT['beta_end'],DEFAULT['t_steps'],device=device)
    abar=torch.cumprod(1-betas,dim=0)
    ckpath=rd/'checkpoint.pt';rows=[];start_epoch=0;global_step=0
    if ckpath.exists():
        ck=torch.load(ckpath,map_location='cpu',weights_only=False)
        if ck.get('protocol')!=protocol:raise RuntimeError('El checkpoint no coincide con el protocolo')
        model.load_state_dict(ck['model']);ema.load_state_dict(ck['ema']);optimizer.load_state_dict(ck['optimizer'])
        start_epoch=ck['epoch_done'];global_step=ck['global_step'];rows=ck['metrics']
        restore_rng(ck['rng'],loader_gen)
        del ck
        print('Reanudando:',key(i),'desde epoca',start_epoch,flush=True)
    else:print('Entrenamiento NUEVO:',key(i),flush=True)
    t_start=time.monotonic()
    for epoch in range(start_epoch,DEFAULT['epochs']):
        model.train();begin=time.monotonic();sum_loss=0.;n_seen=0
        for x0,_ in loader:
            x0=x0.to(device,non_blocking=True)
            t=torch.randint(0,DEFAULT['t_steps'],(x0.shape[0],),device=device)
            noise=torch.randn_like(x0)
            a=abar[t][:,None,None,None]
            xt=a.sqrt()*x0+(1-a).sqrt()*noise
            pred=model(xt,t)
            loss=F.mse_loss(pred,noise)
            optimizer.zero_grad(set_to_none=True)
            loss.backward();optimizer.step()
            with torch.no_grad():
                d=DEFAULT['ema_decay']
                for q,p in zip(ema.parameters(),model.parameters()):
                    q.mul_(d).add_(p,alpha=1-d)
                for q,p in zip(ema.buffers(),model.buffers()):q.copy_(p)
            bs=x0.shape[0];sum_loss+=loss.item()*bs;n_seen+=bs;global_step+=1
        sec=time.monotonic()-begin
        rows.append({'epoch':epoch+1,'train_mse':sum_loss/n_seen,'steps':global_step,'epoch_seconds':sec})
        state={'protocol':protocol,'epoch_done':epoch+1,'global_step':global_step,
               'model':model.state_dict(),'ema':ema.state_dict(),'optimizer':optimizer.state_dict(),
               'rng':rng_state(loader_gen),'metrics':rows}
        atomic_torch(ckpath,state)
        progress_csv(rd/'perdidas_entrenamiento.csv',rows)
        complete=epoch+1==DEFAULT['epochs']
        atomic_json(rd/'estado.json',{'key':key(i),'epochs_done':epoch+1,'total_epochs':DEFAULT['epochs'],
                                      'complete':complete,'last_train_mse':rows[-1]['train_mse']})
        print(f'{key(i)} EPOCA {epoch+1}/{DEFAULT["epochs"]} MSE={rows[-1]["train_mse"]:.6f} duracion={sec:.1f}s',flush=True)
        if complete:
            atomic_torch(rd/'ema_final.pt',ema.state_dict())
            atomic_torch(rd/'raw_final.pt',model.state_dict())
            print('COMPLETADO:',key(i),flush=True);break
        # Margen para guardar checkpoints y liberar GPU antes del tiempo de SLURM
        elapsed=time.monotonic()-t_start
        if elapsed+max(sec*1.3,240)>max_seconds:
            print('PAUSA SEGURA: limite de tiempo cercano. Reenviar array tras consultar status.',flush=True)
            break


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',choices=['check','status','selftest','train'],required=True)
    ap.add_argument('--run-index',type=int,choices=range(4))
    ap.add_argument('--time-budget-seconds',type=int,default=13200)
    args=ap.parse_args()
    if args.mode=='check':check()
    elif args.mode=='selftest':selftest()
    elif args.mode=='status':status()
    else:
        if args.run_index is None:ap.error('--run-index obligatorio')
        train(args.run_index,args.time_budget_seconds)

if __name__=='__main__':main()
