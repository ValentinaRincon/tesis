#!/usr/bin/env python3
"""Comparación controlada de muestreadores EMA CelebA128, con KID opcional.

--mode check/status/selftest son seguros para login (sin importar torch).
--mode preview/generate/evaluate requieren un nodo GPU de SLURM.
No modifica checkpoints ni salidas de experimentos anteriores.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CK = {
    60: ROOT / 'CelebA 128x128 Refinamiento' / 'checkpoint.pt',
    80: ROOT / 'CelebA 128x128 Refinamiento80' / 'checkpoint.pt',
}
OUT = ROOT / 'CelebA 128x128 Mejor Calidad'
BASE = ROOT / 'entrenar_celeba128.py'
SPLIT = ROOT / 'data/celeba/list_eval_partition.txt'
TEST_CACHE = ROOT / 'Evaluacion CelebA128 ep40 vs ep60/features/test.pt'
METHODS = ('ddim250', 'ddim500', 'ddpm1000')
PREVIEW_N = 16
N_KID = 1024
BATCH = 16
MICROBATCH = 2
N_BOOT = 250
SEED_PREVIEW = 20261120
SEED_KID = 20261121
SEED_BOOT = 20261122
D = 2048


def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def atomic_json(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    os.replace(tmp, p)


def atomic_torch(p, obj, torch):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + f'.{os.getpid()}.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, p)


def ck_verified(epoch, torch):
    import entrenar_celeba128 as original
    checkpoint = torch.load(CK[epoch], map_location='cpu', weights_only=False)
    if int(checkpoint.get('epoch', -1)) != epoch:
        raise RuntimeError(f'Checkpoint {CK[epoch]} tiene época incorrecta')
    if epoch == 60:
        p = checkpoint.get('refinement_protocol', {})
        if p.get('initial_epoch') != 40 or p.get('target_epoch') != 60:
            raise RuntimeError('No es el checkpoint refinado 40 -> 60 esperado')
    if epoch == 80:
        p = checkpoint.get('quality_protocol', {})
        if p.get('initial_epoch') != 60 or p.get('target_epoch') != 80:
            raise RuntimeError('No es el checkpoint refinado 60 -> 80 esperado')
    model = original.ResUNet128().cuda().eval()
    model.load_state_dict(checkpoint['ema'], strict=True)
    del checkpoint
    return model, original


def spec(epoch, method, kind):
    return dict(experiment='sampler_celeba128_ema_quality_v1', epoch=epoch,
                sampler=method, kind=kind, source_sha256=digest(BASE),
                checkpoint_sha256=digest(CK[epoch]),
                crop='CenterCrop160_Resize128_RGB_Normalize(-1,1)',
                initial_noise='torch.randn CPU generator, seed=SEED+chunk; same initial xT across samplers',
                ddpm_noise='independent CUDA generator, seed=SEED+chunk+900000',
                seed=SEED_PREVIEW if kind == 'preview' else SEED_KID,
                n=PREVIEW_N if kind == 'preview' else N_KID,
                batch=BATCH, microbatch=MICROBATCH, betas='linear 1e-4 -> 0.02, T=1000',
                deterministic_ddim_eta=0,
                ddpm_variance='posterior beta_tilde, no x0 clipping')


def check():
    necessary = [ROOT/'entrenar_celeba64.py', BASE, CK[60], SPLIT]
    for p in necessary:
        print(('OK    ' if p.is_file() else 'FALTA '), p)
    s60 = CK[60].parent/'estado.json'
    if not s60.is_file(): necessary.append(s60)
    else:
        s = json.loads(s60.read_text(encoding='utf-8'))
        print('Época 60:', s.get('epoch'), 'completo:', s.get('complete'))
        if int(s.get('epoch', -1)) != 60 or not s.get('complete'):
            raise RuntimeError('La época 60 aún no está confirmada')
    if not all(p.is_file() for p in necessary):
        raise FileNotFoundError('Faltan archivos necesarios; revisar rutas de Hypatia')
    print('CHECK OK — sin importar torch ni cargar checkpoint')


def status():
    print('Resultados:', OUT)
    for epoch in (60,80):
        print('Checkpoint', epoch, CK[epoch].is_file())
        for method in METHODS:
            p=OUT/'previas'/f'ep{epoch:03d}_{method}.png'
            print('  preview',method, p.is_file())
            folder=OUT/'kid_muestras'/f'ep{epoch}_{method}'
            count=sum((folder/f'lote_{i:04d}.pt').is_file() for i in range(N_KID//BATCH))
            if count:print('  KID',method, f'{count*BATCH}/{N_KID} imágenes')
    print('KID época 60 (samplers):', (OUT/'KID_ep60_samplers.csv').is_file())
    print('KID época 60 vs 80:', (OUT/'KID_ep60_vs_80.csv').is_file())


def selftest():
    for steps in (250,500):
        ts=list(reversed([round(999*k/(steps-1)) for k in range(steps)]))
        assert len(set(ts)) == steps and ts[0] == 999 and ts[-1] == 0
    for epoch in (60,80):
        for method in METHODS:
            assert (method.startswith('ddim') and int(method[4:]) in (250,500)) or method=='ddpm1000'
    assert N_KID % BATCH == 0 and PREVIEW_N % MICROBATCH == 0
    print('SELFTEST OK: timesteps, lotes y opciones')


def require_gpu():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Este modo requiere GPU asignada por sbatch, no login')
    torch.backends.cuda.matmul.allow_tf32 = False
    return torch


def sample_batch(model, original, method, seed, n, torch):
    """Igual x_T por índice/semilla para todos los métodos; DDPM agrega ruido propio."""
    if n % MICROBATCH: raise ValueError('Lote debe ser múltiplo de microbatch')
    cpu_gen = torch.Generator(device='cpu').manual_seed(int(seed))
    xT = torch.randn((n,3,128,128),generator=cpu_gen,device='cpu')
    ddpm_rng = torch.Generator(device='cuda').manual_seed(int(seed)+900000)
    beta=torch.linspace(original.CONFIG['beta_start'], original.CONFIG['beta_end'],
                        original.CONFIG['t_steps'],device='cuda',dtype=torch.float32)
    alpha=1-beta
    abar=torch.cumprod(alpha,dim=0)
    if method=='ddpm1000':
        ts=list(range(999,-1,-1))
    else:
        steps=int(method[4:])
        ts=torch.linspace(0,999,steps,device='cpu').round().long().flip(0).tolist()
        if len(set(ts)) != steps:raise RuntimeError('Timesteps duplicados')
    result=[]
    model.eval()
    with torch.inference_mode():
        for offset in range(0,n,MICROBATCH):
            x=xT[offset:offset+MICROBATCH].cuda()
            for j,ti in enumerate(ts):
                t=torch.full((len(x),),int(ti),dtype=torch.long,device=x.device)
                with torch.autocast('cuda',dtype=torch.float16):
                    eps=model(x,t).float()
                a=abar[ti]
                if method=='ddpm1000':
                    mean=(x-beta[ti]/(1-a).sqrt()*eps)/alpha[ti].sqrt()
                    if ti>0:
                        abar_prev=abar[ti-1]
                        posterior_var=beta[ti]*(1-abar_prev)/(1-a)
                        z=torch.randn(x.shape,device=x.device,dtype=x.dtype,generator=ddpm_rng)
                        x=mean+posterior_var.sqrt()*z
                    else:x=mean
                else:
                    x0=(x-(1-a).sqrt()*eps)/a.sqrt()
                    an=abar[ts[j+1]] if j+1<len(ts) else a.new_tensor(1.)
                    x=an.sqrt()*x0+(1-an).sqrt()*eps
            result.append((x.clamp(-1,1).add(1).mul(127.5)).round().byte().cpu())
    return torch.cat(result,dim=0)


def save_grid(images,path):
    from torchvision.utils import make_grid,save_image
    path.parent.mkdir(parents=True,exist_ok=True)
    save_image(make_grid(images.float()/255, nrow=4,padding=2),path)


def make_collage(epoch):
    from PIL import Image,ImageDraw
    paths=[OUT/'previas'/f'ep{epoch:03d}_{m}.png' for m in METHODS]
    if not all(p.is_file() for p in paths):return
    imgs=[Image.open(p).convert('RGB') for p in paths]
    gap=14; header=38
    w=sum(im.width for im in imgs)+gap*(len(imgs)-1)
    h=max(im.height for im in imgs)+header
    pic=Image.new('RGB',(w,h),'white');d=ImageDraw.Draw(pic)
    x=0
    for m,im in zip(METHODS,imgs):
        d.text((x+6,12),f'Epoca {epoch} | {m.upper()}',fill='black')
        pic.paste(im,(x,header));x+=im.width+gap
    pic.save(OUT/'previas'/f'comparacion_samplers_ep{epoch:03d}.png')
    print('COMPARACIÓN:', OUT/'previas'/f'comparacion_samplers_ep{epoch:03d}.png',flush=True)
    if epoch==80:
        for m in METHODS:
            left=OUT/'previas'/f'ep060_{m}.png'
            right=OUT/'previas'/f'ep080_{m}.png'
            if not (left.is_file() and right.is_file()):continue
            a=Image.open(left).convert('RGB');b=Image.open(right).convert('RGB')
            canvas=Image.new('RGB',(a.width+b.width+gap,max(a.height,b.height)+header),'white')
            t=ImageDraw.Draw(canvas)
            t.text((5,12),f'Época 60 - {m}',fill='black')
            t.text((a.width+gap+5,12),f'Época 80 - {m}',fill='black')
            canvas.paste(a,(0,header));canvas.paste(b,(a.width+gap,header))
            canvas.save(OUT/'previas'/f'comparacion_ep060_vs_ep080_{m}.png')


def preview(epoch,method):
    torch=require_gpu()
    if not CK[epoch].is_file():raise FileNotFoundError(CK[epoch])
    model, orig=ck_verified(epoch,torch)
    methods=METHODS if method=='all' else (method,)
    for meth in methods:
        path=OUT/'previas'/f'ep{epoch:03d}_{meth}.png'
        meta=path.with_suffix('.json')
        p=spec(epoch,meth,'preview')
        if path.exists() and meta.exists() and json.loads(meta.read_text(encoding='utf8')) == p:
            print('YA EXISTE',path,flush=True)
            continue
        if path.exists() and (not meta.exists() or json.loads(meta.read_text(encoding='utf8')) != p):
            raise RuntimeError('Preview con metadatos incompatibles: '+str(path))
        tic=time.monotonic()
        result=sample_batch(model,orig,meth,SEED_PREVIEW,PREVIEW_N,torch)
        save_grid(result,path)
        atomic_json(meta,p)
        print(f'LISTO {meth}: {path} ({time.monotonic()-tic:.1f}s)',flush=True)
    make_collage(epoch)


def generate(epoch,method,budget):
    torch=require_gpu()
    if not CK[epoch].is_file():raise FileNotFoundError(CK[epoch])
    if method=='all':raise ValueError('Selecciona un método individual para generación KID')
    folder=OUT/'kid_muestras'/f'ep{epoch}_{method}'
    manifest=folder/'protocolo.json'
    meta=spec(epoch,method,'kid')
    if manifest.is_file():
        if json.loads(manifest.read_text(encoding='utf8')) != meta:
            raise RuntimeError('Protocolo de lotes existente incompatible; no sobrescribir')
    else:
        if list(folder.glob('lote_*.pt')):
            raise RuntimeError('Hay lotes sin protocolo en '+str(folder))
        atomic_json(manifest,meta)
    model,orig=ck_verified(epoch,torch)
    start=time.monotonic(); finished=0
    for i in range(N_KID//BATCH):
        file=folder/f'lote_{i:04d}.pt'
        if file.is_file():
            finished+=1;continue
        if time.monotonic()-start>budget:
            print('PAUSA SEGURA',finished,'/',N_KID//BATCH,'lotes. Reenvía el mismo SLURM.',flush=True)
            return
        imgs=sample_batch(model,orig,method,SEED_KID+i,BATCH,torch)
        if imgs.dtype != torch.uint8 or tuple(imgs.shape)!=(BATCH,3,128,128):
            raise RuntimeError('Forma de imágenes incorrecta')
        atomic_torch(file,imgs,torch)
        finished+=1
        if finished%4==0 or finished==N_KID//BATCH:
            print(f'ep{epoch} {method}: {finished*BATCH}/{N_KID}; seg={time.monotonic()-start:.1f}',flush=True)
    print('GENERACIÓN COMPLETA',epoch,method,N_KID,flush=True)


def kid_kernel(x,y,torch, block=256):
    result=torch.empty((len(x),len(y)),device=x.device,dtype=torch.float32)
    yt=y.T.contiguous()
    for i in range(0,len(x),block):
        result[i:i+block].copy_((1+(x[i:i+block]@yt)/D).pow(3))
    return result


def kid_u(Kx,Ky,Kxy,torch):
    nx=len(Kx);ny=len(Ky)
    xx=(Kx.sum(dtype=torch.float64)-Kx.diagonal().sum(dtype=torch.float64))/(nx*(nx-1))
    yy=(Ky.sum(dtype=torch.float64)-Ky.diagonal().sum(dtype=torch.float64))/(ny*(ny-1))
    xy=Kxy.sum(dtype=torch.float64)/(nx*ny)
    return (xx+yy-2*xy).item()


def weighted_self(K,c,torch):
    n=c[0].sum().item()
    w=c.T.contiguous()
    quad=(w.double()*(K@w).double()).sum(dim=0)
    diag=(c.double()*K.diagonal().double().unsqueeze(0)).sum(dim=1)
    return (quad-diag)/(n*(n-1))


def weighted_cross(K,left,right):
    n=left[0].sum().item();m=right[0].sum().item()
    return (left.T.double()*(K@right.T.contiguous()).double()).sum(dim=0)/(n*m)


def read_features_for_test(torch):
    if not TEST_CACHE.is_file():
        raise FileNotFoundError('Falta caché test Inception previa: '+str(TEST_CACHE))
    obj=torch.load(TEST_CACHE,map_location='cpu',weights_only=True)
    meta=obj['meta']
    if (meta.get('kind')!='test' or meta.get('n')!=10000 or
        meta.get('feature_dim')!=D or meta.get('split_sha256')!=digest(SPLIT)
        or meta.get('source_script_128_sha256')!=digest(BASE)
        or meta.get('extraction')!='torchmetrics Inception-v3 KID 2048, uint8 0-255'):
        raise RuntimeError('Caché TEST incompatible con el protocolo previo')
    feats=obj['features']
    if tuple(feats.shape)!=(10000,D) or not torch.isfinite(feats).all():
        raise RuntimeError('Features test corruptas')
    return feats


def extract_case(case, torch, extractor):
    epoch,method=case
    folder=OUT/'kid_muestras'/f'ep{epoch}_{method}'
    meta=spec(epoch,method,'kid')
    file=folder/'protocolo.json'
    if not file.is_file() or json.loads(file.read_text(encoding='utf8')) != meta:
        raise RuntimeError('Protocolo inexistente o modificado en '+str(folder))
    cache=OUT/'kid_features'/f'ep{epoch}_{method}.pt'
    if cache.is_file():
        obj=torch.load(cache,map_location='cpu',weights_only=True)
        if obj.get('meta')!=meta or tuple(obj['features'].shape)!=(N_KID,D):
            raise RuntimeError('Features KID incompatibles con el protocolo: '+str(cache))
        return obj['features']
    parts=[]
    for i in range(N_KID//BATCH):
        f=folder/f'lote_{i:04d}.pt'
        if not f.is_file():raise FileNotFoundError('Falta lote: '+str(f))
        x=torch.load(f,map_location='cpu',weights_only=True)
        if x.dtype != torch.uint8 or tuple(x.shape)!=(BATCH,3,128,128):
            raise RuntimeError('Lote corrupto '+str(f))
        with torch.inference_mode():
            parts.append(extractor(x.cuda()).float().cpu())
    feats=torch.cat(parts)
    if tuple(feats.shape)!=(N_KID,D):raise RuntimeError('Dimensiones features incorrectas')
    atomic_torch(cache,{'meta':meta,'features':feats},torch)
    return feats


def evaluate(cases_str):
    import numpy as np
    torch=require_gpu()
    from torchmetrics.image.kid import KernelInceptionDistance
    cases=[]
    for item in cases_str.split(','):
        raw=item.strip().split(':')
        if len(raw)!=2 or int(raw[0]) not in CK or raw[1] not in METHODS:
            raise ValueError('Caso inválido: '+item)
        cases.append((int(raw[0]),raw[1]))
    if len(cases)<2 or len(set(cases))!=len(cases):
        raise ValueError('Se requieren al menos dos casos distintos')
    if len({e for e,m in cases})==1 and cases[0][0]==60 and len(cases)==3:
        prefix='KID_ep60_samplers'
    elif len(cases)==2 and {e for e,m in cases}=={60,80}:
        prefix='KID_ep60_vs_80'
    else:prefix='KID_otros'
    torch.set_num_threads(int(os.getenv('SLURM_CPUS_PER_TASK','3')))
    test=read_features_for_test(torch).cuda()
    extractor=KernelInceptionDistance(feature=D,subsets=2,subset_size=500,normalize=False).cuda().eval().inception
    images={case:extract_case(case,torch,extractor).cuda() for case in cases}
    del extractor
    torch.cuda.empty_cache()
    # Mismo conjunto TEST y mismo bootstrap de índices reales y de imágenes generadas.
    Kx=kid_kernel(test,test,torch)
    rng=np.random.default_rng(SEED_BOOT)
    cx=torch.as_tensor(rng.multinomial(len(test),np.full(len(test),1/len(test)),size=N_BOOT),device='cuda',dtype=torch.float32)
    ca=torch.as_tensor(rng.multinomial(N_KID,np.full(N_KID,1/N_KID),size=N_BOOT),device='cuda',dtype=torch.float32)
    xx=weighted_self(Kx,cx,torch)
    vals={};draws={}
    for case,features in images.items():
        Ky=kid_kernel(features,features,torch)
        Kxy=kid_kernel(test,features,torch)
        v=kid_u(Kx,Ky,Kxy,torch)
        boot=(xx+weighted_self(Ky,ca,torch)-2*weighted_cross(Kxy,cx,ca)).cpu().numpy()
        vals[case]=v;draws[case]=boot
        print(f'{case}: KIDx1000={v*1000:.5f}',flush=True)
        del Ky,Kxy
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    results=[]
    labels=[]
    for case in cases:
        val=vals[case];arr=draws[case]
        low,high=2*val-np.quantile(arr,.975),2*val-np.quantile(arr,.025)
        label=f'ép{case[0]} {case[1]}'
        labels.append(label)
        results.append(dict(caso=label,kid_u_x1000=val*1000,ic95_inf_x1000=low*1000,
                            ic95_sup_x1000=high*1000,n_fake=N_KID,n_test=len(test),n_boot=N_BOOT,
                            comentario='IC básico exploratorio condicionado a los checkpoints y semillas; menos KID es mejor'))
    OUT.mkdir(exist_ok=True,parents=True)
    csvfile=OUT/(prefix+'.csv')
    with csvfile.open('w',newline='',encoding='utf-8') as f:
        wr=csv.DictWriter(f,fieldnames=list(results[0]));wr.writeheader();wr.writerows(results)
    fig,ax=plt.subplots(figsize=(8,3.7))
    y=np.arange(len(results))
    for j,row in enumerate(results):
        val=row['kid_u_x1000'];lo=row['ic95_inf_x1000'];hi=row['ic95_sup_x1000']
        ax.errorbar([val],[j],xerr=np.array([[max(0,val-lo)],[max(0,hi-val)]]),fmt='o',capsize=5)
    ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlabel('KID U ×1000 (IC bootstrap básico 95 %)')
    ax.set_title('Comparación controlada de muestreadores CelebA128')
    ax.grid(axis='x',alpha=.2)
    fig.tight_layout();fig.savefig(OUT/(prefix+'.png'),dpi=220);plt.close(fig)
    if len(cases)==2:
        a,b=cases
        diff=vals[b]-vals[a]
        bd=draws[b]-draws[a]
        lo,hi=2*diff-np.quantile(bd,.975),2*diff-np.quantile(bd,.025)
        with (OUT/(prefix+'_diferencia.csv')).open('w',newline='',encoding='utf8') as f:
            w=csv.DictWriter(f,fieldnames=['comparacion','delta_kid_x1000','ic95_inf_x1000','ic95_sup_x1000'])
            w.writeheader();w.writerow(dict(comparacion=f'{b} menos {a}',delta_kid_x1000=diff*1000,
                                         ic95_inf_x1000=lo*1000,ic95_sup_x1000=hi*1000))
    print('EVALUACIÓN COMPLETA:',csvfile,flush=True)


def main():
    pa=argparse.ArgumentParser()
    pa.add_argument('--mode',required=True, choices=['check','status','selftest','preview','generate','evaluate'])
    pa.add_argument('--epoch',type=int,choices=[60,80],default=60)
    pa.add_argument('--sampler',choices=[*METHODS,'all'],default='all')
    pa.add_argument('--cases',default='60:ddim250,60:ddim500,60:ddpm1000')
    pa.add_argument('--budget',type=int,default=27000)
    a=pa.parse_args()
    if a.mode=='check':check()
    elif a.mode=='status':status()
    elif a.mode=='selftest':selftest()
    elif a.mode=='preview':preview(a.epoch,a.sampler)
    elif a.mode=='generate':generate(a.epoch,a.sampler,a.budget)
    elif a.mode=='evaluate':evaluate(a.cases)

if __name__=='__main__':main()
