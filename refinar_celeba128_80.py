#!/usr/bin/env python3
"""Refinamiento controlado CelebA128 época 60 -> 80, conservando el checkpoint época 60.

--mode check/status: sin importar PyTorch, seguros en login.
--mode train: solo GPU SLURM.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT/'CelebA 128x128 Refinamiento'/'checkpoint.pt'
OUT = ROOT/'CelebA 128x128 Refinamiento80'
CK80 = OUT/'checkpoint.pt'
PROTOCOL = OUT/'protocolo.json'
START, TARGET = 60, 80
LR_SHARED=1.25e-5
LR_NEW=2.5e-5


def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def save_json(path,content):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(content,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    os.replace(tmp,path)


def protocol(orig):
    return dict(experiment='celeba128_refine60_to80_v1',
                source_checkpoint_sha256=digest(SOURCE),
                source_original_script_sha256=digest(Path(orig.__file__)),
                source_refinement_script_sha256=digest(ROOT/'refinar_celeba128.py'),
                model_experiment=orig.CONFIG['experiment'],
                initial_epoch=START, target_epoch=TARGET,
                lr_shared=LR_SHARED,lr_new=LR_NEW,
                batch_size=orig.CONFIG['batch_size'],
                loss='DDPM noise prediction MSE',
                optimizer='Adam states restored from ep60',
                scaler='fp16 GradScaler restored from ep60',
                ema_decay=orig.CONFIG['ema_decay'],
                rng='restored exactly at epoch boundary ep60',
                preview_every=5,preview_sampler='DDIM eta0 50 steps, fixed seed 20261030')


def check():
    required=[ROOT/'entrenar_celeba64.py', ROOT/'entrenar_celeba128.py',
              ROOT/'refinar_celeba128.py', SOURCE, ROOT/'data/celeba/img_align_celeba',
              ROOT/'data/celeba/list_eval_partition.txt',SOURCE.parent/'estado.json']
    for f in required:print(('OK   ' if f.exists() else 'FALTA'),f)
    if not all(p.exists() for p in required):raise FileNotFoundError('Revisar rutas antes de enviar job')
    state=json.loads((SOURCE.parent/'estado.json').read_text(encoding='utf-8'))
    if int(state.get('epoch',-1))!=START or not state.get('complete'):
        raise RuntimeError('No se confirma época 60 completa en estado.json')
    print('CHECK OK: checkpoint de época 60 no será modificado.')


def status():
    print('Checkpoint 60:',SOURCE.is_file())
    print('Checkpoint 80:',CK80.is_file())
    p=OUT/'estado.json'
    print(p.read_text(encoding='utf-8') if p.is_file() else 'Aún no hay épocas adicionales completas')
    print('Previews:',sorted(p.name for p in (OUT/'previews').glob('epoca_*.png')))


def train(budget):
    import copy
    import torch
    from torch import nn
    from torch.nn import functional as F
    import entrenar_celeba128 as orig
    if not torch.cuda.is_available():
        raise RuntimeError('Se requiere GPU SLURM, no el nodo de login')
    p=protocol(orig)
    if PROTOCOL.is_file() and json.loads(PROTOCOL.read_text(encoding='utf8'))!=p:
        raise RuntimeError('Protocolo existente incompatible; no se sobrescribirá')
    if CK80.exists() and not PROTOCOL.exists():
        raise RuntimeError('Checkpoint sin protocolo: revisar manualmente')
    OUT.mkdir(parents=True,exist_ok=True)
    orig.seed_all()
    torch.backends.cudnn.benchmark=True
    model=orig.ResUNet128().cuda()
    ema=copy.deepcopy(model).eval().requires_grad_(False)
    optimizer=orig.make_optimizer(model)
    scaler=torch.amp.GradScaler('cuda',enabled=True)
    gen=torch.Generator().manual_seed(orig.CONFIG['seed']+500)
    if CK80.exists():
        ck=torch.load(CK80,map_location='cpu',weights_only=False)
        if ck.get('quality_protocol')!=p:raise RuntimeError('Checkpoint de ep80 incompatible')
        print('Reanudando checkpoint de refinamiento80.',flush=True)
    else:
        ck=torch.load(SOURCE,map_location='cpu',weights_only=False)
        if ck.get('epoch')!=START or ck.get('refinement_protocol',{}).get('target_epoch')!=60:
            raise RuntimeError('Se esperaba checkpoint completo de época 60')
        print('Cargando EMA, Adam, GradScaler y RNG del checkpoint ep60.',flush=True)
    start=int(ck['epoch'])
    model.load_state_dict(ck['model'],strict=True)
    ema.load_state_dict(ck['ema'],strict=True)
    optimizer.load_state_dict(ck['optimizer'])
    scaler.load_state_dict(ck['scaler'])
    rows=list(ck['rows'])
    step=int(ck['global_step'])
    orig.rng_restore(ck['rng'],gen)
    del ck
    if len(optimizer.param_groups)!=2:
        raise RuntimeError('Se esperaban dos grupos de Adam')
    optimizer.param_groups[0]['lr']=LR_SHARED
    optimizer.param_groups[1]['lr']=LR_NEW
    save_json(PROTOCOL,p)
    print('LR compartidas/nuevas:', [g['lr'] for g in optimizer.param_groups],flush=True)
    if start>=TARGET:
        print('YA COMPLETO: época',start,flush=True);return
    names=orig.image_list()
    ds=orig.FaceDataset(names)
    loader=torch.utils.data.DataLoader(ds,batch_size=orig.CONFIG['batch_size'],
        shuffle=True,generator=gen,num_workers=orig.CONFIG['workers'],
        pin_memory=True,drop_last=False)
    abar=orig.schedule('cuda')
    began=time.monotonic()
    print('GPU',torch.cuda.get_device_name(0),'muestras',len(ds),'inicio',start+1,'meta',TARGET,flush=True)
    for ep in range(start,TARGET):
        tic=time.monotonic();model.train();total=0.;seen=0
        for x0 in loader:
            x0=x0.cuda(non_blocking=True)
            t=torch.randint(0,orig.CONFIG['t_steps'],(len(x0),),device='cuda')
            eps=torch.randn_like(x0)
            a=abar[t][:,None,None,None]
            xt=a.sqrt()*x0+(1-a).sqrt()*eps
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda',dtype=torch.float16):
                pred=model(xt,t)
                loss=F.mse_loss(pred.float(),eps.float())
            if not torch.isfinite(loss):raise RuntimeError('Loss no finita')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(),orig.CONFIG['max_grad_norm'])
            scale=scaler.get_scale()
            scaler.step(optimizer);scaler.update()
            if scaler.get_scale()>=scale:
                with torch.no_grad():
                    for q,w in zip(ema.parameters(),model.parameters()):
                        q.lerp_(w,1-orig.CONFIG['ema_decay'])
                    for q,w in zip(ema.buffers(),model.buffers()):
                        q.copy_(w)
            total+=float(loss.item())*len(x0);seen+=len(x0);step+=1
        sec=time.monotonic()-tic;mse=total/seen
        rows.append(dict(epoch=ep+1,train_mse=mse,seconds=round(sec,2),global_step=step))
        orig.pt_save(CK80, dict(quality_protocol=p,epoch=ep+1,rows=rows,global_step=step,
            model=model.state_dict(),ema=ema.state_dict(),optimizer=optimizer.state_dict(),
            scaler=scaler.state_dict(),rng=orig.rng_state(gen)))
        save_json(OUT/'estado.json',dict(epoch=ep+1,epochs=TARGET,
            train_mse=round(mse,8),complete=(ep+1==TARGET),
            elapsed_seconds=round(time.monotonic()-began,1)))
        with (OUT/'perdidas_61_80.csv').open('w',encoding='utf8') as f:
            f.write('epoch,train_mse,seconds,global_step\n')
            for r in rows:
                if int(r['epoch'])>START:
                    f.write(f"{r['epoch']},{r['train_mse']},{r['seconds']},{r['global_step']}\n")
        print(f'EPOCA {ep+1}/{TARGET} MSE={mse:.6f} secs={sec:.1f}',flush=True)
        if (ep+1)%5==0:
            images=orig.sample_ddim(ema,steps=50,n=8,seed=20261030)
            orig.save_grid(images,OUT/'previews'/f'epoca_{ep+1:03d}.png')
        if ep+1==TARGET:
            orig.pt_save(OUT/'ema_final.pt',ema.state_dict())
            print('ENTRENAMIENTO 60 -> 80 COMPLETADO',flush=True)
            break
        if time.monotonic()-began+max(sec*1.5,300)>budget:
            print('PAUSA SEGURA: reenviar mismo SLURM para continuar',flush=True)
            break


def main():
    pa=argparse.ArgumentParser()
    pa.add_argument('--mode',required=True,choices=['check','status','train'])
    pa.add_argument('--budget',type=int,default=27000)
    a=pa.parse_args()
    if a.mode=='check':check()
    elif a.mode=='status':status()
    else:train(a.budget)

if __name__=='__main__':main()
