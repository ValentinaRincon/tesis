#!/usr/bin/env python3
"""CelebA RGB 128x128, ResUNet128 con pesos iniciales EMA CelebA64.

Flujo: --mode selftest / check (login), gpu-test / train / sample (SOLO SLURM GPU).
Checkpoint seguro por epoca; no modifica archivos del experimento 64x64.
"""
import argparse
import copy
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
from torch import nn
from torch.nn import functional as F

import entrenar_celeba64 as original

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'celeba'
IMAGES = DATA / 'img_align_celeba'
SPLIT = DATA / 'list_eval_partition.txt'
OLD_CK = ROOT / 'CelebA 64x64 ResUNet' / 'checkpoint.pt'
OUT = ROOT / 'CelebA 128x128 ResUNet'
CONFIG = dict(
    experiment='celeba128_resunet_warmstart64ema_v1',
    size=128, seed=20261030, data_seed=20261029,
    max_train_images=50000, epochs=40, batch_size=4,
    workers=2, learning_rate_shared=5e-5, learning_rate_new=1e-4,
    ema_decay=0.999, t_steps=1000, beta_start=1e-4, beta_end=0.02,
    amp='fp16', max_grad_norm=1.0,
    preview_every=5, preview_n=8, preview_steps=50,
    initialization='weights_from_original_64_ema_for_matching_layers'
)


class ResUNet128(original.ResUNet64):
    """128->64->32->16->8. Atencion solo 16 y 8, nunca 128.

    Conserva las claves/pesos de todos los bloques 64x64 y anade
    encoder/decoder de 128. Son nuevos y se entrenan a esta resolucion.
    """
    def __init__(self):
        super().__init__()
        self.e128a = original.ResBlock(32, 32)
        self.e128b = original.ResBlock(32, 32)
        self.down128 = nn.Conv2d(32, 32, 4, stride=2, padding=1)
        self.up128 = nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1)
        self.d128a = original.ResBlock(64, 32)
        self.d128b = original.ResBlock(32, 32)

    def forward(self, x, t):
        if x.shape[-2:] != (128, 128):
            raise ValueError('ResUNet128 requiere 128x128')
        te = self.time(t)
        top = self.e128b(self.e128a(self.stem(x), te), te)
        a = self.e64b(self.e64a(self.down128(top), te), te)
        b = self.e32b(self.e32a(self.down64(a), te), te)
        c = self.e16b(self.att16(self.e16a(self.down32(b), te)), te)
        h = self.mid2(self.midatt(self.mid1(self.down16(c), te)), te)
        h = self.d16b(self.d16att(self.d16a(torch.cat([self.up16(h), c], dim=1), te)), te)
        h = self.d32b(self.d32a(torch.cat([self.up32(h), b], dim=1), te), te)
        h = self.d64b(self.d64a(torch.cat([self.up64(h), a], dim=1), te), te)
        h = self.d128b(self.d128a(torch.cat([self.up128(h), top], dim=1), te), te)
        return self.out(h)


def schedule(device):
    beta = torch.linspace(CONFIG['beta_start'], CONFIG['beta_end'], CONFIG['t_steps'], device=device)
    return torch.cumprod(1.0 - beta, 0)


def image_list():
    if not SPLIT.is_file() or not IMAGES.is_dir():
        raise FileNotFoundError('Se necesita CelebA extraido y list_eval_partition.txt en '+str(DATA))
    names = []
    with SPLIT.open() as f:
        for line in f:
            bits = line.split()
            if len(bits) == 2 and bits[1] == '0':
                names.append(bits[0])
    if len(names) < CONFIG['max_train_images']:
        raise RuntimeError('Split train CelebA insuficiente')
    rng = random.Random(CONFIG['data_seed'])
    names = sorted(rng.sample(names, CONFIG['max_train_images']))
    for name in (names[0], names[len(names)//2], names[-1]):
        if not (IMAGES/name).is_file():
            raise FileNotFoundError('Falta '+str(IMAGES/name))
    return names


class FaceDataset(torch.utils.data.Dataset):
    def __init__(self, names):
        from torchvision import transforms
        self.names = names
        self.transform = transforms.Compose([
            transforms.CenterCrop(160),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((.5,)*3, (.5,)*3),
        ])

    def __len__(self): return len(self.names)

    def __getitem__(self, idx):
        from PIL import Image
        with Image.open(IMAGES/self.names[idx]) as im:
            return self.transform(im.convert('RGB'))


def make_protocol():
    return dict(config=CONFIG,
                source128_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                source64_sha256=hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest(),
                split_sha256=hashlib.sha256(SPLIT.read_bytes()).hexdigest())


def json_save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf8')
    tmp.replace(path)


def pt_save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    torch.save(data, tmp)
    tmp.replace(path)


def seed_all():
    random.seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'] % (2**32))
    torch.manual_seed(CONFIG['seed'])
    torch.cuda.manual_seed_all(CONFIG['seed'])


def rng_state(gen):
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all(),
                loader=gen.get_state())


def rng_restore(state, gen):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch']); torch.cuda.set_rng_state_all(state['cuda'])
    gen.set_state(state['loader'])


def initialize_from_64(model):
    if not OLD_CK.is_file():
        raise FileNotFoundError('No existe checkpoint EMA 64x64: '+str(OLD_CK))
    source = torch.load(OLD_CK, map_location='cpu', weights_only=False)
    if 'ema' not in source:
        raise RuntimeError('El checkpoint 64x64 no contiene pesos EMA')
    result = model.load_state_dict(source['ema'], strict=False)
    missing = set(result.missing_keys)
    expected = {key for key in model.state_dict() if key.startswith(
        ('e128a.', 'e128b.', 'down128.', 'up128.', 'd128a.', 'd128b.'))}
    if missing != expected or result.unexpected_keys:
        raise RuntimeError(f'Incompatibilidad en la transferencia de 64x64: missing={sorted(missing)}, unexpected={result.unexpected_keys}')
    print(f'Transferencia: {len(model.state_dict())-len(missing)} tensores del modelo EMA 64; {len(missing)} nuevos inicializados.', flush=True)
    del source


def make_optimizer(net):
    prefixes=('e128a.', 'e128b.', 'down128.', 'up128.', 'd128a.', 'd128b.')
    new = [p for name, p in net.named_parameters() if name.startswith(prefixes)]
    shared = [p for name, p in net.named_parameters() if not name.startswith(prefixes)]
    assert len(new) and len(shared)
    return torch.optim.Adam([
        dict(params=shared, lr=CONFIG['learning_rate_shared']),
        dict(params=new, lr=CONFIG['learning_rate_new']),
    ])


@torch.inference_mode()
def sample_ddim(net, steps=50, n=8, seed=20261030, microbatch=2):
    device = next(net.parameters()).device
    abar = schedule(device)
    ts = torch.linspace(0, 999, steps, device='cpu').round().long().flip(0).tolist()
    if len(set(ts)) != steps: raise ValueError('Timesteps duplicados')
    generator = torch.Generator(device=device).manual_seed(seed)
    net.eval()
    batches=[]
    for count in range(0, n, microbatch):
        b=min(microbatch, n-count)
        x=torch.randn(b, 3, 128, 128, device=device, generator=generator)
        for i, tid in enumerate(ts):
            t=torch.full((b,), int(tid), dtype=torch.long, device=device)
            alpha=abar[tid]
            with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=device.type=='cuda'):
                eps=net(x,t)
            eps=eps.float()
            x0=(x - (1-alpha).sqrt()*eps)/alpha.sqrt()
            alpha_prev=abar[ts[i+1]] if i+1<len(ts) else torch.ones_like(alpha)
            x=alpha_prev.sqrt()*x0 + (1-alpha_prev).sqrt()*eps
        batches.append((x.clamp(-1,1).add(1).mul(127.5)).round().byte().cpu())
    return torch.cat(batches)


def save_grid(images, path):
    from torchvision.utils import save_image, make_grid
    path.parent.mkdir(parents=True, exist_ok=True)
    grid=make_grid(images.float()/255, nrow=4, padding=2)
    save_image(grid, path)


def selftest():
    net=ResUNet128()
    assert net.e128a.conv1.in_channels == 32
    assert net.e64a.conv1.in_channels == 32
    assert net.midatt.attn.embed_dim == 256
    abar=schedule('cpu')
    assert abar.shape == (1000,) and 0<abar[-1]<abar[0]<1
    print('SELFTEST OK: ResUNet128, 128->64->32->16->8; schedule DDPM.')
    print('Parámetros:', sum(p.numel() for p in net.parameters()))


def check():
    print('CelebA:', DATA)
    print('Fotos extraídas:', IMAGES.is_dir())
    print('Split oficial:', SPLIT.is_file())
    print('Checkpoint original 64x64:', OLD_CK.is_file())
    names=image_list()
    if not OLD_CK.is_file():
        raise FileNotFoundError('Se requiere el checkpoint final de 64x64 para warm-start')
    print('CHECK OK: train separado; muestra fija de', len(names), 'imagenes. Ejemplo:', names[0])
    print('Carpeta nueva:', OUT)


def gpu_test():
    if not torch.cuda.is_available(): raise RuntimeError('SOLO en nodo GPU asignado por SLURM')
    seed_all()
    model=ResUNet128().cuda()
    initialize_from_64(model)
    model.train()
    opt=make_optimizer(model)
    scaler=torch.amp.GradScaler('cuda')
    torch.cuda.reset_peak_memory_stats()
    before=time.monotonic()
    batch=CONFIG['batch_size']
    x0=torch.randn(batch,3,128,128,device='cuda').clamp(-1,1)
    t=torch.randint(0,1000,(batch,),device='cuda')
    noise=torch.randn_like(x0)
    a=schedule('cuda')[t][:,None,None,None]
    xt=a.sqrt()*x0 + (1-a).sqrt()*noise
    opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.float16):
        pred=model(xt,t)
        loss=F.mse_loss(pred.float(),noise.float())
    assert pred.shape==noise.shape and torch.isfinite(loss)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    nn.utils.clip_grad_norm_(model.parameters(),CONFIG['max_grad_norm'])
    scaler.step(opt); scaler.update()
    torch.cuda.synchronize()
    print(f'GPU TEST OK; batch={batch}; loss={loss.item():.5f}; paso={time.monotonic()-before:.2f}s; peak_gpu={torch.cuda.max_memory_allocated()/1024**3:.2f} GiB',flush=True)


def status():
    p=OUT/'estado.json'
    print(p.read_text(encoding='utf8') if p.is_file() else 'Aún sin épocas completas')
    print('Checkpoint:', (OUT/'checkpoint.pt').is_file())
    print('Previews:',len(list((OUT/'previews').glob('epoca_*.png'))) if (OUT/'previews').is_dir() else 0)


def save_rows(rows):
    p=OUT/'perdidas.csv'; tmp=OUT/'perdidas.csv.tmp'
    with tmp.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['epoch','train_mse','seconds','global_step'])
        writer.writeheader(); writer.writerows(rows)
    tmp.replace(p)


def train(time_budget):
    if not torch.cuda.is_available(): raise RuntimeError('Solo ejecutar train en SLURM GPU')
    names=image_list()
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=make_protocol()
    manifest=OUT/'protocolo.json'
    if manifest.is_file() and json.loads(manifest.read_text())!=protocol:
        raise RuntimeError('Protocolo distinto al experimento existente, no se sobrescribirá')
    json_save(manifest,protocol)
    seed_all()
    torch.backends.cudnn.benchmark=True
    model=ResUNet128().cuda()
    ck_path=OUT/'checkpoint.pt'
    resume=ck_path.is_file()
    if not resume: initialize_from_64(model)
    ema=copy.deepcopy(model).eval().requires_grad_(False)
    optimizer=make_optimizer(model)
    scaler=torch.amp.GradScaler('cuda',enabled=True)
    gen=torch.Generator().manual_seed(CONFIG['seed']+500)
    rows=[]; start=0; step=0
    if resume:
        ck=torch.load(ck_path,map_location='cpu',weights_only=False)
        if ck['protocol']!=protocol: raise RuntimeError('Checkpoint de otro protocolo')
        model.load_state_dict(ck['model']); ema.load_state_dict(ck['ema'])
        optimizer.load_state_dict(ck['optimizer']); scaler.load_state_dict(ck['scaler'])
        start=int(ck['epoch']); rows=ck['rows']; step=ck['global_step']
        rng_restore(ck['rng'],gen)
        print('Reanuda en época',start+1,flush=True)
        del ck
    if start>=CONFIG['epochs']:
        print('ENTRENAMIENTO YA COMPLETO',flush=True); return
    dataset=FaceDataset(names)
    loader=torch.utils.data.DataLoader(dataset,batch_size=CONFIG['batch_size'],shuffle=True,
        generator=gen,num_workers=CONFIG['workers'],pin_memory=True,drop_last=False)
    abar=schedule('cuda')
    print('GPU:',torch.cuda.get_device_name(0),'imgs:',len(dataset),'batch:',CONFIG['batch_size'], 'inicio:',start+1,flush=True)
    began=time.monotonic()
    for epoch in range(start,CONFIG['epochs']):
        tic=time.monotonic(); model.train(); total=0.;seen=0
        for x0 in loader:
            x0=x0.cuda(non_blocking=True)
            t=torch.randint(0,1000,(len(x0),),device='cuda')
            eps=torch.randn_like(x0)
            a=abar[t][:,None,None,None]
            xt=a.sqrt()*x0+(1-a).sqrt()*eps
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):
                pred=model(xt,t)
                loss=F.mse_loss(pred.float(),eps.float())
            if not torch.isfinite(loss): raise RuntimeError('Pérdida no finita')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(),CONFIG['max_grad_norm'])
            prev_scale=scaler.get_scale(); scaler.step(optimizer);scaler.update()
            if scaler.get_scale()>=prev_scale:
                with torch.no_grad():
                    for q,p in zip(ema.parameters(),model.parameters()):q.lerp_(p,1-CONFIG['ema_decay'])
                    for q,p in zip(ema.buffers(),model.buffers()):q.copy_(p)
            total+=loss.item()*len(x0);seen+=len(x0);step+=1
        secs=time.monotonic()-tic
        rows.append(dict(epoch=epoch+1,train_mse=total/seen,seconds=round(secs,2),global_step=step))
        pt_save(ck_path,dict(protocol=protocol,epoch=epoch+1,rows=rows,global_step=step,
            model=model.state_dict(),ema=ema.state_dict(),optimizer=optimizer.state_dict(),
            scaler=scaler.state_dict(),rng=rng_state(gen)))
        save_rows(rows)
        json_save(OUT/'estado.json',dict(epoch=epoch+1,epochs=CONFIG['epochs'],
            train_mse=round(total/seen,8),elapsed_seconds=round(time.monotonic()-began,1),
            complete=(epoch+1==CONFIG['epochs'])))
        print(f'EPOCA {epoch+1}/{CONFIG["epochs"]} MSE={total/seen:.6f} {secs:.1f}s',flush=True)
        if (epoch+1)%CONFIG['preview_every']==0 or epoch+1==CONFIG['epochs']:
            images=sample_ddim(ema,steps=CONFIG['preview_steps'],n=CONFIG['preview_n'],seed=20261030)
            save_grid(images,OUT/'previews'/f'epoca_{epoch+1:03d}.png')
        if epoch+1==CONFIG['epochs']:
            pt_save(OUT/'ema_final.pt',ema.state_dict())
            print('ENTRENAMIENTO COMPLETO',flush=True)
            break
        if time.monotonic()-began+max(secs*1.5,300)>time_budget:
            print('PAUSA SEGURA: reenviar job; reanuda desde checkpoint.',flush=True)
            break


def sample():
    if not torch.cuda.is_available(): raise RuntimeError('SOLO en SLURM GPU')
    ck=OUT/'checkpoint.pt'
    if not ck.is_file():raise FileNotFoundError('No existe checkpoint 128')
    data=torch.load(ck,map_location='cpu',weights_only=False)
    if data['protocol']!=make_protocol():raise RuntimeError('Protocolo incompatible')
    model=ResUNet128().cuda().eval()
    model.load_state_dict(data['ema'])
    images=sample_ddim(model,steps=250,n=16,seed=20261105,microbatch=2)
    path=OUT/'muestras_ddim250.png'
    save_grid(images,path)
    print('NUEVAS MUESTRAS:',path,flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',required=True,choices=['selftest','check','gpu-test','train','sample','status'])
    parser.add_argument('--time-budget-seconds',type=int,default=42000)
    args=parser.parse_args()
    modes=dict(selftest=selftest,check=check,gpu_test=gpu_test,
               train=lambda:train(args.time_budget_seconds),sample=sample,status=status)
    modes[args.mode.replace('-','_')]()

if __name__=='__main__': main()
