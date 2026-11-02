#!/usr/bin/env python3
"""CelebA RGB 64x64: DDPM eps-prediction con ResUNet, attention y EMA.

En login: --mode selftest/check/status. En nodo CPU: --mode prepare.
Solo nodo GPU SLURM: --mode train/sample.
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
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'celeba'
IMAGES = DATA / 'img_align_celeba'
SPLIT = DATA / 'list_eval_partition.txt'
ARCHIVE = DATA / 'img_align_celeba.zip'
OUT = ROOT / 'CelebA 64x64 ResUNet'
CONFIG = dict(dataset='CelebA aligned cropped; official train split', size=64,
              max_train_images=50000, data_seed=20261029, seed=20261029,
              epochs=60, batch_size=16, workers=2, lr=0.0001, ema_decay=0.999,
              t_steps=1000, beta_start=0.0001, beta_end=0.02,
              amp='fp16', max_grad_norm=1.0, preview_every=5,
              preview_steps=50, preview_n=16)


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n', encoding='utf8')
    temp.replace(path)


def save_torch(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    torch.save(obj, tmp)
    tmp.replace(path)


class TimeEmbedding(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.d = d
        self.net = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))

    def forward(self, t):
        half = self.d // 2
        rates = torch.exp(torch.arange(half, device=t.device) * (-math.log(10000) / (half - 1)))
        raw = t.float()[:, None] * rates[None, :]
        return self.net(torch.cat([raw.sin(), raw.cos()], dim=1))


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.time = nn.Sequential(nn.SiLU(), nn.Linear(tdim, out_ch))
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, te):
        h = F.silu(self.norm1(self.conv1(x)))
        h = h + self.time(te)[:, :, None, None]
        h = self.norm2(self.conv2(h))
        return F.silu(h + self.skip(x))


class Attn(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(channels, 4, batch_first=True)

    def forward(self, x):
        batch, channels, height, width = x.shape
        q = self.norm(x).flatten(2).transpose(1, 2)
        a, _ = self.attn(q, q, q, need_weights=False)
        return x + a.transpose(1, 2).reshape(batch, channels, height, width)


class ResUNet64(nn.Module):
    """Encoder 64->32->16->8; attention solo 16x16 y 8x8."""
    def __init__(self):
        super().__init__()
        self.time = TimeEmbedding(64)
        self.stem = nn.Conv2d(3, 32, 3, padding=1)
        self.e64a = ResBlock(32, 32); self.e64b = ResBlock(32, 32)
        self.down64 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.e32a = ResBlock(64, 64); self.e32b = ResBlock(64, 64)
        self.down32 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        self.e16a = ResBlock(128, 128); self.att16 = Attn(128); self.e16b = ResBlock(128, 128)
        self.down16 = nn.Conv2d(128, 256, 4, stride=2, padding=1)
        self.mid1 = ResBlock(256, 256); self.midatt = Attn(256); self.mid2 = ResBlock(256, 256)
        self.up16 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.d16a = ResBlock(256, 128); self.d16att = Attn(128); self.d16b = ResBlock(128, 128)
        self.up32 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.d32a = ResBlock(128, 64); self.d32b = ResBlock(64, 64)
        self.up64 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.d64a = ResBlock(64, 32); self.d64b = ResBlock(32, 32)
        self.out = nn.Conv2d(32, 3, 1)

    def forward(self, x, t):
        te = self.time(t)
        a = self.e64b(self.e64a(self.stem(x), te), te)
        b = self.e32b(self.e32a(self.down64(a), te), te)
        c = self.e16b(self.att16(self.e16a(self.down32(b), te)), te)
        h = self.mid2(self.midatt(self.mid1(self.down16(c), te)), te)
        h = self.d16b(self.d16att(self.d16a(torch.cat([self.up16(h), c], 1), te)), te)
        h = self.d32b(self.d32a(torch.cat([self.up32(h), b], 1), te), te)
        h = self.d64b(self.d64a(torch.cat([self.up64(h), a], 1), te), te)
        return self.out(h)


def schedule(device):
    beta = torch.linspace(CONFIG['beta_start'], CONFIG['beta_end'], CONFIG['t_steps'], device=device)
    return torch.cumprod(1 - beta, dim=0)


def image_list():
    if not SPLIT.exists():
        raise FileNotFoundError('Falta list_eval_partition.txt en '+str(DATA)+' (archivo oficial CelebA).')
    if not IMAGES.exists():
        raise FileNotFoundError('Falta '+str(IMAGES)+'; extrae primero el ZIP usando el job CPU.')
    names = []
    with SPLIT.open() as f:
        for line in f:
            bits = line.split()
            if len(bits) == 2 and bits[1] == '0':
                names.append(bits[0])
    if len(names) < CONFIG['max_train_images']:
        raise RuntimeError('Train split insuficiente: '+str(len(names)))
    rng = random.Random(CONFIG['data_seed'])
    names = sorted(rng.sample(names, CONFIG['max_train_images']))
    for x in (names[0], names[len(names)//2], names[-1]):
        if not (IMAGES/x).is_file():
            raise FileNotFoundError('No existe imagen de entrenamiento '+str(IMAGES/x))
    return names


class FaceDataset(torch.utils.data.Dataset):
    def __init__(self, names):
        from torchvision import transforms
        self.names = names
        self.transform = transforms.Compose([
            transforms.CenterCrop(160), transforms.Resize((64, 64)),
            transforms.ToTensor(), transforms.Normalize((.5,)*3, (.5,)*3)])

    def __len__(self): return len(self.names)

    def __getitem__(self, index):
        from PIL import Image
        with Image.open(IMAGES/self.names[index]) as im:
            return self.transform(im.convert('RGB'))


def seed_all():
    random.seed(CONFIG['seed']); np.random.seed(CONFIG['seed'] % (2**32))
    torch.manual_seed(CONFIG['seed']); torch.cuda.manual_seed_all(CONFIG['seed'])


def get_rng(gen):
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all(), loader=gen.get_state())


def set_rng(s, gen):
    random.setstate(s['python']); np.random.set_state(s['numpy'])
    torch.set_rng_state(s['torch']); torch.cuda.set_rng_state_all(s['cuda'])
    gen.set_state(s['loader'])


def manifest_spec():
    digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if SPLIT.exists():
        split_digest=hashlib.sha256(SPLIT.read_bytes()).hexdigest()
    else:
        split_digest='missing'
    return dict(config=CONFIG, script_sha256=digest, split_sha256=split_digest)


@torch.inference_mode()
def sample_ddim(net, steps=50, n=16, seed=123):
    device = next(net.parameters()).device
    abar = schedule(device)
    ts = torch.linspace(0, 999, steps, device='cpu').round().long().flip(0).tolist()
    if len(set(ts)) != steps: raise ValueError('Tiempos duplicados en DDIM')
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(n, 3, 64, 64, device=device, generator=generator)
    net.eval()
    for k, t_idx in enumerate(ts):
        t = torch.full((n,), t_idx, device=device, dtype=torch.long)
        a = abar[t_idx]
        e = net(x, t)
        x0 = (x - torch.sqrt(1-a)*e) / torch.sqrt(a)
        ap = abar[ts[k+1]] if k+1<len(ts) else torch.ones_like(a)
        x = torch.sqrt(ap)*x0 + torch.sqrt(1-ap)*e
    return (x.clamp(-1,1).add(1).mul(127.5)).round().to(torch.uint8).cpu()


def grid_save(images, path):
    from torchvision.utils import make_grid, save_image
    path.parent.mkdir(parents=True, exist_ok=True)
    grid = make_grid(images.float()/255.0, nrow=4, padding=2)
    save_image(grid, path)


def selftest():
    model = ResUNet64()
    with torch.no_grad():
        pred = model(torch.randn(1, 3, 64, 64), torch.randint(0, 1000, (1,)))
    assert pred.shape == (1,3,64,64)
    ab = schedule('cpu')
    assert 0 < ab[-1] < ab[0] < 1
    print('SELFTEST OK: ResUNet64, shape RGB, DDPM schedule.')
    print('Parametros:', sum(p.numel() for p in model.parameters()))


def check():
    print('Datos:', DATA)
    print('ZIP:', 'PRESENTE' if ARCHIVE.is_file() else 'NO', ARCHIVE)
    print('JPG extraidos:', 'SI' if IMAGES.is_dir() else 'NO', IMAGES)
    print('Particion oficial:', 'SI' if SPLIT.is_file() else 'NO', SPLIT)
    names = image_list()
    print('CHECK OK. Entrenamiento exclusivo train; subset fijo:', len(names), 'ejemplo:', names[0])
    print('Salida:', OUT)


def prepare():
    if not ARCHIVE.is_file():
        raise FileNotFoundError('Descarga primero '+str(ARCHIVE)+' desde la fuente CelebA autorizada.')
    if not SPLIT.is_file():
        raise FileNotFoundError('Descarga tambien '+str(SPLIT)+' (particion oficial).')
    if IMAGES.is_dir() and (IMAGES/'000001.jpg').is_file():
        print('Las imagenes parecen extraidas; verifica --mode check.'); return
    with zipfile.ZipFile(ARCHIVE) as z:
        records = z.namelist()
        print('Archivos en ZIP:', len(records), flush=True)
        # solo archivos JPG dentro de img_align_celeba, sin rutas arbitrarias
        for i, info in enumerate(z.infolist()):
            name = Path(info.filename)
            if info.is_dir() or name.suffix.lower() != '.jpg' or name.parent.name != 'img_align_celeba':
                continue
            target = IMAGES/name.name
            IMAGES.mkdir(parents=True, exist_ok=True)
            if target.is_file() and target.stat().st_size == info.file_size: continue
            temp = target.with_suffix('.jpg.tmp')
            with z.open(info) as src, temp.open('wb') as dst:
                import shutil
                shutil.copyfileobj(src, dst)
            temp.replace(target)
            if i % 20000 == 0: print('Extraccion:', i, '/', len(records), flush=True)
    check()


def status():
    if (OUT/'estado.json').exists():
        print(json.dumps(json.loads((OUT/'estado.json').read_text()), indent=2, ensure_ascii=False))
    else:
        print('Aun sin epocas completas.')
    print('Checkpoints:', (OUT/'checkpoint.pt').exists())
    print('Previews:', len(list((OUT/'previews').glob('epoca_*.png'))) if (OUT/'previews').is_dir() else 0)


def rows_csv(rows):
    path = OUT/'perdidas.csv'; temp = OUT/'perdidas.csv.tmp'
    with temp.open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=['epoch', 'train_mse', 'seconds', 'global_step'])
        w.writeheader(); w.writerows(rows)
    temp.replace(path)


def train(time_budget):
    if not torch.cuda.is_available():
        raise RuntimeError('Requiere una GPU asignada por SLURM.')
    names = image_list()
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = manifest_spec()
    manifest = OUT/'protocolo.json'
    if manifest.is_file() and json.loads(manifest.read_text()) != protocol:
        raise RuntimeError('Protocolo cambiado; no sobrescribir entrenamientos existentes.')
    save_json(manifest, protocol)
    seed_all()
    torch.backends.cudnn.benchmark=True
    device = torch.device('cuda')
    model=ResUNet64().to(device)
    ema=copy.deepcopy(model).eval().requires_grad_(False)
    optimizer=torch.optim.Adam(model.parameters(),lr=CONFIG['lr'])
    scaler=torch.amp.GradScaler('cuda', enabled=CONFIG['amp']=='fp16')
    gen=torch.Generator().manual_seed(CONFIG['seed']+500)
    ck_path=OUT/'checkpoint.pt'; rows=[]; start=0; global_step=0
    if ck_path.is_file():
        ck=torch.load(ck_path, map_location='cpu', weights_only=False)
        if ck['protocol']!=protocol: raise RuntimeError('El checkpoint no corresponde al protocolo.')
        model.load_state_dict(ck['model']); ema.load_state_dict(ck['ema'])
        optimizer.load_state_dict(ck['optimizer']); scaler.load_state_dict(ck['scaler'])
        start=ck['epoch']; rows=ck['rows'];global_step=ck['global_step']
        set_rng(ck['rng'], gen)
        print('Reanuda desde epoca',start, flush=True)
        del ck
    abar=schedule(device)
    ds=FaceDataset(names)
    loader=torch.utils.data.DataLoader(ds,batch_size=CONFIG['batch_size'],shuffle=True,
        generator=gen,num_workers=CONFIG['workers'],pin_memory=True,drop_last=False)
    print('GPU:',torch.cuda.get_device_name(0),'train:',len(ds),'epoca inicial:',start+1,flush=True)
    start_time=time.monotonic()
    for ep in range(start, CONFIG['epochs']):
        tic=time.monotonic(); model.train(); total=0.0; n=0
        for x0 in loader:
            x0=x0.to(device,non_blocking=True)
            times=torch.randint(0,CONFIG['t_steps'],(len(x0),),device=device)
            noise=torch.randn_like(x0)
            a=abar[times][:,None,None,None]
            xt=a.sqrt()*x0+(1-a).sqrt()*noise
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=CONFIG['amp']=='fp16'):
                pred=model(xt,times)
                loss=F.mse_loss(pred.float(),noise.float())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(),CONFIG['max_grad_norm'])
            prev_scale=scaler.get_scale()
            scaler.step(optimizer); scaler.update()
            if scaler.get_scale()>=prev_scale:
                with torch.no_grad():
                    for q,p in zip(ema.parameters(),model.parameters()):
                        q.lerp_(p, 1-CONFIG['ema_decay'])
                    for q,p in zip(ema.buffers(),model.buffers()):q.copy_(p)
            total+=loss.item()*len(x0); n+=len(x0); global_step+=1
        secs=time.monotonic()-tic
        rows.append(dict(epoch=ep+1,train_mse=total/n,seconds=secs,global_step=global_step))
        state=dict(protocol=protocol,epoch=ep+1,rows=rows,global_step=global_step,
                   model=model.state_dict(),ema=ema.state_dict(),optimizer=optimizer.state_dict(),
                   scaler=scaler.state_dict(),rng=get_rng(gen))
        save_torch(ck_path,state)
        rows_csv(rows)
        save_json(OUT/'estado.json',dict(epoch=ep+1,epochs=CONFIG['epochs'],
                  train_mse=total/n,elapsed_seconds=round(time.monotonic()-start_time,1),
                  complete=(ep+1==CONFIG['epochs'])))
        print(f'EPOCA {ep+1}/{CONFIG["epochs"]} MSE={total/n:.6f} {secs:.1f}s',flush=True)
        if (ep+1)%CONFIG['preview_every']==0 or ep+1==CONFIG['epochs']:
            with torch.inference_mode():
                imgs=sample_ddim(ema,CONFIG['preview_steps'],CONFIG['preview_n'])
                grid_save(imgs,OUT/'previews'/f'epoca_{ep+1:03d}.png')
        if ep+1==CONFIG['epochs']:
            save_torch(OUT/'ema_final.pt',ema.state_dict())
            print('ENTRENAMIENTO COMPLETO',flush=True)
            break
        elapsed=time.monotonic()-start_time
        if elapsed+max(secs*1.5,300) > time_budget:
            print('PAUSA SEGURA: guardar y reenviar job para continuar.',flush=True)
            break


def sample():
    if not torch.cuda.is_available():raise RuntimeError('Requiere GPU con SLURM')
    ck=OUT/'checkpoint.pt'
    if not ck.is_file():raise FileNotFoundError('No hay checkpoint de entrenamiento')
    d=torch.load(ck,map_location='cpu',weights_only=False)
    if d['protocol']!=manifest_spec():raise RuntimeError('Protocolo distinto')
    model=ResUNet64().cuda().eval()
    model.load_state_dict(d['ema'])
    with torch.inference_mode():
        imgs=sample_ddim(model,steps=250,n=16,seed=20261029)
    grid_save(imgs,OUT/'muestras_ddim250.png')
    print('Imagenes DDIM-250:',OUT/'muestras_ddim250.png')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',required=True,choices=['selftest','check','prepare','status','train','sample'])
    parser.add_argument('--time-budget-seconds',type=int,default=13200)
    args=parser.parse_args()
    {'selftest':selftest,'check':check,'prepare':prepare,'status':status,
     'train':lambda:train(args.time_budget_seconds),'sample':sample}[args.mode]()


if __name__=='__main__':main()
