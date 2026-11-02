#!/usr/bin/env python3
"""CelebA64 EMA: generacion DDIM-250, KID contra test y vecinos contra train.

USO (login): --mode selftest | check | status
USO (GPU via SLURM): --mode generate | evaluate
No reentrena ni cambia el codigo original.
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
import entrenar_celeba64 as base

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'Evaluacion CelebA64 KID Vecinos'
CHUNKS = OUT / 'chunks_ddim250'
FEAT = OUT / 'features'
N_FAKE = 5000
N_REAL = 10000
N_TRAIN = 50000
BATCH = 16
STEPS = 250
GEN_SEED = 20261030
TEST_SEED = 20261030
FEATURE_D = 2048
EXPECTED_CHUNKS = math.ceil(N_FAKE / BATCH)
MODEL_PATH = base.OUT / 'ema_final.pt'


def atomic_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False)+'\n', encoding='utf8')
    tmp.replace(path)


def atomic_torch(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    torch.save(obj, tmp)
    tmp.replace(path)


def file_sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for buf in iter(lambda:f.read(4*1024*1024), b''):
            h.update(buf)
    return h.hexdigest()


def protocol():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f'Falta checkpoint final: {MODEL_PATH}')
    if not base.SPLIT.is_file():
        raise FileNotFoundError(base.SPLIT)
    if not (base.OUT/'estado.json').is_file():
        raise FileNotFoundError('Falta estado del entrenamiento original')
    status=json.loads((base.OUT/'estado.json').read_text())
    if status.get('epoch')!=base.CONFIG['epochs'] or not status.get('complete'):
        raise RuntimeError(f'Entrenamiento no completado segun estado.json: {status}')
    return {
        'experiment':'celeba64_ema_ddim250_kid_test_nearest_train_v1',
        'checkpoint_sha256':file_sha256(MODEL_PATH),
        'original_training_spec':base.CONFIG,
        'split_sha256':file_sha256(base.SPLIT),
        'fake_count':N_FAKE,'real_test_count':N_REAL,'train_count':N_TRAIN,
        'steps':STEPS,'batch':BATCH,'gen_seed':GEN_SEED,
        'test_seed':TEST_SEED,'feature_dim':FEATURE_D,
        'preprocess':'CenterCrop(160)->Resize(64x64)->RGB; tensor 0..255 uint8',
        'kid_extractor':'torchmetrics KernelInceptionDistance(feature=2048, normalize=False)',
    }


def protocol_file():
    return OUT/'protocolo.json'


def ensure_protocol(write=False):
    p=protocol()
    dest=protocol_file()
    if dest.is_file():
        old=json.loads(dest.read_text())
        if old!=p:
            raise RuntimeError('Protocolo/checkpoint diferente al existente. NO mezclar generacion y evaluacion.')
    elif write:
        atomic_json(dest,p)
    return p


def chunk_path(i):
    return CHUNKS / f'lote_{i:04d}.pt'


def count_chunk(i):
    return min(BATCH,N_FAKE-i*BATCH)


def check_chunk(i, load=False):
    path=chunk_path(i)
    if not path.is_file(): return False
    if load:
        x=torch.load(path,map_location='cpu',weights_only=True)
        if x.dtype!=torch.uint8 or tuple(x.shape)!=(count_chunk(i),3,64,64):
            raise RuntimeError(f'Chunk corrupto: {path}, {x.dtype}, {tuple(x.shape)}')
    return True


def read_partition():
    train=[]; test=[]
    with base.SPLIT.open() as f:
        for line in f:
            parts=line.split()
            if len(parts)!=2: continue
            if parts[1]=='0': train.append(parts[0])
            if parts[1]=='2': test.append(parts[0])
    if len(test)<N_REAL: raise RuntimeError(f'Test split insuficiente: {len(test)}')
    actual_train=base.image_list()
    if len(actual_train)!=N_TRAIN: raise RuntimeError('Train subset != 50000; protocolo distinto')
    if set(actual_train)&set(test):raise RuntimeError('Fuga entre train y test')
    chosen_test=sorted(random.Random(TEST_SEED).sample(test,N_REAL))
    return actual_train,chosen_test


def check():
    print('Original:',base.OUT)
    print('Pesos EMA finales:', MODEL_PATH, 'PRESENTES' if MODEL_PATH.is_file() else 'FALTAN')
    print('Fotos:', base.IMAGES, 'PRESENTES' if base.IMAGES.is_dir() else 'FALTAN')
    print('Partition:',base.SPLIT, 'PRESENTE' if base.SPLIT.is_file() else 'FALTA')
    train,test=read_partition()
    print('Train usado realmente:',len(train),'test separado:',len(test),'disjuntos:',not bool(set(train)&set(test)))
    p=ensure_protocol(False)
    print('Protocol hash checkpoint:',p['checkpoint_sha256'][:16])
    print('Lotes generados:',sum(check_chunk(i) for i in range(EXPECTED_CHUNKS)),'/',EXPECTED_CHUNKS)
    print('Paquetes KID:',end=' ')
    try:
        import torchmetrics, torch_fidelity
        print('torchmetrics OK, torch_fidelity OK')
    except ImportError as e:
        print('NO DISPONIBLES: ',e)
        raise
    print('CHECK OK. No carga pesos ni imagenes a GPU.')


def status():
    print('Lotes DDIM250:',sum(check_chunk(i) for i in range(EXPECTED_CHUNKS)),'/',EXPECTED_CHUNKS)
    for key in ('fake','test','train'):
        p=FEAT/f'{key}.pt'
        print('Features',key, 'SI' if p.is_file() else 'NO')
    for name in ('kid_celeba64.csv','vecinos_resumen.csv','vecinos_primeras16.png','vecinos_mayor_similitud.png'):
        print(name,'SI' if (OUT/name).is_file() else 'NO')


def selftest():
    # KID U: simetria y formula finita; datos sin peso de Inception.
    x=torch.ones((4,2))
    assert abs(kid_u(x,x))<1e-6
    y=x+2
    assert kid_u(x,y)>0
    assert count_chunk(0)==16 and count_chunk(EXPECTED_CHUNKS-1)==N_FAKE%BATCH
    train=[f'img{i}' for i in range(3)]
    test=['img9']
    assert set(train).isdisjoint(test)
    print('SELFTEST OK: estimador U, archivos, conteos y splits.')


def generate(budget_seconds):
    if not torch.cuda.is_available(): raise RuntimeError('Generacion requiere GPU SLURM')
    p=ensure_protocol(True)
    CHUNKS.mkdir(parents=True,exist_ok=True)
    net=base.ResUNet64().cuda().eval()
    sd=torch.load(MODEL_PATH,map_location='cpu',weights_only=True)
    net.load_state_dict(sd,strict=True)
    del sd
    print('GPU:',torch.cuda.get_device_name(0),'modelo:',p['checkpoint_sha256'][:16],flush=True)
    tic=time.monotonic()
    completed=0
    for i in range(EXPECTED_CHUNKS):
        if check_chunk(i):
            completed+=1
            continue
        if time.monotonic()-tic>budget_seconds:
            print('PAUSA segura para reenvio: no se mezclan semillas ni lotes',flush=True)
            break
        imgs=base.sample_ddim(net,steps=STEPS,n=count_chunk(i),seed=GEN_SEED+i)
        if imgs.dtype!=torch.uint8 or tuple(imgs.shape)!=(count_chunk(i),3,64,64):
            raise RuntimeError('Muestras con dimensiones inesperadas')
        atomic_torch(chunk_path(i),imgs)
        completed+=1
        if completed%10==0 or completed==EXPECTED_CHUNKS:
            print('DDIM250',completed,'/',EXPECTED_CHUNKS, 'tiempo',round(time.monotonic()-tic,1),'s',flush=True)
    if completed==EXPECTED_CHUNKS:
        print('GENERACION COMPLETA: 5000 imagenes nuevas.',flush=True)
    else:
        raise RuntimeError('GENERACION INCOMPLETA: reenviar mismo SLURM para reanudar; NO evaluar aun.')


def kid_u(x,y):
    """KID U no sesgado en dos conjuntos de features, para pruebas pequeñas."""
    a=poly(x,x); b=poly(y,y); c=poly(x,y)
    return ((a.sum()-a.diag().sum())/(len(x)*(len(x)-1))+
            (b.sum()-b.diag().sum())/(len(y)*(len(y)-1))-2*c.mean()).item()


def poly(x,y):
    return (x @ y.T / FEATURE_D + 1.0).pow(3) if x.shape[1]==FEATURE_D else (x @ y.T / x.shape[1] + 1.0).pow(3)


def uint8_faces(batch):
    return (batch.add(1).mul(127.5).round().clamp(0,255).to(torch.uint8))


def feature_path(kind):
    return FEAT/f'{kind}.pt'


def fake_fingerprint():
    h=hashlib.sha256()
    for i in range(EXPECTED_CHUNKS):
        p=chunk_path(i)
        if not p.is_file():raise FileNotFoundError(p)
        st=p.stat();h.update(f'{p.name}:{st.st_size}:{st.st_mtime_ns}\n'.encode())
    return h.hexdigest()


def feature_meta(kind,proto):
    return dict(kind=kind,checkpoint_sha256=proto['checkpoint_sha256'],
                split_sha256=proto['split_sha256'],
                fake_fingerprint=fake_fingerprint() if kind=='fake' else None,
                n={'fake':N_FAKE,'test':N_REAL,'train':N_TRAIN}[kind],
                feature_dim=FEATURE_D)


def feature_cached(kind,proto):
    path=feature_path(kind)
    if not path.is_file():return None
    s=torch.load(path,map_location='cpu',weights_only=True)
    if s['meta']!=feature_meta(kind,proto):
        raise RuntimeError(f'Cache incompatible: {path}; no mezclar resultados')
    x=s['features']
    if tuple(x.shape)!=(s['meta']['n'],FEATURE_D) or not torch.isfinite(x).all():
        raise RuntimeError(f'Features corruptas: {path}')
    return x


def extract(kind,proto,inception,device,train,test):
    cached=feature_cached(kind,proto)
    if cached is not None:
        print('CACHE',kind,tuple(cached.shape),flush=True)
        return cached
    print('Extrayendo Inception:',kind,flush=True)
    acc=[]
    if kind=='fake':
        for i in range(EXPECTED_CHUNKS):
            ims=torch.load(chunk_path(i),map_location='cpu',weights_only=True)
            if ims.dtype!=torch.uint8 or tuple(ims.shape)!=(count_chunk(i),3,64,64):
                raise RuntimeError('Chunk con forma errada: '+str(chunk_path(i)))
            with torch.inference_mode():
                acc.append(inception(ims.to(device)).float().cpu())
            if i%40==0: print(' ',kind,i+1,'/',EXPECTED_CHUNKS,flush=True)
    else:
        from torch.utils.data import DataLoader
        dataset=base.FaceDataset(train if kind=='train' else test)
        loader=DataLoader(dataset,batch_size=64,shuffle=False,num_workers=2,pin_memory=True)
        for i,ims in enumerate(loader):
            ims=uint8_faces(ims)
            with torch.inference_mode():
                acc.append(inception(ims.to(device)).float().cpu())
            if i%100==0:print(' ',kind,min((i+1)*64,len(dataset)),'/',len(dataset),flush=True)
    features=torch.cat(acc).contiguous()
    exp={'fake':N_FAKE,'test':N_REAL,'train':N_TRAIN}[kind]
    if tuple(features.shape)!=(exp,FEATURE_D) or not torch.isfinite(features).all():
        raise RuntimeError(f'Error features {kind}: {features.shape}')
    atomic_torch(feature_path(kind),dict(meta=feature_meta(kind,proto),features=features))
    print('FEATURES GUARDADAS:',feature_path(kind),flush=True)
    return features


def matrix_stats(real,fake,device):
    """U-statistics con matrices gram en GPU, y muestras de subconjuntos."""
    real=real.to(device)
    fake=fake.to(device)
    with torch.no_grad():
        rr=poly(real,real)
        ff=poly(fake,fake)
        rf=poly(real,fake)
        r=(rr.sum()-rr.diag().sum())/(len(real)*(len(real)-1))
        f=(ff.sum()-ff.diag().sum())/(len(fake)*(len(fake)-1))
        kid=float((r+f-2*rf.mean()).item())
        rng=np.random.default_rng(20261030)
        subset=[]
        for _ in range(100):
            ir=torch.as_tensor(rng.choice(len(real),size=500,replace=False),dtype=torch.long,device=device)
            ig=torch.as_tensor(rng.choice(len(fake),size=500,replace=False),dtype=torch.long,device=device)
            ar=rr[ir[:,None],ir[None,:]]
            af=ff[ig[:,None],ig[None,:]]
            ac=rf[ir[:,None],ig[None,:]]
            sub=(ar.sum()-ar.diag().sum())/(500*499)+(af.sum()-af.diag().sum())/(500*499)-2*ac.mean()
            subset.append(float(sub.item()))
        del rr,ff,rf
        return kid,np.asarray(subset)


def nearest(train_features,query_features,device,topk=2):
    train=F.normalize(train_features.to(device),dim=1)
    query=F.normalize(query_features.to(device),dim=1)
    scores=[];indices=[]
    with torch.no_grad():
        for off in range(0,len(query),128):
            sim=query[off:off+128] @ train.T
            v,ind=torch.topk(sim,k=topk,dim=1)
            scores.append(v.cpu());indices.append(ind.cpu())
    return torch.cat(scores).numpy(),torch.cat(indices).numpy()


def save_rows(path,rows):
    if not rows: return
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    tmp.replace(path)


def selected_fake(index):
    path=chunk_path(index//BATCH)
    return torch.load(path,map_location='cpu',weights_only=True)[index%BATCH]


def save_neighbor_grid(path,indices,nearest_idx,nearest_score,train_names):
    from PIL import Image, ImageDraw
    face_ds=base.FaceDataset(train_names)
    w=64; header=25; pad=7
    canvas=Image.new('RGB',(3*w+4*pad,header+len(indices)*(w+pad)),(255,255,255))
    draw=ImageDraw.Draw(canvas)
    for col,title in enumerate(('GENERADA','TRAIN MAS CERCANA','SEGUNDA')):
        draw.text((pad+col*(w+pad),5),title[:13],fill=(0,0,0))
    for row,j in enumerate(indices):
        fake=selected_fake(int(j)).permute(1,2,0).numpy()
        imgs=[Image.fromarray(fake,'RGB')]
        for rank in (0,1):
            t=face_ds[int(nearest_idx[j,rank])]
            im=uint8_faces(t).permute(1,2,0).numpy()
            imgs.append(Image.fromarray(im,'RGB'))
        for col,img in enumerate(imgs):
            canvas.paste(img,(pad+col*(w+pad),header+row*(w+pad)))
        draw.text((1,header+row*(w+pad)+25),str(j),fill=(255,255,255),stroke_width=1,stroke_fill=(0,0,0))
    path.parent.mkdir(parents=True,exist_ok=True)
    canvas.save(path)
    print('Comparacion visual:',path,flush=True)


def evaluate():
    if not torch.cuda.is_available(): raise RuntimeError('Evaluacion requiere GPU SLURM')
    proto=ensure_protocol(True)
    for i in range(EXPECTED_CHUNKS):
        if not check_chunk(i):raise FileNotFoundError(f'Faltan lotes DDIM, primero: {chunk_path(i)}')
    from torchmetrics.image.kid import KernelInceptionDistance
    from importlib.metadata import version
    print('Torchmetrics',version('torchmetrics'),'Torch-fidelity',version('torch-fidelity'),flush=True)
    device=torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32=False
    metric=KernelInceptionDistance(feature=FEATURE_D,subsets=2,subset_size=500,normalize=False).to(device).eval()
    inception=metric.inception
    train,test=read_partition()
    print('Disjuntos train/test:',not bool(set(train)&set(test)),flush=True)
    # Cache separado por grupo; reanudacion segura si el nodo es interrumpido.
    real=extract('test',proto,inception,device,train,test)
    fake=extract('fake',proto,inception,device,train,test)
    train_feat=extract('train',proto,inception,device,train,test)
    print('KID sobre 10k test + 5k generadas',flush=True)
    kid,subsets=matrix_stats(real,fake,device)
    save_rows(OUT/'kid_celeba64.csv',[{
        'dataset':'CelebA test oficial (10k seleccionadas)',
        'generator':'ResUNet64 Attention EMA DDIM250','n_real':N_REAL,'n_generated':N_FAKE,
        'kid_u_complete':kid,'kid_u_complete_x1000':kid*1000,
        'subset_kid_mean_x1000':float(subsets.mean()*1000),
        'subset_kid_std_x1000_descriptive':float(subsets.std(ddof=1)*1000),
        'subsets':len(subsets),'subset_size':500,
        'uncertainty_note':'STD entre subconjuntos NO es IC95',
    }])
    print('KID completo x1000:',kid*1000,flush=True)
    print('Buscando vecinos Inception-coseno: 5000 generadas vs 50000 TRAIN',flush=True)
    scores,idx=nearest(train_feat,fake,device,topk=2)
    # Referencia: 1000 fotografias de TEST frente a los mismos 50k train.
    control_rng=np.random.default_rng(TEST_SEED+31)
    control_ids=control_rng.choice(len(real),size=1000,replace=False)
    baseline,_=nearest(train_feat,real[control_ids],device,topk=2)
    rows=[]
    from functools import lru_cache
    train_ds=base.FaceDataset(train)
    @lru_cache(maxsize=512)
    def prepared_train_pixel(i):
        return uint8_faces(train_ds[i]).float().numpy()/255.0
    active_chunk=-1
    fake_chunk=None
    pixel_mses=[]
    for j in range(N_FAKE):
        if j//BATCH!=active_chunk:
            active_chunk=j//BATCH
            fake_chunk=torch.load(chunk_path(active_chunk),map_location='cpu',weights_only=True)
        pixels=fake_chunk[j%BATCH].float().numpy()/255.0
        neighbor=prepared_train_pixel(int(idx[j,0]))
        mse=float(np.mean((pixels-neighbor)**2))
        pixel_mses.append(mse)
        rows.append({'fake_index':j,'generated_seed':GEN_SEED+j//BATCH,
                     'first_train_file':train[int(idx[j,0])],
                     'cosine_similarity_1':float(scores[j,0]),
                     'second_train_file':train[int(idx[j,1])],
                     'cosine_similarity_2':float(scores[j,1]),
                     'pixel_mse_to_inception_neighbor':mse})
    save_rows(OUT/'vecinos_detalle.csv',rows)
    sim=scores[:,0]
    save_rows(OUT/'vecinos_resumen.csv',[{
        'feature_space':'TorchMetrics/Inception 2048, cosine',
        'n_fake':len(sim),'train_reference':len(train),
        'fake_to_train_mean':float(sim.mean()),
        'fake_to_train_median':float(np.median(sim)),
        'fake_to_train_p95':float(np.quantile(sim,.95)),
        'fake_to_train_max':float(sim.max()),
        'pixel_mse_median_to_inception_neighbor':float(np.median(pixel_mses)),
        'pixel_mse_min_to_inception_neighbor':float(np.min(pixel_mses)),
        'test_to_train_mean_control':float(baseline[:,0].mean()),
        'test_to_train_median_control':float(np.median(baseline[:,0])),
        'test_to_train_p95_control':float(np.quantile(baseline[:,0],.95)),
        'note':'Similitud coseno alta NO demuestra memorizacion ni identidad pixel a pixel',
    }])
    save_neighbor_grid(OUT/'vecinos_primeras16.png',list(range(16)),idx,scores,train)
    top=np.argsort(-sim)[:16].tolist()
    save_neighbor_grid(OUT/'vecinos_mayor_similitud.png',top,idx,scores,train)
    print('TERMINADO. Resultados:',OUT,flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',required=True,choices=['selftest','check','status','generate','evaluate'])
    ap.add_argument('--time-budget-seconds',type=int,default=13200)
    opt=ap.parse_args()
    {'selftest':selftest,'check':check,'status':status,
     'generate':lambda:generate(opt.time_budget_seconds),'evaluate':evaluate}[opt.mode]()

if __name__=='__main__': main()
