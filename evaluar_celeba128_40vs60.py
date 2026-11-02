#!/usr/bin/env python3
"""Evaluacion pareada CelebA128 epocas 40/60.

Login: check/status (sin importar PyTorch). GPU SLURM: generate/evaluate.
Solo lectura de checkpoints originales. Muestras y features persistentes, reanudables.
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

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'celeba'
SPLIT = DATA / 'list_eval_partition.txt'
IMAGES = DATA / 'img_align_celeba'
CK40 = ROOT / 'CelebA 128x128 ResUNet' / 'checkpoint.pt'
CK60 = ROOT / 'CelebA 128x128 Refinamiento' / 'checkpoint.pt'
OUT = ROOT / 'Evaluacion CelebA128 ep40 vs ep60'
CHUNKS = OUT / 'chunks'
FEATURES = OUT / 'features'
N_FAKE = 5000
N_TEST = 10000
N_TRAIN = 50000
N_CONTROL = 5000
BATCH = 16
STEPS = 250
FEATURE_D = 2048
N_BOOT = 400
GEN_SEED = 20261107
TEST_SEED = 20261107
BOOT_SEED = 20261108
TRAIN_CONTROL_SEED = 20261109
EXPECTED = math.ceil(N_FAKE / BATCH)


def atomic_json(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n', encoding='utf8')
    os.replace(tmp, p)


def atomic_torch(p, obj, torch):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f'.{os.getpid()}.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, p)


def save_csv(p, rows):
    if not rows: raise ValueError('CSV vacio: '+str(p))
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f'.{os.getpid()}.tmp')
    with tmp.open('w', newline='', encoding='utf8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, p)


def sha256(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def chunk_path(epoch, i):
    return CHUNKS / f'ep{epoch}' / f'lote_{i:04d}.pt'


def chunk_n(i):
    return min(BATCH, N_FAKE - i * BATCH)


def check():
    files = [ROOT/'entrenar_celeba128.py', ROOT/'entrenar_celeba64.py',
             ROOT/'refinar_celeba128.py', CK40, CK60, SPLIT]
    print('COMPROBACION LIGERA (sin cargar PyTorch ni checkpoints)')
    for p in files:
        print('OK   ' if p.is_file() else 'FALTA', p)
    print('Fotografias:', 'OK' if IMAGES.is_dir() else 'FALTA', IMAGES)
    if not all(p.is_file() for p in files) or not IMAGES.is_dir():
        raise FileNotFoundError('Faltan archivos para evaluacion; revise rutas.')
    for file, epoch in ((CK40,40),(CK60,60)):
        state = file.parent / 'estado.json'
        if not state.is_file(): raise FileNotFoundError('Falta estado: '+str(state))
        d = json.loads(state.read_text(encoding='utf8'))
        if int(d.get('epoch', -1)) != epoch or not d.get('complete'):
            raise RuntimeError(f'Checkpoint {epoch} no confirmado por estado.json: {d}')
    print('CHECK OK: checkpoints de epocas 40 y 60 confirmados mediante estado.json.')


def status():
    for epoch in (40, 60):
        count = sum(chunk_path(epoch,i).is_file() for i in range(EXPECTED))
        print(f'Imagenes generadas ep{epoch}: {sum(chunk_n(i) for i in range(EXPECTED) if chunk_path(epoch,i).is_file())}/{N_FAKE} ({count}/{EXPECTED} lotes)')
    for kind in ('test','train','fake40','fake60'):
        print(f'Inception {kind}:', 'SI' if (FEATURES/f'{kind}.pt').is_file() else 'NO')
    for name in ('kid_comparacion.csv','bootstrap_pareado.csv','vecinos_resumen.csv',
                 'kid_ep40_vs_ep60_IC95.png', 'diferencia_kid_60_menos_40_IC95.png',
                 'vecinos_mayor_similitud_40.png', 'vecinos_mayor_similitud_60.png'):
        print(name, 'SI' if (OUT/name).is_file() else 'NO')


def protocol():
    import entrenar_celeba128 as base
    return {
        'experiment': 'celeba128_ema_ep40_vs_60_paired_kid_nearest_v1',
        'checkpoints_sha256': {'40':sha256(CK40),'60':sha256(CK60)},
        'source_script_128_sha256': sha256(Path(base.__file__)),
        'split_sha256':sha256(SPLIT),
        'model_config':base.CONFIG,
        'image_preprocess':'RGB CenterCrop(160) Resize((128,128)) ToTensor Normalize(.5,.5)',
        'fake_count_per_epoch':N_FAKE, 'test_count':N_TEST, 'train_count':N_TRAIN,
        'train_control_count':N_CONTROL,'ddim_steps':STEPS,'eta':0,
        'seed_for_chunk':'20261107 + zero_based_chunk_index (identical initial Gaussian noise across epochs)',
        'chunk_size':BATCH,'microbatch':2, 'feature_dim':FEATURE_D,
        'kid_kernel':'(1 + dot(x,y) / 2048)**3; unbiased U-statistic on full sets',
        'kid_feature_extractor':'torchmetrics KernelInceptionDistance(feature=2048, normalize=False), Inception-v3',
        'bootstrap_replicates':N_BOOT,'bootstrap_seed':BOOT_SEED,
        'test_seed':TEST_SEED,'train_control_seed':TRAIN_CONTROL_SEED,
        'ci_method':'basic bootstrap 95%, paired real and fake indices; exploratory',
    }


def ensure_protocol(write=False):
    p = protocol()
    f = OUT / 'protocolo.json'
    if f.is_file():
        old = json.loads(f.read_text(encoding='utf8'))
        if old != p:
            raise RuntimeError('Protocolo/checkpoints distintos de los ya guardados; no mezclar experimentos.')
    elif write:
        atomic_json(f, p)
    return p


def read_partition():
    import entrenar_celeba128 as base
    train = base.image_list()  # mismas 50000 utilizadas para entrenar
    test = []
    with SPLIT.open(encoding='utf8') as f:
        for line in f:
            b = line.split()
            if len(b)==2 and b[1]=='2':test.append(b[0])
    if len(train)!=N_TRAIN or len(test)<N_TEST or set(train) & set(test):
        raise RuntimeError('Split CelebA inválido, fuga entre train y test o tamaño inesperado')
    test = sorted(random.Random(TEST_SEED).sample(test,N_TEST))
    return train,test


def load_model(epoch,torch,base):
    ck = torch.load(CK40 if epoch==40 else CK60,map_location='cpu',weights_only=False)
    if int(ck.get('epoch',-1))!=epoch:
        raise RuntimeError(f'Checkpoint no corresponde a epoca {epoch}')
    if epoch==40:
        if ck.get('protocol',{}).get('config',{}).get('experiment')!=base.CONFIG['experiment']:
            raise RuntimeError('Checkpoint original no es el del experimento de 128x128')
    else:
        if ck.get('refinement_protocol',{}).get('initial_epoch')!=40 or ck['refinement_protocol'].get('target_epoch')!=60:
            raise RuntimeError('Checkpoint 60 no pertenece al refinamiento esperado')
    model=base.ResUNet128().cuda().eval()
    model.load_state_dict(ck['ema'], strict=True)
    del ck
    return model


def generate(epoch,budget):
    import torch
    import entrenar_celeba128 as base
    if not torch.cuda.is_available():raise RuntimeError('Se necesita GPU mediante SLURM')
    p = ensure_protocol(write=True)
    print(f'Epoca {epoch}, GPU {torch.cuda.get_device_name(0)}, checkpoint {p["checkpoints_sha256"][str(epoch)][:16]}',flush=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = load_model(epoch,torch,base)
    start=time.monotonic()
    count=0
    for i in range(EXPECTED):
        f=chunk_path(epoch,i)
        if f.is_file():
            count+=1
            continue
        if time.monotonic()-start > budget:
            print(f'PAUSA SEGURA: epoca {epoch}, {count}/{EXPECTED} lotes. Reenviar el mismo job.',flush=True)
            return
        # Ambos checkpoints usan las mismas semillas en cada lote.
        images=base.sample_ddim(model,steps=STEPS,n=chunk_n(i),seed=GEN_SEED+i,microbatch=2)
        if images.dtype!=torch.uint8 or tuple(images.shape)!=(chunk_n(i),3,128,128):
            raise RuntimeError(f'Forma erronea lote {i}: {images.shape}')
        atomic_torch(f,images,torch)
        count+=1
        if count%10==0 or count==EXPECTED:
            print(f'ep{epoch}: {count}/{EXPECTED} lotes; {sum(chunk_n(j) for j in range(count))}/{N_FAKE} imgs; {time.monotonic()-start:.0f}s en esta ejecucion',flush=True)
    print(f'GENERACION ep{epoch} COMPLETA: {N_FAKE} imagenes',flush=True)


def verify_chunks(torch):
    for epoch in (40,60):
        for i in range(EXPECTED):
            f=chunk_path(epoch,i)
            if not f.is_file():raise FileNotFoundError(f'Faltan lotes ep{epoch}; primero: {f}')
            if f.stat().st_size < 1000:raise RuntimeError('Lote demasiado pequeño '+str(f))
    print('Todos los lotes presentes: 5000 imágenes por checkpoint.',flush=True)


def chunk_fingerprint(epoch):
    h=hashlib.sha256()
    for i in range(EXPECTED):
        f=chunk_path(epoch,i)
        s=f.stat()
        h.update(f'{f.name}:{s.st_size}:{s.st_mtime_ns}\n'.encode())
    return h.hexdigest()


def feature_meta(kind, p):
    meta={'experiment':p['experiment'],'kind':kind,'split_sha256':p['split_sha256'],
          'source_script_128_sha256':p['source_script_128_sha256'],
          'n':{'test':N_TEST,'train':N_TRAIN,'fake40':N_FAKE,'fake60':N_FAKE}[kind],
          'feature_dim':FEATURE_D,'extraction':'torchmetrics Inception-v3 KID 2048, uint8 0-255'}
    if kind in ('fake40','fake60'):
        epoch=int(kind[-2:]);meta['checkpoint_sha256']=p['checkpoints_sha256'][str(epoch)]
        meta['chunks_fingerprint']=chunk_fingerprint(epoch)
    else:
        meta['selection_seed']=TEST_SEED if kind=='test' else p['model_config']['data_seed']
    return meta


def feature_file(kind):
    return FEATURES / f'{kind}.pt'


def feature_load(kind,p,torch):
    f=feature_file(kind)
    if not f.is_file():return None
    obj=torch.load(f,map_location='cpu',weights_only=True)
    if obj['meta']!=feature_meta(kind,p):raise RuntimeError('Cache incompatible: '+str(f))
    x=obj['features']
    if tuple(x.shape)!=(obj['meta']['n'],FEATURE_D) or not torch.isfinite(x).all():
        raise RuntimeError('Features invalidas: '+str(f))
    print('CACHE',kind,tuple(x.shape),flush=True)
    return x


def extract(kind,p,inception,torch,base,train,test):
    old=feature_load(kind,p,torch)
    if old is not None:return old
    from torch.utils.data import DataLoader
    parts=[]
    tic=time.monotonic()
    if kind.startswith('fake'):
        epoch=int(kind[-2:])
        for i in range(EXPECTED):
            x=torch.load(chunk_path(epoch,i),map_location='cpu',weights_only=True)
            if x.dtype!=torch.uint8 or tuple(x.shape)!=(chunk_n(i),3,128,128):
                raise RuntimeError('Chunk corrupto: '+str(chunk_path(epoch,i)))
            with torch.inference_mode():parts.append(inception(x.cuda(non_blocking=True)).float().cpu())
            if (i+1)%50==0 or i+1==EXPECTED:print(f'Inception {kind}: lote {i+1}/{EXPECTED}',flush=True)
    else:
        ds=base.FaceDataset(train if kind=='train' else test)
        loader=DataLoader(ds,batch_size=32,shuffle=False,num_workers=2,pin_memory=True)
        for i,x in enumerate(loader):
            x=x.add(1).mul(127.5).round().clamp(0,255).to(torch.uint8)
            with torch.inference_mode():parts.append(inception(x.cuda(non_blocking=True)).float().cpu())
            if (i+1)%100==0 or (i+1)*32>=len(ds):print(f'Inception {kind}: {min((i+1)*32,len(ds))}/{len(ds)}',flush=True)
    values=torch.cat(parts).contiguous()
    if tuple(values.shape)!=(feature_meta(kind,p)['n'],FEATURE_D) or not torch.isfinite(values).all():
        raise RuntimeError(f'Inception inválido {kind}: {values.shape}')
    atomic_torch(feature_file(kind),{'meta':feature_meta(kind,p),'features':values},torch)
    print('FEATURES GUARDADAS',kind,tuple(values.shape),f'{time.monotonic()-tic:.0f}s',flush=True)
    return values


def gram(x,y,torch,block=384):
    # Acumula resultado en GPU FP32, matmul TF32 desactivado.
    result=torch.empty((len(x),len(y)),device=x.device,dtype=torch.float32)
    yt=y.T.contiguous()
    for off in range(0,len(x),block):
        z=x[off:off+block] @ yt
        result[off:off+block].copy_(z.div_(FEATURE_D).add_(1).pow_(3))
    return result


def kid_full(kxx,kyy,kxy,torch):
    nx,ny=kxx.shape[0],kyy.shape[0]
    xx=(kxx.sum(dtype=torch.float64)-kxx.diagonal().sum(dtype=torch.float64))/(nx*(nx-1))
    yy=(kyy.sum(dtype=torch.float64)-kyy.diagonal().sum(dtype=torch.float64))/(ny*(ny-1))
    xy=kxy.sum(dtype=torch.float64)/(nx*ny)
    return (xx+yy-2*xy).item()


def weighted_self(k,counts,torch):
    # counts: [replicas, N] con reemplazo; resta una sola diagonal por cada posicion remuestreada.
    n=int(counts[0].sum().item())
    w=counts.T.contiguous()
    a=k @ w
    quad=(w.double()*a.double()).sum(dim=0)
    diag=(counts.double()*k.diagonal().double().unsqueeze(0)).sum(dim=1)
    return (quad-diag)/(n*(n-1))


def weighted_cross(k,left,right,torch):
    n=int(left[0].sum().item());m=int(right[0].sum().item())
    q=k@right.T.contiguous()
    return (left.T.double()*q.double()).sum(dim=0)/(n*m)


def ci_basic(est, draws, np):
    a,b=np.quantile(draws,[.025,.975]);return 2*est-b,2*est-a


def plot_ci(points,filename,title,xlabel):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,ax=plt.subplots(figsize=(8.2,3.3))
    for y,(name,est,lo,hi) in enumerate(points):
        ax.errorbar([est],[y],xerr=np.array([[max(0,est-lo)],[max(0,hi-est)]]),fmt='o',capsize=5)
    ax.set_yticks(list(range(len(points))),[x[0] for x in points]);ax.invert_yaxis()
    ax.axvline(0,linestyle='--',color='gray',alpha=.6)
    ax.grid(axis='x',alpha=.2);ax.set_xlabel(xlabel);ax.set_title(title)
    fig.tight_layout();fig.savefig(OUT/filename,dpi=220);plt.close(fig)


def nearest(train,q,torch,topk=2):
    import torch.nn.functional as F
    t=F.normalize(train,dim=1)
    q=F.normalize(q,dim=1)
    vs=[];ids=[]
    for k in range(0,len(q),128):
        sim=q[k:k+128] @ t.T
        v,i=sim.topk(topk,dim=1)
        vs.append(v.cpu());ids.append(i.cpu())
    return torch.cat(vs).numpy(),torch.cat(ids).numpy()


def selected_fake(epoch,index,torch):
    t=torch.load(chunk_path(epoch,index//BATCH),map_location='cpu',weights_only=True)
    return t[index%BATCH].permute(1,2,0).numpy()


def save_neighbors(epoch,indices,neighbors,train_names,base,torch,path):
    from PIL import Image,ImageDraw
    dataset=base.FaceDataset(train_names)
    w=128;pad=8;head=34
    canvas=Image.new('RGB',(2*w+3*pad,head+len(indices)*(w+pad)),(255,255,255))
    draw=ImageDraw.Draw(canvas)
    draw.text((pad,10),f'Generada ep{epoch}',fill=(0,0,0))
    draw.text((w+2*pad,10),'Train más próxima (coseno)',fill=(0,0,0))
    for j,i in enumerate(indices):
        gen=Image.fromarray(selected_fake(epoch,int(i),torch),'RGB')
        real=dataset[int(neighbors[i,0])]
        real=real.add(1).mul(127.5).round().clamp(0,255).byte().permute(1,2,0).numpy()
        y=head+j*(w+pad)
        canvas.paste(gen,(pad,y));canvas.paste(Image.fromarray(real,'RGB'),(w+2*pad,y))
        draw.text((pad+4,y+4),str(i),fill='white',stroke_width=1,stroke_fill='black')
    canvas.save(path)


def evaluate():
    import numpy as np
    import torch
    import entrenar_celeba128 as base
    from torchmetrics.image.kid import KernelInceptionDistance
    from importlib.metadata import version
    if not torch.cuda.is_available():raise RuntimeError('Evaluacion necesita GPU SLURM')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.set_num_threads(int(os.getenv('SLURM_CPUS_PER_TASK','3')))
    p=ensure_protocol(write=True)
    verify_chunks(torch)
    train_names,test_names=read_partition()
    print('Train/test disjuntos',not bool(set(train_names)&set(test_names)),flush=True)
    device=torch.device('cuda')
    print('TorchMetrics:',version('torchmetrics'),'Torch-fidelity:',version('torch-fidelity'),flush=True)
    extractor=KernelInceptionDistance(feature=FEATURE_D,subsets=2,subset_size=500,normalize=False).to(device).eval().inception
    feats={}
    for kind in ('test','fake40','fake60','train'):
        feats[kind]=extract(kind,p,extractor,torch,base,train_names,test_names)
    del extractor
    torch.cuda.empty_cache()
    print('EXTRACCION COMPLETA. Calculando KID U y bootstrap pareado.',flush=True)
    x=feats['test'].to(device);a=feats['fake40'].to(device);b=feats['fake60'].to(device)
    rng=np.random.default_rng(TRAIN_CONTROL_SEED)
    control_ids=np.sort(rng.choice(N_TRAIN,N_CONTROL,replace=False))
    z=feats['train'][control_ids].to(device)
    Kr=gram(x,x,torch);Ka=gram(a,a,torch);Kb=gram(b,b,torch)
    Kra=gram(x,a,torch);Krb=gram(x,b,torch)
    Kz=gram(z,z,torch);Krz=gram(x,z,torch)
    kid40=kid_full(Kr,Ka,Kra,torch)
    kid60=kid_full(Kr,Kb,Krb,torch)
    kidctl=kid_full(Kr,Kz,Krz,torch)
    delta=kid60-kid40
    print(f'KID U×1000: ep40={kid40*1000:.5f}; ep60={kid60*1000:.5f}; Δ60−40={delta*1000:.5f}; real-vs-real={kidctl*1000:.5f}',flush=True)
    rng=np.random.default_rng(BOOT_SEED)
    cx=torch.as_tensor(rng.multinomial(N_TEST,np.full(N_TEST,1/N_TEST),size=N_BOOT),dtype=torch.float32,device=device)
    ca=torch.as_tensor(rng.multinomial(N_FAKE,np.full(N_FAKE,1/N_FAKE),size=N_BOOT),dtype=torch.float32,device=device)
    cz=torch.as_tensor(rng.multinomial(N_CONTROL,np.full(N_CONTROL,1/N_CONTROL),size=N_BOOT),dtype=torch.float32,device=device)
    xx=weighted_self(Kr,cx,torch)
    boot40=(xx+weighted_self(Ka,ca,torch)-2*weighted_cross(Kra,cx,ca,torch)).cpu().numpy()
    boot60=(xx+weighted_self(Kb,ca,torch)-2*weighted_cross(Krb,cx,ca,torch)).cpu().numpy()
    bootctl=(xx+weighted_self(Kz,cz,torch)-2*weighted_cross(Krz,cx,cz,torch)).cpu().numpy()
    bootdiff=boot60-boot40
    summary=[]
    for name,est,draw in [('ep40 vs test',kid40,boot40),('ep60 vs test',kid60,boot60),('ep60 − ep40',delta,bootdiff),('train real vs test',kidctl,bootctl)]:
        lo,hi=ci_basic(est,draw,np)
        summary.append(dict(comparacion=name,kid_u_x1000=est*1000,ic95_basico_inf_x1000=lo*1000,
                            ic95_basico_sup_x1000=hi*1000,bootstrap_replicas=N_BOOT,
                            n_test=N_TEST,n_modelo=N_FAKE if name!='train real vs test' else N_CONTROL,
                            note='IC exploratorio condicional a checkpoints/semillas; bootstrap básico pareado; KID menor = mejor'))
    save_csv(OUT/'kid_comparacion.csv',summary)
    save_csv(OUT/'bootstrap_pareado.csv',[dict(replica=i,kid40_x1000=boot40[i]*1000,
             kid60_x1000=boot60[i]*1000,diferencia_60_menos_40_x1000=bootdiff[i]*1000,
             baseline_real_x1000=bootctl[i]*1000) for i in range(N_BOOT)])
    plot_ci([('Época 40',*tuple(float(summary[0][k]) for k in ('kid_u_x1000','ic95_basico_inf_x1000','ic95_basico_sup_x1000'))),
             ('Época 60',*tuple(float(summary[1][k]) for k in ('kid_u_x1000','ic95_basico_inf_x1000','ic95_basico_sup_x1000'))),
             ('Real vs real',*tuple(float(summary[3][k]) for k in ('kid_u_x1000','ic95_basico_inf_x1000','ic95_basico_sup_x1000')))],
            'kid_ep40_vs_ep60_IC95.png','CelebA128: KID a lo largo del refinamiento', 'KID U × 1000 (IC básico bootstrap 95%)')
    plot_ci([('Δ = época 60 − época 40',*tuple(float(summary[2][k]) for k in ('kid_u_x1000','ic95_basico_inf_x1000','ic95_basico_sup_x1000')))],
            'diferencia_kid_60_menos_40_IC95.png', 'Diferencia pareada KID (negativo favorece época 60)', 'Δ KID U × 1000 (IC básico bootstrap 95%)')
    del Kr,Ka,Kb,Kra,Krb,Kz,Krz,cx,ca,cz,xx,x,a,b,z
    torch.cuda.empty_cache()
    print('Buscando vecinos train en espacio Inception (coseno)',flush=True)
    t=feats['train'].to(device)
    vals={};inds={}
    for epoch in (40,60):
        q=feats[f'fake{epoch}'].to(device)
        vals[epoch],inds[epoch]=nearest(t,q,torch)
        del q
    control_rng=np.random.default_rng(TEST_SEED+31)
    ids=control_rng.choice(N_TEST,size=1000,replace=False)
    baseline,_=nearest(t,feats['test'][ids].to(device),torch)
    rows=[];summary_n=[]
    for epoch in (40,60):
        s=vals[epoch][:,0]
        summary_n.append(dict(epoca=epoch,n_generated=N_FAKE,n_train=N_TRAIN,
              mean_cosine=float(s.mean()),median_cosine=float(np.median(s)),p95_cosine=float(np.quantile(s,.95)),
              max_cosine=float(s.max()),cosine_ge_095=int(np.sum(s>=.95)),
              test_to_train_mean_control=float(baseline[:,0].mean()),
              note='Similitud de Inception no demuestra ni descarta memorización'))
        for i in range(N_FAKE):
            rows.append(dict(epoca=epoch,indice_generado=i,seed_lote=GEN_SEED+i//BATCH,
                 train_mas_cercana=train_names[int(inds[epoch][i,0])],coseno_1=float(vals[epoch][i,0]),
                 train_segunda=train_names[int(inds[epoch][i,1])],coseno_2=float(vals[epoch][i,1])))
        top=np.argsort(-s)[:12].tolist()
        save_neighbors(epoch,top,inds[epoch],train_names,base,torch,OUT/f'vecinos_mayor_similitud_{epoch}.png')
    save_csv(OUT/'vecinos_resumen.csv',summary_n)
    save_csv(OUT/'vecinos_detalle.csv',rows)
    atomic_json(OUT/'resumen.json',{'KID_x1000':{row['comparacion']:row['kid_u_x1000'] for row in summary},
                'delta_60_menos_40_x1000':delta*1000,'bootstrap':N_BOOT,
                'interpretacion':'Delta negativo favorece epoca 60, pero la importancia depende del IC. IC condicional a checkpoint y semillas.'})
    print('EVALUACION COMPLETA:',OUT,flush=True)


def selftest():
    # Solo NumPy, evita cargar PyTorch en login.
    import numpy as np
    rng=np.random.default_rng(7)
    xa=rng.normal(size=(6,4));ya=rng.normal(size=(7,4))
    fun=lambda u,v: (1+u@v.T/4)**3
    kxx,kyy,kxy=fun(xa,xa),fun(ya,ya),fun(xa,ya)
    u=lambda xx,yy,xy:((xx.sum()-np.trace(xx))/(len(xx)*(len(xx)-1))+
                        (yy.sum()-np.trace(yy))/(len(yy)*(len(yy)-1))-2*xy.mean())
    constant=np.ones((6,4))
    kc=fun(constant,constant)
    assert abs(u(kc,kc,kc))<1e-9  # muestras constantes identicas: U=0
    i=np.array([0,1,1,2,3,5]);j=np.array([0,2,2,3,4,5,6])
    assert np.isfinite(u(kxx[np.ix_(i,i)],kyy[np.ix_(j,j)],kxy[np.ix_(i,j)]))
    assert sum(chunk_n(i) for i in range(EXPECTED))==N_FAKE
    assert chunk_n(EXPECTED-1)==N_FAKE % BATCH
    print('SELFTEST OK: identidad MMD, bootstrap con repeticion, conteos de chunks')


def main():
    pa=argparse.ArgumentParser()
    pa.add_argument('--mode',required=True,choices=('check','status','selftest','generate','evaluate'))
    pa.add_argument('--epoch',type=int,choices=(40,60),default=None)
    pa.add_argument('--time-budget-seconds',type=int,default=20400)
    arg=pa.parse_args()
    if arg.mode=='check':check()
    elif arg.mode=='status':status()
    elif arg.mode=='selftest':selftest()
    elif arg.mode=='generate':
        if arg.epoch is None:pa.error('--epoch 40 o --epoch 60 es obligatorio para generate')
        generate(arg.epoch,arg.time_budget_seconds)
    elif arg.mode=='evaluate':evaluate()

if __name__=='__main__':main()
