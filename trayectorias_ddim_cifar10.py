#!/usr/bin/env python3
"""Fase B, posterior a A: trayectorias DDIM pareadas con los checkpoints EMA originales.
Compara (i) Attention/ResUNet a iguales pasos y (ii) diferencias terminales
al cambiar 1000/250/50 pasos dentro de cada modelo. NO entrena.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

import evaluar_kid_cifar10_5000 as old
import comparar_ddim_cifar10 as ddim

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'Trayectorias DDIM CIFAR10'
STEPS=(1000,250,50)
N=16
SEED=20261103
MODEL_INFO=[('attention',old.SimpleUNetAttention,ROOT/'modelos/cifar10_attention_ema_model.pt'),
            ('resunet',old.ResUNetAttention,ROOT/'modelos/cifar10_resunet_attention_ema_model.pt')]


def hashfile(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def protocol():
    return {'experiment':'DDIM deterministic paired trajectories','weights':{k:hashfile(p) for k,_,p in MODEL_INFO},
            'architecture_source_sha256':hashfile(ROOT/'evaluar_kid_cifar10_5000.py'),
            'ddim_source_sha256':hashfile(ROOT/'comparar_ddim_cifar10.py'),
            'steps':list(STEPS),'n_images':N,'seed':SEED,'T':1000,
            'same_x_T_all_architectures_and_step_counts':True,'eta':0,
            'trajectory_metric':'RMS pixel difference, averaged across images',
            'note':'for different step counts, snapshots may be at different physical t; compare terminal x0 directly'}


def check():
    for _,_,p in MODEL_INFO:
        print('PESOS',p,'OK' if p.is_file() else 'FALTA')
        if not p.is_file():raise FileNotFoundError(p)
    print('Protocolo:',OUT)
    p=OUT/'protocolo.json'
    if p.exists() and json.loads(p.read_text())!=protocol():
        raise RuntimeError('Protocolo incompatible, no mezclar resultados')
    print('CHECK OK. Calculo exclusivamente en GPU SLURM.')


def selftest():
    for steps in STEPS:
        ts=ddim.timetable(steps)
        assert len(ts)==steps and ts[0]==0 and ts[-1]==999
    a=torch.tensor([[[[1.,2.]]]]);b=torch.tensor([[[[0.,1.]]]])
    assert abs((a-b).square().mean().sqrt().item()-1.0)<1e-6
    print('SELFTEST OK.')


def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    temp=path.with_suffix('.tmp')
    with temp.open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    temp.replace(path)


@torch.inference_mode()
def run():
    if not torch.cuda.is_available():raise RuntimeError('Requiere GPU SLURM')
    check();OUT.mkdir(parents=True,exist_ok=True)
    ddim.atomic_json(OUT/'protocolo.json',protocol())
    device=torch.device('cuda')
    models={}
    for name, cls,path in MODEL_INFO:
        m=cls().to(device)
        m.load_state_dict(old.load_state_dict(path,device),strict=True)
        m.eval();models[name]=m
    gen=torch.Generator(device=device).manual_seed(SEED)
    x_initial=torch.randn((N,3,32,32),generator=gen,device=device)
    _,_,abar=old.get_schedule(device)
    rows=[];ends={};snaprows=[]
    targets=[999,900,750,500,250,100,50,0]
    for steps in STEPS:
        print('TRAYECTORIAS',steps,'pasos',flush=True)
        grid=ddim.timetable(steps)
        xs={name:x_initial.clone() for name in models}
        # Ambos predictores usan tiempos y x_T inicial idénticos dentro de cada configuración.
        for j in range(len(grid)-1,-1,-1):
            t=grid[j];prev=grid[j-1] if j>0 else -1
            ts=torch.full((N,),t,device=device,dtype=torch.long)
            predictions={name:models[name](xs[name],ts) for name in models}
            rms=(xs['attention']-xs['resunet']).square().flatten(1).mean(1).sqrt()
            norms={name:xs[name].square().flatten(1).mean(1).sqrt().mean().item() for name in xs}
            cos=F.cosine_similarity(predictions['attention'].flatten(1),
                                    predictions['resunet'].flatten(1),dim=1).mean().item()
            rows.append({'steps':steps,'t':t,'next_t':prev,
                         'rms_separation':rms.mean().item(),
                         'std_separation_over_images':rms.std(unbiased=True).item(),
                         'rms_x_attention':norms['attention'],'rms_x_resunet':norms['resunet'],
                         'cos_epsilon_different_states':cos})
            for name in xs:
                earlier=abar[prev] if prev>=0 else xs[name].new_tensor(1.)
                xs[name]=ddim.ddim_update(xs[name],predictions[name],abar[t],earlier)
        for name in xs:ends[(name,steps)]=xs[name].detach().cpu().clone()
        print('FINAL',steps,'RMS diferencia entre arquitecturas',
              (xs['attention']-xs['resunet']).square().mean().sqrt().item(),flush=True)
    write_csv(OUT/'trayectorias_todas.csv',rows)
    # Reportar el estado físico más cercano a cada objetivo por separado.
    for steps in STEPS:
        rr=[r for r in rows if r['steps']==steps]
        for target in targets:
            r=min(rr,key=lambda z:abs(z['t']-target))
            snaprows.append({'steps':steps,'target_t':target,'actual_t':r['t'],
                             'rms_separation':r['rms_separation'],'rms_x_attention':r['rms_x_attention'],
                             'rms_x_resunet':r['rms_x_resunet']})
    write_csv(OUT/'snapshots_aproximados.csv',snaprows)
    terminal=[]
    for name in models:
        base=ends[(name,1000)]
        for steps in STEPS:
            val=ends[(name,steps)];d=(val-base).square().flatten(1).mean(1).sqrt()
            terminal.append({'architecture':name,'steps':steps,'reference_steps':1000,
                             'terminal_rms_vs_1000':d.mean().item(),
                             'terminal_std_over_images':d.std(unbiased=True).item()})
    for steps in STEPS:
        d=(ends[('attention',steps)]-ends[('resunet',steps)]).square().flatten(1).mean(1).sqrt()
        terminal.append({'architecture':'attention_vs_resunet','steps':steps,'reference_steps':steps,
                         'terminal_rms_vs_1000':d.mean().item(),
                         'terminal_std_over_images':d.std(unbiased=True).item()})
    write_csv(OUT/'distancias_finales.csv',terminal)
    torch.save({f'{name}_{steps}':t for (name,steps),t in ends.items()},OUT/'estados_finales.pt')
    plot(rows,ends)
    print('EXPERIMENTO B COMPLETADO:',OUT,flush=True)


def plot(rows,ends):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,5))
    for s in STEPS:
        rr=sorted((r for r in rows if r['steps']==s),key=lambda z:z['t'])
        ax.plot([r['t'] for r in rr],[r['rms_separation'] for r in rr],label=f'{s} pasos')
    ax.set_xlabel('t (al muestrear se recorre de derecha a izquierda)')
    ax.set_ylabel('RMS separación Attention–ResUNet (mismo t)')
    ax.set_title('Separación temporal DDIM determinista entre arquitecturas')
    ax.legend();ax.grid(alpha=.25);fig.tight_layout()
    fig.savefig(OUT/'separacion_trayectorias_vs_t.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(10,7))
    for i,name in enumerate(('attention','resunet')):
        for j,s in enumerate(STEPS):
            images=ends[(name,s)][:4]
            images=((images.clamp(-1,1)+1)*127.5).round().to(torch.uint8)
            tiles=torch.cat([torch.cat([images[0],images[1]],dim=2),
                             torch.cat([images[2],images[3]],dim=2)],dim=1)
            axes[i,j].imshow(tiles.permute(1,2,0).numpy(),interpolation='nearest')
            axes[i,j].axis('off');axes[i,j].set_title(f'{name} - {s} pasos')
    fig.suptitle('Mismas condiciones iniciales con distintas discretizaciones DDIM')
    fig.tight_layout();fig.savefig(OUT/'comparacion_terminal_pareada.png',dpi=180);plt.close(fig)


def status():
    for name in ('trayectorias_todas.csv','distancias_finales.csv',
                 'separacion_trayectorias_vs_t.png','comparacion_terminal_pareada.png'):
        print(name,'OK' if (OUT/name).is_file() else 'PENDIENTE')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',required=True,choices=['check','selftest','status','run'])
    a=p.parse_args()
    {'check':check,'selftest':selftest,'status':status,'run':run}[a.mode]()

if __name__=='__main__':main()
