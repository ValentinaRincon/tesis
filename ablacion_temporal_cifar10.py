#!/usr/bin/env python3
"""Ablación temporal de predictores DDPM sobre CIFAR-10.

Usa Attention+EMA (A) y ResUNet+Attention+EMA (B) y cambia de
predictor en t=50,250,500. No vuelve a entrenar modelos.

COMPARACIÓN PAREADA:
  - Reutiliza las 5.000 muestras de A/A y B/B del experimento previo.
  - En cada híbrido, usa la misma semilla por lote, el mismo x_T y
    las mismas realizaciones z_t que la evaluación KID de 5.000.
  - Usa exactamente el mismo sampler DDPM ancestral y beta schedule.

Desde ~/tesis, login node:
    python ablacion_temporal_cifar10.py --mode check
    python ablacion_temporal_cifar10.py --mode status
En nodo GPU SLURM:
    python -u ablacion_temporal_cifar10.py --mode sample --variant-index 0
    python -u ablacion_temporal_cifar10.py --mode kid

No modifica modelos ni archivos de KID anteriores.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import torch

# Se importa el script utilizado previamente en Hypatia para mantener las
# arquitecturas y las convenciones del sampler exactamente iguales.
import evaluar_kid_cifar10_5000 as previous

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'Ablacion Temporal CIFAR10'
CHUNKS = OUT / 'chunks'
MODELS_DIR = ROOT / 'modelos'
DATA_DIR = ROOT / 'data'

N_SAMPLES = previous.N_SAMPLES  # 5000
BATCH_SIZE = previous.SAMPLE_BATCH_SIZE  # 32
T = previous.T  # 1000
BETA_START = previous.BETA_START
BETA_END = previous.BETA_END
SAMPLING_SEED = previous.SAMPLE_SEED  # 20261020
KID_SEED = previous.METRIC_SEED  # 42
KID_SUBSETS = previous.KID_SUBSETS  # 100
KID_SUBSET_SIZE = previous.KID_SUBSET_SIZE  # 500
KID_BATCH_SIZE = previous.KID_BATCH_SIZE

WEIGHTS_A = MODELS_DIR / 'cifar10_attention_ema_model.pt'
WEIGHTS_B = MODELS_DIR / 'cifar10_resunet_attention_ema_model.pt'

# t<=cutoff = "bajo ruido" durante el proceso inverso (pasos tardíos de generación).
CUTS = (50, 250, 500)
HYBRIDS = []
for cutoff in CUTS:
    HYBRIDS.append({
        'key': f'B_low_A_high_t{cutoff:03d}',
        'label': f'ResUNet bajo ruido, Attention alto ruido (t≤{cutoff})',
        'low': 'B', 'high': 'A', 'cutoff': cutoff,
    })
    HYBRIDS.append({
        'key': f'A_low_B_high_t{cutoff:03d}',
        'label': f'Attention bajo ruido, ResUNet alto ruido (t≤{cutoff})',
        'low': 'A', 'high': 'B', 'cutoff': cutoff,
    })

CONTROLS = [
    {'key': 'A_all', 'label': 'Attention + EMA, todo t',
     'low': 'A', 'high': 'A', 'cutoff': None, 'reference_key': 'attention_ema'},
    {'key': 'B_all', 'label': 'ResUNet + Attention + EMA, todo t',
     'low': 'B', 'high': 'B', 'cutoff': None, 'reference_key': 'resunet_ema'},
]


def hash_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def sha_weights():
    return {'A': hash_file(WEIGHTS_A), 'B': hash_file(WEIGHTS_B)}


def protocol():
    return {
        'experiment': 'time-window expert ablation',
        'n_samples': N_SAMPLES,
        'batch_size': BATCH_SIZE,
        'timesteps': T,
        'beta_start': BETA_START,
        'beta_end': BETA_END,
        'sampling_seed': SAMPLING_SEED,
        'sampler': 'DDPM ancestral posterior variance',
        'dtype': 'uint8 RGB',
        'cutoffs': list(CUTS),
        'rule': 'low model for t<=cutoff; high model for t>cutoff',
        'checkpoint_sha256': sha_weights(),
        'reference_script': 'evaluar_kid_cifar10_5000.py',
    }


def atomic_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def ensure_protocol():
    path = OUT / 'protocol.json'
    desired = protocol()
    if path.is_file():
        actual = json.loads(path.read_text(encoding='utf-8'))
        if actual != desired:
            raise RuntimeError('Protocolo o pesos diferentes en carpeta existente. No mezcles resultados.')
    else:
        atomic_json(path, desired)


def verify_controls(validate_data=True):
    """Control baselines reuse old cached arrays but MUST match all conditions."""
    if not WEIGHTS_A.is_file() or not WEIGHTS_B.is_file():
        raise FileNotFoundError(f'Faltan checkpoints EMA: {WEIGHTS_A}, {WEIGHTS_B}')
    hashes = sha_weights()
    target_count = math.ceil(N_SAMPLES / BATCH_SIZE)
    for c in CONTROLS:
        refkey = c['reference_key']
        manifest_path = previous.CHUNK_DIR / refkey / 'config.json'
        if not manifest_path.is_file():
            raise FileNotFoundError(f'Falta control previo: {manifest_path}')
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        model = c['low']
        filename = (WEIGHTS_A if model == 'A' else WEIGHTS_B).name
        expected = {'model': refkey, 'weights_file': filename,
                    'weights_sha256': hashes[model], **previous.settings()}
        if manifest != expected:
            raise RuntimeError(f'El control previo {refkey} no coincide en pesos/semilla/sampler.')
        if validate_data:
            count = previous.count_complete(refkey)
            if count != target_count:
                raise RuntimeError(f'{refkey}: {count}/{target_count} lotes; no usar como control.')
    return target_count


def destination(key, index):
    return CHUNKS / key / f'batch_{index:04d}.pt'


def expected_batch_size(index):
    return min(BATCH_SIZE, N_SAMPLES - index * BATCH_SIZE)


def good_chunk(key, index):
    return previous.validate_chunk(destination(key, index), expected_batch_size(index))


def count_chunks(key):
    return sum(good_chunk(key, i) for i in range(math.ceil(N_SAMPLES / BATCH_SIZE)))


def check():
    print('Ablación temporal CIFAR-10 — sin GPU / sin entrenamiento')
    print('Salida:', OUT)
    print('T:', T, 'N muestras/configuración:', N_SAMPLES)
    if previous.N_SAMPLES != 5000 or previous.SAMPLE_BATCH_SIZE != 32 or previous.T != 1000:
        raise RuntimeError('Se modificó el script base de KID; revisar antes de ejecutar.')
    for path in (WEIGHTS_A, WEIGHTS_B, DATA_DIR / 'cifar-10-batches-py' / 'test_batch'):
        print(('OK ' if path.is_file() else 'FALTA '), path)
        if not path.is_file():
            raise FileNotFoundError(path)
    for module in ('torchmetrics', 'torch_fidelity', 'torchvision'):
        if importlib.util.find_spec(module) is None:
            raise RuntimeError('Falta dependencia: ' + module)
    n = verify_controls()
    print(f'Dos controles anteriores validados: {n} lotes cada uno.')
    for i, c in enumerate(HYBRIDS):
        print(f'  {i}: {c["key"]}  (bajo={c["low"]}, alto={c["high"]})')
    print('Comprobación OK. Enviar SLURM; no ejecutar --mode sample en login.')


def status():
    total = math.ceil(N_SAMPLES / BATCH_SIZE)
    print(f'Ablación temporal: {N_SAMPLES} muestras / condición, {total} lotes')
    print('Controles previos:')
    for c in CONTROLS:
        print(' ', c['label'], '(reutilizado)', previous.count_complete(c['reference_key']), '/', total)
    print('Híbridos:')
    for c in HYBRIDS:
        print(' ', c['key'], count_chunks(c['key']), '/', total)
    csvfile = OUT / 'kid_ablation.csv'
    print('CSV:', csvfile if csvfile.exists() else 'todavía no disponible')


@torch.inference_mode()
def generate_hybrid_batch(models, variant, batch_size, generator, device, schedule):
    """Same DDPM update and RNG draw order as previous.generate_batch()."""
    betas, alphas, abar = schedule
    x = torch.randn((batch_size, 3, 32, 32), device=device, generator=generator)
    for step in range(T - 1, -1, -1):
        net = models[variant['low'] if step <= variant['cutoff'] else variant['high']]
        timesteps = torch.full((batch_size,), step, dtype=torch.long, device=device)
        eps_hat = net(x, timesteps)
        beta = betas[step]
        alpha = alphas[step]
        alpha_bar = abar[step]
        mean = (x - beta / torch.sqrt(1.0 - alpha_bar) * eps_hat) / torch.sqrt(alpha)
        if step == 0:
            x = mean
        else:
            previous_alpha_bar = abar[step - 1]
            posterior_var = beta * (1.0 - previous_alpha_bar) / (1.0 - alpha_bar)
            z = torch.randn(x.shape, dtype=x.dtype, device=device, generator=generator)
            x = mean + torch.sqrt(posterior_var) * z
    return ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).cpu()


def sample(index):
    if not torch.cuda.is_available():
        raise RuntimeError('Se necesita una GPU asignada por SLURM.')
    target_count = verify_controls(validate_data=False)
    ensure_protocol()
    variant = HYBRIDS[index]
    key = variant['key']
    manifest_path = CHUNKS / key / 'manifest.json'
    manifest = {'key': key, 'low': variant['low'], 'high': variant['high'],
                'cutoff': variant['cutoff'], 'protocol': protocol()}
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding='utf-8')) != manifest:
            raise RuntimeError('El manifiesto del híbrido no coincide; no sobrescribir.')
    else:
        atomic_json(manifest_path, manifest)
    missing = [i for i in range(target_count) if not good_chunk(key, i)]
    print(f'{key}: {target_count-len(missing)}/{target_count} lotes existentes', flush=True)
    if not missing:
        return
    device = torch.device('cuda:0')
    print('GPU:', torch.cuda.get_device_name(0), flush=True)
    models = {}
    for model_code, constructor, weights in (
        ('A', previous.SimpleUNetAttention, WEIGHTS_A),
        ('B', previous.ResUNetAttention, WEIGHTS_B),
    ):
        model = constructor().to(device)
        model.load_state_dict(previous.load_state_dict(weights, device), strict=True)
        model.eval()
        models[model_code] = model
    schedule = previous.get_schedule(device)
    for i in missing:
        generator = torch.Generator(device=device)
        generator.manual_seed(SAMPLING_SEED + 1009 * i)
        images = generate_hybrid_batch(models, variant, expected_batch_size(i), generator, device, schedule)
        output = destination(key, i)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix('.tmp')
        torch.save(images, temporary)
        temporary.replace(output)
        if i % 10 == 0 or i == target_count - 1:
            print(f'{key}: {i+1}/{target_count} lotes | {min((i+1)*BATCH_SIZE,N_SAMPLES)}/{N_SAMPLES} muestras', flush=True)
    print('MUESTREO COMPLETO:', key, flush=True)


def all_variants():
    return CONTROLS + HYBRIDS


def source_chunk(variant, i):
    if 'reference_key' in variant:
        return previous.chunk_path(variant['reference_key'], i)
    return destination(variant['key'], i)


def validate_all():
    verify_controls(validate_data=True)
    ensure_protocol()
    total = math.ceil(N_SAMPLES / BATCH_SIZE)
    for c in HYBRIDS:
        manifest = CHUNKS / c['key'] / 'manifest.json'
        if not manifest.is_file():
            raise RuntimeError('No existe manifiesto de ' + c['key'])
        correct = {'key': c['key'], 'low': c['low'], 'high': c['high'],
                   'cutoff': c['cutoff'], 'protocol': protocol()}
        if json.loads(manifest.read_text(encoding='utf-8')) != correct:
            raise RuntimeError('Manifiesto incompatible: ' + c['key'])
        n = count_chunks(c['key'])
        if n != total:
            raise RuntimeError(f'{c["key"]}: {n}/{total} lotes listos. Reanuda generación.')


def csv_save(rows):
    p = OUT / 'kid_ablation.csv'
    fields = ['variant','key','low_noise_model','high_noise_model','cutoff_t',
              'KID_mean','KID_std_subsets','KID_x1000','generated_count','real_count',
              'subset_size','subsets','feature_dim','kid_seed','reference_reused']
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix('.tmp')
    with tmp.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(p)


def draw_plots(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    OUT.mkdir(parents=True, exist_ok=True)
    x = list(range(len(rows)))
    means = [r['KID_x1000'] for r in rows]
    sd = [r['KID_std_subsets'] * 1000 for r in rows]
    names = [r['key'] for r in rows]
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x, means, yerr=sd, capsize=3)
    ax.set_xticks(x, names, rotation=30, ha='right')
    ax.set_ylabel('KID × 1000 (menor es mejor)')
    ax.set_title('Ablación temporal: predictores de ruido A y B (EMA)')
    ax.grid(axis='y', alpha=.25)
    fig.text(.01,.01,'Barras = desviación entre subconjuntos KID; NO son intervalos de confianza',fontsize=9)
    fig.tight_layout(rect=[0,.04,1,1])
    fig.savefig(OUT/'kid_ablation.png', dpi=220)
    plt.close(fig)

    keys = {r['key']: r for r in rows}
    fig, ax = plt.subplots(figsize=(9,5.5))
    for prefix, textlabel in (('B_low_A_high', 'ResUNet en bajo ruido / Attention en alto'),
                              ('A_low_B_high', 'Attention en bajo ruido / ResUNet en alto')):
        group = [keys[f'{prefix}_t{x:03d}'] for x in CUTS]
        ax.errorbar(CUTS, [r['KID_x1000'] for r in group],
                    yerr=[r['KID_std_subsets']*1000 for r in group],
                    marker='o', capsize=3, label=textlabel)
    ax.axhline(keys['A_all']['KID_x1000'], linestyle='--', label='Attention todo t')
    ax.axhline(keys['B_all']['KID_x1000'], linestyle=':', label='ResUNet todo t')
    ax.set_xlabel(r'Corte $t_c$: predictor de bajo ruido se usa para $t\leq t_c$')
    ax.set_ylabel('KID × 1000 (menor es mejor)')
    ax.set_title('Dependencia de la calidad generativa con la ventana temporal')
    ax.grid(alpha=.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT/'kid_vs_cutoff.png',dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(len(rows),4,figsize=(9,2.2*len(rows)))
    for ri, row in enumerate(rows):
        variant = next(v for v in all_variants() if v['key'] == row['key'])
        rgb = torch.load(source_chunk(variant, 0), map_location='cpu', weights_only=True)
        for col in range(4):
            img = rgb[col].permute(1,2,0).numpy()
            axes[ri,col].imshow(img,interpolation='nearest')
            axes[ri,col].axis('off')
        axes[ri,0].set_ylabel(row['key'], rotation=0, labelpad=90,va='center',fontsize=8)
    fig.suptitle('Primeras cuatro semillas pareadas por configuración')
    fig.tight_layout(rect=[0,0,1,.98])
    fig.savefig(OUT/'muestras_pareadas.png',dpi=200,bbox_inches='tight')
    plt.close(fig)


def calculate_kid():
    validate_all()
    if not torch.cuda.is_available():
        raise RuntimeError('El cálculo KID debe ejecutarse con GPU asignada.')
    from torchmetrics.image.kid import KernelInceptionDistance
    from torchvision.datasets import CIFAR10
    device = torch.device('cuda:0')
    # Identical implementation and features to previous KID 5000 experiment.
    metric = KernelInceptionDistance(
        feature=2048, subsets=KID_SUBSETS, subset_size=KID_SUBSET_SIZE,
        normalize=False, reset_real_features=False).to(device)
    cifar = CIFAR10(root=str(DATA_DIR), train=False, download=False)
    real_images = cifar.data
    with torch.inference_mode():
        for i in range(0, len(real_images), KID_BATCH_SIZE):
            real = torch.from_numpy(real_images[i:i+KID_BATCH_SIZE].copy()).permute(0,3,1,2).contiguous().to(device)
            metric.update(real, real=True)
    rows=[]
    for v in all_variants():
        print('KID para:', v['key'], flush=True)
        with torch.inference_mode():
            for i in range(math.ceil(N_SAMPLES/BATCH_SIZE)):
                batch = torch.load(source_chunk(v,i), map_location='cpu', weights_only=True).to(device)
                metric.update(batch, real=False)
            torch.manual_seed(KID_SEED)
            torch.cuda.manual_seed_all(KID_SEED)
            mean,std = metric.compute()
        rows.append({
            'variant':v['label'],'key':v['key'],
            'low_noise_model':v['low'],'high_noise_model':v['high'],
            'cutoff_t': v['cutoff'] if v['cutoff'] is not None else '',
            'KID_mean':float(mean.cpu()), 'KID_std_subsets':float(std.cpu()),
            'KID_x1000':float(mean.cpu())*1000,
            'generated_count':N_SAMPLES,'real_count':len(real_images),
            'subset_size':KID_SUBSET_SIZE,'subsets':KID_SUBSETS,
            'feature_dim':2048,'kid_seed':KID_SEED,
            'reference_reused':bool('reference_key' in v),
        })
        csv_save(rows)
        print(f'KID×1000={rows[-1]["KID_x1000"]:.4f}',flush=True)
        metric.reset()
    draw_plots(rows)
    print('EXPERIMENTO FINALIZADO:', OUT, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['check','status','sample','kid'], required=True)
    parser.add_argument('--variant-index',type=int,choices=range(len(HYBRIDS)))
    args=parser.parse_args()
    if args.mode == 'check': check()
    elif args.mode == 'status': status()
    elif args.mode == 'sample':
        if args.variant_index is None:
            parser.error('--variant-index 0..5 obligatorio para sample')
        sample(args.variant_index)
    elif args.mode == 'kid': calculate_kid()


if __name__ == '__main__':
    main()
