#!/usr/bin/env python3
"""Continuacion controlada CelebA128, epoch 40 -> 60, sin modificar el experimento original.

En nodo login SOLO --mode check/status. El entrenamiento y la generacion
requieren un nodo GPU asignado por SLURM.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ORIGINAL_DIR = ROOT / 'CelebA 128x128 ResUNet'
ORIGINAL_CK = ORIGINAL_DIR / 'checkpoint.pt'
OUT = ROOT / 'CelebA 128x128 Refinamiento'
CHECKPOINT = OUT / 'checkpoint.pt'
PROTOCOL = OUT / 'protocolo.json'
START_EPOCH = 40
TARGET_EPOCH = 60
LR_SHARED = 2.5e-5
LR_NEW = 5.0e-5
SEED_COMPARE = 20261106


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def status():
    p = OUT / 'estado.json'
    print('Refinamiento:', OUT)
    if p.is_file():
        print(p.read_text(encoding='utf-8'))
    else:
        print('Aún no se ha iniciado el refinamiento. El entrenamiento original permanece intacto.')
    print('Checkpoint de refinamiento:', CHECKPOINT.is_file())
    print('Comparación DDIM250:', (OUT / 'comparacion_40_vs_60_ddim250.png').is_file())


def check():
    print('Script original 128:', (ROOT / 'entrenar_celeba128.py').is_file())
    print('Checkpoint original época 40:', ORIGINAL_CK.is_file())
    print('Estado original:', (ORIGINAL_DIR / 'estado.json').is_file())
    print('Datos CelebA:', (ROOT / 'data/celeba/img_align_celeba').is_dir())
    missing = [str(p) for p in [ROOT / 'entrenar_celeba128.py', ORIGINAL_CK,
                                 ROOT / 'data/celeba/img_align_celeba'] if not p.exists()]
    if missing:
        raise FileNotFoundError('Faltan archivos: ' + ', '.join(missing))
    state = ORIGINAL_DIR / 'estado.json'
    if state.is_file():
        info = json.loads(state.read_text(encoding='utf-8'))
        print('Estado del entrenamiento original:', info.get('epoch'), '/', info.get('epochs'))
        if int(info.get('epoch', -1)) != START_EPOCH:
            raise RuntimeError('Se esperaba checkpoint inicial de época 40: verificar antes de continuar')
    print('CHECK OK (rutas, sin cargar checkpoint ni PyTorch).')


def protocol_for_original(orig):
    return {
        'experiment': 'celeba128_finetune_epoch40_to60_v1',
        'original_checkpoint_sha256': sha256(ORIGINAL_CK),
        'source_128_script_sha256': sha256(Path(orig.__file__)),
        'initial_epoch': START_EPOCH,
        'target_epoch': TARGET_EPOCH,
        'lr_shared': LR_SHARED,
        'lr_new': LR_NEW,
        'preserve_optimizer_and_scaler': True,
        'preserve_ema': True,
        'previews_every_epochs': 5,
        'preview_ddim_steps': 50,
        'comparison_ddim_steps': [250, 500],
        'comparison_seed': SEED_COMPARE,
        'source_training_config': orig.CONFIG,
    }


def gpu_preflight(orig, torch):
    if not torch.cuda.is_available():
        raise RuntimeError('Este modo requiere GPU asignada por SLURM')
    if not ORIGINAL_CK.is_file():
        raise FileNotFoundError(ORIGINAL_CK)
    return protocol_for_original(orig)


def load_source(orig, torch, protocol):
    source = torch.load(ORIGINAL_CK, map_location='cpu', weights_only=False)
    if int(source.get('epoch', -1)) != START_EPOCH:
        raise RuntimeError(f'El checkpoint original está en época {source.get("epoch")}, no 40')
    if source.get('protocol', {}).get('config', {}).get('experiment') != orig.CONFIG['experiment']:
        raise RuntimeError('El checkpoint original no coincide con la arquitectura/protocolo de 128')
    return source


def set_refinement_lr(optimizer):
    if len(optimizer.param_groups) != 2:
        raise RuntimeError('Se esperaban dos grupos del optimizador (capas compartidas/nuevas)')
    optimizer.param_groups[0]['lr'] = LR_SHARED
    optimizer.param_groups[1]['lr'] = LR_NEW


def train(time_budget):
    import copy
    import time
    import torch
    from torch import nn
    from torch.nn import functional as F
    import entrenar_celeba128 as orig

    protocol = gpu_preflight(orig, torch)
    if PROTOCOL.is_file() and json.loads(PROTOCOL.read_text(encoding='utf-8')) != protocol:
        raise RuntimeError('Protocolo de refinamiento distinto: no se sobreescriben resultados anteriores')
    if CHECKPOINT.exists() and not PROTOCOL.exists():
        raise RuntimeError('Hay un checkpoint de refinamiento sin protocolo: revisar manualmente')
    OUT.mkdir(parents=True, exist_ok=True)
    orig.seed_all()
    torch.backends.cudnn.benchmark = True
    model = orig.ResUNet128().cuda()
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = orig.make_optimizer(model)
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    generator = torch.Generator().manual_seed(orig.CONFIG['seed'] + 500)
    rows = []
    step = 0
    if CHECKPOINT.exists():
        ck = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
        if ck.get('refinement_protocol') != protocol:
            raise RuntimeError('Checkpoint de refinamiento incompatible')
        start = int(ck['epoch'])
        model.load_state_dict(ck['model'])
        ema.load_state_dict(ck['ema'])
        optimizer.load_state_dict(ck['optimizer'])
        scaler.load_state_dict(ck['scaler'])
        rows = ck['rows']
        step = ck['global_step']
        orig.rng_restore(ck['rng'], generator)
        print(f'Reanuda refinamiento desde época {start}', flush=True)
    else:
        ck = load_source(orig, torch, protocol)
        start = int(ck['epoch'])
        model.load_state_dict(ck['model'])
        ema.load_state_dict(ck['ema'])
        optimizer.load_state_dict(ck['optimizer'])
        scaler.load_state_dict(ck['scaler'])
        rows = list(ck['rows'])
        step = int(ck['global_step'])
        orig.rng_restore(ck['rng'], generator)
        print('Inicializado desde checkpoint COMPLETO de época 40, incluidos EMA, Adam y estados RNG.', flush=True)
    del ck
    set_refinement_lr(optimizer)  # después de load_state_dict para no perder el cambio
    save_json(PROTOCOL, protocol)
    print('Learning rates:', [g['lr'] for g in optimizer.param_groups], flush=True)
    if start >= TARGET_EPOCH:
        print('REFINAMIENTO YA COMPLETO', flush=True)
        return

    names = orig.image_list()  # exactamente las mismas 50.000 imágenes, en el mismo orden
    ds = orig.FaceDataset(names)
    loader = torch.utils.data.DataLoader(ds,
        batch_size=orig.CONFIG['batch_size'], shuffle=True, generator=generator,
        num_workers=orig.CONFIG['workers'], pin_memory=True, drop_last=False)
    abar = orig.schedule('cuda')
    began = time.monotonic()
    print('GPU:', torch.cuda.get_device_name(0), 'muestras:', len(ds),
          'inicio:', start+1, 'meta:', TARGET_EPOCH, flush=True)
    for epoch in range(start, TARGET_EPOCH):
        tic = time.monotonic()
        model.train()
        total, seen = 0., 0
        for x0 in loader:
            x0 = x0.cuda(non_blocking=True)
            t = torch.randint(0, 1000, (len(x0),), device='cuda')
            eps = torch.randn_like(x0)
            a = abar[t][:, None, None, None]
            xt = a.sqrt()*x0 + (1-a).sqrt()*eps
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                pred = model(xt, t)
                loss = F.mse_loss(pred.float(), eps.float())
            if not torch.isfinite(loss):
                raise RuntimeError('Loss no finita')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), orig.CONFIG['max_grad_norm'])
            prev_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= prev_scale:
                with torch.no_grad():
                    for q, p in zip(ema.parameters(), model.parameters()):
                        q.lerp_(p, 1-orig.CONFIG['ema_decay'])
                    for q, p in zip(ema.buffers(), model.buffers()):
                        q.copy_(p)
            total += float(loss.item()) * len(x0)
            seen += len(x0)
            step += 1
        secs = time.monotonic()-tic
        mse = total/seen
        rows.append(dict(epoch=epoch+1,train_mse=mse,seconds=round(secs, 2),global_step=step))
        orig.pt_save(CHECKPOINT, dict(refinement_protocol=protocol,
            epoch=epoch+1, rows=rows, global_step=step,
            model=model.state_dict(), ema=ema.state_dict(),optimizer=optimizer.state_dict(),
            scaler=scaler.state_dict(),rng=orig.rng_state(generator)))
        # El estado se escribe solo después del checkpoint, para no anunciar progreso inexistente.
        save_json(OUT / 'estado.json',dict(epoch=epoch+1, target=TARGET_EPOCH,
            train_mse=round(mse, 8), elapsed_seconds=round(time.monotonic()-began, 1),
            complete=(epoch+1 == TARGET_EPOCH)))
        with (OUT / 'perdidas_adicionales.csv').open('w', encoding='utf-8') as f:
            f.write('epoch,train_mse,seconds,global_step\n')
            for row in rows:
                if int(row['epoch']) > START_EPOCH:
                    f.write(f"{row['epoch']},{row['train_mse']},{row['seconds']},{row['global_step']}\n")
        print(f'EPOCA {epoch+1}/{TARGET_EPOCH} MSE={mse:.6f} tiempo={secs:.1f}s', flush=True)
        if (epoch+1) % 5 == 0:
            imgs = orig.sample_ddim(ema, steps=50, n=8, seed=20261030)
            orig.save_grid(imgs, OUT / 'previews' / f'epoca_{epoch+1:03d}.png')
        if epoch+1 == TARGET_EPOCH:
            orig.pt_save(OUT / 'ema_final.pt', ema.state_dict())
            print('REFINAMIENTO COMPLETO, época 60', flush=True)
            break
        if time.monotonic()-began+max(secs*1.5, 300)>time_budget:
            print('PAUSA SEGURA; reenviar mismo SLURM para continuar.', flush=True)
            break


def render_pair(a, b, steps):
    from PIL import Image, ImageDraw
    from torchvision.utils import make_grid
    from torchvision.transforms.functional import to_pil_image
    lp = to_pil_image(make_grid(a.float()/255, nrow=4, padding=2))
    rp = to_pil_image(make_grid(b.float()/255, nrow=4, padding=2))
    canvas = Image.new('RGB', (lp.width + rp.width + 16, max(lp.height, rp.height) + 36), 'white')
    canvas.paste(lp, (0, 36))
    canvas.paste(rp, (lp.width+16, 36))
    draw = ImageDraw.Draw(canvas)
    draw.text((6,8), f'EMA epoca 40 | DDIM {steps}', fill='black')
    draw.text((lp.width+22,8), f'EMA epoca 60 | DDIM {steps}', fill='black')
    canvas.save(OUT / f'comparacion_40_vs_60_ddim{steps}.png')


def generate(mode):
    import torch
    import entrenar_celeba128 as orig
    protocol = gpu_preflight(orig, torch)
    if mode == 'compare':
        if not CHECKPOINT.is_file():
            raise FileNotFoundError('Falta checkpoint del refinamiento')
        refined = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
        if refined.get('refinement_protocol') != protocol:
            raise RuntimeError('Protocolo de refinamiento incompatible')
        if int(refined['epoch']) != TARGET_EPOCH:
            raise RuntimeError('Aún no termina el refinamiento hasta época 60')
    base = load_source(orig, torch, protocol)
    model = orig.ResUNet128().cuda().eval()
    base_ema = base['ema']
    model.load_state_dict(base_ema)
    del base
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for steps in (250, 500):
        a = orig.sample_ddim(model, steps=steps, n=16, seed=SEED_COMPARE, microbatch=2)
        orig.save_grid(a, OUT / f'epoca_040_ddim{steps}.png')
        if mode == 'baseline':
            print(f'BASELINE {steps} pasos:', OUT / f'epoca_040_ddim{steps}.png',flush=True)
            continue
        model.load_state_dict(refined['ema'])
        b = orig.sample_ddim(model, steps=steps, n=16, seed=SEED_COMPARE, microbatch=2)
        orig.save_grid(b, OUT / f'epoca_060_ddim{steps}.png')
        render_pair(a,b,steps)
        rms = ((a.float()/255 - b.float()/255)**2).mean().sqrt().item()
        results.append(dict(steps=steps, rms_normalized_0_1=rms))
        print(f'Comparación DDIM {steps}:', OUT / f'comparacion_40_vs_60_ddim{steps}.png', flush=True)
        model.load_state_dict(base_ema)  # Restaurar pesos de epoca 40 para la proxima discretizacion
    if mode == 'compare':
        save_json(OUT / 'comparacion.json', dict(
            steps=results, n=16, seed=SEED_COMPARE,
            note='RMS no es indicador de mejora de calidad: comparar visualmente y cuantificar si hace falta.'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, choices=['check','status','train','baseline','compare'])
    parser.add_argument('--time-budget-seconds', type=int, default=27600)
    args = parser.parse_args()
    if args.mode=='check': check()
    elif args.mode=='status': status()
    elif args.mode=='train': train(args.time_budget_seconds)
    elif args.mode in ('baseline','compare'): generate(args.mode)

if __name__ == '__main__': main()
