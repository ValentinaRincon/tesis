#!/usr/bin/env python3
"""Trayectorias x_t y estimación denoised x0 para CelebA128 EMA ep60.

Pensado para ejecutarse en ~/tesis con el código de calidad_celeba128.py ya
instalado. Modos selftest/check/status no importan torch y son seguros en login.
SOLO --mode run debe ejecutarse en un nodo GPU de SLURM.

Reutiliza la misma arquitectura, normalización, schedule, seeds y ecuaciones de
muestreo del experimento de calidad. No escribe en carpetas de checkpoints.
"""
import argparse
import hashlib
import json
import os
import shutil
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'entrenar_celeba128.py'
QUALITY = ROOT / 'calidad_celeba128.py'
CKPT = ROOT / 'CelebA 128x128 Refinamiento' / 'checkpoint.pt'
OUT = ROOT / 'CelebA 128x128 Apendice Trayectorias'
ARCHIVE = ROOT / 'CelebA128_apendice_trayectorias_ep60.zip'
N = 4
MICROBATCH = 2
SEED = 20261120  # Igual al experimento visual ep60 anterior.
METHODS = ('ddim250', 'ddim500', 'ddpm1000')
TARGET_T = (999, 900, 800, 700, 600, 500, 400, 300, 200, 150, 100, 50, 10)
# Al final se añade una columna etiquetada "final", t=0 después del último paso.
TILE = 192


def sha256_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def timestamps_python(method):
    if method == 'ddpm1000':
        return list(range(999, -1, -1))
    count = int(method.replace('ddim', ''))
    return list(reversed([round(999 * k / (count - 1)) for k in range(count)]))


def select_stages(timesteps):
    """Elige t real más cercano a cada objetivo y conserva el orden descendente."""
    actual = [min(timesteps, key=lambda t: abs(t - target)) for target in TARGET_T]
    if len(actual) != len(set(actual)) or actual != sorted(actual, reverse=True):
        raise RuntimeError(f'Capturas duplicadas o desordenadas: {actual}')
    return actual


def expected_files(method):
    return [
        OUT / f'trayectoria_{method}_xt.png',
        OUT / f'trayectoria_{method}_x0_estimado.png',
        OUT / f'trayectoria_{method}_x0_detalle.png',
        OUT / f'protocolo_{method}.json',
    ]


def selftest():
    for m in METHODS:
        ts = timestamps_python(m)
        stages = select_stages(ts)
        assert ts[0] == 999 and ts[-1] == 0
        assert len(ts) == (1000 if m == 'ddpm1000' else int(m.replace('ddim', '')))
        assert len(ts) == len(set(ts)) and len(stages) == len(TARGET_T)
        assert max(abs(a - b) for a, b in zip(stages, TARGET_T)) <= 3
        print(f'{m}: {len(ts)} pasos; t de capturas = {stages}')
    assert N % MICROBATCH == 0
    print('SELFTEST OK (sin torch ni GPU)')


def check():
    paths = [BASE, QUALITY, CKPT]
    for p in paths:
        print(('OK   ' if p.is_file() else 'FALTA'), p)
    if not all(p.is_file() for p in paths):
        raise FileNotFoundError('Falta un archivo del proyecto de CelebA128')
    # Los estados exactos se validan al cargar el checkpoint en GPU.
    print('CHECK OK: script de muestreo y checkpoint EMA de ep60 disponibles')


def status():
    for m in METHODS:
        files = expected_files(m)
        print(m, ':', 'LISTO' if all(p.is_file() for p in files) else 'PENDIENTE')
        for p in files:
            print('  ', 'SI' if p.is_file() else 'NO', p.name)
    print('ZIP completo:', 'SI' if ARCHIVE.is_file() else 'NO', ARCHIVE)


def to_byte_tensor(x):
    """Convierte valores [-1,1] a UINT8, clip fijo (NO auto-normalización)."""
    return (x.detach().float().clamp(-1, 1).add(1).mul(127.5)).round().byte().cpu()


def load_gpu():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Se requiere una GPU asignada con sbatch; no ejecutar en login')
    import calidad_celeba128 as quality
    model, original = quality.ck_verified(60, torch)
    model.eval()
    return torch, model, original


def run_sampler(torch, model, original, method):
    # Semillas, semántica y ruido DDPM equivalentes al código de calidad_celeba128.
    xT = torch.randn((N, 3, 128, 128), generator=torch.Generator(device='cpu').manual_seed(SEED))
    rng_ddpm = torch.Generator(device='cuda').manual_seed(SEED + 900000)
    beta = torch.linspace(original.CONFIG['beta_start'], original.CONFIG['beta_end'],
                          original.CONFIG['t_steps'], dtype=torch.float32, device='cuda')
    alpha = 1.0 - beta
    abar = torch.cumprod(alpha, dim=0)
    if method == 'ddpm1000':
        ts = list(range(999, -1, -1))
    else:
        count = int(method.replace('ddim', ''))
        ts = torch.linspace(0, 999, count, device='cpu').round().long().flip(0).tolist()
    if len(set(ts)) != len(ts):
        raise RuntimeError('Timesteps duplicados')
    stages = select_stages(ts)
    capture = set(stages)
    xs = {t: [] for t in stages}
    estimates = {t: [] for t in stages}
    xs['final'] = []
    estimates['final'] = []
    begin = time.monotonic()
    with torch.inference_mode():
        for start in range(0, N, MICROBATCH):
            x = xT[start:start + MICROBATCH].cuda()
            for j, ti in enumerate(ts):
                t_tensor = torch.full((len(x),), int(ti), dtype=torch.long, device=x.device)
                with torch.autocast('cuda', dtype=torch.float16):
                    eps = model(x, t_tensor).float()
                a = abar[ti]
                # xhat_0 para visualización: se calcula antes del paso inverso.
                # Clip fijo únicamente al exportar PNG, no durante el muestreo.
                if ti in capture:
                    x0_hat = (x - (1 - a).sqrt() * eps) / a.sqrt()
                    xs[ti].append(to_byte_tensor(x))
                    estimates[ti].append(to_byte_tensor(x0_hat))
                if method == 'ddpm1000':
                    mean = (x - beta[ti] / (1 - a).sqrt() * eps) / alpha[ti].sqrt()
                    if ti > 0:
                        abar_prev = abar[ti - 1]
                        postvar = beta[ti] * (1 - abar_prev) / (1 - a)
                        z = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=rng_ddpm)
                        x = mean + postvar.sqrt() * z
                    else:
                        x = mean
                else:
                    x0_hat = (x - (1 - a).sqrt() * eps) / a.sqrt()
                    a_next = abar[ts[j + 1]] if j + 1 < len(ts) else a.new_tensor(1.0)
                    x = a_next.sqrt() * x0_hat + (1 - a_next).sqrt() * eps
                if j % 250 == 0:
                    print(f'  {method}: muestra {start + 1}-{start + len(x)} / {N}, '
                          f'paso {j + 1}/{len(ts)}, t={ti}', flush=True)
            f = to_byte_tensor(x)
            xs['final'].append(f)
            estimates['final'].append(f)
    # Consolidación por muestra: 4 caras fijadas, mismas entradas para todos los métodos.
    xt = {k: torch.cat(v, dim=0) for k, v in xs.items()}
    est = {k: torch.cat(v, dim=0) for k, v in estimates.items()}
    print(f'{method}: COMPLETO en {time.monotonic() - begin:.1f} s', flush=True)
    return stages, xt, est


def _font(size, heavy=False):
    from PIL import ImageFont
    variants = ['DejaVuSans-Bold.ttf', 'DejaVuSans.ttf'] if heavy else ['DejaVuSans.ttf']
    for name in variants:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def make_panel(data, stages, method, mode, filepath, only_late=False):
    """PNG a resolución alta: columnas t, filas diferentes muestras."""
    from PIL import Image, ImageDraw
    import numpy as np
    if only_late:
        order = [t for t in stages if t <= 500] + ['final']
    else:
        order = stages + ['final']
    cols = len(order)
    rows = N
    margin = 22
    label_w = 114
    header_h = 94
    footer_h = 92
    gap = 8
    w = margin * 2 + label_w + cols * TILE + (cols - 1) * gap
    h = header_h + rows * TILE + (rows - 1) * gap + footer_h
    canvas = Image.new('RGB', (w, h), '#ffffff')
    draw = ImageDraw.Draw(canvas)
    font = _font(22)
    bold = _font(27, True)
    small = _font(17)
    mode_label = 'Estado x_t' if mode == 'xt' else 'Estimación x0 desde x_t'
    draw.text((margin, 16), f'CelebA 128x128 | EMA época 60 | {method.upper()} | {mode_label}',
              fill='#111111', font=bold)
    draw.text((margin, 55), 'Mismo ruido inicial x_T para los tres muestreadores; t es el tiempo de difusión.',
              fill='#444444', font=small)
    xorigin = margin + label_w
    for j, label in enumerate(order):
        x = xorigin + j * (TILE + gap)
        tick = f't={label}' if label != 'final' else 'FINAL'
        if label == 'final':
            draw.rounded_rectangle((x - 2, header_h - 35, x + TILE + 2, header_h - 4),
                                   radius=6, fill='#e9f2fc')
        text_width = draw.textbbox((0, 0), tick, font=font)[2]
        draw.text((x + (TILE - text_width) / 2, header_h - 31), tick,
                  fill='#111111', font=font)
    for row in range(rows):
        y = header_h + row * (TILE + gap)
        draw.text((margin, y + 77), f'Muestra {row + 1}', fill='#222222', font=_font(19))
        for j, label in enumerate(order):
            img_arr = data[label][row].permute(1, 2, 0).numpy()
            assert img_arr.dtype == np.uint8 and img_arr.shape == (128, 128, 3)
            pic = Image.fromarray(img_arr, 'RGB').resize((TILE, TILE), Image.Resampling.BICUBIC)
            x = xorigin + j * (TILE + gap)
            canvas.paste(pic, (x, y))
    note = ('x_t = estado ruidoso antes del paso inverso; FINAL = imagen tras el último paso.'
            if mode == 'xt' else
            'x0 estimado = (x_t - sqrt(1-alpha_bar_t)*epsilon_theta)/sqrt(alpha_bar_t); '
            'recorte [-1,1] solo al visualizar.')
    draw.text((margin, h - 66), note, fill='#444444', font=small)
    draw.text((margin, h - 36), 'Figura exploratoria: mismo checkpoint, condiciones iniciales pareadas. '
              'DDPM incluye ruido adicional por paso.', fill='#666666', font=small)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    tmp = filepath.with_name(filepath.stem + '.tmp.png')
    canvas.save(tmp, format='PNG', compress_level=5)
    os.replace(tmp, filepath)
    print('FIGURA:', filepath.name, f'({w}x{h})', flush=True)


def manifest(method, stages, torch):
    return {
        'experiment': 'celeba128_appendix_trajectories_ema60_v1',
        'checkpoint': str(CKPT.relative_to(ROOT)),
        'checkpoint_sha256': sha256_file(CKPT),
        'base_model_sha256': sha256_file(BASE),
        'sampler_model_sha256': sha256_file(QUALITY),
        'sampler': method,
        'n_reverse_steps': len(timestamps_python(method)),
        'n_images': N,
        'initial_noise_seed': SEED,
        'noise_sampling': 'CPU torch.randn, same initial xT for all samplers',
        'ddpm_random_seed': SEED + 900000,
        'ddim_eta': 0,
        'states_t_actual': stages,
        'terminal_state': 'after t=0 reverse step',
        'normalization': 'model uses [-1,1]; display-only clamp [-1,1] -> 0..255',
        'interpretation': 'x_t noisy state vs x0 estimate; x0 estimate is not a physical path state',
        'torch_version': torch.__version__,
    }


def produce_archive():
    needed = [p for method in METHODS for p in expected_files(method)]
    if not all(p.is_file() for p in needed):
        print('ZIP no creado: faltan aún métodos o figuras', flush=True)
        return
    readme = ROOT / 'README_APENDICE_TRAYECTORIAS_128.md'
    tmp = ARCHIVE.with_name(ARCHIVE.name + '.tmp')
    with zipfile.ZipFile(tmp, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in needed:
            z.write(path, arcname=path.name)
        if readme.is_file():
            z.write(readme, arcname=readme.name)
    os.replace(tmp, ARCHIVE)
    print('ZIP DESCARGABLE:', ARCHIVE, flush=True)


def run(method):
    check()
    torch, model, original = load_gpu()
    methods = METHODS if method == 'all' else (method,)
    for m in methods:
        all_files = expected_files(m)
        data_path = all_files[-1]
        if data_path.is_file():
            # Si ya hay resultados completos, no toca el contenido anterior.
            if all(p.is_file() for p in all_files):
                prior = json.loads(data_path.read_text(encoding='utf8'))
                if prior.get('checkpoint_sha256') != sha256_file(CKPT) or prior.get('sampler') != m:
                    raise RuntimeError('El resultado existente corresponde a otro checkpoint/protocolo')
                print('YA COMPLETO', m, flush=True)
                continue
            print('Regenerando las figuras incompletas de', m, flush=True)
        stages, xt, est = run_sampler(torch, model, original, m)
        make_panel(xt, stages, m, 'xt', OUT/f'trayectoria_{m}_xt.png')
        make_panel(est, stages, m, 'x0_est', OUT/f'trayectoria_{m}_x0_estimado.png')
        make_panel(est, stages, m, 'x0_est', OUT/f'trayectoria_{m}_x0_detalle.png', only_late=True)
        payload = manifest(m, stages, torch)
        tmp = data_path.with_name(data_path.name + '.tmp')
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
        os.replace(tmp, data_path)
        print('MÉTODO COMPLETO:', m, flush=True)
    produce_archive()
    print('TRAYECTORIAS: FINALIZADO', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('check', 'status', 'selftest', 'run'), required=True)
    p.add_argument('--sampler', choices=(*METHODS, 'all'), default='all')
    args = p.parse_args()
    if args.mode == 'check':
        check()
    elif args.mode == 'status':
        status()
    elif args.mode == 'selftest':
        selftest()
    else:
        run(args.sampler)


if __name__ == '__main__':
    main()
