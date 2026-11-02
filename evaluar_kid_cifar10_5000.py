#!/usr/bin/env python3
"""KID con 5.000 muestras por cada uno de cuatro DDPM CIFAR-10 ya entrenados.

Uso en nodo login (operaciones ligeras):
    python evaluar_kid_cifar10_5000.py --mode check
Uso exclusivo en nodo GPU mediante SLURM:
    python -u evaluar_kid_cifar10_5000.py --mode sample --model-index $SLURM_ARRAY_TASK_ID
    python -u evaluar_kid_cifar10_5000.py --mode kid

No modifica los pesos originales. Guarda lotes de muestras uint8
para reanudar tras un corte sin repetir los lotes ya finalizados.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Solo importamos torchvision.datasets en --mode kid, para no descargar ni
# cargar imágenes de referencia en la etapa de generación.

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / 'modelos'
DATA_DIR = ROOT / 'data'
OUTPUT_DIR = ROOT / 'KID CIFAR10 5000'
CHUNK_DIR = OUTPUT_DIR / 'chunks'

T = 1000
BETA_START = 1e-4
BETA_END = 0.02
N_SAMPLES = 5000
SAMPLE_BATCH_SIZE = 32
KID_BATCH_SIZE = 64
KID_SUBSETS = 100
KID_SUBSET_SIZE = 500
SAMPLE_SEED = 20261020
METRIC_SEED = 42

# La arquitectura coincide EXACTAMENTE con los scripts previos.
# No incluir el baseline: el archivo asociado anteriormente
# contenía pesos de attention, incompatibles con SimpleUNet.

class SinusoidalTimeEmbedding(nn.Module):

    def __init__(
        self,
        dim
    ):

        super().__init__()

        self.dim = dim


    def forward(
        self,
        t
    ):

        half_dim = self.dim // 2

        frequencies = (
            math.log(10000)
            /
            (half_dim - 1)
        )

        frequencies = torch.exp(
            torch.arange(
                half_dim,
                device=t.device
            )
            *
            -frequencies
        )

        angles = (
            t.float()[:, None]
            *
            frequencies[None, :]
        )

        return torch.cat(
            [
                angles.sin(),
                angles.cos()
            ],
            dim=1
        )

class ResBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        time_dim
    ):

        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )

        self.norm1 = nn.GroupNorm(
            8,
            out_channels
        )

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )

        self.norm2 = nn.GroupNorm(
            8,
            out_channels
        )

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                time_dim,
                out_channels
            )
        )

        if in_channels != out_channels:

            self.residual = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1
            )

        else:

            self.residual = nn.Identity()


    def forward(
        self,
        x,
        t_emb
    ):

        h = self.conv1(x)

        h = self.norm1(h)

        h = F.silu(h)

        time_information = self.time_mlp(
            t_emb
        )[:, :, None, None]

        h = h + time_information

        h = self.conv2(h)

        h = self.norm2(h)

        h = h + self.residual(x)

        return F.silu(h)

class SelfAttentionBlock(nn.Module):

    def __init__(
        self,
        channels,
        num_heads=4
    ):

        super().__init__()

        self.norm = nn.GroupNorm(
            8,
            channels
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True
        )


    def forward(
        self,
        x
    ):

        residual = x

        x = self.norm(x)

        B, C, H, W = x.shape

        # [B,C,H,W] -> [B,H*W,C]
        x = x.flatten(2).transpose(1, 2)

        x_attended, _ = self.attention(
            x,
            x,
            x,
            need_weights=False
        )

        # [B,H*W,C] -> [B,C,H,W]
        x_attended = (
            x_attended
            .transpose(1, 2)
            .reshape(B, C, H, W)
        )

        return residual + x_attended

class SimpleUNetAttention(nn.Module):

    def __init__(
        self,
        time_dim=64
    ):

        super().__init__()

        self.time_embedding = nn.Sequential(

            SinusoidalTimeEmbedding(
                time_dim
            ),

            nn.Linear(
                time_dim,
                time_dim
            ),

            nn.SiLU(),

            nn.Linear(
                time_dim,
                time_dim
            )
        )

        self.input_conv = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            padding=1
        )

        self.block1 = ResBlock(
            64,
            64,
            time_dim
        )

        self.down1 = nn.Conv2d(
            64,
            128,
            kernel_size=4,
            stride=2,
            padding=1
        )

        self.block2 = ResBlock(
            128,
            128,
            time_dim
        )

        self.down2 = nn.Conv2d(
            128,
            256,
            kernel_size=4,
            stride=2,
            padding=1
        )

        self.middle = ResBlock(
            256,
            256,
            time_dim
        )

        self.attention = SelfAttentionBlock(
            channels=256,
            num_heads=4
        )

        self.up1 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=4,
            stride=2,
            padding=1
        )

        self.block3 = ResBlock(
            256,
            128,
            time_dim
        )

        self.up2 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=4,
            stride=2,
            padding=1
        )

        self.block4 = ResBlock(
            128,
            64,
            time_dim
        )

        self.output_conv = nn.Conv2d(
            64,
            3,
            kernel_size=1
        )


    def forward(
        self,
        x,
        t
    ):

        t_emb = self.time_embedding(t)

        x = self.input_conv(x)

        skip1 = self.block1(
            x,
            t_emb
        )

        x = self.down1(
            skip1
        )

        skip2 = self.block2(
            x,
            t_emb
        )

        x = self.down2(
            skip2
        )

        x = self.middle(
            x,
            t_emb
        )

        x = self.attention(
            x
        )

        x = self.up1(x)

        x = torch.cat(
            [
                x,
                skip2
            ],
            dim=1
        )

        x = self.block3(
            x,
            t_emb
        )

        x = self.up2(x)

        x = torch.cat(
            [
                x,
                skip1
            ],
            dim=1
        )

        x = self.block4(
            x,
            t_emb
        )

        return self.output_conv(x)

class ResUNetAttention(nn.Module):

    def __init__(
        self,
        time_dim=64
    ):

        super().__init__()

        # Mantenemos el mismo embedding temporal del experimento anterior
        # para que el cambio principal sea la arquitectura espacial.
        self.time_embedding = nn.Sequential(

            SinusoidalTimeEmbedding(
                time_dim
            ),

            nn.Linear(
                time_dim,
                time_dim
            ),

            nn.SiLU(),

            nn.Linear(
                time_dim,
                time_dim
            )
        )

        # -------------------------
        # Entrada: 32x32
        # -------------------------
        self.input_conv = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            padding=1
        )

        self.enc32_res1 = ResBlock(
            64,
            64,
            time_dim
        )

        self.enc32_res2 = ResBlock(
            64,
            64,
            time_dim
        )

        # 32x32 -> 16x16
        self.down32_to_16 = nn.Conv2d(
            64,
            128,
            kernel_size=4,
            stride=2,
            padding=1
        )

        # -------------------------
        # Encoder: 16x16
        # -------------------------
        self.enc16_res1 = ResBlock(
            128,
            128,
            time_dim
        )

        self.enc16_attention = SelfAttentionBlock(
            channels=128,
            num_heads=4
        )

        self.enc16_res2 = ResBlock(
            128,
            128,
            time_dim
        )

        # 16x16 -> 8x8
        self.down16_to_8 = nn.Conv2d(
            128,
            256,
            kernel_size=4,
            stride=2,
            padding=1
        )

        # -------------------------
        # Bottleneck: 8x8
        # -------------------------
        self.mid_res1 = ResBlock(
            256,
            256,
            time_dim
        )

        self.mid_attention = SelfAttentionBlock(
            channels=256,
            num_heads=4
        )

        self.mid_res2 = ResBlock(
            256,
            256,
            time_dim
        )

        # -------------------------
        # Decoder: 8x8 -> 16x16
        # -------------------------
        self.up8_to_16 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=4,
            stride=2,
            padding=1
        )

        # Tras concatenar con skip16:
        # 128 + 128 = 256 canales.
        self.dec16_res1 = ResBlock(
            256,
            128,
            time_dim
        )

        self.dec16_attention = SelfAttentionBlock(
            channels=128,
            num_heads=4
        )

        self.dec16_res2 = ResBlock(
            128,
            128,
            time_dim
        )

        # -------------------------
        # Decoder: 16x16 -> 32x32
        # -------------------------
        self.up16_to_32 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=4,
            stride=2,
            padding=1
        )

        # Tras concatenar con skip32:
        # 64 + 64 = 128 canales.
        self.dec32_res1 = ResBlock(
            128,
            64,
            time_dim
        )

        self.dec32_res2 = ResBlock(
            64,
            64,
            time_dim
        )

        self.output_conv = nn.Conv2d(
            64,
            3,
            kernel_size=1
        )


    def forward(
        self,
        x,
        t
    ):

        t_emb = self.time_embedding(t)

        # 32x32
        x = self.input_conv(x)

        x = self.enc32_res1(
            x,
            t_emb
        )

        x = self.enc32_res2(
            x,
            t_emb
        )

        skip32 = x

        # 16x16
        x = self.down32_to_16(
            x
        )

        x = self.enc16_res1(
            x,
            t_emb
        )

        x = self.enc16_attention(
            x
        )

        x = self.enc16_res2(
            x,
            t_emb
        )

        skip16 = x

        # 8x8
        x = self.down16_to_8(
            x
        )

        x = self.mid_res1(
            x,
            t_emb
        )

        x = self.mid_attention(
            x
        )

        x = self.mid_res2(
            x,
            t_emb
        )

        # 16x16 + skip
        x = self.up8_to_16(
            x
        )

        x = torch.cat(
            [
                x,
                skip16
            ],
            dim=1
        )

        x = self.dec16_res1(
            x,
            t_emb
        )

        x = self.dec16_attention(
            x
        )

        x = self.dec16_res2(
            x,
            t_emb
        )

        # 32x32 + skip
        x = self.up16_to_32(
            x
        )

        x = torch.cat(
            [
                x,
                skip32
            ],
            dim=1
        )

        x = self.dec32_res1(
            x,
            t_emb
        )

        x = self.dec32_res2(
            x,
            t_emb
        )

        return self.output_conv(
            x
        )

MODELS = [
    ('attention_raw', 'Attention (raw)', SimpleUNetAttention,
     'cifar10_attention_ema_raw_model.pt'),
    ('attention_ema', 'Attention + EMA', SimpleUNetAttention,
     'cifar10_attention_ema_model.pt'),
    ('resunet_raw', 'ResUNet + Attention (raw)', ResUNetAttention,
     'cifar10_resunet_attention_ema_raw_model.pt'),
    ('resunet_ema', 'ResUNet + Attention + EMA', ResUNetAttention,
     'cifar10_resunet_attention_ema_model.pt'),
]


def settings():
    return dict(
        T=T,
        beta_start=BETA_START,
        beta_end=BETA_END,
        n_samples=N_SAMPLES,
        batch_size=SAMPLE_BATCH_SIZE,
        seed=SAMPLE_SEED,
        sampler='DDPM ancestral posterior variance',
        output_dtype='uint8',
    )


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['check', 'status', 'sample', 'kid'], required=True)
    p.add_argument('--model-index', type=int, choices=range(4))
    return p.parse_args()


def check():
    print(f'CONFIGURACIÓN: {N_SAMPLES} muestras por modelo; {KID_SUBSETS} subconjuntos de {KID_SUBSET_SIZE}')
    print(f'SALIDA: {OUTPUT_DIR}')
    print('Modelos requeridos:')
    ok = True
    for key, label, _, filename in MODELS:
        path = MODEL_DIR / filename
        present = path.is_file()
        print(f'  {"OK" if present else "FALTA"}: {label} -> {path}')
        ok &= present
    test_batch = DATA_DIR / 'cifar-10-batches-py' / 'test_batch'
    print('CIFAR-10 test:', 'OK' if test_batch.exists() else 'FALTA', test_batch)
    ok &= test_batch.exists()
    print('Dependencias para KID:')
    for module in ('torchmetrics', 'torch_fidelity'):
        present = importlib.util.find_spec(module) is not None
        print(' ', module, 'OK' if present else 'FALTA')
        ok &= present
    print('Esta comprobación NO descarga pesos Inception ni reserva GPU.')
    print('Las ponderaciones de Inception deben estar en la caché accesible al job, o permitir su descarga.')
    if not ok:
        raise SystemExit('Faltan archivos o dependencias: revisa antes de mandar jobs.')
    print('Archivos y dependencias presentes.')


def status():
    """Consulta lotes guardados sin solicitar GPU ni cargar las redes."""
    expected = math.ceil(N_SAMPLES / SAMPLE_BATCH_SIZE)
    print(f'Avance para {N_SAMPLES} imágenes/modelo — {OUTPUT_DIR}')
    for key, label, _, _ in MODELS:
        n = count_complete(key)
        print(f'{label}: {n}/{expected} lotes válidos')
    csv_path = OUTPUT_DIR / 'kid_comparison.csv'
    print('CSV KID:', csv_path if csv_path.is_file() else 'todavía no disponible')


def get_schedule(device):
    betas = torch.linspace(BETA_START, BETA_END, T, device=device)
    alphas = 1.0 - betas
    abar = torch.cumprod(alphas, dim=0)
    return betas, alphas, abar


def load_state_dict(path, device):
    # Son archivos de pesos propios; weights_only=True evita deserialización arbitraria.
    obj = torch.load(path, map_location='cpu', weights_only=True)
    if isinstance(obj, dict) and 'model_state_dict' in obj:
        return obj['model_state_dict']
    if isinstance(obj, dict) and 'ema_model_state_dict' in obj:
        return obj['ema_model_state_dict']
    if isinstance(obj, dict) and obj and all(isinstance(x, torch.Tensor) for x in obj.values()):
        return obj
    raise TypeError(f'No se reconoce un state_dict en {path}')


@torch.inference_mode()
def generate_batch(model, batch_size, generator, schedule, device):
    betas, alphas, abar = schedule
    x = torch.randn((batch_size, 3, 32, 32), device=device, generator=generator)

    for step in range(T - 1, -1, -1):
        timesteps = torch.full((batch_size,), step, dtype=torch.long, device=device)
        eps_hat = model(x, timesteps)
        beta = betas[step]
        alpha = alphas[step]
        alpha_bar = abar[step]
        mean = (x - beta / torch.sqrt(1.0 - alpha_bar) * eps_hat) / torch.sqrt(alpha)

        if step == 0:
            x = mean
        else:
            previous = abar[step - 1]
            posterior_var = beta * (1 - previous) / (1 - alpha_bar)
            z = torch.randn(x.shape, dtype=x.dtype, device=device, generator=generator)
            x = mean + torch.sqrt(posterior_var) * z

    # Imagen generada en [-1,1] -> uint8 RGB (torchmetrics normalize=False).
    return ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).cpu()


def chunk_path(key, chunk_i):
    return CHUNK_DIR / key / f'batch_{chunk_i:04d}.pt'


def expected_size(chunk_i):
    return min(SAMPLE_BATCH_SIZE, N_SAMPLES - chunk_i * SAMPLE_BATCH_SIZE)


def validate_chunk(path, expected):
    if not path.is_file():
        return False
    try:
        x = torch.load(path, map_location='cpu', weights_only=True)
        return (isinstance(x, torch.Tensor) and x.dtype == torch.uint8
                and tuple(x.shape) == (expected, 3, 32, 32))
    except Exception:
        return False


def count_complete(key):
    nchunks = math.ceil(N_SAMPLES / SAMPLE_BATCH_SIZE)
    return sum(validate_chunk(chunk_path(key, i), expected_size(i))
               for i in range(nchunks))


def sample_model(model_index):
    if not torch.cuda.is_available():
        raise RuntimeError('Muestreo programado requiere GPU; ejecuta este modo mediante sbatch.')
    device = torch.device('cuda:0')
    key, label, constructor, filename = MODELS[model_index]
    weight_path = MODEL_DIR / filename
    if not weight_path.is_file():
        raise FileNotFoundError(weight_path)

    target = CHUNK_DIR / key
    target.mkdir(parents=True, exist_ok=True)
    manifest = target / 'config.json'
    config = {'model': key, 'weights_file': filename, 'weights_sha256': file_sha256(weight_path), **settings()}
    if manifest.exists():
        existing = json.loads(manifest.read_text())
        if existing != config:
            raise RuntimeError(f'Configuración cambió para {key}: usa otra carpeta antes de continuar.')
    else:
        manifest.write_text(json.dumps(config, indent=2) + '\n')

    nchunks = math.ceil(N_SAMPLES / SAMPLE_BATCH_SIZE)
    incomplete = [i for i in range(nchunks) if not validate_chunk(chunk_path(key,i),expected_size(i))]
    print(f'{label}: {nchunks-len(incomplete)}/{nchunks} lotes ya listos', flush=True)
    if not incomplete:
        print('Nada que generar: ya están todas las muestras.', flush=True)
        return

    print('GPU:', torch.cuda.get_device_name(0), flush=True)
    model = constructor().to(device)
    model.load_state_dict(load_state_dict(weight_path, device), strict=True)
    model.eval()
    schedule = get_schedule(device)

    # Misma semilla por lote en TODAS las variantes de red.
    # No depende de la cantidad de lotes previamente guardados.
    for i in incomplete:
        generator = torch.Generator(device=device)
        generator.manual_seed(SAMPLE_SEED + 1009 * i)
        expected = expected_size(i)
        batch = generate_batch(model, expected, generator, schedule, device)
        dst = chunk_path(key, i)
        tmp = dst.with_suffix('.tmp')
        torch.save(batch, tmp)
        tmp.replace(dst)  # escritura atómica: no marcamos lotes parciales como completos
        print(f'[{label}] lote {i+1}/{nchunks} | {min((i+1)*SAMPLE_BATCH_SIZE,N_SAMPLES)}/{N_SAMPLES} imágenes', flush=True)

    print(f'TERMINADO: {label} | {count_complete(key)} lotes guardados', flush=True)


def ensure_chunks():
    for key, label, _, _ in MODELS:
        good = count_complete(key)
        expected = math.ceil(N_SAMPLES / SAMPLE_BATCH_SIZE)
        if good != expected:
            raise RuntimeError(f'{label}: solo {good}/{expected} lotes. Reanuda el array de sampling.')
        manifest = CHUNK_DIR / key / 'config.json'
        filename = next(fn for k, _, _, fn in MODELS if k == key)
        model_path = MODEL_DIR / filename
        config = {'model': key, 'weights_file': filename, 'weights_sha256': file_sha256(model_path), **settings()}
        if not manifest.exists() or json.loads(manifest.read_text()) != config:
            raise RuntimeError(f'Manifiesto inválido para {key}.')


def calculate_kid():
    ensure_chunks()
    if not torch.cuda.is_available():
        raise RuntimeError('KID está programado para el job con GPU de SLURM.')
    device = torch.device('cuda:0')

    try:
        from torchmetrics.image.kid import KernelInceptionDistance
        from torchvision.datasets import CIFAR10
    except ImportError as exc:
        raise RuntimeError('Instala primero torchmetrics y torch-fidelity en ~/venvs/thesis') from exc

    print('Iniciando extractor Inception-v3 oficial de torch-fidelity...', flush=True)
    print('Si los pesos no están en caché y Hypatia bloquea descargas, habrá que copiarlos.', flush=True)
    metric = KernelInceptionDistance(
        feature=2048,
        subsets=KID_SUBSETS,
        subset_size=KID_SUBSET_SIZE,
        normalize=False,
        reset_real_features=False,
    ).to(device)

    # Uso de CIFAR-10 test (10000 imágenes) real, nunca train.
    cifar = CIFAR10(root=str(DATA_DIR), train=False, download=False)
    images = cifar.data
    print('CIFAR-10 test real:', len(images), flush=True)

    with torch.inference_mode():
        for i in range(0, len(images), KID_BATCH_SIZE):
            # dataset.data uint8 RGB NHWC -> BCHW
            real = torch.from_numpy(images[i:i+KID_BATCH_SIZE].copy()).permute(0,3,1,2).contiguous().to(device)
            metric.update(real, real=True)
            if i % (KID_BATCH_SIZE * 30) == 0:
                print('Imágenes reales procesadas:', min(i + KID_BATCH_SIZE, len(images)), flush=True)

    results = []
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for key, label, _, _ in MODELS:
        print('Calculando KID:', label, flush=True)
        with torch.inference_mode():
            for i in range(math.ceil(N_SAMPLES / SAMPLE_BATCH_SIZE)):
                fake = torch.load(chunk_path(key, i), map_location='cpu', weights_only=True).to(device)
                metric.update(fake, real=False)

            # Semilla fija para el muestreo de subconjuntos KID.
            torch.manual_seed(METRIC_SEED)
            torch.cuda.manual_seed_all(METRIC_SEED)
            mean, std = metric.compute()

        row = {
            'model': label,
            'model_key': key,
            'KID_mean': float(mean.detach().cpu()),
            'KID_std_subsets': float(std.detach().cpu()),
            'KID_x1000': float(mean.detach().cpu()) * 1000.0,
            'generated_count': N_SAMPLES,
            'real_count': len(images),
            'subset_size': KID_SUBSET_SIZE,
            'subsets': KID_SUBSETS,
            'feature_dim': 2048,
            'seed': METRIC_SEED,
        }
        results.append(row)
        print(row, flush=True)
        metric.reset()  # conserva características reales (reset_real_features=False)

        # Persistimos después de cada modelo por si el job se interrumpe.
        write_csv(results)

    plot_results(results)
    (OUTPUT_DIR / 'experiment_config.json').write_text(json.dumps({
        'n_generated_per_model': N_SAMPLES,
        'n_real_test': len(images),
        'sample_batch_size': SAMPLE_BATCH_SIZE,
        'kid_feature_dim': 2048,
        'kid_subsets': KID_SUBSETS,
        'kid_subset_size': KID_SUBSET_SIZE,
        'sampling_seed': SAMPLE_SEED,
        'kid_seed': METRIC_SEED,
        'T': T,
        'beta_start': BETA_START,
        'beta_end': BETA_END,
        'generated_image_dtype': 'uint8',
    }, indent=2) + '\n', encoding='utf-8')
    print('KID completo:', OUTPUT_DIR.resolve(), flush=True)


def write_csv(rows):
    dest = OUTPUT_DIR / 'kid_comparison.csv'
    cols = ['model','model_key','KID_mean','KID_std_subsets','KID_x1000',
            'generated_count','real_count','subset_size','subsets','feature_dim','seed']
    tmp = dest.with_suffix('.tmp')
    with tmp.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(dest)


def plot_results(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    names = [x['model'] for x in rows]
    values = [x['KID_mean']*1000 for x in rows]
    errors = [x['KID_std_subsets']*1000 for x in rows]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    xpos = list(range(len(names)))
    ax.bar(xpos, values, yerr=errors, capsize=4)
    ax.set_xticks(xpos)
    ax.set_xticklabels(names, rotation=14, ha='right')
    ax.set_ylabel('KID × 1000 (menor es mejor)')
    ax.set_title('KID — 5.000 muestras generadas por modelo')
    ax.text(0.02, -0.31, 'Barras: desviación entre subconjuntos (no intervalos de confianza)',
            transform=ax.transAxes, fontsize=9)
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR/'kid_comparison.png', dpi=250, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    cli = parse_args()
    if cli.mode == 'check':
        check()
    elif cli.mode == 'status':
        status()
    elif cli.mode == 'sample':
        if cli.model_index is None:
            raise SystemExit('Falta --model-index 0, 1, 2 o 3')
        sample_model(cli.model_index)
    elif cli.mode == 'kid':
        calculate_kid()
