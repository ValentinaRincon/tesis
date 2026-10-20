# ============================================================
# EVALUACIÓN COMÚN DE MODELOS DDPM EN CIFAR-10
# Baseline / Attention raw / Attention EMA / ResUNet raw / ResUNet EMA
# ============================================================

import argparse
import csv
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SEED = 42
SAMPLING_SEED = 2026
DIAGNOSTIC_SEED = 31415

T = 1000
BETA_START = 1e-4
BETA_END = 0.02

BATCH_SIZE_EVAL = 128
NUM_WORKERS = 2
N_SCORE_SAMPLES = 256

EVAL_TIMES = [999, 750, 500, 250, 50]
VIS_T = 250

N_GENERATED = 16
N_COMPARISON = 4

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "modelos"
OUT_DIR = ROOT / "Evaluacion CIFAR10"
SCORE_RGB_DIR = OUT_DIR / "score_rgb"
GENERATED_DIR = OUT_DIR / "generated"

OUT_DIR.mkdir(parents=True, exist_ok=True)
SCORE_RGB_DIR.mkdir(parents=True, exist_ok=True)
GENERATED_DIR.mkdir(parents=True, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require-resunet",
        action="store_true",
        help="Exige que el modelo ResUNet final y su COMPLETE.flag existan.",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Omite el reverse DDPM de 1000 pasos para una prueba rápida.",
    )
    return parser.parse_args()


args = parse_args()

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda")
    print("Usando GPU:", torch.cuda.get_device_name(0), flush=True)
else:
    device = torch.device("cpu")
    print("ADVERTENCIA: usando CPU; la generación será lenta.", flush=True)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])

test_dataset = datasets.CIFAR10(
    root=str(DATA_DIR),
    train=False,
    download=False,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE_EVAL,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(NUM_WORKERS > 0),
)

print(f"CIFAR-10 test: {len(test_dataset)} imágenes", flush=True)


betas = torch.linspace(
    BETA_START,
    BETA_END,
    T,
    device=device
)

alphas = 1.0 - betas

alpha_bars = torch.cumprod(
    alphas,
    dim=0
)


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

class SimpleUNet(nn.Module):

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


MODEL_SPECS = [
    {
        "key": "baseline",
        "label": "Baseline",
        "architecture": SimpleUNet,
        "weights": MODEL_DIR / "cifar10_diffusion_model.pt",
        "losses": MODEL_DIR / "cifar10_epoch_losses.pt",
        "training_family": "Baseline",
        "variant": "raw",
    },
    {
        "key": "attention_raw",
        "label": "Attention (raw)",
        "architecture": SimpleUNetAttention,
        "weights": MODEL_DIR / "cifar10_attention_ema_raw_model.pt",
        "losses": MODEL_DIR / "cifar10_attention_ema_epoch_losses.pt",
        "training_family": "Attention",
        "variant": "raw",
    },
    {
        "key": "attention_ema",
        "label": "Attention + EMA",
        "architecture": SimpleUNetAttention,
        "weights": MODEL_DIR / "cifar10_attention_ema_model.pt",
        "losses": MODEL_DIR / "cifar10_attention_ema_epoch_losses.pt",
        "training_family": "Attention",
        "variant": "ema",
    },
    {
        "key": "resunet_raw",
        "label": "ResUNet + Attention (raw)",
        "architecture": ResUNetAttention,
        "weights": MODEL_DIR / "cifar10_resunet_attention_ema_raw_model.pt",
        "losses": MODEL_DIR / "cifar10_resunet_attention_ema_epoch_losses.pt",
        "training_family": "ResUNet + Attention",
        "variant": "raw",
    },
    {
        "key": "resunet_ema",
        "label": "ResUNet + Attention + EMA",
        "architecture": ResUNetAttention,
        "weights": MODEL_DIR / "cifar10_resunet_attention_ema_model.pt",
        "losses": MODEL_DIR / "cifar10_resunet_attention_ema_epoch_losses.pt",
        "training_family": "ResUNet + Attention",
        "variant": "ema",
    },
]


def safe_torch_load(path, map_location):
    return torch.load(
        path,
        map_location=map_location,
        weights_only=False
    )


def extract_state_dict(obj):
    if isinstance(obj, dict):
        if "model_state_dict" in obj:
            return obj["model_state_dict"]
        if "ema_model_state_dict" in obj:
            return obj["ema_model_state_dict"]
        if obj and all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj

    raise TypeError("El archivo no contiene un state_dict reconocido.")


def load_model(spec):
    path = spec["weights"]

    if not path.exists():
        return None, f"No existe: {path.name}"

    model = spec["architecture"]().to(device)

    obj = safe_torch_load(
        path,
        map_location=device
    )

    state_dict = extract_state_dict(obj)

    try:
        model.load_state_dict(
            state_dict,
            strict=True
        )
    except RuntimeError as exc:
        return None, (
            f"Pesos incompatibles con {spec['architecture'].__name__}: {exc}"
        )

    model.eval()
    return model, None


def load_losses(path):
    if not path.exists():
        return None

    obj = safe_torch_load(
        path,
        map_location="cpu"
    )

    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().flatten().tolist()

    if isinstance(obj, (list, tuple)):
        return [float(x) for x in obj]

    if isinstance(obj, dict):
        for key in ("epoch_losses", "losses", "train_losses"):
            if key in obj:
                values = obj[key]
                if isinstance(values, torch.Tensor):
                    values = values.detach().cpu().flatten().tolist()
                return [float(x) for x in values]

    return None


def tensor_to_rgb(x):
    return ((x.detach().cpu() + 1.0) / 2.0).clamp(0.0, 1.0)


def save_grid(images, path, title, ncols=4):
    images = tensor_to_rgb(images)

    n = images.shape[0]
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.0 * ncols, 3.0 * nrows)
    )

    if nrows == 1 and ncols == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes]
    elif ncols == 1:
        axes = [[ax] for ax in axes]

    for idx in range(nrows * ncols):
        ax = axes[idx // ncols][idx % ncols]
        ax.axis("off")

        if idx < n:
            img = images[idx].permute(1, 2, 0)
            ax.imshow(img, interpolation="nearest")

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def fixed_noise_like(x, seed):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    noise = torch.randn(
        x.shape,
        generator=gen,
        device="cpu",
        dtype=x.dtype
    )

    return noise.to(
        x.device,
        non_blocking=True
    )


@torch.no_grad()
def sample_reverse(model, initial_x, seed=SAMPLING_SEED):
    model.eval()

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    x = initial_x.clone()
    n_samples = x.shape[0]

    for timestep in reversed(range(T)):
        t = torch.full(
            (n_samples,),
            timestep,
            device=device,
            dtype=torch.long
        )

        predicted_noise = model(x, t)

        beta_t = betas[timestep]
        alpha_t = alphas[timestep]
        alpha_bar_t = alpha_bars[timestep]

        mean = (
            1.0 / torch.sqrt(alpha_t)
        ) * (
            x
            -
            beta_t
            / torch.sqrt(1.0 - alpha_bar_t)
            * predicted_noise
        )

        if timestep > 0:
            alpha_bar_previous = alpha_bars[timestep - 1]

            posterior_variance = (
                beta_t
                * (1.0 - alpha_bar_previous)
                / (1.0 - alpha_bar_t)
            )

            z = torch.randn_like(x)

            x = (
                mean
                +
                torch.sqrt(posterior_variance)
                * z
            )
        else:
            x = mean

    return x


batches = []
n_collected = 0

for x, _ in test_loader:
    batches.append(x)
    n_collected += x.shape[0]

    if n_collected >= N_SCORE_SAMPLES:
        break

x_eval = torch.cat(batches, dim=0)[:N_SCORE_SAMPLES].to(
    device,
    non_blocking=True
)

x_vis = x_eval[:1]


available_specs = []
load_errors = []

for spec in MODEL_SPECS:
    if spec["weights"].exists():
        available_specs.append(spec)
    else:
        load_errors.append(
            (spec["label"], f"Archivo ausente: {spec['weights'].name}")
        )


if args.require_resunet:
    required = MODEL_DIR / "cifar10_resunet_attention_ema_model.pt"
    complete_flag = MODEL_DIR / "cifar10_resunet_attention_ema_COMPLETE.flag"

    if not required.exists() or not complete_flag.exists():
        raise SystemExit(
            "ResUNet todavía no está completo. "
            "No se ejecuta la evaluación definitiva."
        )


if not available_specs:
    raise SystemExit("No encontré ningún archivo de pesos conocido.")


print(
    "Modelos candidatos:",
    ", ".join(spec["label"] for spec in available_specs),
    flush=True
)


loss_curves = {}
seen_training_families = set()

for spec in available_specs:
    family = spec["training_family"]

    if family in seen_training_families:
        continue

    losses = load_losses(spec["losses"])

    if losses:
        loss_curves[family] = losses
        seen_training_families.add(family)


if loss_curves:
    fig = plt.figure(figsize=(10, 6))

    for label, losses in loss_curves.items():
        plt.plot(
            range(1, len(losses) + 1),
            losses,
            marker="o",
            markersize=2,
            linewidth=1.4,
            label=label
        )

    plt.xlabel("Época")
    plt.ylabel("MSE de predicción de ruido")
    plt.title("Comparación de pérdidas de entrenamiento — CIFAR-10")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUT_DIR / "comparacion_loss.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)


diagnostic_rows = []
model_summary_rows = []
generated_for_comparison = []


torch.manual_seed(SAMPLING_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SAMPLING_SEED)

initial_xT = torch.randn(
    N_GENERATED,
    3,
    32,
    32,
    device=device
)


for spec in available_specs:

    print(f"\nEvaluando: {spec['label']}", flush=True)

    model, error = load_model(spec)

    if error is not None:
        print("  OMITIDO:", error, flush=True)
        load_errors.append((spec["label"], error))
        continue

    n_params = sum(p.numel() for p in model.parameters())
    losses = load_losses(spec["losses"])

    model_summary_rows.append({
        "model": spec["label"],
        "architecture": spec["architecture"].__name__,
        "variant": spec["variant"],
        "parameters": n_params,
        "epochs_with_loss": len(losses) if losses else "",
        "final_train_loss": float(losses[-1]) if losses else "",
        "min_train_loss": float(min(losses)) if losses else "",
    })

    for timestep in EVAL_TIMES:

        t = torch.full(
            (x_eval.shape[0],),
            timestep,
            device=device,
            dtype=torch.long
        )

        eps = fixed_noise_like(
            x_eval,
            DIAGNOSTIC_SEED + timestep
        )

        alpha_bar_t = alpha_bars[timestep]
        sqrt_ab = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_ab = torch.sqrt(1.0 - alpha_bar_t)

        xt = (
            sqrt_ab * x_eval
            +
            sqrt_one_minus_ab * eps
        )

        with torch.no_grad():
            eps_pred = model(xt, t)

        eps_mse = F.mse_loss(
            eps_pred,
            eps
        ).item()

        eps_cos = F.cosine_similarity(
            eps_pred.flatten(1),
            eps.flatten(1),
            dim=1
        ).mean().item()

        score_pred = (
            -eps_pred
            / sqrt_one_minus_ab
        )

        score_norm_mean = torch.sqrt(
            torch.sum(
                score_pred ** 2,
                dim=1
            )
        ).mean().item()

        x0_hat = (
            xt
            -
            sqrt_one_minus_ab
            * eps_pred
        ) / sqrt_ab

        x0hat_mse = F.mse_loss(
            x0_hat,
            x_eval
        ).item()

        diagnostic_rows.append({
            "model": spec["label"],
            "t": timestep,
            "epsilon_mse": eps_mse,
            "epsilon_cosine": eps_cos,
            "score_norm_mean": score_norm_mean,
            "x0hat_mse": x0hat_mse,
        })


    t_vis = torch.full(
        (1,),
        VIS_T,
        device=device,
        dtype=torch.long
    )

    eps_vis = fixed_noise_like(
        x_vis,
        DIAGNOSTIC_SEED + VIS_T
    )

    ab_vis = alpha_bars[VIS_T]
    sigma_vis = torch.sqrt(1.0 - ab_vis)

    xt_vis = (
        torch.sqrt(ab_vis) * x_vis
        +
        sigma_vis * eps_vis
    )

    with torch.no_grad():
        eps_pred_vis = model(
            xt_vis,
            t_vis
        )

    score_vis = (
        -eps_pred_vis
        / sigma_vis
    )

    x0_hat_vis = (
        xt_vis
        -
        sigma_vis * eps_pred_vis
    ) / torch.sqrt(ab_vis)

    score_np = score_vis[0].detach().cpu()

    vmax = float(score_np.abs().max())

    score_norm = torch.sqrt(
        torch.sum(
            score_np ** 2,
            dim=0
        )
    )

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(12, 8)
    )

    axes[0, 0].imshow(
        tensor_to_rgb(
            xt_vis[0]
        ).permute(1, 2, 0),
        interpolation="nearest"
    )
    axes[0, 0].set_title(f"$x_t$ (t={VIS_T})")

    for ax, channel, name in zip(
        [axes[0, 1], axes[0, 2], axes[1, 0]],
        [score_np[0], score_np[1], score_np[2]],
        ["$s_R$", "$s_G$", "$s_B$"]
    ):
        ax.imshow(
            channel,
            cmap="coolwarm",
            vmin=-vmax,
            vmax=vmax,
            interpolation="nearest"
        )
        ax.set_title(name)

    axes[1, 1].imshow(
        score_norm,
        interpolation="nearest"
    )
    axes[1, 1].set_title(r"$||s_\theta||_2$")

    axes[1, 2].imshow(
        tensor_to_rgb(
            x0_hat_vis[0]
        ).permute(1, 2, 0),
        interpolation="nearest"
    )
    axes[1, 2].set_title(r"$\hat{x}_0$")

    for ax in axes.flat:
        ax.axis("off")

    fig.suptitle(
        f"{spec['label']} — score RGB",
        fontsize=15
    )

    plt.tight_layout()
    plt.savefig(
        SCORE_RGB_DIR / f"score_rgb_{spec['key']}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)


    if not args.skip_generation:
        print("  Generando 16 muestras...", flush=True)

        generated = sample_reverse(
            model,
            initial_xT,
            seed=SAMPLING_SEED
        )

        save_grid(
            generated,
            GENERATED_DIR / f"generated_{spec['key']}.png",
            title=spec["label"],
            ncols=4
        )

        generated_for_comparison.append(
            (
                spec["label"],
                tensor_to_rgb(
                    generated[:N_COMPARISON]
                )
            )
        )

    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if diagnostic_rows:
    with open(
        OUT_DIR / "score_metrics.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "t",
                "epsilon_mse",
                "epsilon_cosine",
                "score_norm_mean",
                "x0hat_mse",
            ]
        )
        writer.writeheader()
        writer.writerows(diagnostic_rows)


if model_summary_rows:
    with open(
        OUT_DIR / "model_summary.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "architecture",
                "variant",
                "parameters",
                "epochs_with_loss",
                "final_train_loss",
                "min_train_loss",
            ]
        )
        writer.writeheader()
        writer.writerows(model_summary_rows)


if load_errors:
    with open(
        OUT_DIR / "load_warnings.txt",
        "w",
        encoding="utf-8"
    ) as f:
        for label, error in load_errors:
            f.write(f"{label}: {error}\n")


def plot_metric(
    metric_key,
    ylabel,
    filename,
    log_y=False
):
    if not diagnostic_rows:
        return

    labels = []

    for row in diagnostic_rows:
        if row["model"] not in labels:
            labels.append(row["model"])

    fig = plt.figure(figsize=(10, 6))

    for label in labels:
        rows = [
            r
            for r in diagnostic_rows
            if r["model"] == label
        ]

        rows = sorted(
            rows,
            key=lambda r: r["t"]
        )

        xs = [r["t"] for r in rows]
        ys = [r[metric_key] for r in rows]

        plt.plot(
            xs,
            ys,
            marker="o",
            linewidth=1.5,
            label=label
        )

    plt.xlabel("t")
    plt.ylabel(ylabel)

    if log_y:
        plt.yscale("log")

    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUT_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)


plot_metric(
    "epsilon_mse",
    r"MSE de $\epsilon_\theta$",
    "epsilon_mse_vs_t.png"
)

plot_metric(
    "epsilon_cosine",
    r"Similitud coseno de $\epsilon_\theta$",
    "epsilon_cosine_vs_t.png"
)

plot_metric(
    "score_norm_mean",
    r"Promedio de $||s_\theta||_2$ por píxel",
    "score_norm_vs_t.png",
    log_y=True
)

plot_metric(
    "x0hat_mse",
    r"MSE de $\hat{x}_0$",
    "x0hat_mse_vs_t.png",
    log_y=True
)


if generated_for_comparison:

    nrows = len(generated_for_comparison)

    fig, axes = plt.subplots(
        nrows,
        N_COMPARISON,
        figsize=(
            2.7 * N_COMPARISON,
            2.7 * nrows
        )
    )

    if nrows == 1:
        axes = [axes]

    for row_idx, (label, images) in enumerate(generated_for_comparison):
        for col_idx in range(N_COMPARISON):
            ax = axes[row_idx][col_idx]

            img = images[col_idx].permute(
                1,
                2,
                0
            )

            ax.imshow(
                img,
                interpolation="nearest"
            )

            ax.axis("off")

            if col_idx == 0:
                ax.set_ylabel(
                    label,
                    fontsize=11
                )

    fig.suptitle(
        "Mismas semillas de muestreo — comparación de modelos",
        fontsize=15
    )

    plt.tight_layout()

    plt.savefig(
        OUT_DIR / "generated_samples_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


print("\nEvaluación terminada.", flush=True)
print("Resultados:", OUT_DIR.resolve(), flush=True)
