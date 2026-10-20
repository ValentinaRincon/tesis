# ============================================================
# CIFAR-10 DDPM + SELF-ATTENTION + EMA
# VERSION ROBUSTA PARA HYPATIA / SLURM
#
# - CIFAR-10 ya debe existir en ./data (download=False)
# - Guarda checkpoint completo después de cada época
# - Reanuda automáticamente desde el checkpoint
# - Mantiene un modelo EMA para generación
# - Guarda previews, curva de loss y muestras finales
# - Puede detenerse de forma limpia antes del walltime
#   para que el .slurm envíe automáticamente la continuación
# ============================================================

import copy
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 0. CONFIGURACIÓN
# ============================================================

SEED = 42

T = 1000
BETA_START = 1e-4
BETA_END = 0.02

BATCH_SIZE = 64
LEARNING_RATE = 1e-4
EPOCHS = 200

EMA_DECAY = 0.9999

NUM_WORKERS = 2

SAMPLE_EVERY = 25
PREVIEW_SAMPLES = 4

# El .slurm lo fija en 210 min para dejar margen antes
# del límite de 4 h. Puede cambiarse desde el entorno.
MAX_RUNTIME_MINUTES = int(
    os.environ.get("MAX_RUNTIME_MINUTES", "210")
)

# Usar siempre la carpeta donde está este script.
ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "modelos"
IMG_DIR = ROOT / "Img CIFAR10 Attention EMA"
PREVIEW_DIR = IMG_DIR / "previews"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = (
    MODEL_DIR / "checkpoint_cifar10_attention_ema_latest.pt"
)

RAW_MODEL_PATH = (
    MODEL_DIR / "cifar10_attention_ema_raw_model.pt"
)

EMA_MODEL_PATH = (
    MODEL_DIR / "cifar10_attention_ema_model.pt"
)

LOSS_DATA_PATH = (
    MODEL_DIR / "cifar10_attention_ema_epoch_losses.pt"
)

COMPLETE_FLAG = (
    MODEL_DIR / "cifar10_attention_ema_COMPLETE.flag"
)


# ============================================================
# 1. SEMILLAS
# ============================================================

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 2. DISPOSITIVO
# ============================================================

if torch.cuda.is_available():

    device = torch.device("cuda")

    print(
        "Usando GPU:",
        torch.cuda.get_device_name(0),
        flush=True
    )

else:

    device = torch.device("cpu")

    print(
        "Usando CPU",
        flush=True
    )


# ============================================================
# 3. CIFAR-10
# ============================================================

transform = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])


dataset = datasets.CIFAR10(
    root=str(DATA_DIR),
    train=True,
    download=False,
    transform=transform
)


loader_generator = torch.Generator()
loader_generator.manual_seed(SEED)


loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(NUM_WORKERS > 0),
    generator=loader_generator
)


print(
    "Dataset:",
    len(dataset),
    "imágenes",
    flush=True
)


# ============================================================
# 4. SCHEDULE
# ============================================================

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


def forward_diffusion_sample(
    x0,
    t
):

    noise = torch.randn_like(x0)

    alpha_bar_t = alpha_bars[t][
        :,
        None,
        None,
        None
    ]

    xt = (
        torch.sqrt(alpha_bar_t) * x0
        +
        torch.sqrt(1.0 - alpha_bar_t) * noise
    )

    return xt, noise


# ============================================================
# 5. TIME EMBEDDING
# ============================================================

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


# ============================================================
# 6. RESBLOCK
# ============================================================

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


# ============================================================
# 7. SELF-ATTENTION
# ============================================================

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


# ============================================================
# 8. U-NET RGB + SELF-ATTENTION
# ============================================================

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


# ============================================================
# 9. MODELO + EMA
# ============================================================

model = SimpleUNetAttention().to(
    device
)

ema_model = copy.deepcopy(
    model
).to(device)

ema_model.eval()

for parameter in ema_model.parameters():
    parameter.requires_grad_(False)


@torch.no_grad()
def update_ema(
    ema_model,
    model,
    decay=EMA_DECAY
):

    ema_params = dict(
        ema_model.named_parameters()
    )

    model_params = dict(
        model.named_parameters()
    )

    for name, param in model_params.items():

        ema_params[name].mul_(
            decay
        ).add_(
            param,
            alpha=1.0 - decay
        )


n_parameters = sum(
    p.numel()
    for p in model.parameters()
)

print(
    "Número de parámetros:",
    f"{n_parameters:,}",
    flush=True
)


# ============================================================
# 10. OPTIMIZADOR
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# 11. VISUALIZACIÓN
# ============================================================

def tensor_to_rgb(
    image
):

    image = image.detach().cpu()

    image = (
        image + 1
    ) / 2

    image = image.clamp(
        0,
        1
    )

    return image.permute(
        1,
        2,
        0
    )


def save_loss_curve(
    losses
):

    if not losses:
        return

    fig = plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        range(
            1,
            len(losses) + 1
        ),
        losses,
        marker="o"
    )

    plt.xlabel(
        "Época"
    )

    plt.ylabel(
        "MSE Pérdida promedio"
    )

    plt.title(
        "DDPM CIFAR-10 + Self-Attention + EMA"
    )

    plt.grid()

    plt.savefig(
        IMG_DIR / "training_loss.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


def save_generated_grid(
    images,
    path,
    title,
    max_images=16
):

    images = images.detach().cpu()

    n_images = min(
        len(images),
        max_images
    )

    n_cols = math.ceil(
        math.sqrt(n_images)
    )

    n_rows = math.ceil(
        n_images / n_cols
    )

    fig = plt.figure(
        figsize=(
            2 * n_cols,
            2 * n_rows
        )
    )

    for i in range(n_images):

        plt.subplot(
            n_rows,
            n_cols,
            i + 1
        )

        plt.imshow(
            tensor_to_rgb(
                images[i]
            )
        )

        plt.axis("off")


    plt.suptitle(
        title
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# 12. SAMPLER REVERSE
# ============================================================

@torch.no_grad()
def sample_reverse(
    sampling_model,
    n_samples=16
):

    sampling_model.eval()

    x = torch.randn(
        n_samples,
        3,
        32,
        32,
        device=device
    )

    for timestep in reversed(
        range(T)
    ):

        t = torch.full(
            (n_samples,),
            timestep,
            device=device,
            dtype=torch.long
        )

        predicted_noise = sampling_model(
            x,
            t
        )

        beta_t = betas[
            timestep
        ]

        alpha_t = alphas[
            timestep
        ]

        alpha_bar_t = alpha_bars[
            timestep
        ]

        mean = (
            1.0
            /
            torch.sqrt(
                alpha_t
            )
        ) * (
            x
            -
            beta_t
            /
            torch.sqrt(
                1.0 - alpha_bar_t
            )
            *
            predicted_noise
        )

        if timestep > 0:

            alpha_bar_previous = alpha_bars[
                timestep - 1
            ]

            posterior_variance = (
                beta_t
                *
                (
                    1.0
                    -
                    alpha_bar_previous
                )
                /
                (
                    1.0
                    -
                    alpha_bar_t
                )
            )

            x = (
                mean
                +
                torch.sqrt(
                    posterior_variance
                )
                *
                torch.randn_like(x)
            )

        else:

            x = mean


    return x


# ============================================================
# 13. CHECKPOINT
# ============================================================

def save_checkpoint(
    epoch,
    epoch_losses
):

    checkpoint = {

        # Índice (0-based) de la última época terminada.
        "epoch": epoch,

        "model_state_dict":
            model.state_dict(),

        "ema_model_state_dict":
            ema_model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "epoch_losses":
            epoch_losses,

        "torch_rng_state":
            torch.get_rng_state(),

        "loader_generator_state":
            loader_generator.get_state(),

        "config": {
            "T": T,
            "beta_start": BETA_START,
            "beta_end": BETA_END,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "ema_decay": EMA_DECAY,
            "seed": SEED
        }
    }


    if torch.cuda.is_available():

        checkpoint[
            "cuda_rng_state_all"
        ] = torch.cuda.get_rng_state_all()


    torch.save(
        checkpoint,
        CHECKPOINT_PATH
    )

    torch.save(
        model.state_dict(),
        RAW_MODEL_PATH
    )

    torch.save(
        ema_model.state_dict(),
        EMA_MODEL_PATH
    )

    torch.save(
        epoch_losses,
        LOSS_DATA_PATH
    )


# ============================================================
# 14. REANUDAR SI HAY CHECKPOINT
# ============================================================

start_epoch = 0
epoch_losses = []


if CHECKPOINT_PATH.exists():

    print(
        "Checkpoint encontrado:",
        CHECKPOINT_PATH,
        flush=True
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    ema_model.load_state_dict(
        checkpoint[
            "ema_model_state_dict"
        ]
    )

    optimizer.load_state_dict(
        checkpoint[
            "optimizer_state_dict"
        ]
    )

    epoch_losses = checkpoint.get(
        "epoch_losses",
        []
    )

    start_epoch = (
        checkpoint[
            "epoch"
        ]
        +
        1
    )


    if "torch_rng_state" in checkpoint:

        torch.set_rng_state(
            checkpoint[
                "torch_rng_state"
            ].cpu()
        )


    if (
        torch.cuda.is_available()
        and
        "cuda_rng_state_all"
        in checkpoint
    ):

        cuda_rng_states = [
            state.cpu()
            for state in checkpoint[
                "cuda_rng_state_all"
            ]
        ]

        torch.cuda.set_rng_state_all(
            cuda_rng_states
        )


    if "loader_generator_state" in checkpoint:

        loader_generator.set_state(
            checkpoint[
                "loader_generator_state"
            ].cpu()
        )


    print(
        f"Se reanudará desde la época "
        f"{start_epoch + 1}/{EPOCHS}.",
        flush=True
    )


# Si hay un flag viejo pero el entrenamiento no está terminado,
# eliminarlo para permitir continuaciones automáticas.
if (
    COMPLETE_FLAG.exists()
    and
    start_epoch < EPOCHS
):

    COMPLETE_FLAG.unlink()


# ============================================================
# 15. ENTRENAMIENTO
# ============================================================

job_start_time = time.time()


if start_epoch < EPOCHS:

    for epoch in range(
        start_epoch,
        EPOCHS
    ):

        epoch_start = time.time()

        model.train()

        total_loss = 0.0


        for x0, _ in loader:

            x0 = x0.to(
                device,
                non_blocking=True
            )

            t = torch.randint(
                0,
                T,
                (x0.shape[0],),
                device=device
            )

            xt, noise = forward_diffusion_sample(
                x0,
                t
            )

            predicted_noise = model(
                xt,
                t
            )

            loss = F.mse_loss(
                predicted_noise,
                noise
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            optimizer.step()

            # Actualizar EMA después de cada paso del optimizador.
            update_ema(
                ema_model,
                model
            )

            total_loss += loss.item()


        average_loss = (
            total_loss
            /
            len(loader)
        )

        epoch_losses.append(
            average_loss
        )

        epoch_minutes = (
            time.time()
            -
            epoch_start
        ) / 60

        total_minutes = (
            time.time()
            -
            job_start_time
        ) / 60


        print(
            f"Época "
            f"{epoch + 1:2d}/"
            f"{EPOCHS} | "
            f"Loss: "
            f"{average_loss:.6f} | "
            f"Época: "
            f"{epoch_minutes:.1f} min | "
            f"Job: "
            f"{total_minutes:.1f} min",
            flush=True
        )


        # Guardar al terminar CADA época.
        save_checkpoint(
            epoch,
            epoch_losses
        )

        save_loss_curve(
            epoch_losses
        )


        # Preview usando EMA.
        if (
            (epoch + 1) % SAMPLE_EVERY == 0
            or
            epoch == 0
        ):

            devices = []

            if torch.cuda.is_available():
                devices = [
                    torch.cuda.current_device()
                ]

            with torch.random.fork_rng(
                devices=devices
            ):

                torch.manual_seed(
                    SEED + 1000
                )

                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(
                        SEED + 1000
                    )

                preview = sample_reverse(
                    ema_model,
                    n_samples=PREVIEW_SAMPLES
                )


            save_generated_grid(
                preview,
                PREVIEW_DIR
                /
                f"preview_epoch_{epoch + 1:03d}.png",
                title=(
                    "CIFAR-10 Attention + EMA "
                    f"— época {epoch + 1}"
                ),
                max_images=PREVIEW_SAMPLES
            )


        # ----------------------------------------------------
        # SALIDA LIMPIA ANTES DEL WALLTIME
        #
        # El .slurm detectará que no existe COMPLETE.flag
        # y enviará automáticamente otro job que reanuda.
        # ----------------------------------------------------

        if (
            epoch + 1 < EPOCHS
            and
            total_minutes >= MAX_RUNTIME_MINUTES
        ):

            print(
                "Se alcanzó el límite interno de tiempo. "
                "El checkpoint está guardado y el job "
                "puede continuar automáticamente.",
                flush=True
            )

            sys.exit(0)


# ============================================================
# 16. ENTRENAMIENTO COMPLETO
# ============================================================

COMPLETE_FLAG.write_text(
    "Entrenamiento completo\n",
    encoding="utf-8"
)


# ============================================================
# 17. GENERACIÓN FINAL CON EMA
# ============================================================

devices = []

if torch.cuda.is_available():
    devices = [
        torch.cuda.current_device()
    ]


with torch.random.fork_rng(
    devices=devices
):

    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    generated_images = sample_reverse(
        ema_model,
        n_samples=16
    )


save_generated_grid(
    generated_images,
    IMG_DIR / "generated_cifar10_attention_ema.png",
    title=(
        "Muestras generadas por DDPM sobre CIFAR-10 "
        "con Self-Attention + EMA"
    ),
    max_images=16
)


print(
    "Entrenamiento COMPLETO.",
    flush=True
)

print(
    "Modelo EMA:",
    EMA_MODEL_PATH,
    flush=True
)

print(
    "Checkpoint:",
    CHECKPOINT_PATH,
    flush=True
)

print(
    "Resultados:",
    IMG_DIR,
    flush=True
)
