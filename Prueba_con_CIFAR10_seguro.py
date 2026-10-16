# ============================================================
# DDPM BASELINE SOBRE CIFAR-10 — VERSION ROBUSTA PARA HYPATIA
#
# Cambios principales:
# - Guarda todas las figuras en disco.
# - Guarda un checkpoint completo al terminar cada época.
# - Puede reanudar automáticamente desde el último checkpoint.
# - Guarda optimizer, pérdidas y estados aleatorios.
# - Guarda una curva de loss después de cada época.
# - Genera previews periódicos durante el entrenamiento.
# - Está pensado para ejecutarse con sbatch sin depender de SSH/Jupyter.
# ============================================================

import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Backend no interactivo: las figuras se guardan aunque no haya sesión gráfica.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 0. CONFIGURACIÓN GENERAL
# ============================================================

SEED = 42

T = 1000
BETA_START = 1e-4
BETA_END = 0.02

BATCH_SIZE = 64
LEARNING_RATE = 1e-4
EPOCHS = 50

# Cada cuántas épocas guardar una pequeña muestra generada.
SAMPLE_EVERY = 5
PREVIEW_SAMPLES = 4

# Rutas relativas a la carpeta desde la que se ejecuta el script.
ROOT = Path.cwd()

MODEL_DIR = ROOT / "modelos"
IMG_DIR = ROOT / "Img CIFAR10"
PREVIEW_DIR = IMG_DIR / "previews"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = MODEL_DIR / "checkpoint_cifar10_latest.pt"
MODEL_PATH = MODEL_DIR / "cifar10_diffusion_model.pt"
LOSS_DATA_PATH = MODEL_DIR / "cifar10_epoch_losses.pt"


# ============================================================
# 1. SEMILLAS
# ============================================================

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 2. DISPOSITIVO
# ============================================================

def get_device():

    if torch.cuda.is_available():

        print(
            "Usando GPU:",
            torch.cuda.get_device_name(0),
            flush=True
        )

        return torch.device("cuda")

    print("Usando CPU", flush=True)

    return torch.device("cpu")


device = get_device()


# ============================================================
# 3. DATOS: CIFAR-10
# ============================================================

transform = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])


dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=False,
    transform=transform
)


# Generador separado para el shuffle del DataLoader.
loader_generator = torch.Generator()
loader_generator.manual_seed(SEED)


loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=True,
    generator=loader_generator
)


x0, labels = next(iter(loader))

print(
    "Shape de un batch:",
    x0.shape,
    flush=True
)

print(
    "Labels:",
    labels[:10],
    flush=True
)


# ============================================================
# 4. UTILIDAD PARA MOSTRAR RGB
# ============================================================

def tensor_to_rgb(image):

    """
    Convierte una imagen [3,H,W] normalizada en [-1,1]
    a [H,W,3] en [0,1].
    """

    image = image.detach().cpu()

    image = (image + 1) / 2

    image = image.clamp(0, 1)

    image = image.permute(1, 2, 0)

    return image


# ============================================================
# 5. GUARDAR MUESTRAS ORIGINALES
# ============================================================

fig = plt.figure(figsize=(8, 8))

for i in range(16):

    plt.subplot(4, 4, i + 1)

    plt.imshow(
        tensor_to_rgb(x0[i])
    )

    plt.axis("off")


plt.suptitle(
    "Muestras originales de CIFAR-10"
)

plt.tight_layout()

plt.savefig(
    IMG_DIR / "cifar10_originales.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# 6. SCHEDULE DE DIFUSIÓN
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


print(
    "beta inicial:",
    betas[0].item(),
    flush=True
)

print(
    "beta final:",
    betas[-1].item(),
    flush=True
)

print(
    "alpha_bar final:",
    alpha_bars[-1].item(),
    flush=True
)


# ============================================================
# 7. PROCESO FORWARD
# ============================================================

def forward_diffusion_sample(x0, t):

    noise = torch.randn_like(x0)

    alpha_bar_t = alpha_bars[t]

    alpha_bar_t = alpha_bar_t[
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
# 8. GUARDAR EJEMPLO DEL FORWARD
# ============================================================

x0_test, _ = next(iter(loader))

x0_test = x0_test[:1].to(device)


times = [
    0,
    100,
    300,
    600,
    999
]


fig = plt.figure(figsize=(14, 3))


for i, timestep in enumerate(times):

    t = torch.tensor(
        [timestep],
        device=device
    )

    xt, _ = forward_diffusion_sample(
        x0_test,
        t
    )

    plt.subplot(
        1,
        len(times),
        i + 1
    )

    plt.imshow(
        tensor_to_rgb(xt[0])
    )

    plt.title(
        f"t={timestep}"
    )

    plt.axis("off")


plt.suptitle(
    "Proceso forward sobre CIFAR-10"
)

plt.tight_layout()

plt.savefig(
    IMG_DIR / "forward_cifar10.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# 9. TIME EMBEDDING
# ============================================================

class SinusoidalTimeEmbedding(nn.Module):

    def __init__(self, dim):

        super().__init__()

        self.dim = dim


    def forward(self, t):

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

        embedding = torch.cat(
            [
                angles.sin(),
                angles.cos()
            ],
            dim=1
        )

        return embedding


# ============================================================
# 10. BLOQUE RESIDUAL
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
        )

        time_information = time_information[
            :,
            :,
            None,
            None
        ]

        h = h + time_information

        h = self.conv2(h)
        h = self.norm2(h)

        h = h + self.residual(x)

        return F.silu(h)


# ============================================================
# 11. U-NET RGB
# ============================================================

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


# ============================================================
# 12. MODELO
# ============================================================

model = SimpleUNet().to(device)


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
# 13. PRUEBA DE UNA PASADA
# ============================================================

x0_test, _ = next(iter(loader))

x0_test = x0_test.to(device)


t_test = torch.randint(
    0,
    T,
    (x0_test.shape[0],),
    device=device
)


xt_test, noise_test = forward_diffusion_sample(
    x0_test,
    t_test
)


with torch.no_grad():

    predicted_noise_test = model(
        xt_test,
        t_test
    )


print("x0:", x0_test.shape, flush=True)
print("xt:", xt_test.shape, flush=True)
print("Ruido real:", noise_test.shape, flush=True)
print(
    "Ruido predicho:",
    predicted_noise_test.shape,
    flush=True
)


test_loss = F.mse_loss(
    predicted_noise_test,
    noise_test
)


print(
    "Loss antes de entrenar:",
    test_loss.item(),
    flush=True
)


# ============================================================
# 14. REVERSE DDPM
#
# Se define antes del entrenamiento para poder generar previews.
# ============================================================

@torch.no_grad()
def sample_reverse(
    model,
    n_samples=16,
    save_snapshots=False
):

    model.eval()

    x = torch.randn(
        n_samples,
        3,
        32,
        32,
        device=device
    )

    snapshots = {}

    save_times = [
        999,
        750,
        500,
        250,
        0
    ]


    for timestep in reversed(range(T)):

        t = torch.full(
            (n_samples,),
            timestep,
            device=device,
            dtype=torch.long
        )

        predicted_noise = model(
            x,
            t
        )

        beta_t = betas[timestep]

        alpha_t = alphas[timestep]

        alpha_bar_t = alpha_bars[timestep]


        mean = (
            1.0
            /
            torch.sqrt(alpha_t)
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

            z = torch.randn_like(x)

            x = (
                mean
                +
                torch.sqrt(
                    posterior_variance
                )
                *
                z
            )

        else:

            x = mean


        if (
            save_snapshots
            and
            timestep in save_times
        ):

            snapshots[
                timestep
            ] = (
                x.detach()
                .cpu()
                .clone()
            )


    return x, snapshots


# ============================================================
# 15. UTILIDADES DE GUARDADO
# ============================================================

def save_loss_curve(
    losses,
    path
):

    if len(losses) == 0:
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

    plt.xlabel("Época")
    plt.ylabel("MSE Pérdida promedio")

    plt.title(
        "Entrenamiento del DDPM sobre CIFAR-10"
    )

    plt.grid()

    plt.savefig(
        path,
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


    plt.suptitle(title)

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


def save_reverse_trajectory(
    snapshots,
    path
):

    times_to_show = [
        999,
        750,
        500,
        250,
        0
    ]

    fig = plt.figure(
        figsize=(14, 3)
    )

    for i, timestep in enumerate(
        times_to_show
    ):

        image = snapshots[
            timestep
        ][0]

        plt.subplot(
            1,
            len(times_to_show),
            i + 1
        )

        plt.imshow(
            tensor_to_rgb(
                image
            )
        )

        plt.title(
            f"t={timestep}"
        )

        plt.axis("off")


    plt.suptitle(
        "Proceso reverse del DDPM sobre CIFAR-10"
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# 16. OPTIMIZADOR
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# 17. CHECKPOINT COMPLETO
# ============================================================

def save_checkpoint(
    epoch,
    epoch_losses
):

    checkpoint = {

        # Última época COMPLETADA.
        "epoch": epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "epoch_losses":
            epoch_losses,

        # RNG para reanudar de forma más reproducible.
        "torch_rng_state":
            torch.get_rng_state(),

        "loader_generator_state":
            loader_generator.get_state(),

        # Información suficiente para detectar configuraciones distintas.
        "config": {
            "T": T,
            "beta_start": BETA_START,
            "beta_end": BETA_END,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
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


    # Guardamos también solo los pesos.
    # Este archivo es cómodo para inferencia/análisis posterior.
    torch.save(
        model.state_dict(),
        MODEL_PATH
    )


    torch.save(
        epoch_losses,
        LOSS_DATA_PATH
    )


# ============================================================
# 18. REANUDAR AUTOMÁTICAMENTE SI EXISTE CHECKPOINT
# ============================================================

start_epoch = 0
epoch_losses = []


if CHECKPOINT_PATH.exists():

    print(
        "\nCheckpoint encontrado:",
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


    # Restaurar RNG.
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

        torch.cuda.set_rng_state_all(
            checkpoint[
                "cuda_rng_state_all"
            ]
        )


    if (
        "loader_generator_state"
        in checkpoint
    ):

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


else:

    print(
        "\nNo hay checkpoint previo. "
        "El entrenamiento comenzará desde cero.",
        flush=True
    )


# ============================================================
# 19. ENTRENAMIENTO
# ============================================================

if start_epoch >= EPOCHS:

    print(
        "Las 50 épocas ya aparecen como completadas. "
        "No se vuelve a entrenar.",
        flush=True
    )

else:

    for epoch in range(
        start_epoch,
        EPOCHS
    ):

        model.train()

        epoch_start = time.time()

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


            total_loss += loss.item()


        average_loss = (
            total_loss
            /
            len(loader)
        )


        epoch_losses.append(
            average_loss
        )


        elapsed = (
            time.time()
            -
            epoch_start
        )


        print(
            f"Época "
            f"{epoch + 1:2d}/"
            f"{EPOCHS} | "
            f"Pérdida promedio: "
            f"{average_loss:.6f} | "
            f"Tiempo: "
            f"{elapsed / 60:.1f} min",
            flush=True
        )


        # ----------------------------------------------------
        # GUARDAR DESPUÉS DE CADA ÉPOCA
        # ----------------------------------------------------

        save_checkpoint(
            epoch,
            epoch_losses
        )


        save_loss_curve(
            epoch_losses,
            IMG_DIR
            /
            "cifar10_training_loss.png"
        )


        # ----------------------------------------------------
        # PREVIEW PERIÓDICO
        #
        # fork_rng evita que el muestreo altere el RNG
        # utilizado después para entrenar.
        # ----------------------------------------------------

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


                preview_images, _ = sample_reverse(
                    model,
                    n_samples=PREVIEW_SAMPLES,
                    save_snapshots=False
                )


            save_generated_grid(
                preview_images,
                PREVIEW_DIR
                /
                f"preview_epoch_{epoch + 1:03d}.png",
                title=(
                    f"CIFAR-10 — época {epoch + 1}"
                ),
                max_images=PREVIEW_SAMPLES
            )


            print(
                "Preview guardado:",
                PREVIEW_DIR
                /
                f"preview_epoch_{epoch + 1:03d}.png",
                flush=True
            )


# ============================================================
# 20. RECARGAR LOS PESOS FINALES
# ============================================================

model = SimpleUNet().to(
    device
)


model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )
)


model.eval()


print(
    "Modelo CIFAR-10 cargado correctamente.",
    flush=True
)


# ============================================================
# 21. GENERACIÓN FINAL
# ============================================================

# fork_rng permite obtener siempre la misma evaluación final
# sin alterar estados guardados del entrenamiento.
devices = []

if torch.cuda.is_available():
    devices = [
        torch.cuda.current_device()
    ]


with torch.random.fork_rng(
    devices=devices
):

    torch.manual_seed(
        SEED
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            SEED
        )


    generated_images, snapshots = sample_reverse(
        model,
        n_samples=16,
        save_snapshots=True
    )


# ============================================================
# 22. GUARDAR LAS 16 MUESTRAS FINALES
# ============================================================

save_generated_grid(
    generated_images,
    IMG_DIR
    /
    "generated_cifar10.png",
    title=(
        "Muestras generadas por el DDPM sobre CIFAR-10"
    ),
    max_images=16
)


# ============================================================
# 23. GUARDAR UNA TRAYECTORIA REVERSE
# ============================================================

save_reverse_trajectory(
    snapshots,
    IMG_DIR
    /
    "reverse_process_cifar10.png"
)


print(
    "\nProceso CIFAR-10 terminado.",
    flush=True
)

print(
    "Checkpoint:",
    CHECKPOINT_PATH.resolve(),
    flush=True
)

print(
    "Pesos:",
    MODEL_PATH.resolve(),
    flush=True
)

print(
    "Figuras:",
    IMG_DIR.resolve(),
    flush=True
)
