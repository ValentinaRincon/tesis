import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import math
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

def get_device():
    if torch.cuda.is_available():
        print("Usando GPU:", torch.cuda.get_device_name(0))
        return torch.device("cuda")
    else:
        print("Usando CPU")
        return torch.device("cpu")

device = get_device()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) #Normalización de los datos de MNIST
])

dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True
)

x0, labels = next(iter(loader))

print("1. x0 antes de moverlo a la GPU:", x0.shape) #Datos de MNIST están en escala de grises, por lo que solo hay un canal   

# Proceso forward

T = 1000

beta_start = 1e-4
beta_end = 0.02

betas = torch.linspace(
    beta_start,
    beta_end,
    T,
    device=device
)

alphas = 1.0 - betas

alpha_bars = torch.cumprod(
    alphas,
    dim=0
)

def forward_diffusion_sample(x0, t):
    """
    x0: [B, C, H, W]
    t:  [B]
    """

    noise = torch.randn_like(x0)

    alpha_bar_t = alpha_bars[t]

    alpha_bar_t = alpha_bar_t[:, None, None, None]

    xt = (                             # Proceso de difusión hacia adelante
        torch.sqrt(alpha_bar_t) * x0
        +
        torch.sqrt(1 - alpha_bar_t) * noise
    )

    return xt, noise   # Para entrenamiento, necesitamos el ruido para calcular la pérdida

x0 = x0.to(device)

t = torch.randint(
    0,
    T,
    (x0.shape[0],),
    device=device
)

xt, noise = forward_diffusion_sample(x0, t)

print("2. x0 después de moverlo a la GPU:", x0.shape)
print("3. xt después de aplicar difusión forward:", xt.shape)
print("4. noise generado:", noise.shape)

def show_tensor_image(image):
    image = image.detach().cpu()

    image = (image + 1) / 2
    image = image.clamp(0, 1)

    plt.imshow(
        image.squeeze(),
        cmap="gray"
    )
    plt.axis("off")

x0, _ = next(iter(loader))
x0 = x0[:1].to(device)

times = [0, 100, 300, 600, 999]

plt.figure(figsize=(12, 3))

for i, timestep in enumerate(times):

    t = torch.tensor(
        [timestep],
        device=device
    )

    xt, _ = forward_diffusion_sample(x0, t)

    plt.subplot(1, len(times), i + 1)

    image = (xt[0].detach().cpu() + 1) / 2
    image = image.clamp(0, 1)

    plt.imshow(image.squeeze(), cmap="gray")
    plt.title(f"t={timestep}")
    plt.axis("off")

plt.show()

# Time embedding

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2

        frequencies = math.log(10000) / (half_dim - 1)

        frequencies = torch.exp(
            torch.arange(
                half_dim,
                device=t.device
            ) * -frequencies
        )

        angles = t.float()[:, None] * frequencies[None, :]

        embedding = torch.cat(
            [angles.sin(), angles.cos()],
            dim=1
        )

        return embedding

time_embedding = SinusoidalTimeEmbedding(64).to(device)

test_t = torch.tensor(
    [0, 100, 300, 999],
    device=device
)

emb = time_embedding(test_t)

print("Forma de t:", test_t.shape)
print("Forma del embedding:", emb.shape)

# Bloques de la red

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
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
            nn.Linear(time_dim, out_channels)
        )

        if in_channels != out_channels:
            self.residual = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x, t_emb):

        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)

        time_information = self.time_mlp(t_emb)
        time_information = time_information[:, :, None, None]

        h = h + time_information

        h = self.conv2(h)
        h = self.norm2(h)

        h = h + self.residual(x)

        return F.silu(h)

# Mini U-Net

class SimpleUNet(nn.Module):
    def __init__(self, time_dim=64):
        super().__init__()

        # Embedding temporal
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        # Entrada: 1 canal (MNIST)
        self.input_conv = nn.Conv2d(
            1,
            64,
            kernel_size=3,
            padding=1
        )

        # Encoder
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

        # Parte más profunda
        self.middle = ResBlock(
            256,
            256,
            time_dim
        )

        # Decoder
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

        # Salida: ruido predicho
        self.output_conv = nn.Conv2d(
            64,
            1,
            kernel_size=1
        )

    def forward(self, x, t):

        t_emb = self.time_embedding(t)

        # Entrada
        x = self.input_conv(x)

        # Encoder
        skip1 = self.block1(x, t_emb)

        x = self.down1(skip1)

        skip2 = self.block2(x, t_emb)

        x = self.down2(skip2)

        # Centro
        x = self.middle(x, t_emb)

        # Decoder
        x = self.up1(x)

        x = torch.cat(
            [x, skip2],
            dim=1
        )

        x = self.block3(x, t_emb)

        x = self.up2(x)

        x = torch.cat(
            [x, skip1],
            dim=1
        )

        x = self.block4(x, t_emb)

        return self.output_conv(x)

# Red

model = SimpleUNet().to(device)

x0, _ = next(iter(loader))
x0 = x0.to(device)

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

# - Predicción del ruido

predicted_noise = model(
    xt,
    t
)

print("x0:", x0.shape)
print("xt:", xt.shape)
print("Ruido real:", noise.shape)
print("Ruido predicho:", predicted_noise.shape)

# - Función de pérdida

loss = F.mse_loss(
    predicted_noise,
    noise
)

print("Loss:", loss.item())

if predicted_noise.shape == noise.shape:
    print("Las formas coinciden. La pérdida se puede calcular correctamente.")

model = SimpleUNet().to(device)

torch.save(model.state_dict(), "modelo_mnist_ddpm.pth")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# =========================
# Prueba de entrenamiento
# =========================

model.train()

# Tomamos un solo batch de 64 imágenes
x0, _ = next(iter(loader))
x0 = x0.to(device)

loss_history = []

for step in range(300):

    # 1. Elegir un tiempo aleatorio para cada imagen
    t = torch.randint(
        0,
        T,
        (x0.shape[0],),
        device=device
    )

    # 2. Aplicar el proceso forward
    xt, noise = forward_diffusion_sample(x0, t)

    # 3. La U-Net intenta predecir el ruido
    predicted_noise = model(xt, t)

    # 4. Comparar ruido real y ruido predicho
    loss = F.mse_loss(predicted_noise, noise)

    # 5. Borrar gradientes de la iteración anterior
    optimizer.zero_grad()

    # 6. Calcular nuevos gradientes
    loss.backward()

    # 7. Actualizar los parámetros de la red
    optimizer.step()

    # Guardar pérdida
    loss_history.append(loss.item())

    if step % 25 == 0:
        print(
            f"Step {step:3d} | "
            f"Loss: {loss.item():.4f}"
        )

plt.figure(figsize=(8, 5))

plt.plot(loss_history)

plt.xlabel("Paso")
plt.ylabel("MSE Pérdida")
plt.title("Entrenamiento de prueba sobre un batch")

plt.grid()

plt.show()

model = SimpleUNet().to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4
)

# =========================
# Entrenamiento con MNIST
# =========================

epochs = 50

epoch_losses = []

for epoch in range(epochs):

    model.train()

    total_loss = 0.0

    for x0, _ in loader:

        # Mover imágenes al dispositivo
        x0 = x0.to(device)

        # Elegir un tiempo aleatorio para cada imagen
        t = torch.randint(
            0,
            T,
            (x0.shape[0],),
            device=device
        )

        # Proceso forward
        xt, noise = forward_diffusion_sample(
            x0,
            t
        )

        # La U-Net predice el ruido
        predicted_noise = model(
            xt,
            t
        )

        # MSE entre ruido verdadero y predicho
        loss = F.mse_loss(
            predicted_noise,
            noise
        )

        # Backpropagation
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    # Loss promedio del epoch
    average_loss = total_loss / len(loader)

    epoch_losses.append(average_loss)

    print(
        f"Época {epoch + 1:2d}/{epochs} | "
        f"Pérdida promedio: {average_loss:.6f}"
    )

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, epochs + 1),
    epoch_losses,
    marker="o"
)

plt.xlabel("Época")
plt.ylabel("MSE Pérdida promedio")
plt.title("Entrenamiento del modelo sobre MNIST")

plt.grid()

plt.show()

Path("modelos").mkdir(exist_ok=True)
torch.save(
    model.state_dict(),
    "modelos/mnist_diffusion_model.pt"
)

print("Modelo guardado.")

plt.savefig(
    "Img MNIST No Fourier/mnist_training_loss.png",
    dpi=300,
    bbox_inches="tight"
)

model = SimpleUNet().to(device)

model.load_state_dict(
    torch.load(
        "modelos/mnist_diffusion_model.pt",
        map_location=device
    )
)

model.eval()

print("Modelo cargado correctamente.")

@torch.no_grad()
def sample_reverse(model, n_samples=64):

    model.eval()

    # Empezamos desde ruido gaussiano puro
    x = torch.randn(
        n_samples,
        1,
        28,
        28,
        device=device
    )

    # Guardaremos algunos estados intermedios
    snapshots = {}

    save_times = [999, 750, 500, 250, 0]

    for timestep in reversed(range(T)):

        # Todos los elementos están en el mismo timestep
        t = torch.full(
            (n_samples,),
            timestep,
            device=device,
            dtype=torch.long
        )

        # La U-Net estima el ruido presente
        predicted_noise = model(x, t)

        beta_t = betas[timestep]
        alpha_t = alphas[timestep]
        alpha_bar_t = alpha_bars[timestep]

        # Media de p(x_{t-1} | x_t)
        mean = (
            1 / torch.sqrt(alpha_t)
        ) * (
            x
            -
            beta_t
            / torch.sqrt(1 - alpha_bar_t)
            * predicted_noise
        )

        if timestep > 0:

            alpha_bar_previous = alpha_bars[timestep - 1]

            posterior_variance = (
                beta_t
                * (1 - alpha_bar_previous)
                / (1 - alpha_bar_t)
            )

            z = torch.randn_like(x)

            x = (
                mean
                +
                torch.sqrt(posterior_variance) * z
            )

        else:
            x = mean

        if timestep in save_times:
            snapshots[timestep] = x.detach().cpu()

    return x, snapshots

generated_images, snapshots = sample_reverse(
    model,
    n_samples=16
)
Path("Img MNIST No Fourier").mkdir(exist_ok=True)
plt.figure(figsize=(8, 8))

for i in range(16):

    plt.subplot(4, 4, i + 1)

    image = generated_images[i].detach().cpu()

    image = (image + 1) / 2
    image = image.clamp(0, 1)

    plt.imshow(
        image.squeeze(),
        cmap="gray"
    )

    plt.axis("off")

plt.tight_layout()

plt.savefig(
    "Img MNIST No Fourier/generated_mnist.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Imágenes generadas guardadas.")

times_to_show = [999, 750, 500, 250, 0]

plt.figure(figsize=(12, 3))

for i, timestep in enumerate(times_to_show):

    image = snapshots[timestep][0]

    image = (image + 1) / 2
    image = image.clamp(0, 1)

    plt.subplot(1, 5, i + 1)

    plt.imshow(
        image.squeeze(),
        cmap="gray"
    )

    plt.title(f"t={timestep}")
    plt.axis("off")

plt.tight_layout()

plt.savefig(
    "Img MNIST No Fourier/reverse_process.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Proceso reverse guardado.")