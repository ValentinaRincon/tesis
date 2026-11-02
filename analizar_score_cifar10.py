#!/usr/bin/env python3
"""Comparación de score en CIFAR-10: Attention+EMA vs ResUNet+Attention+EMA.

NO ENTRENA modelos. Usa los pesos existentes; entrada compartida para comparar
predicción de ruido y score en distintos instantes. Además genera dos
trayectorias DDPM usando exactamente las mismas muestras de ruido aleatorio.

En login (solo comprobación de archivos):
    python analizar_score_cifar10.py --mode check
En una GPU reservada mediante SLURM:
    python -u analizar_score_cifar10.py --mode all
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"
MODEL_ROOT = ROOT / "modelos"
OUT = ROOT / "Analisis Score CIFAR10"
OUT_FIG = OUT / "figuras"

T = 1000
BETA_START = 1e-4
BETA_END = 0.02
TEST_COUNT = 512
BATCH = 64
SEED_DATA = 20261024
SEED_NOISE = 314159
SEED_SAMPLING = 260124
T_VALUES = [0, 1, 5, 10, 25, 50, 100, 150, 200, 250, 350, 500, 650, 750, 900, 999]
FIG_T = [50, 250, 750]
SNAPSHOTS = [999, 750, 500, 250, 50, 0]
TRAJECTORY_N = 4

MODEL_SPECS = [
    ("attention_ema", "Attention + EMA", "cifar10_attention_ema_model.pt"),
    ("resunet_ema", "ResUNet + Attention + EMA", "cifar10_resunet_attention_ema_model.pt"),
]

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


def cli():
    p = argparse.ArgumentParser(description="Análisis comparativo del score")
    p.add_argument("--mode", choices=("check", "all"), required=True)
    return p.parse_args()


def check():
    print("Comparación sobre 512 imágenes de CIFAR-10 test, 16 tiempos")
    print("No se ejecuta entrenamiento ni se reserva GPU en modo check.")
    ok=True
    for _,label,fname in MODEL_SPECS:
        path=MODEL_ROOT/fname
        print(f"{'OK' if path.is_file() else 'FALTA'} {label}: {path}")
        ok &= path.is_file()
    dataset = DATA_ROOT / "cifar-10-batches-py" / "test_batch"
    print(f"{'OK' if dataset.is_file() else 'FALTA'} CIFAR-10 test: {dataset}")
    ok &= dataset.is_file()
    if not ok:
        raise SystemExit("Hay archivos faltantes. No envíes el job todavía.")
    print("Comprobación terminada. Los pesos se validarán en el nodo GPU.")


def model_from_weights(constructor, fname, device):
    # Archivos de pesos propios (state_dict), sin ejecutar código arbitrario.
    payload = torch.load(MODEL_ROOT/fname, map_location="cpu", weights_only=True)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        payload = payload["model_state_dict"]
    elif isinstance(payload, dict) and "ema_model_state_dict" in payload:
        payload = payload["ema_model_state_dict"]
    if not isinstance(payload, dict) or not payload or not all(
        isinstance(value, torch.Tensor) for value in payload.values()
    ):
        raise ValueError(f"Archivo de pesos inválido: {fname}")
    model=constructor().to(device)
    model.load_state_dict(payload, strict=True)
    model.eval()
    return model


def get_dataset():
    from torchvision.datasets import CIFAR10
    from torchvision.transforms import Compose, ToTensor, Normalize
    transform=Compose([ToTensor(), Normalize((0.5,)*3, (0.5,)*3)])
    dataset=CIFAR10(root=str(DATA_ROOT), train=False, transform=transform, download=False)
    gen=torch.Generator(device="cpu").manual_seed(SEED_DATA)
    indices=torch.randperm(len(dataset), generator=gen)[:TEST_COUNT].tolist()
    x=torch.stack([dataset[i][0] for i in indices])
    return x, indices


def get_schedule(device):
    betas=torch.linspace(BETA_START, BETA_END, T, device=device)
    alphas=1.0-betas
    alpha_bars=torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars


def gaussian_like_cpu(x, seed, device):
    # Un mismo eps fijo para cada imagen y tiempo; nunca se sortea por modelo.
    g=torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(x.shape, dtype=torch.float32, generator=g).to(device)


def score_from_prediction(eps_hat, abar):
    return -eps_hat / torch.sqrt(1.0 - abar)


def sample_metrics(x0, eps, xt, e_a, e_b, abar):
    score_a = score_from_prediction(e_a, abar)
    score_b = score_from_prediction(e_b, abar)
    vec_a = score_a.flatten(start_dim=1)
    vec_b = score_b.flatten(start_dim=1)
    abs_a = vec_a.norm(dim=1)
    abs_b = vec_b.norm(dim=1)
    denom = torch.clamp((abs_a+abs_b)/2.0, min=1e-8)
    sdiff = (vec_b-vec_a).norm(dim=1) / denom
    x0h_a=(xt-torch.sqrt(1-abar)*e_a)/torch.sqrt(abar)
    x0h_b=(xt-torch.sqrt(1-abar)*e_b)/torch.sqrt(abar)
    return {
        "epsilon_mse_attention": (e_a-eps).square().mean(dim=(1,2,3)),
        "epsilon_mse_resunet": (e_b-eps).square().mean(dim=(1,2,3)),
        "epsilon_cos_attention": F.cosine_similarity(e_a.flatten(1), eps.flatten(1), dim=1),
        "epsilon_cos_resunet": F.cosine_similarity(e_b.flatten(1), eps.flatten(1), dim=1),
        "score_norm_attention": score_a.square().sum(dim=1).sqrt().mean(dim=(1,2)),
        "score_norm_resunet": score_b.square().sum(dim=1).sqrt().mean(dim=(1,2)),
        "score_cosine_between": F.cosine_similarity(vec_a, vec_b, dim=1),
        "score_relative_l2_difference": sdiff,
        "x0_mse_attention": (x0h_a-x0).square().mean(dim=(1,2,3)),
        "x0_mse_resunet": (x0h_b-x0).square().mean(dim=(1,2,3)),
    }


def write_csv(path, rows):
    if not rows: return
    fields=list(rows[0].keys())
    with path.open('w', newline='', encoding='utf-8') as fp:
        writer=csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def as_image(x):
    return ((x.detach().cpu().permute(1,2,0)+1.0)/2.0).clamp(0,1).numpy()


def score_fig(t, xt, s_a, s_b, abar):
    # Primera imagen de evaluación, MISMOS x_t en las dos filas.
    s_a=s_a[0].detach().cpu().numpy()
    s_b=s_b[0].detach().cpu().numpy()
    sa=np.sqrt(np.sum(s_a**2,axis=0))
    sb=np.sqrt(np.sum(s_b**2,axis=0))
    vmax=max(float(np.quantile(np.abs(s_a), 0.99)), float(np.quantile(np.abs(s_b), 0.99)), 1e-7)
    normmax=max(float(np.quantile(sa,.99)), float(np.quantile(sb,.99)),1e-7)
    fig, axes=plt.subplots(2,6,figsize=(16,6.1),constrained_layout=True)
    rows=[("Attention + EMA",s_a,sa), ("ResUNet + EMA",s_b,sb)]
    for r, (label,channels,norm) in enumerate(rows):
        axes[r,0].imshow(as_image(xt[0]),interpolation='nearest')
        for c in range(3):
            axes[r,c+1].imshow(channels[c],cmap='coolwarm',vmin=-vmax,vmax=vmax,interpolation='nearest')
        axes[r,4].imshow(norm,cmap='magma',vmin=0,vmax=normmax,interpolation='nearest')
        # s=-epsilon_hat/sigma => x0hat = (xt + sigma**2*s)/sqrt(abar).
        x0hat=(xt[0]+(1-abar)*torch.as_tensor(channels,device=xt.device))/torch.sqrt(abar)
        axes[r,5].imshow(as_image(x0hat),interpolation='nearest')
        axes[r,0].set_ylabel(label,fontsize=10)
        for ax in axes[r]: ax.set_xticks([]); ax.set_yticks([])
    for col,ttl in enumerate([r'$x_t$',r'$s_R$',r'$s_G$',r'$s_B$',r'$\|s_\theta\|_2$',r'$\hat{x}_0$']):
        axes[0,col].set_title(ttl)
    fig.suptitle(f"Campos de score RGB sobre la MISMA muestra — t={t}",fontsize=14)
    fig.savefig(OUT_FIG/f"score_rgb_t{t:03d}.png",dpi=220,bbox_inches='tight')
    plt.close(fig)

    # Campo diferencia evaluado sobre el mismo x_t (ResUNet - Attention).
    delta=s_b-s_a
    dnorm=np.sqrt(np.sum(delta**2,axis=0))
    vmax_d=max(float(np.quantile(np.abs(delta),.99)),1e-7)
    fig,axes=plt.subplots(1,4,figsize=(13,3.2),constrained_layout=True)
    for c,name in enumerate(['R','G','B']):
        im=axes[c].imshow(delta[c],cmap='coolwarm',vmin=-vmax_d,vmax=vmax_d,interpolation='nearest')
        axes[c].set_title(rf'$\Delta s_{{{name}}}$')
    axes[3].imshow(dnorm,cmap='magma',interpolation='nearest')
    axes[3].set_title(r'$\|s_B-s_A\|_2$ por píxel')
    for ax in axes: ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle(f'Diferencia espacial de score (ResUNet - Attention), t={t}')
    fig.savefig(OUT_FIG/f"diferencia_score_t{t:03d}.png",dpi=220,bbox_inches='tight')
    plt.close(fig)


def plot_curves(rows, summary):
    xs=[r['t'] for r in summary]
    for fname, ylabel, keys, logarithmic in [
        ('epsilon_mse_vs_t.png',r'MSE de $\epsilon_\theta$',["epsilon_mse_attention_mean","epsilon_mse_resunet_mean"],True),
        ('score_norm_vs_t.png',r'Media de $\|s_\theta\|_2$ RGB/píxel',["score_norm_attention_mean","score_norm_resunet_mean"],True),
        ('x0_mse_vs_t.png',r'MSE de $\hat{x}_0$',["x0_mse_attention_mean","x0_mse_resunet_mean"],True),
    ]:
        fig,ax=plt.subplots(figsize=(9,5))
        for k,label in zip(keys,['Attention + EMA','ResUNet + EMA']):
            ax.plot(xs,[r[k] for r in summary],marker='o',label=label)
        if logarithmic: ax.set_yscale('log')
        ax.set_xlabel('t'); ax.set_ylabel(ylabel)
        ax.grid(alpha=.28); ax.legend(); fig.tight_layout(); fig.savefig(OUT_FIG/fname,dpi=230);plt.close(fig)

    fig,ax=plt.subplots(figsize=(9,5))
    ax.plot(xs,[r['score_cosine_between_mean'] for r in summary],marker='o',label='Coseno score A vs B')
    ax.plot(xs,[r['score_relative_l2_difference_mean'] for r in summary],marker='s',label='Diferencia L2 relativa')
    ax.set_xlabel('t');ax.set_ylabel('Valor promedio (sobre mismas entradas)')
    ax.grid(alpha=.28);ax.legend();fig.tight_layout();fig.savefig(OUT_FIG/'comparacion_campos_score.png',dpi=230);plt.close(fig)

    fig,ax=plt.subplots(figsize=(9,5))
    delta=np.array([r['delta_mse_mean'] for r in summary])
    lo=np.array([r['delta_mse_boot_ci95_low'] for r in summary]); hi=np.array([r['delta_mse_boot_ci95_high'] for r in summary])
    ax.plot(xs,delta,'o-',label=r'$\Delta\mathrm{MSE}=\mathrm{ResUNet}-\mathrm{Attention}$')
    ax.fill_between(xs,lo,hi,alpha=.2,label='IC bootstrap 95% sobre imágenes/ruidos fijos')
    ax.axhline(0,color='black',lw=.8)
    ax.set_xlabel('t');ax.set_ylabel(r'$\Delta$ MSE de $\epsilon$')
    ax.grid(alpha=.28);ax.legend();fig.tight_layout();fig.savefig(OUT_FIG/'delta_mse_con_incertidumbre.png',dpi=230);plt.close(fig)


def eval_forward(models, device, abar):
    xall, indices=get_dataset()
    print(f"Test: {len(indices)} imágenes, semilla de índices {SEED_DATA}",flush=True)
    OUT.mkdir(exist_ok=True,parents=True)
    OUT_FIG.mkdir(exist_ok=True,parents=True)
    summary=[]
    all_rows=[]
    for step in T_VALUES:
        per_metric={}
        for batch_i in range(0,len(xall),BATCH):
            x0=xall[batch_i:batch_i+BATCH].to(device,non_blocking=True)
            # ε independiente por imagen y tiempo, pero compartido entre modelos.
            eps=gaussian_like_cpu(x0,SEED_NOISE+step*1000003+batch_i,device)
            alpha_bar=abar[step]
            xt=torch.sqrt(alpha_bar)*x0+torch.sqrt(1-alpha_bar)*eps
            ts=torch.full((len(x0),),step,dtype=torch.long,device=device)
            with torch.inference_mode():
                e_a=models[0](xt,ts)
                e_b=models[1](xt,ts)
                metrics=sample_metrics(x0,eps,xt,e_a,e_b,alpha_bar)
                if batch_i==0 and step in FIG_T:
                    score_fig(step,xt,-e_a/torch.sqrt(1-alpha_bar),-e_b/torch.sqrt(1-alpha_bar),alpha_bar)
            for key,val in metrics.items():
                per_metric.setdefault(key,[]).append(val.detach().cpu())
        per_metric={k:torch.cat(v).numpy() for k,v in per_metric.items()}
        # Guardar estimaciones por imagen permite estimar error de forma pareada.
        for i in range(len(indices)):
            all_rows.append({"t":step,"test_index":indices[i],**{k:float(vals[i]) for k,vals in per_metric.items()}})
        row={"t":step,"n_images":len(indices)}
        for k,v in per_metric.items():
            row[k+'_mean']=float(np.mean(v))
            row[k+'_sd_across_images']=float(np.std(v,ddof=1))
        delta=per_metric['epsilon_mse_resunet']-per_metric['epsilon_mse_attention']
        row['delta_mse_mean']=float(np.mean(delta))
        row['win_fraction_resunet']=float(np.mean(delta<0))
        rng=np.random.default_rng(SEED_DATA+step)
        indices_boot=rng.integers(0,len(delta),size=(500,len(delta)))
        boot=delta[indices_boot].mean(axis=1)
        row['delta_mse_boot_ci95_low']=float(np.quantile(boot,.025))
        row['delta_mse_boot_ci95_high']=float(np.quantile(boot,.975))
        summary.append(row)
        print(f"t={step:>3} | MSE A={row['epsilon_mse_attention_mean']:.5g} | B={row['epsilon_mse_resunet_mean']:.5g} | score cos={row['score_cosine_between_mean']:.4f}",flush=True)
    write_csv(OUT/'score_por_imagen.csv',all_rows)
    write_csv(OUT/'score_resumen_por_t.csv',summary)
    plot_curves(all_rows,summary)
    return summary


@torch.inference_mode()
def paired_trajectories(models,device,betas,alphas,abar):
    # Dos procesos DDPM independientes con los MISMOS z_t en cada paso.
    # No evalúa similitud de modelos sobre el mismo x_t a menos que se indique.
    gen=torch.Generator(device=device).manual_seed(SEED_SAMPLING)
    common_xT=torch.randn((TRAJECTORY_N,3,32,32),device=device,generator=gen)
    xa=common_xT.clone()
    xb=common_xT.clone()
    frames_a={}; frames_b={}; rows=[]
    for step in reversed(range(T)):
        if step in SNAPSHOTS:
            frames_a[step]=xa.detach().cpu().clone()
            frames_b[step]=xb.detach().cpu().clone()
        ts=torch.full((TRAJECTORY_N,),step,dtype=torch.long,device=device)
        ea=models[0](xa,ts)
        eb=models[1](xb,ts)
        # Comparar score en un MISMO punto xa de referencia (distinto de eb(xb)).
        if step in SNAPSHOTS:
            eb_on_xa=models[1](xa,ts)
            sa=-ea/torch.sqrt(1-abar[step])
            sb_same=-eb_on_xa/torch.sqrt(1-abar[step])
            rows.append({
                't':step,
                'trajectory_rms_gap':float((xa-xb).square().mean().sqrt()),
                'score_cosine_on_attention_path':float(F.cosine_similarity(sa.flatten(1),sb_same.flatten(1),dim=1).mean()),
                'score_rms_diff_on_attention_path':float((sa-sb_same).square().mean().sqrt()),
            })
        beta=betas[step]; alpha=alphas[step]; abar_t=abar[step]
        mean_a=(xa-beta/torch.sqrt(1-abar_t)*ea)/torch.sqrt(alpha)
        mean_b=(xb-beta/torch.sqrt(1-abar_t)*eb)/torch.sqrt(alpha)
        if step>0:
            abprev=abar[step-1]
            var=beta*(1-abprev)/(1-abar_t)
            z=torch.randn(xa.shape,device=device,generator=gen)
            xa=mean_a+torch.sqrt(var)*z
            xb=mean_b+torch.sqrt(var)*z
        else:
            xa=mean_a;xb=mean_b
        if step in (999,750,500,250,50,1):
            print('Trayectorias DDPM paso',step,flush=True)
    # Una muestra ilustrativa, dos filas. x0 final tiene otro significado que x_t intermedio.
    frames_a['final']=xa.detach().cpu()
    frames_b['final']=xb.detach().cpu()
    fig,axs=plt.subplots(2,len(SNAPSHOTS),figsize=(15,5.5),constrained_layout=True)
    for row,(label,frames) in enumerate([('Attention + EMA',frames_a),('ResUNet + EMA',frames_b)]):
        for col,step in enumerate(SNAPSHOTS):
            img=frames['final'][0] if step==0 else frames[step][0]
            axs[row,col].imshow(as_image(img),interpolation='nearest')
            axs[row,col].set_xticks([]);axs[row,col].set_yticks([])
            if row==0: axs[row,col].set_title(f't={step}')
            if col==0: axs[row,col].set_ylabel(label,fontsize=10)
    fig.suptitle('Mismas x_T y realizaciones de ruido; trayectorias pueden divergir')
    fig.savefig(OUT_FIG/'trayectorias_pareadas.png',dpi=240,bbox_inches='tight');plt.close(fig)
    fig,axs=plt.subplots(2,TRAJECTORY_N,figsize=(10,5.5),constrained_layout=True)
    for row,(label,imgs) in enumerate([('Attention + EMA',xa),('ResUNet + EMA',xb)]):
        for col in range(TRAJECTORY_N):
            axs[row,col].imshow(as_image(imgs[col]),interpolation='nearest')
            axs[row,col].set_xticks([]);axs[row,col].set_yticks([])
            if col==0: axs[row,col].set_ylabel(label)
    fig.suptitle('Muestras finales con ruido inicial y z_t compartidos')
    fig.savefig(OUT_FIG/'muestras_pareadas.png',dpi=240,bbox_inches='tight');plt.close(fig)
    write_csv(OUT/'trayectorias_metricas.csv',sorted(rows,key=lambda r:r['t']))
    print('Trayectorias guardadas',flush=True)


def run():
    check()
    if not torch.cuda.is_available():
        raise RuntimeError('Este modo requiere una GPU reservada con sbatch.')
    device=torch.device('cuda:0')
    print('GPU:',torch.cuda.get_device_name(device),flush=True)
    OUT.mkdir(parents=True,exist_ok=True); OUT_FIG.mkdir(parents=True,exist_ok=True)
    models=[
        model_from_weights(SimpleUNetAttention,MODEL_SPECS[0][2],device),
        model_from_weights(ResUNetAttention,MODEL_SPECS[1][2],device),
    ]
    print('Pesos cargados en modo eval',flush=True)
    betas,alphas,abar=get_schedule(device)
    summary=eval_forward(models,device,abar)
    paired_trajectories(models,device,betas,alphas,abar)
    def digest_file(fname):
        h=hashlib.sha256()
        with (MODEL_ROOT/fname).open('rb') as fh:
            for chunk in iter(lambda:fh.read(1024*1024),b''):
                h.update(chunk)
        return h.hexdigest()
    metadata={
        'models':[{'label':x[1], 'file':x[2], 'sha256':digest_file(x[2])} for x in MODEL_SPECS],
        'dataset':'CIFAR10 test', 'n_test':TEST_COUNT,
        't_values':T_VALUES, 'fig_t':FIG_T, 'trajectory_snapshots':SNAPSHOTS,
        'n_trajectories':TRAJECTORY_N,
        'sampling_seed':SEED_SAMPLING,'data_seed':SEED_DATA,'noise_seed':SEED_NOISE,
        'T':T,'beta_start':BETA_START,'beta_end':BETA_END,
        'note':'epsilon target per realization is conditional denoising target, not marginal score truth. Bootstrap across fixed held-out examples, not retraining replicates.',
        'note_trajectories':'DDPM ancestral sampling with paired noises; images on different paths are not an apples-to-apples score field comparison.',
    }
    (OUT/'configuracion.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('FINALIZADO. RESULTADOS:',OUT,flush=True)


if __name__=='__main__':
    args=cli()
    if args.mode=='check':check()
    else:run()
