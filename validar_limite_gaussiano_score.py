#!/usr/bin/env python3
"""Verificacion directa del score gaussiano en DDPM CIFAR-10.

Para p_t ~ N(0,I), s(x,t) ~ -x y J_s ~ -I en D=3072 dimensiones.
Usa VJP / productos jacobiano-vector transpuestos y probes Rademacher:
  E_v ||J_s^T v + v||^2 = ||J_s + I||_F^2.
La raiz del promedio de dicha magnitud / D estima E_J.

Importa los loaders y la arquitectura ya probados de analizar_score_cifar10.py.
No entrena, ni modifica checkpoints ni el analisis geometrico anterior.

LOGIN: --mode check / --mode selftest / --mode status
GPU/SLURM: --mode all
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import analizar_score_cifar10 as previous

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "Limite Gaussiano Score CIFAR10"
ROWS = OUT / "mediciones_gaussianas.csv"
SUMMARY = OUT / "resumen_limite_gaussiano.csv"
DIFFS = OUT / "diferencias_pareadas_gaussiano.csv"
FIGS = OUT / "figuras"
PROTOCOL = OUT / "protocolo.json"
D = 3 * 32 * 32
MODEL_NAMES = ["Attention + EMA", "ResUNet + Attention + EMA"]
DEFAULT_T = (500, 750, 900, 950, 975, 999)
ROW_FIELDS = ["model", "t", "image_index", "cifar10_test_index", "n_probes",
              "frob_j_plus_i_sq_mc", "frob_j_sq_mc", "trace_j_mc",
              "score_plus_x_sq", "x_sq"]


def get_args():
    p = argparse.ArgumentParser(description="Verifica J_score ~ -I y score ~ -x con CIFAR-10")
    p.add_argument("--mode", required=True, choices=("check", "selftest", "status", "all"))
    p.add_argument("--images", type=int, default=24)
    p.add_argument("--probes", type=int, default=16, help="Direcciones Rademacher por imagen")
    p.add_argument("--bootstrap", type=int, default=1500)
    p.add_argument("--times", default=",".join(map(str, DEFAULT_T)))
    p.add_argument("--seed", type=int, default=20261026)
    return p.parse_args()


def times_of(args):
    try:
        times = [int(q.strip()) for q in args.times.split(",")]
    except ValueError as exc:
        raise ValueError("--times debe contener enteros separados por comas") from exc
    if not times or len(set(times)) != len(times) or any(not 0 <= t < previous.T for t in times):
        raise ValueError("Tiempos no validos o duplicados; usar 0..999")
    return sorted(times)


def filehash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest(args):
    return {
        "analysis": "direct_gaussian_limit_score_jacobian_vjp_v1",
        "architecture_file_sha256": filehash(previous.__file__),
        "checkpoints_sha256": {key: filehash(previous.MODEL_ROOT / name)
                              for key, _, name in previous.MODEL_SPECS},
        "images": args.images,
        "probes_per_image": args.probes,
        "bootstrap_reps": args.bootstrap,
        "times": times_of(args),
        "seed": args.seed,
        "dataset_selection_seed": previous.SEED_DATA,
        "data": "CIFAR10 test, Normalize(0.5,0.5) transform to [-1,1]",
        "schedule": {"T": previous.T, "beta_start": previous.BETA_START,
                     "beta_end": previous.BETA_END},
        "jacobian_coordinates": "all 3072 normalized RGB components",
        "same_images_noise_probes_for_both_models": True,
        "gaussian_prediction": "for p_t=N(0,I), s_t=-x, J_s=-I",
        "E_J": "sqrt(mean_v ||J_s^T v+v||^2 / 3072)",
        "E_score": "sqrt(sum ||s_theta+x_t||^2 / sum ||x_t||^2)",
        "uncertainty": "paired image bootstrap conditional on trained checkpoints and fixed probes",
        "torch": torch.__version__,
    }


def require_same_protocol(args):
    new = manifest(args)
    if PROTOCOL.exists():
        old = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        if old != new:
            raise RuntimeError("Protocolo o checkpoints diferentes. NO mezclar mediciones existentes; "
                               "usar mismos argumentos y pesos, o renombrar la carpeta de salida completa.")
    else:
        OUT.mkdir(parents=True, exist_ok=True)
        PROTOCOL.write_text(json.dumps(new, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def atomic_csv_write(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def read_rows():
    if not ROWS.exists():
        return []
    with ROWS.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def rademacher(shape, g, device):
    # CPU generador con seed fijo, evitando dependencia del RNG de GPU.
    return torch.randint(0, 2, shape, generator=g, dtype=torch.int8).to(
        device=device, dtype=torch.float32).mul_(2).sub_(1)


def vjp_measure(s, x, probes):
    """Estimadores cuadrados por imagen, sin materializar matrices D x D.

    Equivale a estimator de ||J+I||_F^2, ||J||_F^2 y tr(J).
    s debe mantener gradiente con respecto a x.
    """
    if len(probes) == 0:
        raise ValueError("Se necesita al menos una direccion aleatoria")
    ej_sq, j_sq, traces = [], [], []
    for k, v in enumerate(probes):
        # Mantener grafo hasta el ultimo VJP del mismo forward.
        jt_v = torch.autograd.grad((s * v).sum(), x,
                                    retain_graph=(k < len(probes) - 1))[0]
        with torch.no_grad():
            ej_sq.append(float((jt_v + v).square().sum().item()))
            j_sq.append(float(jt_v.square().sum().item()))
            traces.append(float((v * jt_v).sum().item()))
    return np.mean(ej_sq), np.mean(j_sq), np.mean(traces)


def verify_math():
    # Deterministico: para J=-I, todos los VJP son -v y E_J=0.
    d = 4
    g = torch.Generator().manual_seed(314159)
    probes = [(torch.randint(0, 2, (1, d), generator=g).double() * 2 - 1)
              for _ in range(10)]
    for factor, target in [(1.0, 0.0), (2.0, 1.0), (0.0, 1.0)]:
        x = torch.randn((1, d), dtype=torch.float64, generator=g).requires_grad_(True)
        score = -factor * x
        # Para score=0, autograd debe reconocer dependencia de x (factor multiplicado).
        err2, j2, trace = vjp_measure(score, x, probes)
        assert abs(err2 / d - target ** 2) < 1e-12, (factor, err2, target)
        assert abs(j2 / d - factor ** 2) < 1e-12
        assert abs(trace / d + factor) < 1e-12
    # Otra validacion: todas las 2**3 direcciones Rademacher reproducen Frobenius exacto.
    mat = torch.tensor([[.3, -1.7, 2.4], [1.2, .4, -.6], [-.1, 1.9, -2.]], dtype=torch.float64)
    x = torch.randn((1, 3), dtype=torch.float64, generator=g).requires_grad_(True)
    s = (mat @ x.flatten()).reshape_as(x)
    probes = [torch.tensor([p], dtype=torch.float64) for p in itertools.product((-1., 1.), repeat=3)]
    err2, j2, trace = vjp_measure(s, x, probes)
    assert abs(err2 - float(((mat + torch.eye(3, dtype=torch.float64))**2).sum())) < 1e-11
    assert abs(j2 - float((mat**2).sum())) < 1e-11
    assert abs(trace - float(mat.trace())) < 1e-11
    print("SELFTEST OK: J=-I, J=-2I, J=0 y matriz general verificados por VJP.")


def check():
    previous.check()
    print("Arquitecturas importadas desde:", Path(previous.__file__).resolve())
    print("Este comando solo inspecciona rutas; no calcula derivadas ni usa GPU.")


def status(args):
    rows = read_rows()
    counts = defaultdict(int)
    for row in rows:
        counts[(int(row["t"]), row["model"])] += 1
    print("Mediciones guardadas:", len(rows))
    for t in times_of(args):
        for name in MODEL_NAMES:
            print(f"t={t:3d}: {name}: {counts[t, name]}/{args.images}")
    print("Resumen listo:", SUMMARY.exists(), "Diferencias listas:", DIFFS.exists())


def bootstrap_analysis(args, rows):
    times = times_of(args)
    lookup = {(int(r["t"]), int(r["image_index"]), r["model"]): r for r in rows}
    for t in times:
        for model in MODEL_NAMES:
            if not all((t, i, model) in lookup for i in range(args.images)):
                print("Esperando mediciones completas. Resumen aun no disponible.")
                return
    rng = np.random.default_rng(args.seed + 1009)
    summary, diffs = [], []
    stats = ["frob_j_plus_i_sq_mc", "frob_j_sq_mc", "trace_j_mc", "score_plus_x_sq", "x_sq"]
    for t in times:
        per_model = {}
        boots = {}
        index = rng.integers(0, args.images, size=(args.bootstrap, args.images))
        for model in MODEL_NAMES:
            a = np.asarray([[float(lookup[t, i, model][s]) for s in stats]
                            for i in range(args.images)], dtype=np.float64)
            if not np.isfinite(a).all() or (a[:, 0] < 0).any() or (a[:, 1] < 0).any() or (a[:, 4] <= 0).any():
                raise RuntimeError(f"Mediciones invalidas para {model} t={t}")
            means = a.mean(axis=0)
            # Cociente de los promedios, no promedio de cocientes.
            ej = np.sqrt(max(0., means[0]) / D)
            jnorm = np.sqrt(max(0., means[1]))
            score_err = np.sqrt(max(0., means[3]) / means[4])
            resampled_means = a[index].mean(axis=1)
            ej_bs = np.sqrt(np.maximum(resampled_means[:, 0], 0) / D)
            score_bs = np.sqrt(np.maximum(resampled_means[:, 3], 0) / resampled_means[:, 4])
            boots[model] = ej_bs
            lo, hi = np.quantile(ej_bs, [.025, .975])
            score_lo, score_hi = np.quantile(score_bs, [.025, .975])
            summary.append({"t": t, "model": model, "n_images": args.images,
                            "probes_per_image": args.probes, "E_J": ej, "E_J_low95": lo,
                            "E_J_high95": hi, "J_frobenius_mc": jnorm,
                            "trace_J_mc": means[2], "score_rel_E": score_err,
                            "score_rel_low95": score_lo, "score_rel_high95": score_hi})
            per_model[model] = ej
        diff = per_model[MODEL_NAMES[1]] - per_model[MODEL_NAMES[0]]
        diff_bs = boots[MODEL_NAMES[1]] - boots[MODEL_NAMES[0]]
        low, high = np.quantile(diff_bs, [.025, .975])
        diffs.append({"t": t, "E_J_B_minus_A": diff,
                      "IC95_low": low, "IC95_high": high,
                      "n_images": args.images, "probes_per_image": args.probes})
    atomic_csv_write(SUMMARY, summary, list(summary[0]))
    atomic_csv_write(DIFFS, diffs, list(diffs[0]))
    FIGS.mkdir(parents=True, exist_ok=True)
    xx = np.asarray(times)
    fig, ax = plt.subplots(figsize=(9, 5))
    for name in MODEL_NAMES:
        arr = [z for z in summary if z["model"] == name]
        y = np.array([z["E_J"] for z in arr]); lo = np.array([z["E_J_low95"] for z in arr]); hi = np.array([z["E_J_high95"] for z in arr])
        ax.errorbar(xx, y, yerr=[np.maximum(y-lo, 0), np.maximum(hi-y, 0)],
                    marker="o", capsize=3, linewidth=1.5, label=name)
    ax.axhline(0, ls="--", color="black", lw=1, label="Teoria: J = -I")
    ax.set_xlabel("Paso de difusion t"); ax.set_ylabel(r"$E_J=\|J_s+I\|_F/\sqrt{3072}$")
    ax.set_title("Desviacion directa respecto al jacobiano gaussiano")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(FIGS / "error_jacobiano_vs_t.png", dpi=210); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    d = np.array([z["E_J_B_minus_A"] for z in diffs]); lo = np.array([z["IC95_low"] for z in diffs]); hi = np.array([z["IC95_high"] for z in diffs])
    ax.errorbar(xx, d, yerr=[np.maximum(d-lo, 0), np.maximum(hi-d, 0)], capsize=3, marker="o", linewidth=1.5)
    ax.axhline(0, ls="--", color="black", lw=1)
    ax.set_xlabel("Paso de difusion t"); ax.set_ylabel(r"$E_{J,\mathrm{ResUNet}}-E_{J,\mathrm{Attention}}$")
    ax.set_title("Diferencia pareada del error de jacobiano (IC bootstrap 95%)")
    ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(FIGS / "diferencia_error_IC95.png", dpi=210); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for name in MODEL_NAMES:
        arr = [z for z in summary if z["model"] == name]
        y = np.array([z["score_rel_E"] for z in arr]); lo=np.array([z["score_rel_low95"] for z in arr]); hi=np.array([z["score_rel_high95"] for z in arr])
        ax.errorbar(xx, y, yerr=[np.maximum(y-lo, 0), np.maximum(hi-y, 0)], label=name,
                    marker="o", capsize=3, linewidth=1.5)
    ax.axhline(0, ls="--", color="black", lw=1, label="Teoria: s = -x")
    ax.set_xlabel("Paso de difusion t"); ax.set_ylabel(r"$\sqrt{\mathbb{E}\|s_\theta+x\|^2/\mathbb{E}\|x\|^2}$")
    ax.set_title("Error de campo score frente al score gaussiano")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(FIGS / "error_score_gaussiano_vs_t.png", dpi=210); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for name in MODEL_NAMES:
        arr = [z for z in summary if z["model"] == name]
        ax.plot(xx, [z["J_frobenius_mc"] for z in arr], marker="o", label=name)
    ax.axhline(np.sqrt(D), ls="--", color="black", lw=1,
               label=rf"Referencia $\sqrt{{3072}}={np.sqrt(D):.3f}$")
    ax.set_xlabel("Paso de difusion t"); ax.set_ylabel(r"$\|J_s\|_F$ (estimado)")
    ax.set_yscale("log"); ax.set_title("Norma del jacobiano frente a la referencia gaussiana")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(FIGS / "norma_jacobiano_con_referencia.png", dpi=210); plt.close(fig)
    print("TABLAS Y GRAFICAS GUARDADAS", SUMMARY, DIFFS, FIGS, flush=True)


def all_run(args):
    if not 2 <= args.images <= previous.TEST_COUNT:
        raise ValueError(f"--images debe estar entre 2 y {previous.TEST_COUNT}")
    if args.probes < 1 or args.bootstrap < 100:
        raise ValueError("--probes >=1 y --bootstrap >=100")
    if not torch.cuda.is_available():
        raise RuntimeError("Este experimento debe lanzarse en un nodo GPU mediante SLURM")
    check()
    require_same_protocol(args)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(2)
    dev = torch.device("cuda:0")
    print("GPU:", torch.cuda.get_device_name(dev), flush=True)
    models = [previous.model_from_weights(previous.SimpleUNetAttention, previous.MODEL_SPECS[0][2], dev),
              previous.model_from_weights(previous.ResUNetAttention, previous.MODEL_SPECS[1][2], dev)]
    for model in models:
        model.eval().requires_grad_(False)
    clean, chosen_indices = previous.get_dataset()
    clean = clean[:args.images]
    chosen_indices = chosen_indices[:args.images]
    _, _, alpha_bar = previous.get_schedule(dev)
    times = times_of(args)
    rows = read_rows()
    done = {(str(r["model"]), int(r["t"]), int(r["image_index"])) for r in rows}
    print(f"COMIENZO: {args.images} imagenes x {len(times)} tiempos x 2 modelos x {args.probes} sondas", flush=True)
    for t in times:
        sigma = torch.sqrt(1 - alpha_bar[t])
        sqrt_ab = torch.sqrt(alpha_bar[t])
        for i in range(args.images):
            if all((name, t, i) in done for name in MODEL_NAMES):
                continue
            rng_noise = torch.Generator(device="cpu").manual_seed(args.seed + 1000003 * t + i * 97)
            noise = torch.randn(clean[i:i+1].shape, generator=rng_noise).to(dev)
            xt = sqrt_ab * clean[i:i+1].to(dev) + sigma * noise
            rng_probe = torch.Generator(device="cpu").manual_seed(args.seed + 2000003 * t + 131 * i + 42)
            probes = [rademacher(xt.shape, rng_probe, dev) for _ in range(args.probes)]
            for model, name in zip(models, MODEL_NAMES):
                if (name, t, i) in done:
                    continue
                x = xt.detach().clone().requires_grad_(True)
                ts = torch.full((1,), t, dtype=torch.long, device=dev)
                score = -model(x, ts) / sigma
                with torch.no_grad():
                    sx_sq = float((score + x).square().sum().item())
                    x_sq = float(x.square().sum().item())
                err_j, norm_j, trace_j = vjp_measure(score, x, probes)
                rows.append({"model": name, "t": t, "image_index": i,
                             "cifar10_test_index": chosen_indices[i], "n_probes": args.probes,
                             "frob_j_plus_i_sq_mc": err_j, "frob_j_sq_mc": norm_j,
                             "trace_j_mc": trace_j, "score_plus_x_sq": sx_sq, "x_sq": x_sq})
                done.add((name, t, i))
                del x, score
            rows.sort(key=lambda r: (int(r["t"]), int(r["image_index"]), str(r["model"])))
            atomic_csv_write(ROWS, rows, ROW_FIELDS)
            if (i+1) % 4 == 0 or i+1 == args.images:
                print(f"t={t:3d} imagen={i+1:3d}/{args.images} filas={len(rows)}", flush=True)
            del xt, probes, noise
        print(f"COMPLETADO t={t}", flush=True)
    bootstrap_analysis(args, rows)
    print("FINALIZADO. RESULTADOS:", OUT, flush=True)


if __name__ == "__main__":
    args = get_args()
    if args.mode == "check": check()
    elif args.mode == "selftest": verify_math()
    elif args.mode == "status": status(args)
    elif args.mode == "all": all_run(args)
