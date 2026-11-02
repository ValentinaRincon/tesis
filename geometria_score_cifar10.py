#!/usr/bin/env python3
"""Geometria diferencial del score en CIFAR-10 (no entrena modelos).

Se importan las ARQUITECTURAS y loaders verificados de
`analizar_score_cifar10.py`, y se utilizan los checkpoints EMA existentes.

El score estimado es s_theta(x,t)=-eps_theta(x,t)/sqrt(1-alphabar_t).
J=ds/dx es la derivada respecto a LOS 3072 COMPONENTES de una imagen,
NO un gradiente espacial 2D sobre posiciones de pixeles.

Para v,w independientes, isotropos y centrados (Rademacher +/-1):
  d=v^T J w - w^T J v = w^T (J^T v)-v^T (J^T w)
  E[d^2] = ||J-J^T||_F^2
  E[||J^T v||^2] = ||J||_F^2.

Asi, ||(J-J^T)/2||_F^2 se estima con 0.25*promedio(d^2).
Solo requiere dos productos vector-jacobiano (VJP) por pareja; jamas
construye la matriz 3072x3072. Es una estimacion Monte Carlo,
no una certificacion de conservatividad del campo global.

En nodo LOGIN (sin utilizar GPU):
  python geometria_score_cifar10.py --mode selftest
  python geometria_score_cifar10.py --mode check
  python geometria_score_cifar10.py --mode status
En nodo COMPUTO por sbatch:
  python -u geometria_score_cifar10.py --mode all
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import analizar_score_cifar10 as previous

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "Geometria Score CIFAR10"
ROWS = OUT / "mediciones_individuales.csv"
SUMMARY = OUT / "resumen_geom_score.csv"
DIFFS = OUT / "diferencias_pareadas.csv"
MANIFEST = OUT / "protocolo.json"
FIGS = OUT / "figuras"

MODEL_NAMES = ["Attention + EMA", "ResUNet + Attention + EMA"]
DEFAULT_T = [0, 50, 250, 500, 750, 999]
DEFAULT_SEED = 20261025


def parse_args():
    ap = argparse.ArgumentParser(description="Asimetria del jacobiano del score (CIFAR-10)")
    ap.add_argument("--mode", choices=["check", "selftest", "status", "all"], required=True)
    ap.add_argument("--images", type=int, default=24, help="Imagenes CIFAR-10 test (piloto: 8)")
    ap.add_argument("--probes", type=int, default=6, help="Parejas de direcciones aleatorias por imagen")
    ap.add_argument("--times", type=str, default=",".join(map(str, DEFAULT_T)))
    ap.add_argument("--bootstrap", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return ap.parse_args()


def csv_write_atomic(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


FIELDS = ["model", "t", "image_index", "cifar10_test_index", "n_probes",
          "asym_frob2_est", "jac_frob2_est", "rho_image_est", "score_l2"]


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get_timepoints(args):
    try:
        vals = [int(z.strip()) for z in args.times.split(",")]
    except ValueError as err:
        raise SystemExit("--times debe contener enteros separados por comas") from err
    if not vals or len(set(vals)) != len(vals) or not all(0 <= z < previous.T for z in vals):
        raise SystemExit("Cada tiempo debe ser unico y estar entre 0 y 999")
    return sorted(vals)


def file_check():
    print("Chequeo de rutas; no ejecuta modelos ni reserva GPU.")
    previous.check()
    print("Modulo de arquitecturas:", Path(previous.__file__).resolve())
    print("Dimensiones: CIFAR-10 RGB 3x32x32 = 3072.")
    print("La ejecucion --mode all requiere GPU mediante SLURM.")


def get_manifest(args):
    model_hashes = {
        key: file_sha256(previous.MODEL_ROOT / filename)
        for key, _, filename in previous.MODEL_SPECS
    }
    return {
        "analysis": "score_jacobian_antisymmetry_mc_vjp_v1",
        "models": model_hashes,
        "architecture_script_sha256": file_sha256(previous.__file__),
        "n_images": args.images,
        "n_probe_pairs": args.probes,
        "times": get_timepoints(args),
        "bootstrap": args.bootstrap,
        "seed": args.seed,
        "cifar10_selection_seed": previous.SEED_DATA,
        "cifar10_source": "test; Normalize([.5]*3,[.5]*3)",
        "beta_start": previous.BETA_START,
        "beta_end": previous.BETA_END,
        "T": previous.T,
        "jacobian_wrt": "all_3072_normalized_image_components",
        "data_sampling": "same_clean_images_noises_and_rademacher_probes_for_both_models",
        "definition_rho": "sqrt(0.25*mean(delta^2)/mean(0.5*(norm(JT_v)^2+norm(JT_w)^2)))",
        "interpretation": "stochastic estimate; finite probes/images; local symmetry necessary, not sufficient, for true score",
        "torch_version": torch.__version__,
    }


def save_or_validate_manifest(args):
    new = get_manifest(args)
    if MANIFEST.exists():
        old = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if old != new:
            raise RuntimeError(
                "El protocolo de la carpeta existente es distinto. "
                "No mezcles datos: usa la configuracion previa o mueve la carpeta completa."
            )
    else:
        OUT.mkdir(exist_ok=True, parents=True)
        MANIFEST.write_text(json.dumps(new, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def rademacher(shape, generator, device):
    # Isotropia E[v_i v_j]=delta_ij; direccion compartida entre modelos.
    return (torch.randint(0, 2, shape, generator=generator, dtype=torch.int8)
            .to(device=device, dtype=torch.float32).mul_(2).sub_(1))


def vjp_pair(s, x, v, w, release_graph):
    """Devuelve cuadrados Frobenius MC de parte antisimetrica y jacobiano."""
    # J^T v=grad_x <v,s>; no es necesario construir J.
    jt_v = torch.autograd.grad((v * s).sum(), x, retain_graph=True)[0]
    jt_w = torch.autograd.grad((w * s).sum(), x, retain_graph=not release_graph)[0]
    with torch.no_grad():
        # d = v^T J w - w^T J v
        d = (w * jt_v).sum() - (v * jt_w).sum()
        numerator = 0.25 * d.square()
        denominator = 0.5 * (jt_v.square().sum() + jt_w.square().sum())
        return float(numerator.detach().item()), float(denominator.detach().item())


def measure(model, xt, t, sigma, pairs):
    x = xt.detach().clone().requires_grad_(True)
    t_tensor = torch.full((1,), t, dtype=torch.long, device=x.device)
    # No usar no_grad: necesitamos derivadas respecto a x, pero NO pesos.
    s = -model(x, t_tensor) / sigma
    s_norm = float(s.detach().flatten().norm().item())
    nums, dens = [], []
    for i, (v, w) in enumerate(pairs):
        num, den = vjp_pair(s, x, v, w, release_graph=(i == len(pairs) - 1))
        nums.append(num)
        dens.append(den)
    numerator = float(np.mean(nums))
    denominator = float(np.mean(dens))
    return numerator, denominator, float(np.sqrt(numerator / denominator)) if denominator > 0 else float("nan"), s_norm


def existing_rows():
    if not ROWS.is_file():
        return []
    with ROWS.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run(args):
    if args.images < 2 or args.images > previous.TEST_COUNT:
        raise ValueError(f"--images entre 2 y {previous.TEST_COUNT}")
    if args.probes < 1 or args.bootstrap < 100:
        raise ValueError("--probes >=1 y --bootstrap >=100")
    if not torch.cuda.is_available():
        raise RuntimeError("--mode all solo se ejecuta con GPU asignada por SLURM")
    file_check()
    save_or_validate_manifest(args)
    FIGS.mkdir(exist_ok=True, parents=True)

    # Evitar TF32 para comparacion mas estable de productos y derivadas.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(2)
    dev = torch.device("cuda:0")
    print("GPU:", torch.cuda.get_device_name(dev), flush=True)
    models = [
        previous.model_from_weights(previous.SimpleUNetAttention, previous.MODEL_SPECS[0][2], dev),
        previous.model_from_weights(previous.ResUNetAttention, previous.MODEL_SPECS[1][2], dev),
    ]
    for m in models:
        m.eval().requires_grad_(False)

    clean, source_indices = previous.get_dataset()
    clean = clean[:args.images]
    source_indices = source_indices[:args.images]
    _, _, abar = previous.get_schedule(dev)
    rows = existing_rows()
    done = {(str(r["model"]), int(r["t"]), int(r["image_index"])) for r in rows}
    times = get_timepoints(args)
    print(f"Iniciando {len(times)} tiempos x {args.images} imagenes x 2 modelos x {args.probes} parejas", flush=True)
    print("Progreso recuperable por imagen; usando probes y ruido iguales entre modelos", flush=True)

    for t in times:
        sigma = torch.sqrt(1 - abar[t])
        sqrtabar = torch.sqrt(abar[t])
        for i in range(args.images):
            if all((name, t, i) in done for name in MODEL_NAMES):
                continue
            rng_noise = torch.Generator(device="cpu").manual_seed(args.seed + 1000003 * t + i * 97)
            eps = torch.randn(clean[i:i+1].shape, generator=rng_noise).to(dev)
            xt = sqrtabar * clean[i:i+1].to(dev) + sigma * eps
            rng_probe = torch.Generator(device="cpu").manual_seed(args.seed + 2000003 * t + 131 * i + 42)
            pairs = [(rademacher(xt.shape, rng_probe, dev), rademacher(xt.shape, rng_probe, dev))
                     for _ in range(args.probes)]
            for model, label in zip(models, MODEL_NAMES):
                if (label, t, i) in done:
                    continue
                num, den, rho_img, s_norm = measure(model, xt, t, sigma, pairs)
                r = dict(model=label, t=t, image_index=i, cifar10_test_index=source_indices[i],
                         n_probes=args.probes, asym_frob2_est=num, jac_frob2_est=den,
                         rho_image_est=rho_img, score_l2=s_norm)
                rows.append(r)
                done.add((label, t, i))
            # Guardar solo si ambos modelos de la imagen se completaron.
            rows = sorted(rows, key=lambda r: (int(r["t"]), int(r["image_index"]), str(r["model"])))
            csv_write_atomic(ROWS, rows, FIELDS)
            if (i + 1) % 4 == 0 or i + 1 == args.images:
                print(f"t={t:3d} | imagen {i+1:3d}/{args.images} | filas {len(rows)}", flush=True)
            del xt, eps, pairs
        print(f"Tiempo t={t} completado", flush=True)
    make_summary_and_plots(args, rows)
    print("FINALIZADO. RESULTADOS:", OUT, flush=True)


def read_pair_tables(rows, times):
    pairs = {}
    for row in rows:
        key = (int(row["t"]), int(row["image_index"]), str(row["model"]))
        pairs[key] = row
    data = {}
    for t in times:
        vals = {}
        for model in MODEL_NAMES:
            vals[model] = np.array([
                [float(pairs[(t, i, model)]["asym_frob2_est"]),
                 float(pairs[(t, i, model)]["jac_frob2_est"])]
                for i in range(len({r["image_index"] for r in rows if int(r["t"]) == t and str(r["model"]) == model}))
            ], dtype=float)
        data[t] = vals
    return data


def metric_rho(values):
    return float(np.sqrt(max(0, np.mean(values[:, 0])) / np.mean(values[:, 1])))


def make_summary_and_plots(args, rows):
    times = get_timepoints(args)
    counts = {(int(t), name): 0 for t in times for name in MODEL_NAMES}
    for r in rows:
        key = (int(r["t"]), str(r["model"]))
        if key in counts:
            counts[key] += 1
    if not all(v == args.images for v in counts.values()):
        print("El experimento no esta completo; resumen omitido.", counts, flush=True)
        return
    lookup = {(int(r["t"]), int(r["image_index"]), str(r["model"])): r for r in rows}
    rng = np.random.default_rng(args.seed + 987)
    out = []
    diffs = []
    for t in times:
        table = {}
        boots = {}
        for name in MODEL_NAMES:
            a = np.array([[float(lookup[(t, i, name)]["asym_frob2_est"]),
                           float(lookup[(t, i, name)]["jac_frob2_est"])] for i in range(args.images)], dtype=np.float64)
            if not np.isfinite(a).all() or (a[:, 1] <= 0).any():
                raise RuntimeError(f"Valores no finitos o jacobiano nulo para {name}, t={t}")
            table[name] = a
        # MISMO indice bootstrap para ambos modelos, preservando pares.
        ix = rng.integers(args.images, size=(args.bootstrap, args.images))
        for name in MODEL_NAMES:
            vals = table[name]
            est = metric_rho(vals)
            samp = vals[ix].mean(axis=1)
            boot = np.sqrt(np.maximum(samp[:, 0], 0) / samp[:, 1])
            boots[name] = boot
            lo, hi = np.quantile(boot, [0.025, 0.975])
            out.append({"t": t, "model": name, "n_images": args.images, "probe_pairs_per_image": args.probes,
                        "rho": est, "rho_lo95": lo, "rho_hi95": hi,
                        "asym_frob2_mc": vals[:, 0].mean(), "jac_frob2_mc": vals[:, 1].mean()})
        delta = metric_rho(table[MODEL_NAMES[1]]) - metric_rho(table[MODEL_NAMES[0]])
        diff_bs = boots[MODEL_NAMES[1]] - boots[MODEL_NAMES[0]]
        dlo, dhi = np.quantile(diff_bs, [0.025, 0.975])
        diffs.append({"t": t, "rho_B_minus_A": delta, "ci95_low": dlo, "ci95_high": dhi,
                      "n_images": args.images, "probe_pairs_per_image": args.probes})

    csv_write_atomic(SUMMARY, out, list(out[0].keys()))
    csv_write_atomic(DIFFS, diffs, list(diffs[0].keys()))
    xs = np.array(times)
    fig, ax = plt.subplots(figsize=(9, 5.3))
    for name, label in zip(MODEL_NAMES, ["Attention + EMA", "ResUNet + EMA"]):
        rr = [r for r in out if r["model"] == name]
        mids = np.array([r["rho"] for r in rr])
        lo = np.array([r["rho_lo95"] for r in rr]); hi = np.array([r["rho_hi95"] for r in rr])
        ax.errorbar(xs, mids, yerr=[np.maximum(0, mids-lo), np.maximum(0, hi-mids)],
                    marker="o", capsize=4, linewidth=1.5, label=label)
    ax.set_xlabel("Paso de difusion t")
    ax.set_ylabel(r"Fraccion antisimetrica del jacobiano $\rho=\|A\|_F/\|J\|_F$")
    ax.set_title("Geometria del score: estimacion Monte Carlo del jacobiano")
    ax.grid(alpha=0.25); ax.legend(); fig.tight_layout()
    fig.savefig(FIGS / "antisimetria_vs_t.png", dpi=220); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    dd = np.array([r["rho_B_minus_A"] for r in diffs]); lo = np.array([r["ci95_low"] for r in diffs]); hi = np.array([r["ci95_high"] for r in diffs])
    ax.errorbar(xs, dd, yerr=[np.maximum(0, dd-lo), np.maximum(0, hi-dd)],
                marker="o", capsize=4, linewidth=1.5)
    ax.axhline(0, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel("Paso de difusion t")
    ax.set_ylabel(r"$\rho_{\mathrm{ResUNet}} - \rho_{\mathrm{Attention}}$")
    ax.set_title("Diferencia pareada de fraccion antisimetrica (IC bootstrap)")
    ax.grid(alpha=0.25); fig.tight_layout()
    fig.savefig(FIGS / "diferencia_rho_IC95.png", dpi=220); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for name in MODEL_NAMES:
        rr = [r for r in out if r["model"] == name]
        ax.plot(xs, np.sqrt([r["jac_frob2_mc"] for r in rr]), marker="o", label=name)
    ax.set_xlabel("Paso t"); ax.set_ylabel(r"Estimacion de $\|J_s\|_F$")
    ax.set_yscale("log"); ax.set_title("Magnitud del jacobiano del score")
    ax.grid(alpha=0.25); ax.legend(); fig.tight_layout()
    fig.savefig(FIGS / "norma_jacobiano_vs_t.png", dpi=220); plt.close(fig)
    print("Tablas y figuras guardadas:", SUMMARY, DIFFS, FIGS, flush=True)


def status(args):
    if not ROWS.is_file():
        print("Todavia no existe archivo de mediciones:", ROWS)
        return
    rows = existing_rows()
    counts = defaultdict(int)
    for row in rows:
        counts[(int(row["t"]), row["model"])] += 1
    print("Mediciones guardadas:", ROWS, "| Total", len(rows))
    for t in get_timepoints(args):
        for name in MODEL_NAMES:
            print(f"t={t:3d} | {name}: {counts[(t, name)]}/{args.images}")
    print("Resumen disponible:", SUMMARY.exists())


def selftest():
    torch.set_default_dtype(torch.float64)
    # s(x) = M x. Si M es simetrica, d=0 para todo v,w.
    symmetric = torch.tensor([[2., 0.4, -0.7], [0.4, -1.2, 0.2], [-0.7, 0.2, 1.]])
    skew = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 0.]])
    v = torch.tensor([1., -0.3, 0.9]); w = torch.tensor([0.5, 0.7, -0.4])
    for mat in [symmetric, skew]:
        x = torch.tensor([0.1, -0.2, 0.6], requires_grad=True)
        y = mat @ x
        jtv = torch.autograd.grad((v*y).sum(), x, retain_graph=True)[0]
        jtw = torch.autograd.grad((w*y).sum(), x)[0]
        d = w@jtv - v@jtw
        expected = v @ (mat @ w) - w @ (mat @ v)
        assert torch.allclose(d, expected, rtol=1e-12, atol=1e-12)
        if mat is symmetric:
            assert abs(float(d)) < 1e-12
    # Momento de bilineal con Rademacher: E d² = ||J-Jᵀ||_F².
    g = torch.Generator().manual_seed(40)
    directions = (torch.randint(0,2,(30000,3),generator=g,dtype=torch.int64).double()*2-1)
    directions2 = (torch.randint(0,2,(30000,3),generator=g,dtype=torch.int64).double()*2-1)
    skew_delta = torch.einsum("bi,ij,bj->b", directions, skew-skew.T, directions2)
    mc = float((skew_delta**2).mean())
    exact = float(((skew-skew.T)**2).sum())
    assert abs(mc/exact-1) < .06, (mc, exact)
    # Verifica el helper VJP sin usar pesos entrenados.
    x = torch.randn((1,3,2,2), dtype=torch.float64, requires_grad=True)
    mat = torch.eye(12, dtype=torch.float64)
    s = (mat@x.flatten()).reshape_as(x)
    p = torch.randn(x.shape, dtype=torch.float64)
    q = torch.randn(x.shape, dtype=torch.float64)
    num, den = vjp_pair(s, x, p, q, release_graph=True)
    assert abs(num) < 1e-20 and den > 0
    torch.set_default_dtype(torch.float32)
    print("SELFTEST OK: VJP y estimador de jacobiano antisimetrico", flush=True)


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "selftest": selftest()
    elif args.mode == "check": file_check()
    elif args.mode == "status": status(args)
    else: run(args)
