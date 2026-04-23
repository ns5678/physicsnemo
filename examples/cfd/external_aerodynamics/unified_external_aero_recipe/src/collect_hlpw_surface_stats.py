#!/usr/bin/env python3
"""Collect per-field normalization stats for the HLPW surface pipeline.

Reads the HLPW boundary fields from the .pdmsh file, applies the same
non-dimensionalization formulas that ``NonDimensionalizeByMetadata``
would apply (driven by the freestream metadata hardcoded in
``conf/dataset/hlpw_domino_surface.yaml``), and writes a
``.pt`` file suitable for ``NormalizeMeshFields.stats_file``.

Stats schema matches ``physicsnemo.datapipes.transforms.mesh.transforms.NormalizeMeshFields``:

    {
        "<field_name>": {
            "type": "scalar" | "vector",
            "mean": tensor,   # scalar for scalar fields, (D,) for vector
            "std":  tensor,   # scalar (direction-preserving for vectors)
        },
        ...
    }

For vector fields the std is a single scalar equal to
``sqrt(mean((x - mean_per_component)**2))`` averaged over all
components AND all points -- same convention as the inline vector
stats in ``highlift_surface.yaml`` and ``drivaer_ml_surface.yaml``.

Run on a compute node (not the login node).  Single pass over the
one HLPW case (~139M surface points).

Usage:
    python collect_hlpw_surface_stats.py
"""

from __future__ import annotations

import glob
from pathlib import Path

import torch

from physicsnemo.mesh import DomainMesh


# -- HLPW metadata (matches conf/dataset/hlpw_domino_surface.yaml metadata block) ---
U_INF = [2672.95, 0.0, 186.92]   # freestream velocity, in/s
P_INF = 176.352                   # freestream pressure, slug/(in*s^2)
RHO_INF = 1.3756e-6               # freestream density, slug/in^3
T_INF = 518.67                    # freestream temperature, degR

PMSH_GLOB = (
    "/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/users/snidhan/"
    "HLPW-Benchmarking/data/pmsh/**/domain_*.pdmsh"
)

OUT_PATH = Path(
    "/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/users/snidhan/"
    "HLPW-Benchmarking/physicsnemo/examples/cfd/external_aerodynamics/"
    "unified_external_aero_recipe/stats/hlpw_domino_surface.pt"
)


def scalar_stats(x: torch.Tensor) -> dict:
    return {
        "type": "scalar",
        "mean": x.mean().detach().to(torch.float32),
        "std":  x.std().detach().to(torch.float32),
    }


def vector_stats(x: torch.Tensor) -> dict:
    # x: (N, D).  Per-component mean; shared scalar std =
    # sqrt(mean((x - mean)^2)) averaged over components and points.
    mean = x.mean(dim=0).detach().to(torch.float32)          # (D,)
    centered = x - mean.view(1, -1)
    std = centered.pow(2).mean().sqrt().detach().to(torch.float32)  # scalar
    return {"type": "vector", "mean": mean, "std": std}


def main() -> None:
    paths = sorted(glob.glob(PMSH_GLOB, recursive=True))
    if not paths:
        raise FileNotFoundError(f"No .pdmsh under {PMSH_GLOB}")
    if len(paths) > 1:
        print(f"WARNING: found {len(paths)} .pdmsh files; using all of them.")

    U_inf = torch.tensor(U_INF, dtype=torch.float64)
    p_inf = torch.tensor(P_INF, dtype=torch.float64)
    rho_inf = torch.tensor(RHO_INF, dtype=torch.float64)
    T_inf = torch.tensor(T_INF, dtype=torch.float64)
    q_inf = 0.5 * rho_inf * (U_inf * U_inf).sum()
    print(f"q_inf = {q_inf.item():.6g}  (freestream dynamic pressure)")

    # Accumulators in fp64 for numerical stability across large N.
    n_total = 0
    sum_T = torch.zeros((), dtype=torch.float64)
    sumsq_T = torch.zeros((), dtype=torch.float64)
    sum_P = torch.zeros((), dtype=torch.float64)
    sumsq_P = torch.zeros((), dtype=torch.float64)
    sum_TW = torch.zeros(3, dtype=torch.float64)   # per-component
    sumsq_TW = torch.zeros((), dtype=torch.float64)  # scalar: sum_{i,c} tw_ic^2

    for path in paths:
        print(f"\nLoading: {path}")
        d = DomainMesh.load(path)
        pd = d.boundaries["boundary"].point_data
        T_raw = pd["PROJ(AVG(T))"].to(torch.float64)       # (N,)
        P_raw = pd["PROJ(AVG(P))"].to(torch.float64)       # (N,)
        TW_raw = pd["AVG(TAU_WALL)"].to(torch.float64)     # (N, 3)
        n = T_raw.shape[0]

        # Apply the SAME non-dim formulas as NonDimensionalizeByMetadata:
        #   temperature: x / T_inf
        #   pressure:    (x - p_inf) / q_inf
        #   stress:      x / q_inf   (applies to tau_wall)
        T_nd = T_raw / T_inf
        P_nd = (P_raw - p_inf) / q_inf
        TW_nd = TW_raw / q_inf

        sum_T += T_nd.sum()
        sumsq_T += (T_nd * T_nd).sum()
        sum_P += P_nd.sum()
        sumsq_P += (P_nd * P_nd).sum()
        sum_TW += TW_nd.sum(dim=0)
        sumsq_TW += (TW_nd * TW_nd).sum()
        n_total += n
        print(f"  added {n:,d} points (running total: {n_total:,d})")

    # Close-form mean/std from running sums (two-pass not needed: the
    # single-case here means one file, one pass; for multi-case this
    # still works since we accumulate sum/sumsq across files).
    mean_T = sum_T / n_total
    var_T = sumsq_T / n_total - mean_T * mean_T
    std_T = var_T.clamp(min=0.0).sqrt()

    mean_P = sum_P / n_total
    var_P = sumsq_P / n_total - mean_P * mean_P
    std_P = var_P.clamp(min=0.0).sqrt()

    # Vector: per-component mean; shared scalar std.
    mean_TW = sum_TW / n_total                    # (3,)
    # E[|x|^2] = sumsq_TW / (n_total * 3)  (already component-flat)
    # Var_shared = E[(x-mean)^2] averaged over components & points =
    #   sumsq_TW / (n_total * 3) - (1/3) * ||mean_TW||^2
    var_TW = sumsq_TW / (n_total * 3) - (mean_TW * mean_TW).sum() / 3
    std_TW = var_TW.clamp(min=0.0).sqrt()

    stats = {
        "temperature": {
            "type": "scalar",
            "mean": mean_T.to(torch.float32),
            "std":  std_T.to(torch.float32),
        },
        "pressure": {
            "type": "scalar",
            "mean": mean_P.to(torch.float32),
            "std":  std_P.to(torch.float32),
        },
        "tau_wall": {
            "type": "vector",
            "mean": mean_TW.to(torch.float32),
            "std":  std_TW.to(torch.float32),
        },
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stats, OUT_PATH)

    print(f"\n==> wrote {OUT_PATH}")
    print(f"    total points: {n_total:,d}")
    for k, v in stats.items():
        mean_str = v["mean"].tolist() if v["mean"].ndim > 0 else f"{v['mean'].item():.6g}"
        print(f"    {k:12s}  type={v['type']:6s}  mean={mean_str}  std={v['std'].item():.6g}")


if __name__ == "__main__":
    main()
