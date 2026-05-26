#!/usr/bin/env python3
"""Collect per-field normalization stats for the HLPW surface pipeline.

Reads the HLPW boundary fields from the .pdmsh file, applies the same
non-dimensionalization formulas that ``NonDimensionalizeByMetadata``
would apply (driven by the freestream metadata hardcoded in
``datasets/hlpw_domino_surface.yaml``), and writes a
``.pt`` file suitable for ``MinMaxNormalizeMeshFields.stats_file``.

Stats schema matches ``physicsnemo.datapipes.transforms.mesh.transforms.MinMaxNormalizeMeshFields``::

    {
        "<field_name>": {
            "type": "scalar" | "vector",
            "min": tensor,   # scalar for scalar fields, (D,) for vector
            "max": tensor,   # same shape as min
        },
        ...
    }

Vector fields use per-component min/max so each component is rescaled
independently to ``[-1, 1]``, matching the legacy
``scale_model_targets`` / ``unscale_model_outputs`` contract in
``physicsnemo/datapipes/cae/domino_datapipe.py``.

Fields are averaged from vertices to cells before stats are computed,
matching the ``PointDataToCellData`` step in the training pipeline
(``boundary.cell_data[f][c] = boundary.point_data[f][cells[c]].mean()``).
This keeps the stats consistent with the cell-averaged values the model
actually sees at training time.

Run on a compute node (not the login node).  Each case is a ~275M-cell
surface (averaged from ~139M vertices); wall time scales linearly with
the number of cases in the selected split.

Usage:
    python collect_hlpw_surface_stats.py \
        --dataset-root /path/to/PhysicsNeMo-HighLiftAeroML \
        --manifest     /path/to/PhysicsNeMo-HighLiftAeroML/manifest.json \
        --split        single_aoa_4_train \
        --out-path     stats/hlpw_domino_surface.pt

Omit --manifest and --split to use every domain_*.pdmsh under --dataset-root.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import torch

from physicsnemo.mesh import DomainMesh


# -- HLPW metadata (matches datasets/hlpw_domino_surface.yaml metadata block) ---
U_INF = [2672.95, 0.0, 186.92]   # freestream velocity, in/s
P_INF = 176.352                   # freestream pressure, slug/(in*s^2)
RHO_INF = 1.3756e-6               # freestream density, slug/in^3
T_INF = 518.67                    # freestream temperature, degR


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collect per-field min/max stats for HLPW surface fields.",
    )
    p.add_argument(
        "--dataset-root", type=Path, required=True,
        help="Directory containing geo_LHC*_AoA_*/domain_*.pdmsh case dirs.",
    )
    p.add_argument(
        "--manifest", type=Path, default=None,
        help="Path to manifest.json. Requires --split.",
    )
    p.add_argument(
        "--split", type=str, default=None,
        help="Manifest key (e.g. single_aoa_4_train). Required with --manifest.",
    )
    p.add_argument(
        "--out-path", type=Path, required=True,
        help="Output .pt stats file. Parent dir is created if missing.",
    )
    args = p.parse_args()
    if args.manifest is not None and args.split is None:
        p.error("--split is required when --manifest is given.")
    return args


def discover_paths(
    dataset_root: Path,
    manifest: Path | None,
    split: str | None,
) -> list[str]:
    """Return sorted .pdmsh paths under dataset_root, optionally filtered by manifest split.

    Manifest entries are matched against each path's parent directory name,
    mirroring ``resolve_manifest_indices`` in datasets.py:336-351 so the
    stats run sees exactly the cases the training loader will see.
    """
    pattern = str(dataset_root / "**" / "domain_*.pdmsh")
    paths = sorted(glob.glob(pattern, recursive=True))
    if not paths:
        raise FileNotFoundError(f"No domain_*.pdmsh files under {dataset_root}")
    if manifest is None:
        return paths
    with open(manifest) as f:
        manifest_data = json.load(f)
    if split not in manifest_data:
        raise KeyError(
            f"Split {split!r} not in manifest. Available: {list(manifest_data)}"
        )
    case_ids = set(manifest_data[split])
    filtered = [p for p in paths if Path(p).parent.name in case_ids]
    if not filtered:
        raise ValueError(
            f"No paths matched split {split!r} under {dataset_root}. "
            f"Manifest has {len(case_ids)} cases; example: {next(iter(case_ids))}. "
            f"Discovered example: {Path(paths[0]).parent.name}"
        )
    missing = case_ids - {Path(p).parent.name for p in filtered}
    if missing:
        print(
            f"WARNING: {len(missing)} manifest entries not found under "
            f"{dataset_root}. Example missing: {next(iter(missing))}"
        )
    return filtered


def main() -> None:
    args = parse_args()
    paths = discover_paths(args.dataset_root, args.manifest, args.split)
    print(f"Selected {len(paths)} cases for stats collection.")

    U_inf = torch.tensor(U_INF, dtype=torch.float64)
    p_inf = torch.tensor(P_INF, dtype=torch.float64)
    rho_inf = torch.tensor(RHO_INF, dtype=torch.float64)
    T_inf = torch.tensor(T_INF, dtype=torch.float64)
    q_inf = 0.5 * rho_inf * (U_inf * U_inf).sum()
    print(f"q_inf = {q_inf.item():.6g}  (freestream dynamic pressure)")

    # Running min/max accumulators in fp64.  Scalar fields track 0-d
    # min/max; vector fields track per-component min/max so each
    # component is rescaled to [-1, 1] independently (legacy
    # scale_model_targets behaviour).
    inf64 = float("inf")
    min_T = torch.full((), inf64, dtype=torch.float64)
    max_T = torch.full((), -inf64, dtype=torch.float64)
    min_P = torch.full((), inf64, dtype=torch.float64)
    max_P = torch.full((), -inf64, dtype=torch.float64)
    min_TW = torch.full((3,), inf64, dtype=torch.float64)
    max_TW = torch.full((3,), -inf64, dtype=torch.float64)
    n_total = 0

    for path in paths:
        print(f"\nLoading: {path}")
        d = DomainMesh.load(path)
        boundary = d.boundaries["boundary"]
        pd = boundary.point_data
        cells = boundary.cells.to(torch.int64)              # (N_c, 3) vertex ids
        T_pt = pd["PROJ(AVG(T))"].to(torch.float64)         # (N_p,)
        P_pt = pd["PROJ(AVG(P))"].to(torch.float64)         # (N_p,)
        TW_pt = pd["AVG(TAU_WALL)"].to(torch.float64)       # (N_p, 3)

        # Cell-average each field via the mean over the three triangle
        # vertices.  Matches the pipeline's PointDataToCellData step (a
        # thin wrapper over Mesh.point_data_to_cell_data, mesh.py:1462),
        # so the stats here reflect the fields the model actually sees at
        # training time.
        T_cell = T_pt[cells].mean(dim=1)                    # (N_c,)
        P_cell = P_pt[cells].mean(dim=1)                    # (N_c,)
        TW_cell = TW_pt[cells].mean(dim=1)                  # (N_c, 3)
        n = T_cell.shape[0]

        # Apply the SAME non-dim formulas as NonDimensionalizeByMetadata:
        #   temperature: x / T_inf
        #   pressure:    (x - p_inf) / q_inf
        #   stress:      x / q_inf   (applies to tau_wall)
        T_nd = T_cell / T_inf
        P_nd = (P_cell - p_inf) / q_inf
        TW_nd = TW_cell / q_inf

        min_T = torch.minimum(min_T, T_nd.min())
        max_T = torch.maximum(max_T, T_nd.max())
        min_P = torch.minimum(min_P, P_nd.min())
        max_P = torch.maximum(max_P, P_nd.max())
        min_TW = torch.minimum(min_TW, TW_nd.amin(dim=0))
        max_TW = torch.maximum(max_TW, TW_nd.amax(dim=0))
        n_total += n
        print(f"  added {n:,d} cells (running total: {n_total:,d})")

    stats = {
        "temperature": {
            "type": "scalar",
            "min": min_T.to(torch.float32),
            "max": max_T.to(torch.float32),
        },
        "pressure": {
            "type": "scalar",
            "min": min_P.to(torch.float32),
            "max": max_P.to(torch.float32),
        },
        "tau_wall": {
            "type": "vector",
            "min": min_TW.to(torch.float32),
            "max": max_TW.to(torch.float32),
        },
    }

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stats, args.out_path)

    print(f"\n==> wrote {args.out_path}")
    print(f"    total points: {n_total:,d}")
    for k, v in stats.items():
        min_str = v["min"].tolist() if v["min"].ndim > 0 else f"{v['min'].item():.6g}"
        max_str = v["max"].tolist() if v["max"].ndim > 0 else f"{v['max'].item():.6g}"
        print(f"    {k:12s}  type={v['type']:6s}  min={min_str}  max={max_str}")


if __name__ == "__main__":
    main()
