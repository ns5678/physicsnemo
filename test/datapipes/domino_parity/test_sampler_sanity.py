# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Phase 2.5 Axis 2: sampler sanity check.

This is **not** a parity check against the legacy DoMINODataPipe.  It is a
standalone sanity test that asserts :class:`SubsampleMesh` on pre-shuffled
``.pmsh`` data behaves as uniform-over-the-full-mesh sampling, which is the
precondition under which the PR's contiguous-block sampler (and its
``torch.randperm``-based general path) is equivalent to legacy
``shuffle_array`` for DoMINO's downstream statistics.

If this test fails, the most likely cause is **incomplete on-disk shuffling**
of the ``.pmsh`` files (data-prep contract broken), not a code bug in
``SubsampleMesh``.

Runtime
-------
Synthetic fixture only.  ~50 lines, runs in under a second on CPU.  No
cluster required.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from .fixtures.make_synthetic_sample import build_synthetic_sample

if TYPE_CHECKING:
    from physicsnemo.mesh import Mesh  # noqa: F401  type hints only


N_REPS = 100


@pytest.fixture(scope="module")
def shuffled_boundary_mesh() -> "Mesh":
    """Return the synthetic surface boundary mesh with on-disk shuffling applied."""
    pytest.importorskip("warp")  # Mesh ctor pulls warp via physicsnemo package __init__
    _, domain = build_synthetic_sample(seed=42, shuffle_on_disk=True)
    return domain.boundaries["boundary"]


def _ks_statistic(a: torch.Tensor, b: torch.Tensor) -> float:
    r"""Two-sample Kolmogorov-Smirnov statistic :math:`\sup_x |F_a(x) - F_b(x)|`.

    Implemented with sort + merge so there are no extra dependencies.

    Parameters
    ----------
    a, b : 1-D float tensors

    Returns
    -------
    float
        KS statistic in :math:`[0, 1]`.  Small values ⇒ same distribution.
    """
    a_sorted = torch.sort(a.reshape(-1))[0]
    b_sorted = torch.sort(b.reshape(-1))[0]
    all_sorted = torch.sort(torch.cat([a_sorted, b_sorted]))[0]

    cdf_a = torch.searchsorted(a_sorted, all_sorted, right=True).float() / a.numel()
    cdf_b = torch.searchsorted(b_sorted, all_sorted, right=True).float() / b.numel()

    return float((cdf_a - cdf_b).abs().max().item())


def test_subsample_preserves_cell_area_distribution(
    shuffled_boundary_mesh,
) -> None:
    r"""KS test on sampled cell areas.

    Under uniform sampling from a mesh with area distribution :math:`F`, the
    sampled subset's empirical area distribution converges to :math:`F`.
    Under contiguous-block sampling on spatially-ordered data it would
    converge to a *restricted* distribution (only large or only small
    triangles in a spatial cluster).  The KS statistic picks this up.

    Threshold of 0.05 is generous; for shuffled on-disk order we expect
    KS ~ 1/sqrt(N_sampled) ~ 0.03 at n_cells_sample = 80.
    """
    pytest.importorskip("tensordict")
    from physicsnemo.datapipes.transforms.mesh import SubsampleMesh

    mesh = shuffled_boundary_mesh
    full_areas = mesh.cell_areas.detach().cpu()

    sample_size = max(min(80, mesh.n_cells // 4), 16)
    sampler = SubsampleMesh(n_cells=sample_size)

    sampled_areas: list[torch.Tensor] = []
    for rep in range(N_REPS):
        gen = torch.Generator().manual_seed(12345 + rep)
        sampler.set_generator(gen)
        sub = sampler(mesh)
        sampled_areas.append(sub.cell_areas.detach().cpu())

    pooled = torch.cat(sampled_areas, dim=0)
    ks = _ks_statistic(pooled, full_areas)
    assert ks < 0.05, (
        f"KS statistic {ks:.3f} on pooled sampled cell_areas exceeds 0.05.  "
        f"Either the on-disk ordering is not shuffled enough for "
        f"contiguous-block sampling (expected: no-op on torch.randperm path), "
        f"or SubsampleMesh semantics changed."
    )


def test_subsample_covers_full_bbox(shuffled_boundary_mesh) -> None:
    r"""Sampled-centroid bbox should cover the full mesh bbox over ``N_REPS`` runs.

    Under uniform sampling, the union of sampled-cell centroid bboxes across
    N repetitions converges (quickly, as N grows) to the full-mesh bbox.
    Under spatial-cluster sampling it converges to the bbox of one cluster.

    Threshold: each axis's coverage must be within 5% of the full span.
    """
    pytest.importorskip("tensordict")
    from physicsnemo.datapipes.transforms.mesh import SubsampleMesh

    mesh = shuffled_boundary_mesh
    all_centroids = mesh.cell_centroids.detach().cpu()
    full_min = all_centroids.min(dim=0).values
    full_max = all_centroids.max(dim=0).values
    full_span = full_max - full_min

    sample_size = max(min(80, mesh.n_cells // 4), 16)
    sampler = SubsampleMesh(n_cells=sample_size)

    union_min = torch.full_like(full_min, float("inf"))
    union_max = torch.full_like(full_max, float("-inf"))
    for rep in range(N_REPS):
        gen = torch.Generator().manual_seed(12345 + rep)
        sampler.set_generator(gen)
        sub = sampler(mesh)
        c = sub.cell_centroids.detach().cpu()
        union_min = torch.minimum(union_min, c.min(dim=0).values)
        union_max = torch.maximum(union_max, c.max(dim=0).values)

    union_span = union_max - union_min
    # Coverage fraction per axis; 1.0 means perfect coverage.
    coverage = union_span / full_span.clamp(min=1e-6)
    assert coverage.min() > 0.95, (
        f"Sampled centroid bbox covers only {coverage.tolist()} fraction of "
        f"the full mesh bbox over {N_REPS} repetitions.  Likely cause: "
        f"on-disk data not shuffled; contiguous-block sampling is returning "
        f"spatial patches instead of uniform samples."
    )
