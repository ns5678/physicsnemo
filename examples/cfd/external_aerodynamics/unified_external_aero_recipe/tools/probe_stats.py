"""Inspect the contents of stats/hlpw_domino_surface.pt.

Purpose: confirm or refute hypothesis D2 (target-space mismatch). The new HLPW
pipeline applies NormalizeMeshFields with stats loaded from this file. If the
file contains per-field mean/std in the expected shape, z-score normalization
is active and the loss sees N(0, 1)-ish targets -- which matches the observed
loss = 1.0 plateau signature.

Three outcomes we care about:
  1. Keys {temperature, pressure, tau_wall} present with reasonable mean/std.
  2. Keys present but std is ~0 or pathological.
  3. Keys missing or file empty (NormalizeMeshFields silently no-ops).

Run from the recipe root (unified_external_aero_recipe/). No side effects.
"""

from __future__ import annotations

from pathlib import Path
from pprint import pformat

import torch


STATS_PATH = Path(__file__).resolve().parent.parent / "stats" / "hlpw_domino_surface.pt"


def describe(obj, indent: int = 0) -> str:
    pad = "  " * indent
    if isinstance(obj, torch.Tensor):
        t = obj.detach().cpu().float()
        if t.numel() <= 8:
            return (
                f"Tensor(shape={tuple(t.shape)}, dtype={obj.dtype}, "
                f"values={t.tolist()})"
            )
        return (
            f"Tensor(shape={tuple(t.shape)}, dtype={obj.dtype}, "
            f"mean={t.mean().item():.6g}, std={t.std().item():.6g}, "
            f"min={t.min().item():.6g}, max={t.max().item():.6g})"
        )
    if isinstance(obj, dict):
        lines = ["{"]
        for k, v in obj.items():
            lines.append(f"{pad}  {k!r}: {describe(v, indent + 1)},")
        lines.append(f"{pad}}}")
        return "\n".join(lines)
    if isinstance(obj, (list, tuple)):
        inner = ", ".join(describe(x, indent + 1) for x in obj)
        return f"{type(obj).__name__}({inner})"
    return pformat(obj)


def main() -> None:
    print(f"Stats file: {STATS_PATH}")
    print(f"Exists: {STATS_PATH.exists()}  Size: "
          f"{STATS_PATH.stat().st_size if STATS_PATH.exists() else 'N/A'} bytes")
    print()

    if not STATS_PATH.exists():
        print("[FAIL] Stats file does not exist. NormalizeMeshFields would fail "
              "or no-op, depending on implementation.")
        return

    obj = torch.load(STATS_PATH, map_location="cpu", weights_only=False)

    print(f"Top-level type: {type(obj).__name__}")
    if isinstance(obj, dict):
        print(f"Top-level keys: {list(obj.keys())}")
        print()
        for k in obj:
            print(f"--- {k} ---")
            print(describe(obj[k], indent=1))
            print()
    else:
        print(describe(obj))

    print()
    print("Hypothesis check (D2):")
    print("  Expect keys at some level: temperature, pressure, tau_wall")
    print("  Expect each entry to carry mean + std (scalar or per-component)")
    print("  If std ~ O(1) and targets have been non-dim'd first, the loss is")
    print("  operating on N(0, 1) targets. MSE=1 plateau = predict-the-mean.")


if __name__ == "__main__":
    main()
