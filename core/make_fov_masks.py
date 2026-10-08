#!/usr/bin/env python3
"""Objective CBCT FOV mask (paper-ready, anatomy-agnostic).

M_FOV is the reconstructed CBCT field of view, derived only from the CBCT
volume with a fixed rule (no CT, no head/lung branching):

  1) Estimate padding from border voxels (mode)
  2) Support = voxels differing from padding by > eps HU
  3) Morphological closing (~2 mm)
  4) Largest connected component
  5) Per axial slice: hole-fill + 2D convex hull  (seals lung air / FOV disk)
  6) Largest CC; optional small dilation

This is more objective for reporting than soft-tissue body masks, because it
depends only on the CBCT reconstruction support.

Example:
  python tools/make_fov_masks.py \\
    --cbct-dir data/work/rigid/CBCT \\
    --out-dir  data/work/rigid/fovMasks
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from loguru import logger
from scipy import ndimage
from scipy.stats import mode as scipy_mode
from skimage.morphology import convex_hull_image


def case_id_from_path(p: Path) -> str:
    name = p.name
    if name.endswith("_0000.nii.gz"):
        return name[: -len("_0000.nii.gz")]
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    return p.stem


def largest_cc(mask: np.ndarray) -> np.ndarray:
    lab, n = ndimage.label(mask)
    if n == 0:
        return mask
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    return lab == int(counts.argmax())


def estimate_padding(arr: np.ndarray, border: int = 2) -> float:
    """Padding HU ≈ mode of a thin border frame (objective, CBCT-only)."""
    z, y, x = arr.shape
    b = max(1, border)
    parts = [
        arr[:b],
        arr[-b:],
        arr[:, :b],
        arr[:, -b:],
        arr[:, :, :b],
        arr[:, :, -b:],
    ]
    border_vals = np.concatenate([p.ravel() for p in parts])
    # round to 1 HU for stable mode
    rounded = np.rint(border_vals).astype(np.int32)
    m = scipy_mode(rounded, keepdims=True)
    return float(m.mode[0])


def fov_mask_from_cbct(img: sitk.Image, eps: float = 1.0, dilate_mm: float = 0.0) -> sitk.Image:
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    sp = img.GetSpacing()  # x,y,z
    spacing_zyx = (float(sp[2]), float(sp[1]), float(sp[0]))

    pad = estimate_padding(arr, border=2)
    support = np.abs(arr - pad) > eps

    # In-plane (2D) closing only. 3D closing erodes away the first/last slices
    # because the structuring element has no neighbors outside the volume.
    it_xy = max(1, int(round(2.0 / min(spacing_zyx[1], spacing_zyx[2]))))
    se2 = ndimage.generate_binary_structure(2, 1)
    closed = np.zeros_like(support, dtype=bool)
    for z in range(support.shape[0]):
        if support[z].any():
            closed[z] = ndimage.binary_closing(support[z], structure=se2, iterations=it_xy)
    support = largest_cc(closed)

    # Axial seal: include interior of the FOV disk (lung air == padding HU)
    out = np.zeros_like(support, dtype=bool)
    for z in range(support.shape[0]):
        sl = support[z]
        if not sl.any():
            continue
        sl = ndimage.binary_fill_holes(sl)
        out[z] = convex_hull_image(sl)
    out = largest_cc(out)

    if dilate_mm > 0:
        sitk_m = sitk.GetImageFromArray(out.astype(np.uint8))
        sitk_m.CopyInformation(img)
        rad = [max(1, int(round(dilate_mm / s))) for s in spacing_zyx]
        sitk_m = sitk.BinaryDilate(sitk_m, [rad[2], rad[1], rad[0]])
        out = sitk.GetArrayFromImage(sitk_m).astype(bool)
        out = largest_cc(out)

    m = sitk.GetImageFromArray(out.astype(np.uint8))
    m.CopyInformation(img)
    return m


def process_one(args: tuple) -> tuple[str, float, float]:
    case, cbct_path, out_path, eps, dilate_mm = args
    img = sitk.ReadImage(str(cbct_path))
    arr = sitk.GetArrayFromImage(img)
    pad = estimate_padding(arr.astype(np.float32))
    mask = fov_mask_from_cbct(img, eps=eps, dilate_mm=dilate_mm)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(mask, str(out_path), useCompression=True)
    frac = float(sitk.GetArrayFromImage(mask).mean())
    return case, frac, pad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cbct-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--eps", type=float, default=1.0, help="|HU - padding| threshold")
    ap.add_argument("--dilate-mm", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    files = [p for p in args.cbct_dir.iterdir() if p.is_file()]
    paths = sorted(p for p in files if p.name.endswith("_0000.nii.gz"))
    if not paths:
        paths = sorted(
            p for p in files if p.name.endswith(".nii.gz") or p.name.endswith(".nrrd")
        )
    if not paths:
        raise SystemExit(f"No CBCT in {args.cbct_dir} ({len(files)} file(s) there)")

    jobs = [
        (case_id_from_path(p), str(p), str(args.out_dir / f"{case_id_from_path(p)}.nii.gz"), args.eps, args.dilate_mm)
        for p in paths
    ]
    logger.info(f"Building {len(jobs)} FOV masks -> {args.out_dir} (eps={args.eps}, dilate={args.dilate_mm}mm)")
    fracs, pads = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(process_one, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            case, frac, pad = fut.result()
            fracs.append(frac)
            pads.append(pad)
            logger.info(f"[{i}/{len(jobs)}] {case}: FOV={100*frac:.1f}%  padHU={pad:.0f}")
    logger.info(f"Done. mean FOV {100*np.mean(fracs):.1f}%, padHU median {np.median(pads):.0f}")
    from viz_fov import save_fov_vis

    save_fov_vis(args.cbct_dir, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
