"""CBCT with the geometric FOV mask on axial, coronal, and sagittal slices.

Same layout as the experiment three-plane mask figure. Writes PNGs under the mask directory.
"""
import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from loguru import logger
from scipy import ndimage


def _load(path: Path) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32)


def _show(ax, img2d, m2d, lo, hi, title: str) -> None:
    ax.imshow(img2d, cmap="gray", vmin=lo, vmax=hi, origin="lower", aspect="auto")
    if m2d.any():
        tint = np.zeros((*m2d.shape, 4), dtype=np.float32)
        tint[m2d > 0] = (1.0, 0.15, 0.05, 0.25)
        ax.imshow(tint, origin="lower", aspect="auto")
        edge = m2d.astype(bool) & ~ndimage.binary_erosion(m2d.astype(bool), iterations=1)
        er = np.zeros((*m2d.shape, 4), dtype=np.float32)
        er[edge] = (1.0, 0.9, 0.1, 1.0)
        ax.imshow(er, origin="lower", aspect="auto")
    ax.set_title(title, fontsize=11)
    ax.axis("off")


def _one(cbct_path: Path, mask_path: Path, out_path: Path, axial_k: int) -> None:
    cbct = _load(cbct_path)
    mask = (_load(mask_path) > 0).astype(np.uint8)
    if cbct.shape != mask.shape:
        raise SystemExit(f"shape mismatch {cbct_path.name} {cbct.shape} vs {mask.shape}")
    vals = cbct[mask > 0] if mask.any() else cbct.ravel()
    lo, hi = np.percentile(vals, [2, 98])
    if hi <= lo:
        lo, hi = float(cbct.min()), float(cbct.max() + 1)
    if mask.any():
        zz, yy, xx = np.where(mask)
        z, y, x = int(np.median(zz)), int(np.median(yy)), int(np.median(xx))
    else:
        z, y, x = (s // 2 for s in cbct.shape)
    extent_y = int(np.any(mask[:, y, :], axis=0).sum())
    extent_x = int(np.any(mask[:, :, x], axis=0).sum())
    if extent_x >= extent_y:
        cor_img, cor_m, cor_t = cbct[:, :, x], mask[:, :, x], f"Coronal  (x={x})"
        sag_img, sag_m, sag_t = cbct[:, y, :], mask[:, y, :], f"Sagittal  (y={y})"
    else:
        cor_img, cor_m, cor_t = cbct[:, y, :], mask[:, y, :], f"Coronal  (y={y})"
        sag_img, sag_m, sag_t = cbct[:, :, x], mask[:, :, x], f"Sagittal  (x={x})"
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), constrained_layout=True)
    _show(axes[0], np.rot90(cbct[z], axial_k), np.rot90(mask[z], axial_k), lo, hi, f"Axial  (z={z})")
    _show(axes[1], cor_img, cor_m, lo, hi, cor_t)
    _show(axes[2], sag_img, sag_m, lo, hi, sag_t)
    fig.suptitle(f"{cbct_path.name} | yellow edge = M_FOV on CBCT", fontsize=12)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_fov_vis(cbct_dir: Path, mask_dir: Path, axial_k: int = 2) -> Path:
    vis = mask_dir / "vis"
    vis.mkdir(parents=True, exist_ok=True)
    masks = sorted(mask_dir.glob("*.nii.gz"))
    saved = []
    for mask_path in masks:
        case = mask_path.name[: -len(".nii.gz")]
        cbct_path = cbct_dir / f"{case}.nii.gz"
        if not cbct_path.is_file():
            logger.warning("skip FOV vis {}: missing CBCT", case)
            continue
        _one(cbct_path, mask_path, vis / f"{case}.png", axial_k)
        saved.append(case)
    body = "\n".join(
        f'<div style="margin:12px 0"><h3>{c}</h3><img src="{c}.png" style="max-width:100%"/></div>' for c in saved
    )
    (vis / "index.html").write_text(
        "<html><body style='font-family:sans-serif;max-width:1400px;margin:auto'>"
        f"<h1>FOV masks (n={len(saved)})</h1>{body}</body></html>\n"
    )
    logger.info("saved FOV vis: {} ({} cases)", vis, len(saved))
    return vis
