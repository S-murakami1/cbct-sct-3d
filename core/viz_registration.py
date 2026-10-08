"""Registration overview: CBCT+FOV | rigid CT | deformed CT | overlays.

Same columns as core/viz_registration.py. Writes PNGs under the deformable directory.
"""
import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from loguru import logger

HU_WIN = (-200, 1000)
COL_TITLES = ["Input + Mask", "CT", "CT def", "Overlay", "Overlay def"]
PLANES = ["axial", "coronal", "sagittal"]


def _load(path: Path) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32)


def _mid(vol: np.ndarray, axial_k: int) -> dict[str, np.ndarray]:
    z, y, x = vol.shape
    return {
        "axial": np.rot90(vol[z // 2], axial_k),
        "coronal": vol[:, y // 2, :],
        "sagittal": vol[:, :, x // 2],
    }


def _limits(vol: np.ndarray) -> tuple[float, float]:
    lo, hi = float(np.percentile(vol, 0.1)), float(np.percentile(vol, 99.9))
    return max(lo, HU_WIN[0]), min(hi, HU_WIN[1] + 500)


def _label(ax, text: str, case: str) -> None:
    props = dict(facecolor="white", alpha=0.9, edgecolor="white", boxstyle="round,pad=0.35")
    ax.text(0.04, 0.96, text, transform=ax.transAxes, fontsize=9, va="top", bbox=props)
    ax.text(0.96, 0.96, case, transform=ax.transAxes, fontsize=8, va="top", ha="right", bbox=props)


def _plot(case: str, cbct, ct, ctdef, fov, axial_k: int, out: Path) -> None:
    vin_lo, vin_hi = _limits(cbct)
    vct_lo, vct_hi = _limits(ct)
    planes_cbct, planes_ct, planes_def = _mid(cbct, axial_k), _mid(ct, axial_k), _mid(ctdef, axial_k)
    planes_fov = _mid(fov, axial_k) if fov is not None else None
    fig, axes = plt.subplots(3, 5, figsize=(16.0, 9.5))
    fig.suptitle(case, fontsize=13)
    for r, plane in enumerate(PLANES):
        sl_in, sl_ct, sl_def = planes_cbct[plane], planes_ct[plane], planes_def[plane]
        sl_fov = planes_fov[plane] if planes_fov is not None else None
        axes[r, 0].imshow(sl_in, cmap="gray", vmin=vin_lo, vmax=vin_hi, origin="lower", aspect="auto")
        if sl_fov is not None and sl_fov.max() > 0:
            axes[r, 0].contour(sl_fov, levels=[0.5], colors="r", linewidths=1.0, origin="lower")
        axes[r, 1].imshow(sl_ct, cmap="gray", vmin=vct_lo, vmax=vct_hi, origin="lower", aspect="auto")
        axes[r, 2].imshow(sl_def, cmap="gray", vmin=vct_lo, vmax=vct_hi, origin="lower", aspect="auto")
        axes[r, 3].imshow(sl_in, cmap="Reds", alpha=0.5, vmin=vin_lo, vmax=vin_hi, origin="lower", aspect="auto")
        axes[r, 3].imshow(sl_ct, cmap="Blues", alpha=0.5, vmin=vct_lo, vmax=vct_hi, origin="lower", aspect="auto")
        axes[r, 4].imshow(sl_in, cmap="Reds", alpha=0.5, vmin=vin_lo, vmax=vin_hi, origin="lower", aspect="auto")
        axes[r, 4].imshow(sl_def, cmap="Blues", alpha=0.5, vmin=vct_lo, vmax=vct_hi, origin="lower", aspect="auto")
        for c in range(5):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            _label(axes[r, c], COL_TITLES[c], case)
        axes[r, 0].set_ylabel(plane, fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def save_registration_vis(rigid: Path, deformed: Path, fov_dir: Path | None, axial_k: int = 2) -> Path:
    vis = deformed / "vis"
    vis.mkdir(parents=True, exist_ok=True)
    cases = sorted(p.name[: -len(".nii.gz")] for p in (deformed / "CBCT").glob("*.nii.gz"))
    saved = []
    for case in cases:
        p_cbct = deformed / "CBCT" / f"{case}.nii.gz"
        p_rigid = rigid / "CT" / f"{case}.nii.gz"
        p_def = deformed / "CT" / f"{case}.nii.gz"
        if not all(p.is_file() for p in (p_cbct, p_rigid, p_def)):
            logger.warning("skip registration vis {}: missing volume", case)
            continue
        cbct, ct, ctdef = _load(p_cbct), _load(p_rigid), _load(p_def)
        if not (cbct.shape == ct.shape == ctdef.shape):
            logger.warning("skip registration vis {}: shape mismatch", case)
            continue
        fov = None
        fov_path = None if fov_dir is None else fov_dir / f"{case}.nii.gz"
        if fov_path is not None and fov_path.is_file():
            fov = _load(fov_path)
            if fov.shape != cbct.shape:
                ref = sitk.ReadImage(str(p_cbct))
                fimg = sitk.Resample(sitk.ReadImage(str(fov_path)), ref, sitk.Transform(), sitk.sitkNearestNeighbor, 0.0)
                fov = sitk.GetArrayFromImage(fimg).astype(np.float32)
        out = vis / f"{case}.png"
        _plot(case, cbct, ct, ctdef, fov, axial_k, out)
        saved.append(case)
    body = "\n".join(
        f'<div style="margin:12px 0"><h3>{c}</h3><img src="{c}.png" style="max-width:100%"/></div>' for c in saved
    )
    (vis / "index.html").write_text(
        "<html><body style='font-family:sans-serif;max-width:1400px;margin:auto'>"
        f"<h1>Registration (n={len(saved)})</h1>"
        "<p>Input+Mask | CT | CT def | Overlay | Overlay def. Overlay: CBCT=Reds, CT=Blues.</p>"
        f"{body}</body></html>\n"
    )
    logger.info("saved registration vis: {} ({} cases)", vis, len(saved))
    return vis
