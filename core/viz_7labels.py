"""CT with the 7-class pseudo label on axial, coronal, and sagittal slices.

Same two-column layout as the experiment 7-label overlay. Writes PNGs under the label directory.
"""
import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from matplotlib.colors import ListedColormap

from loguru import logger

NAMES = {
    0: "background",
    1: "organs",
    2: "cardiac",
    3: "muscles",
    4: "bones",
    5: "ribs",
    6: "vertebrae",
}
COLORS = [
    (0.00, 0.00, 0.00),
    (0.90, 0.62, 0.00),
    (0.84, 0.37, 0.00),
    (0.80, 0.47, 0.65),
    (0.94, 0.89, 0.26),
    (0.34, 0.71, 0.91),
    (0.00, 0.62, 0.45),
]
CMAP = ListedColormap(COLORS)
HU_WIN = (-200.0, 1000.0)


def _load(path: Path) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32)


def _slice(vol: np.ndarray, z: int, y: int, x: int, plane: str) -> np.ndarray:
    if plane == "axial":
        return np.rot90(vol[z], 2)
    if plane == "coronal":
        return vol[:, y, :]
    return vol[:, :, x]


def _one(ct_path: Path, seg_path: Path, out_path: Path) -> None:
    ct = np.clip(_load(ct_path), *HU_WIN)
    seg = np.rint(_load(seg_path)).astype(np.int32)
    if ct.shape != seg.shape:
        raise SystemExit(f"shape mismatch {ct_path.name} {ct.shape} vs {seg.shape}")
    z, y, x = (s // 2 for s in seg.shape)
    present = [NAMES[i] for i in sorted(int(u) for u in np.unique(seg) if u in NAMES)]
    fig, axes = plt.subplots(3, 2, figsize=(8, 10))
    for i, plane in enumerate(("axial", "coronal", "sagittal")):
        img = _slice(ct, z, y, x, plane)
        mask = _slice(seg, z, y, x, plane)
        axes[i, 0].imshow(img, cmap="gray", origin="lower", aspect="auto")
        axes[i, 1].imshow(img, cmap="gray", origin="lower", aspect="auto")
        axes[i, 1].imshow(
            np.ma.masked_where(mask == 0, mask),
            cmap=CMAP,
            origin="lower",
            aspect="auto",
            alpha=0.55,
            vmin=0,
            vmax=6,
        )
        axes[i, 0].set_ylabel(plane, fontsize=10)
        if i == 0:
            axes[i, 0].set_title("CT")
            axes[i, 1].set_title("overlay")
        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(f"{seg_path.name}  labels: {present}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_7label_vis(ct_dir: Path, label_dir: Path) -> Path:
    vis = label_dir / "vis"
    vis.mkdir(parents=True, exist_ok=True)
    saved = []
    for seg_path in sorted(label_dir.glob("*.nii.gz")):
        case = seg_path.name[: -len(".nii.gz")]
        ct_path = ct_dir / f"{case}_0000.nii.gz"
        if not ct_path.is_file():
            ct_path = ct_dir / f"{case}.nii.gz"
        if not ct_path.is_file():
            logger.warning("skip 7-label vis {}: missing CT", case)
            continue
        _one(ct_path, seg_path, vis / f"{case}.png")
        saved.append(case)
    body = "\n".join(
        f'<div style="margin:12px 0"><h3>{c}</h3><img src="{c}.png" style="max-width:100%"/></div>' for c in saved
    )
    (vis / "index.html").write_text(
        "<html><body style='font-family:sans-serif;max-width:1100px;margin:auto'>"
        f"<h1>7-class labels (n={len(saved)})</h1>"
        "<p>organs orange, cardiac vermillion, muscles pink, bones yellow, ribs sky, vertebrae green.</p>"
        f"{body}</body></html>\n"
    )
    logger.info("saved 7-label vis: {} ({} cases)", vis, len(saved))
    return vis
