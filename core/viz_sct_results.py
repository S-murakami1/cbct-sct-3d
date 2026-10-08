#!/usr/bin/env python3
"""Visualize sCT results: CBCT | pred | GT | |pred-GT| | |CBCT-GT| (3 planes).

Example:
  python core/viz_sct_results.py \\
    --pred-dir results/pred_mae_HU \\
    --cbct-dir raw/Dataset093_Head_CBCT/imagesTr \\
    --gt-dir raw/Dataset094_Head_CT/imagesTr \\
    --mask-dir data/work/rigid/fovMasks \\
    --metrics results/pred_mae_HU/metrics_geom.json \\
    --out-dir results/pred_mae_HU/viz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

REPO = Path(__file__).resolve().parents[1]


def load(path: Path) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32)


def find_file(folder: Path, case: str) -> Path | None:
    for name in (f"{case}_0000.nii.gz", f"{case}.nii.gz"):
        p = folder / name
        if p.exists():
            return p
    return None


def center_idx(mask: np.ndarray) -> tuple[int, int, int]:
    if not mask.any():
        z, y, x = mask.shape
        return z // 2, y // 2, x // 2
    zz, yy, xx = np.where(mask)
    return int(np.median(zz)), int(np.median(yy)), int(np.median(xx))


def three_planes(vol: np.ndarray, mask: np.ndarray | None, axial_k: int) -> dict[str, np.ndarray]:
    m = mask if mask is not None else np.ones(vol.shape, dtype=bool)
    z, y, x = center_idx(m.astype(np.uint8))
    return {
        "axial": np.rot90(vol[z], axial_k),
        "coronal": vol[:, y, :],
        "sagittal": vol[:, :, x],
    }


def error_vmax(err: np.ndarray, mask: np.ndarray | None, pct: float = 95.0) -> float:
    vals = err[mask > 0] if mask is not None and mask.any() else err.ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 500.0
    return float(np.ceil(np.percentile(vals, pct) / 10.0) * 10.0)


def viz_case(
    case: str,
    pred_dir: Path,
    cbct_dir: Path,
    gt_dir: Path,
    mask_dir: Path | None,
    out_dir: Path,
    hu_win: tuple[float, float],
    axial_k: int,
    err_vmax: float,
    metrics_row: dict | None,
) -> Path:
    pred_p = find_file(pred_dir, case)
    cbct_p = find_file(cbct_dir, case)
    gt_p = find_file(gt_dir, case)
    if pred_p is None or cbct_p is None or gt_p is None:
        raise FileNotFoundError(f"missing volume for {case}")

    pred = load(pred_p)
    cbct = load(cbct_p)
    gt = load(gt_p)
    mask = None
    if mask_dir is not None:
        mask_p = find_file(mask_dir, case)
        if mask_p is not None:
            mask = load(mask_p) > 0

    err_pred = np.abs(pred - gt)
    err_cbct = np.abs(cbct - gt)

    cols = [
        ("CBCT", cbct, "gray", hu_win[0], hu_win[1]),
        ("sCT (HU)", pred, "gray", hu_win[0], hu_win[1]),
        ("CT GT (HU)", gt, "gray", hu_win[0], hu_win[1]),
        ("|sCT−GT|", err_pred, "hot", 0.0, err_vmax),
        ("|CBCT−GT|", err_cbct, "hot", 0.0, err_vmax),
    ]
    planes = ["axial", "coronal", "sagittal"]

    fig, axes = plt.subplots(len(planes), len(cols), figsize=(3.0 * len(cols), 3.0 * len(planes)))
    stats = []
    if metrics_row:
        p = metrics_row.get("pred_vs_gt", {})
        if "mae" in p:
            stats.append(f"MAE={p['mae']:.1f}")
        if "psnr" in p:
            stats.append(f"PSNR={p['psnr']:.2f}")
        if "ncc" in p:
            stats.append(f"NCC={p['ncc']:.3f}")
    fig.suptitle(f"{case}  |  " + "  ".join(stats), fontsize=11)

    for i, plane in enumerate(planes):
        for j, (title, vol, cmap, vmin, vmax) in enumerate(cols):
            ax = axes[i, j]
            sl = three_planes(vol, mask, axial_k)[plane]
            if mask is not None and title.startswith("|"):
                msl = three_planes(mask.astype(np.float32), mask, axial_k)[plane] > 0
                sl = np.where(msl, sl, np.nan)
            im = ax.imshow(sl, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower", aspect="auto")
            if title.startswith("|") and i == 0:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            if i == 0:
                ax.set_title(title, fontsize=9)
            if j == 0:
                ax.set_ylabel(plane, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{case}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-dir", type=Path, required=True)
    ap.add_argument("--cbct-dir", type=Path, required=True)
    ap.add_argument("--gt-dir", type=Path, required=True)
    ap.add_argument("--mask-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--metrics", type=Path, default=None, help="metrics_geom.json for titles")
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--hu-win", type=float, nargs=2, default=(-200, 1000))
    ap.add_argument("--axial-k", type=int, default=2)
    ap.add_argument("--err-vmax", type=float, default=None, help="Shared error color scale max (HU)")
    ap.add_argument("--title", type=str, default="sCT results")
    args = ap.parse_args()

    preds = sorted(p for p in args.pred_dir.glob("*.nii.gz") if p.name != "dataset.json")
    cases = args.cases or [p.name.replace(".nii.gz", "") for p in preds]
    metrics_by_case: dict[str, dict] = {}
    if args.metrics and args.metrics.is_file():
        js = json.loads(args.metrics.read_text())
        for row in js.get("cases", []):
            if "case" in row:
                metrics_by_case[row["case"]] = row

    err_max = args.err_vmax
    if err_max is None:
        vals = []
        for case in cases:
            row = metrics_by_case.get(case, {})
            for key in ("pred_vs_gt", "cbct_vs_gt"):
                if key in row and "p95_abs" in row[key]:
                    vals.append(float(row[key]["p95_abs"]))
        err_max = float(np.ceil(max(vals) / 10.0) * 10.0) if vals else 500.0
    print(f"Error color scale: 0 – {err_max:.0f} HU")

    cards = []
    for case in cases:
        try:
            out = viz_case(
                case,
                args.pred_dir,
                args.cbct_dir,
                args.gt_dir,
                args.mask_dir,
                args.out_dir,
                tuple(args.hu_win),
                args.axial_k,
                err_max,
                metrics_by_case.get(case),
            )
            print(f"  {out}")
            cards.append(case)
        except FileNotFoundError as e:
            print(f"  skip {case}: {e}")

    html = args.out_dir / "index.html"
    body = "\n".join(
        f'<div style="margin:12px 0"><h3>{c}</h3><img src="{c}.png" style="max-width:100%"/></div>'
        for c in cards
    )
    html.write_text(
        "<html><body style='font-family:sans-serif;max-width:1800px;margin:auto'>"
        f"<h1>{args.title}</h1>"
        "<p>Rows: axial / coronal / sagittal. "
        "Columns: CBCT | sCT | GT | |sCT−GT| | |CBCT−GT|.</p>"
        f"{body}</body></html>\n"
    )
    print(f"Index: {html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
