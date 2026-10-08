#!/usr/bin/env python3
"""Evaluate sCT / CBCT vs GT CT inside a binary mask.

The reported score is the geometric CBCT FOV (M_FOV) and is written to --out
(use metrics_geom.json). metrics_fov.json is the earlier intensity-mask score
and is not replaced.

Example:
  python core/eval_mae_body_mask.py \\
    --pred-dir results/pred_mae_HU \\
    --gt-dir raw/Dataset094_Head_CT/imagesTr \\
    --cbct-dir raw/Dataset093_Head_CBCT/imagesTr \\
    --mask-dir data/work/rigid/fovMasks \\
    --out results/pred_mae_HU/metrics_geom.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def case_id(p: Path) -> str:
    name = p.name
    if name.endswith("_0000.nii.gz"):
        return name[: -len("_0000.nii.gz")]
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    return p.stem


def load(path: Path) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32)


def find(folder: Path, case: str) -> Path:
    candidates = [
        folder / f"{case}_0000.nii.gz",
        folder / f"{case}.nii.gz",
    ]
    # If given .../labelsTs or imagesTs, also try the Tr sibling (and vice versa).
    name = folder.name
    if name.endswith("Ts"):
        alt = folder.parent / (name[:-2] + "Tr")
        candidates += [alt / f"{case}_0000.nii.gz", alt / f"{case}.nii.gz"]
    elif name.endswith("Tr"):
        alt = folder.parent / (name[:-2] + "Ts")
        candidates += [alt / f"{case}_0000.nii.gz", alt / f"{case}.nii.gz"]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"{case} not in {folder}")


def psnr_from_mse(mse: float, data_range: float) -> float:
    if mse <= 0:
        return float("inf")
    return float(10.0 * np.log10((data_range**2) / mse))


def metrics(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, data_range: float) -> dict:
    m = mask.astype(bool)
    if m.sum() < 100:
        return {"n": int(m.sum())}
    d = pred[m] - gt[m]
    mse = float(np.mean(d**2))
    ncc = float(np.corrcoef(pred[m].ravel(), gt[m].ravel())[0, 1])
    return {
        "n": int(m.sum()),
        "mae": float(np.mean(np.abs(d))),
        "psnr": psnr_from_mse(mse, data_range),
        "ncc": ncc,
        "data_range": float(data_range),
    }


def evaluate(
    pred_dir: Path,
    gt_dir: Path,
    mask_dir: Path,
    cbct_dir: Path | None,
    exclude: list[str],
    data_range: float,
    mask_kind: str,
) -> dict:
    preds = sorted(p for p in pred_dir.glob("*.nii.gz") if p.name != "dataset.json")
    rows = []
    print(f"\n[{mask_kind}] {mask_dir}")
    for pred_path in preds:
        case = case_id(pred_path)
        if case in exclude:
            continue
        gt = load(find(gt_dir, case))
        pred = load(pred_path)
        mask = load(find(mask_dir, case)) > 0
        if pred.shape != gt.shape or pred.shape != mask.shape:
            raise SystemExit(f"{case}: shape mismatch pred{pred.shape} gt{gt.shape} mask{mask.shape}")
        row = {"case": case, "pred_vs_gt": metrics(pred, gt, mask, data_range)}
        if cbct_dir:
            cbct = load(find(cbct_dir, case))
            row["cbct_vs_gt"] = metrics(cbct, gt, mask, data_range)
        rows.append(row)
        p = row["pred_vs_gt"]
        extra = ""
        if cbct_dir:
            c = row["cbct_vs_gt"]
            extra = f"  | CBCT MAE={c['mae']:.1f} PSNR={c['psnr']:.2f} NCC={c['ncc']:.3f}"
        print(f"{case}: MAE={p['mae']:.1f} PSNR={p['psnr']:.2f} NCC={p['ncc']:.3f}{extra}")

    def mean_key(pair: tuple[str, str]):
        vals = [r[pair[0]][pair[1]] for r in rows if pair[1] in r[pair[0]] and r[pair[0]][pair[1]] is not None]
        return float(np.mean(vals)) if vals else None

    keys = ("mae", "psnr", "ncc")
    summary = {
        "n": len(rows),
        "mask_kind": mask_kind,
        "mask_dir": str(mask_dir),
        "exclude": exclude,
        "data_range": data_range,
        "notes": {
            "psnr": "10*log10(data_range^2 / MSE) on masked voxels",
            "ncc": "Pearson correlation on masked voxels",
        },
    }
    for k in keys:
        summary[f"pred_{k}_mean"] = mean_key(("pred_vs_gt", k))
        if cbct_dir:
            summary[f"cbct_{k}_mean"] = mean_key(("cbct_vs_gt", k))
    summary["cases"] = rows
    return summary


def write_summary(path: Path, summary: dict) -> None:
    if path.name == "metrics_fov.json":
        raise SystemExit(
            f"Refusing to write {path}. metrics_fov.json is the intensity-mask score and is kept. "
            "Write the geometric score to metrics_geom.json."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"\nSummary [{summary['mask_kind']}] n={summary['n']} (data_range={summary['data_range']:g}): "
        f"MAE={summary['pred_mae_mean']:.2f}  PSNR={summary['pred_psnr_mean']:.2f}  "
        f"NCC={summary['pred_ncc_mean']:.4f}"
        + (
            f"  | CBCT MAE={summary['cbct_mae_mean']:.2f} PSNR={summary['cbct_psnr_mean']:.2f} "
            f"NCC={summary['cbct_ncc_mean']:.4f}"
            if "cbct_mae_mean" in summary
            else ""
        )
    )
    print(f"Wrote {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-dir", type=Path, required=True, help="HU predictions")
    ap.add_argument("--gt-dir", type=Path, required=True)
    ap.add_argument("--mask-dir", type=Path, required=True, help="Geometric CBCT FOV (M_FOV). This is the reported score.")
    ap.add_argument("--cbct-dir", type=Path, default=None, help="Optional CBCT baseline")
    ap.add_argument("--out", type=Path, required=True, help="Geometric score. Use metrics_geom.json.")
    ap.add_argument("--exclude", nargs="*", default=[], help="Case IDs to exclude")
    ap.add_argument(
        "--data-range",
        type=float,
        default=2000.0,
        help="Peak intensity span for PSNR (default 2000 HU, e.g. [-1000,1000])",
    )
    args = ap.parse_args()

    summary = evaluate(
        args.pred_dir, args.gt_dir, args.mask_dir, args.cbct_dir, args.exclude, args.data_range, "geometric"
    )
    write_summary(args.out, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
