#!/usr/bin/env python3
"""Denormalize nnUNet_translation predictions back to approximate HU.

Per-case (default): HU ~= pred * std + mean using each GT CT volume.

Dataset-level (--fingerprint): use dataset_fingerprint.json mean/std
(for CT_clip / CTNormalization_clip on all datasets).

Example:
  python core/denorm_predictions_to_hu.py \\
    -i results/pred_mae \\
    -g raw/Dataset094_Head_CT/imagesTr \\
    -o results/pred_mae_HU \\
    --fingerprint preprocessed/Dataset094_Head_CT/dataset_fingerprint.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def case_id_from_pred(path: Path) -> str:
    # case_0000.nii.gz -> case
    name = path.name
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    return path.stem


def find_gt(gt_dir: Path, case: str) -> Path:
    cands = [
        gt_dir / f"{case}_0000.nii.gz",
        gt_dir / f"{case}.nii.gz",
    ]
    for c in cands:
        if c.exists():
            return c
    raise FileNotFoundError(f"GT CT not found for {case} under {gt_dir}")


def zscore_stats(arr: np.ndarray) -> tuple[float, float]:
    arr = arr.astype(np.float64, copy=False)
    mean = float(arr.mean())
    std = float(arr.std())
    return mean, max(std, 1e-8)


def stats_from_fingerprint(path: Path, channel: int = 0) -> tuple[float, float]:
    js = json.loads(path.read_text())
    ch = js["foreground_intensity_properties_per_channel"][str(channel)]
    return float(ch["mean"]), max(float(ch["std"]), 1e-8)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--pred-dir", type=Path, required=True, help="Folder with predicted .nii.gz")
    ap.add_argument(
        "-g",
        "--gt-dir",
        type=Path,
        required=True,
        help="Paired GT CT folder (imagesTs or imagesTr); per-case stats unless --fingerprint",
    )
    ap.add_argument(
        "--fingerprint",
        type=Path,
        default=None,
        help="preprocessed/.../dataset_fingerprint.json (required for CT_clip denorm)",
    )
    ap.add_argument("--fingerprint-channel", type=int, default=0)
    ap.add_argument("-o", "--out-dir", type=Path, required=True, help="Output folder for HU NIfTIs")
    ap.add_argument(
        "--metrics",
        action="store_true",
        help="Also compute MAE/RMSE vs GT in HU (FOV = GT finite voxels)",
    )
    ap.add_argument("--clip", type=float, nargs=2, default=None, metavar=("LO", "HI"), help="Optional HU clip")
    args = ap.parse_args()

    preds = sorted(args.pred_dir.glob("*.nii.gz"))
    preds = [p for p in preds if p.name != "dataset.json"]
    if not preds:
        raise SystemExit(f"No .nii.gz in {args.pred_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    ds_mean, ds_std = (None, None)
    if args.fingerprint is not None:
        ds_mean, ds_std = stats_from_fingerprint(args.fingerprint, args.fingerprint_channel)
        print(f"Dataset-level denorm: mean={ds_mean:.3f} std={ds_std:.3f} from {args.fingerprint}")

    for pred_path in preds:
        case = case_id_from_pred(pred_path)
        gt_path = find_gt(args.gt_dir, case)

        pred_img = sitk.ReadImage(str(pred_path))
        gt_img = sitk.ReadImage(str(gt_path))
        pred = sitk.GetArrayFromImage(pred_img).astype(np.float32)
        gt = sitk.GetArrayFromImage(gt_img).astype(np.float32)

        mean, std = (ds_mean, ds_std) if ds_mean is not None else zscore_stats(gt)
        hu = pred * np.float32(std) + np.float32(mean)
        if args.clip is not None:
            hu = np.clip(hu, args.clip[0], args.clip[1])

        out = sitk.GetImageFromArray(hu)
        out.CopyInformation(pred_img)
        out_path = args.out_dir / f"{case}.nii.gz"
        sitk.WriteImage(out, str(out_path), useCompression=True)

        row = {
            "case": case,
            "mean": mean,
            "std": std,
            "pred_min": float(pred.min()),
            "pred_max": float(pred.max()),
            "hu_min": float(hu.min()),
            "hu_max": float(hu.max()),
        }
        if args.metrics:
            if pred.shape != gt.shape:
                row["mae_hu"] = None
                row["rmse_hu"] = None
                row["note"] = f"shape mismatch pred{pred.shape} gt{gt.shape}"
            else:
                diff = hu - gt
                row["mae_hu"] = float(np.mean(np.abs(diff)))
                row["rmse_hu"] = float(np.sqrt(np.mean(diff ** 2)))
        rows.append(row)
        extra = f"  MAE={row['mae_hu']:.1f}" if row.get("mae_hu") is not None else ""
        print(f"{case}: mu={mean:.1f} std={std:.1f} HU=[{row['hu_min']:.1f},{row['hu_max']:.1f}]{extra}")

    metrics_path = args.out_dir / "denorm_metrics.json"
    summary = {"n": len(rows), "cases": rows}
    if args.metrics:
        maes = [r["mae_hu"] for r in rows if r.get("mae_hu") is not None]
        if maes:
            summary["mae_hu_mean"] = float(np.mean(maes))
            summary["mae_hu_median"] = float(np.median(maes))
            print(f"\nMAE HU mean={summary['mae_hu_mean']:.2f}  median={summary['mae_hu_median']:.2f}")
    metrics_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {len(rows)} volumes -> {args.out_dir}")
    print(f"Metrics JSON -> {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
