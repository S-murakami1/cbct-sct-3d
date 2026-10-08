#!/usr/bin/env python3
"""Paired CBCT/CT nnUNet raw. Labels are the geometric CBCT FOV.

HU clip is --hu-clip, the same range as CTNormalization_clip. Channel name is CT_clip.
Does not rewrite nnunet_paths.sh.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from loguru import logger

NORM_CHANNEL = "CT_clip"


def clip_hu(img: sitk.Image, lo: float, hi: float) -> sitk.Image:
    arr = np.clip(sitk.GetArrayFromImage(img).astype(np.float32), lo, hi)
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(img)
    return out


def write_fov_label(fov_src: Path, ref: sitk.Image, dst: Path) -> None:
    fov = sitk.ReadImage(str(fov_src))
    if fov.GetSize() != ref.GetSize():
        raise RuntimeError(f"{fov_src.name}: size {fov.GetSize()} vs volume {ref.GetSize()}")
    arr = (sitk.GetArrayFromImage(fov) > 0).astype(np.uint8)
    mask = sitk.GetImageFromArray(arr)
    mask.CopyInformation(ref)
    sitk.WriteImage(mask, str(dst), useCompression=True)


def write_case(args: tuple) -> str:
    case, cbct_src, ct_src, fov_src, cbct_img_dst, ct_img_dst, cbct_lbl_dst, ct_lbl_dst, lo, hi = args
    cbct = sitk.ReadImage(str(cbct_src))
    ct = sitk.ReadImage(str(ct_src))
    if cbct.GetSize() != ct.GetSize():
        raise RuntimeError(f"{case}: CBCT/CT size mismatch {cbct.GetSize()} vs {ct.GetSize()}")
    cbct = clip_hu(cbct, lo, hi)
    ct = clip_hu(ct, lo, hi)
    for dst in (cbct_img_dst, ct_img_dst, cbct_lbl_dst, ct_lbl_dst):
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(cbct, str(cbct_img_dst), useCompression=True)
    sitk.WriteImage(ct, str(ct_img_dst), useCompression=True)
    write_fov_label(Path(fov_src), cbct, Path(cbct_lbl_dst))
    write_fov_label(Path(fov_src), cbct, Path(ct_lbl_dst))
    return case


def dataset_json(path: Path, n: int) -> None:
    path.write_text(
        json.dumps(
            {
                "labels": {"background": 0, "foreground": 1},
                "channel_names": {"0": NORM_CHANNEL},
                "numTraining": n,
                "file_ending": ".nii.gz",
            },
            indent=2,
        )
        + "\n"
    )


def patient_id(case: str) -> str:
    return case.split("_")[0]


def qc_bad_cases(csv_path: Path | None) -> set[str]:
    if csv_path is None or not csv_path.is_file():
        return set()
    bad: set[str] = set()
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if (row.get("registration") or "").strip():
                case = (row.get("case") or "").strip()
                if case:
                    bad.add(case)
    return bad


def make_patient_splits(cases: list[str], n_folds: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    patients = sorted({patient_id(c) for c in cases})
    rng.shuffle(patients)
    folds = [patients[i::n_folds] for i in range(n_folds)]
    case_by_pat: dict[str, list[str]] = {}
    for c in cases:
        case_by_pat.setdefault(patient_id(c), []).append(c)
    splits = []
    for i in range(n_folds):
        val_pats = set(folds[i])
        train_pats = set(patients) - val_pats
        splits.append(
            {
                "train": sorted(c for p in train_pats for c in case_by_pat[p]),
                "val": sorted(c for p in val_pats for c in case_by_pat[p]),
            }
        )
    return splits


def filter_splits(src: Path, train_cases: set[str]) -> list[dict]:
    payload = json.loads(src.read_text())
    if isinstance(payload, dict):
        payload = [payload]
    out = []
    for fold in payload:
        out.append(
            {
                "train": [c for c in fold["train"] if c in train_cases],
                "val": [c for c in fold["val"] if c in train_cases],
            }
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, required=True, help="deformable CBCT/CT root")
    ap.add_argument("--fov-mask-dir", type=Path, required=True, help="geometric CBCT FOV from make_fov_masks.py")
    ap.add_argument("--raw-root", type=Path, required=True)
    ap.add_argument("--dataset-name", required=True)
    ap.add_argument("--dataset-id", type=int, required=True, help="CBCT id; CT uses id+1")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--exclude-csv", type=Path, default=None)
    ap.add_argument("--test-cases", default="", help="comma-separated case ids for imagesTs; empty = all imagesTr")
    ap.add_argument("--splits-json", type=Path, default=None)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--cases", default="", help="comma-separated case ids; empty = all")
    ap.add_argument("--hu-clip", type=float, nargs=2, required=True, metavar=("LOW", "HIGH"))
    args = ap.parse_args()
    lo, hi = args.hu_clip
    if lo >= hi:
        raise SystemExit(f"hu_clip low must be below high: {lo}, {hi}")

    src_cbct = args.src / "CBCT"
    src_ct = args.src / "CT"
    id_in, id_out = args.dataset_id, args.dataset_id + 1
    name_in = f"Dataset{id_in:03d}_{args.dataset_name}_CBCT"
    name_out = f"Dataset{id_out:03d}_{args.dataset_name}_CT"
    ds_in = args.raw_root / name_in
    ds_out = args.raw_root / name_out

    for ds in (ds_in, ds_out):
        if ds.exists():
            if not args.overwrite:
                raise SystemExit(f"{ds} exists. Set overwrite: true to replace.")
            shutil.rmtree(ds)
        for sub in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
            (ds / sub).mkdir(parents=True)

    cases = sorted(p.name.replace(".nii.gz", "") for p in src_cbct.glob("*.nii.gz"))
    if args.cases.strip():
        wanted = {c.strip() for c in args.cases.split(",") if c.strip()}
        cases = [c for c in cases if c in wanted]
    bad = qc_bad_cases(args.exclude_csv)
    if bad:
        before = len(cases)
        cases = [c for c in cases if c not in bad]
        logger.info(f"registration QC: skip {before - len(cases)}")
    missing = [c for c in cases if not (src_ct / f"{c}.nii.gz").exists()]
    if missing:
        raise SystemExit(f"Missing CT for: {missing[:10]}")
    missing_fov = [c for c in cases if not (args.fov_mask_dir / f"{c}.nii.gz").is_file()]
    if missing_fov:
        raise SystemExit(f"Missing geometric FOV for: {missing_fov[:10]}")
    if not cases:
        raise SystemExit("no cases to write")

    held = {c.strip() for c in args.test_cases.split(",") if c.strip()}
    unknown = sorted(held - set(cases))
    if unknown:
        raise SystemExit(f"test_cases not in the dataset: {unknown}")
    test_cases = [c for c in cases if c in held]
    train_cases = [c for c in cases if c not in held]

    jobs = []
    for case in cases:
        split = "Ts" if case in test_cases else "Tr"
        jobs.append(
            (
                case,
                src_cbct / f"{case}.nii.gz",
                src_ct / f"{case}.nii.gz",
                args.fov_mask_dir / f"{case}.nii.gz",
                ds_in / f"images{split}" / f"{case}_0000.nii.gz",
                ds_out / f"images{split}" / f"{case}_0000.nii.gz",
                ds_in / f"labels{split}" / f"{case}.nii.gz",
                ds_out / f"labels{split}" / f"{case}.nii.gz",
                lo,
                hi,
            )
        )

    logger.info(f"Building {name_in} + {name_out} ({len(train_cases)} train, {len(test_cases)} test)")
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(write_case, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            fut.result()
            if i % 20 == 0 or i == len(jobs):
                logger.info(f"[{i}/{len(jobs)}]")

    dataset_json(ds_in / "dataset.json", len(train_cases))
    dataset_json(ds_out / "dataset.json", len(train_cases))

    if args.splits_json and args.splits_json.is_file():
        splits = filter_splits(args.splits_json, set(train_cases))
    else:
        splits = make_patient_splits(train_cases, n_folds=5, seed=args.split_seed)
    splits_path = args.raw_root / f"Dataset{id_in:03d}_{args.dataset_name}_splits_final.json"
    splits_path.write_text(json.dumps(splits, indent=2) + "\n")
    logger.info(f"splits: {splits_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
