#!/usr/bin/env python3
"""TotalSegmentator `total` -> 7 AFP classes, then optional muscle-task union into class 3."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from totalsegmentator.python_api import totalsegmentator
from loguru import logger

LABELS = Path(__file__).resolve().parent
if str(LABELS) not in sys.path:
    sys.path.insert(0, str(LABELS))

from remap_totalseg_7labels import remap_one  # noqa: E402
from totalseg_7labels import old_to_new_lut  # noqa: E402

MUSCLE_ID = 3


def case_id(path: Path) -> str:
    name = path.name
    if name.endswith("_0000.nii.gz"):
        return name[: -len("_0000.nii.gz")]
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    return path.stem


def list_cts(in_dir: Path) -> list[Path]:
    cts = sorted(in_dir.glob("*_0000.nii.gz"))
    return cts if cts else sorted(p for p in in_dir.glob("*.nii.gz") if p.is_file())


def save_nii(data: np.ndarray, ref: nib.Nifti1Image, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    nii = nib.Nifti1Image(data.astype(np.uint8, copy=False), ref.affine, ref.header)
    nii.set_data_dtype(np.uint8)
    try:
        nii.header.extensions.clear()
    except Exception:
        pass
    nib.save(nii, dst)


def load_bool(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    data = np.asanyarray(nib.load(path).dataobj) > 0
    if data.shape != shape:
        raise SystemExit(f"shape mismatch {path.name}: {data.shape} vs {shape}")
    return data


def merge_muscles(labels7: np.ndarray, extra: np.ndarray) -> np.ndarray:
    out = labels7.copy()
    extra = extra > 0
    conflict = extra & (out != 0) & (out != MUSCLE_ID)
    paint = extra & ~conflict
    out[paint] = MUSCLE_ID
    return out


def infer_task(ct: Path, out_path: Path, task: str, device: str, overwrite: bool) -> None:
    if out_path.exists() and not overwrite:
        logger.info(f"skip {task} {out_path.name}")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"infer {task} -> {out_path.name}")
    kwargs = {"ml": True, "device": device, "quiet": True}
    if task != "total":
        kwargs["task"] = task
    totalsegmentator(str(ct), str(out_path), **kwargs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--labels-out", type=Path, required=True)
    ap.add_argument("--device", default="gpu:0")
    ap.add_argument("--muscle-tasks", default="", help="comma-separated TotalSeg tasks unioned into muscles")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    tasks = [t.strip() for t in args.muscle_tasks.split(",") if t.strip()]
    cases = list_cts(args.input)
    if not cases:
        raise SystemExit(f"no CT in {args.input}")
    args.labels_out.mkdir(parents=True, exist_ok=True)
    lut = np.asarray(old_to_new_lut(), dtype=np.uint8)

    logger.info(f"input : {args.input.resolve()}")
    logger.info(f"n     : {len(cases)}  muscle_tasks={tasks or 'none'}")
    for i, ct in enumerate(cases, 1):
        cid = case_id(ct)
        logger.info(f"[{i}/{len(cases)}] {cid}")
        seg117 = args.work / "totalseg117" / f"{cid}.nii.gz"
        labels7 = args.work / "labels7" / f"{cid}.nii.gz"
        infer_task(ct, seg117, "total", args.device, args.overwrite)
        if not labels7.exists() or args.overwrite:
            logger.info(f"remap 7labels -> {labels7.name}")
            labels7.parent.mkdir(parents=True, exist_ok=True)
            remap_one(seg117, labels7, lut)

        ref = nib.load(ct)
        shape = np.asanyarray(ref.dataobj).shape
        lab = np.asanyarray(nib.load(labels7).dataobj).astype(np.uint8)
        if lab.shape != shape:
            raise SystemExit(f"{cid}: labels7 {lab.shape} vs CT {shape}")
        if tasks:
            union = np.zeros(shape, dtype=bool)
            for task in tasks:
                part = args.work / "muscles" / task / f"{cid}.nii.gz"
                infer_task(ct, part, task, args.device, args.overwrite)
                union |= load_bool(part, shape)
            merged = merge_muscles(lab, union)
        else:
            merged = lab
        uniq = sorted(int(v) for v in np.unique(merged))
        if any(v < 0 or v > 6 for v in uniq):
            raise SystemExit(f"{cid}: label values {uniq} outside 0-6")
        n_mus = int((merged == MUSCLE_ID).sum())
        if n_mus == 0:
            raise SystemExit(f"{cid}: muscles (id 3) is empty")
        save_nii(merged, ref, args.labels_out / f"{cid}.nii.gz")
        logger.info(f"labels={uniq} muscles={n_mus}")
    from viz_7labels import save_7label_vis

    save_7label_vis(args.input, args.labels_out)
    logger.info(f"done {args.labels_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
