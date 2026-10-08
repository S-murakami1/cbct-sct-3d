#!/usr/bin/env python3
"""After plan_and_preprocess + unpack, wire CT targets into CBCT dataset.

Mirrors notebooks/nnUNetv2_translation_tutorial_singlemod.ipynb linking steps.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from loguru import logger


def case_id_from_ct(path: Path) -> str:
    name = path.name
    if name.endswith("_0000.nii.gz"):
        return name[: -len("_0000.nii.gz")]
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    return path.stem


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preprocessed", type=Path, required=True)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--id-in", type=int, required=True)
    ap.add_argument("--id-out", type=int, required=True)
    ap.add_argument("--name-in", type=str, required=True)
    ap.add_argument("--name-out", type=str, required=True)
    ap.add_argument("--config", type=str, default="nnUNetPlans_3d_fullres")
    args = ap.parse_args()

    ds_in = f"Dataset{args.id_in:03d}_{args.name_in}"
    ds_out = f"Dataset{args.id_out:03d}_{args.name_out}"
    pre_in = args.preprocessed / ds_in
    pre_out = args.preprocessed / ds_out
    raw_out_images = args.raw / ds_out / "imagesTr"

    # 1) raw CT (imagesTr) -> CBCT gt_segmentations (match by case id)
    gt_dir = pre_in / "gt_segmentations"
    gt_dir.mkdir(parents=True, exist_ok=True)
    ct_files = sorted(raw_out_images.glob("*_0000.nii.gz"))
    if not ct_files:
        raise SystemExit(f"No CT images in {raw_out_images}")
    keep_cases = {case_id_from_ct(ct) for ct in ct_files}
    for ct in ct_files:
        case = case_id_from_ct(ct)
        dst = gt_dir / f"{case}.nii.gz"
        shutil.copy2(ct, dst)
        logger.info(f"gt: {ct.name} -> {dst.name}")
    for stale in sorted(gt_dir.glob("*.nii.gz")):
        if stale.name[: -len(".nii.gz")] not in keep_cases:
            stale.unlink()
            logger.info(f"gt remove stale: {stale.name}")

    # 2) preprocessed CT .npy -> CBCT *_seg.npy (match by case id)
    # Requires prior unpack (nnUNetv2_unpack / unpack_dataset) so CT .npy exist.
    cfg_in = pre_in / args.config
    cfg_out = pre_out / args.config
    if not cfg_out.is_dir():
        raise SystemExit(f"Missing CT preprocessed dir: {cfg_out}")
    cbct_cases = {
        p.name.replace(".npz", "").replace(".npy", "").replace("_seg", "")
        for p in cfg_in.glob("*")
        if p.suffix in {".npz", ".npy"} and not p.name.endswith("_seg.npy")
    }
    imgs = sorted(p for p in cfg_out.glob("*.npy") if not p.name.endswith("_seg.npy"))
    if not imgs:
        raise SystemExit(
            f"No preprocessed CT npy in {cfg_out} (run unpack first, e.g. nnUNetv2_unpack {args.id_out} 3d_fullres 0)"
        )
    n_linked = 0
    for img in imgs:
        case = img.stem
        if cbct_cases and case not in cbct_cases:
            logger.info(f"seg skip (no CBCT case): {case}")
            continue
        seg = cfg_in / f"{case}_seg.npy"
        shutil.copy2(img, seg)
        logger.info(f"seg: {img.name} -> {seg.name}")
        n_linked += 1
    if n_linked == 0:
        raise SystemExit(f"No CT npy linked into {cfg_in}")

    # 3) optional patient splits
    # Head_CBCT -> Head; SynthRAD_HN_CBCT -> SynthRAD_HN; Lung_CBCT -> Lung
    region = args.name_in.rsplit("_", 1)[0]
    splits_src = args.raw / f"Dataset{args.id_in:03d}_{region}_splits_final.json"
    if splits_src.exists():
        shutil.copy2(splits_src, pre_in / "splits_final.json")
        logger.info(f"copied splits -> {pre_in / 'splits_final.json'}")
    else:
        logger.info(f"no splits at {splits_src} (skipped)")
    logger.info("Linking done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
