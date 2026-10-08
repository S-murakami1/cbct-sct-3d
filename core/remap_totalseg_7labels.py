"""Remap TotalSegmentator `total` NIfTIs to 7 AFP labels. Original files are not modified."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from loguru import logger

from totalseg_7labels import NEW_LABELS, correspondence_payload, old_to_new_lut


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True, help="directory of 117-class segs (left unchanged)")
    p.add_argument("--output", type=Path, default=None, help="new directory for 7-class segs")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def default_output_dir(in_dir: Path) -> Path:
    return in_dir.parent / f"{in_dir.name}_7labels"


def list_segs(in_dir: Path) -> list[Path]:
    return sorted(p for p in in_dir.glob("*.nii.gz") if p.is_file())


def write_tables(out_dir: Path) -> None:
    payload = correspondence_payload()
    (out_dir / "correspondence.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (out_dir / "correspondence.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["old_id", "old_name", "new_id", "new_name"])
        w.writeheader()
        w.writerows(payload["old_to_new"])
    labels = {name: int(i) for i, name in NEW_LABELS.items()}
    (out_dir / "labels.json").write_text(json.dumps(labels, indent=2) + "\n")


def remap_one(src: Path, dst: Path, lut: np.ndarray) -> None:
    img = nib.load(src)
    data = np.asanyarray(img.dataobj)
    if data.max() >= lut.size:
        raise SystemExit(f"{src.name}: label {int(data.max())} exceeds LUT")
    out = lut[data.astype(np.int32)]
    nii = nib.Nifti1Image(out.astype(np.uint8, copy=False), img.affine, img.header)
    nii.set_data_dtype(np.uint8)
    try:
        nii.header.extensions.clear()
    except Exception:
        pass
    nib.save(nii, dst)


def main() -> None:
    args = parse_args()
    in_dir = args.input.resolve()
    out_dir = (args.output or default_output_dir(in_dir)).resolve()
    if out_dir == in_dir:
        raise SystemExit("output directory must differ from input (old labels are preserved)")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_tables(out_dir)

    lut = np.asarray(old_to_new_lut(), dtype=np.uint8)
    cases = list_segs(in_dir)
    if not cases:
        raise SystemExit(f"no .nii.gz in {in_dir}")

    logger.info(f"input (kept) : {in_dir}")
    logger.info(f"output (new) : {out_dir}")
    logger.info(f"n cases      : {len(cases)}")

    for i, src in enumerate(cases, 1):
        dst = out_dir / src.name
        if dst.exists() and not args.overwrite:
            logger.info(f"[{i}/{len(cases)}] skip  {dst.name}")
            continue
        logger.info(f"[{i}/{len(cases)}] remap {src.name}")
        remap_one(src, dst, lut)

    logger.info(f"done {out_dir}")


if __name__ == "__main__":
    main()
