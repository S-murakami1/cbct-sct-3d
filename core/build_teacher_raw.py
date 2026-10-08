#!/usr/bin/env python3
"""Symlink deformed CT and 7-class labels into an nnUNet segmentation raw dataset."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from loguru import logger

LABELS = {
    "background": 0,
    "organs": 1,
    "cardiac": 2,
    "muscles": 3,
    "bones": 4,
    "ribs": 5,
    "vertebrae": 6,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ct-dir", type=Path, required=True, help="imagesTr (*_0000.nii.gz)")
    ap.add_argument("--seg-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.out.exists():
        if not args.overwrite:
            raise SystemExit(f"{args.out} exists. Set overwrite: true to replace.")
        shutil.rmtree(args.out)
    (args.out / "imagesTr").mkdir(parents=True)
    (args.out / "labelsTr").mkdir(parents=True)

    cts = sorted(args.ct_dir.glob("*_0000.nii.gz"))
    if not cts:
        raise SystemExit(f"no CT in {args.ct_dir}")
    for ct in cts:
        case = ct.name[: -len("_0000.nii.gz")]
        seg = args.seg_dir / f"{case}.nii.gz"
        if not seg.is_file():
            raise SystemExit(f"missing 7-class label: {seg}")
        (args.out / "imagesTr" / ct.name).symlink_to(ct.resolve())
        (args.out / "labelsTr" / f"{case}.nii.gz").symlink_to(seg.resolve())

    payload = {
        "channel_names": {"0": "CT_clip"},
        "labels": LABELS,
        "numTraining": len(cts),
        "file_ending": ".nii.gz",
        "description": "Deformed CT (CT_clip) + AFP 7-class TotalSeg labels",
    }
    (args.out / "dataset.json").write_text(json.dumps(payload, indent=2) + "\n")
    logger.info(f"wrote {args.out} n={len(cts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
