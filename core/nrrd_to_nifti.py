#!/usr/bin/env python3
"""Convert rigid-aligned NRRD pairs to NIfTI (no resampling).

Expects ``<src>/{CBCT,CT}/*.nrrd`` and writes ``<dst>/{CBCT,CT}/*.nii.gz``.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import SimpleITK as sitk
from loguru import logger


def convert_one(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return f"skip {dst.name}"
    sitk.WriteImage(sitk.ReadImage(str(src)), str(dst), useCompression=True)
    return f"ok   {dst}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, required=True, help="Input root with CBCT/ and CT/ NRRD")
    ap.add_argument("--dst", type=Path, required=True, help="Output root for CBCT/ and CT/ NIfTI")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    src, dst = args.src.resolve(), args.dst.resolve()

    jobs: list[tuple[Path, Path]] = []
    for modality in ("CBCT", "CT"):
        src_dir = src / modality
        if not src_dir.is_dir():
            raise SystemExit(f"Missing {src_dir}")
        for p in sorted(src_dir.glob("*.nrrd")):
            jobs.append((p, dst / modality / f"{p.stem}.nii.gz"))
    if not jobs:
        raise SystemExit(f"No .nrrd under {src}")

    logger.info(f"{src} -> {dst} ({len(jobs)} files)")
    ok = skip = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(convert_one, s, d) for s, d in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            msg = fut.result()
            if msg.startswith("skip"):
                skip += 1
            else:
                ok += 1
            if i % 20 == 0 or i == len(jobs):
                logger.info(f"[{i}/{len(jobs)}] {msg}")
    logger.info(f"Done. wrote={ok} skipped={skip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
