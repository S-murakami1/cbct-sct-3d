#!/usr/bin/env python3
"""SynthRAD2025-style deformable registration: CT (moving) -> CBCT (fixed).

Mirrors https://github.com/SynthRAD2025/preprocessing stage2 deformable:
  - Official param_def_cbct_HN.txt / param_def_cbct_TH.txt
  - Fixed = CBCT, Moving = CT
  - Fixed mask / post-fill mask = geometric CBCT FOV (M_FOV), not intensity body
  - After warp: mask outside FOV to DefaultPixelValue (-1000) on deformed CT and CBCT
    (SynthRAD2025 Task2 fills ct_deformed / ct / cbct outside FOV)

No affine in this stage (SynthRAD does rigid separately; our *_rigid is already aligned).

Pass --fov-mask-dir to use precomputed M_FOV (fovMasks*/labels*). With
--refill-only, reuse existing _elastix/*/result.0.nii.gz and re-apply FOV fill
to both CT and CBCT outputs.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from loguru import logger

CORE = Path(__file__).resolve().parent
REPO = CORE.parent


def _elastix_runtime_dir() -> Path:
    """Use core/bin if present, else fall back to repo tools/bin."""
    for d in (CORE, REPO / "tools"):
        if (d / "bin" / "elastix").exists():
            return d
    return CORE


TOOLS = _elastix_runtime_dir()
ELASTIX = TOOLS / "bin" / "elastix"
CONFIGS = CORE / "synthrad2025_configs"
LIB = TOOLS / "lib"

REGION_PARAM = {
    "HN": CONFIGS / "param_def_cbct_HN.txt",
    "TH": CONFIGS / "param_def_cbct_TH.txt",
    "AB": CONFIGS / "param_def_cbct_AB.txt",
}


def env_with_libs() -> dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = f"{TOOLS / 'bin'}:{env.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = f"{LIB}:{env.get('LD_LIBRARY_PATH', '')}"
    return env


def try_lock(lock_path: Path) -> bool:
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        os.write(fd, f"{os.getpid()} {time.time()}\n".encode())
        os.close(fd)
        return True
    except FileExistsError:
        return False


def find_fov_mask(case: str, fov_mask_dir: Path | None) -> Path | None:
    """Locate geometric M_FOV for a case."""
    names = [f"{case}.nii.gz", f"{case}_0000.nii.gz"]
    if fov_mask_dir is not None:
        for name in names:
            p = fov_mask_dir / name
            if p.exists():
                return p
        for sub in ("fovMasksTr", "fovMasksTs", "labelsTr", "labelsTs", "."):
            d = fov_mask_dir / sub if sub != "." else fov_mask_dir
            if not d.is_dir():
                continue
            for name in names:
                p = d / name
                if p.exists():
                    return p
    return None


def intensity_fov_mask(cbct_path: Path) -> sitk.Image:
    """Legacy fallback: intensity above lower-decile + 50 HU (bad for lung air)."""
    img = sitk.ReadImage(str(cbct_path))
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    thr = float(np.percentile(arr, 10)) + 50.0
    mask_arr = (arr > thr).astype(np.uint8)
    mask = sitk.GetImageFromArray(mask_arr)
    mask.CopyInformation(img)
    return mask


def load_or_build_fov_mask(
    case: str,
    cbct_path: Path,
    mask_path: Path,
    fov_mask_dir: Path | None,
    allow_intensity_fallback: bool,
) -> tuple[sitk.Image, str]:
    src = find_fov_mask(case, fov_mask_dir)
    if src is not None:
        mask = sitk.ReadImage(str(src))
        # binarize
        arr = (sitk.GetArrayFromImage(mask) > 0).astype(np.uint8)
        out = sitk.GetImageFromArray(arr)
        ref = sitk.ReadImage(str(cbct_path))
        if out.GetSize() != ref.GetSize():
            raise RuntimeError(
                f"FOV mask size {out.GetSize()} != CBCT {ref.GetSize()} for {case}"
            )
        out.CopyInformation(ref)
        sitk.WriteImage(out, str(mask_path), useCompression=True)
        return out, f"M_FOV:{src}"
    if not allow_intensity_fallback:
        raise FileNotFoundError(f"No M_FOV for {case} under {fov_mask_dir}")
    mask = intensity_fov_mask(cbct_path)
    sitk.WriteImage(mask, str(mask_path), useCompression=True)
    return mask, "intensity_fallback"


def prepare_param(template: Path, out_path: Path, default_pixel: float) -> None:
    """Ensure WriteResultImage true and DefaultPixelValue set."""
    lines = []
    for line in template.read_text().splitlines():
        s = line.strip()
        if s.startswith("(WriteResultImage"):
            lines.append('(WriteResultImage "true")')
        elif s.startswith("(DefaultPixelValue"):
            lines.append(f"(DefaultPixelValue {default_pixel:.4f})")
        elif s.startswith("(ResultImageFormat"):
            lines.append('(ResultImageFormat "nii.gz")')
        elif s.startswith("(ResultImagePixelType"):
            lines.append('(ResultImagePixelType "float")')
        else:
            lines.append(line)
    text = "\n".join(lines) + "\n"
    if "(WriteResultImage" not in text:
        text += '(WriteResultImage "true")\n'
    out_path.write_text(text)


def mask_outside_fov(ct: sitk.Image, mask: sitk.Image, fill: float) -> sitk.Image:
    mask_u8 = sitk.Cast(mask, sitk.sitkUInt8)
    return sitk.Mask(ct, mask_u8, outsideValue=float(fill))


def write_masked(img: sitk.Image, mask: sitk.Image, fill: float, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    masked = mask_outside_fov(img, mask, fill)
    tmp = out_path.with_name(out_path.name.replace(".nii.gz", ".tmp.nii.gz"))
    sitk.WriteImage(masked, str(tmp), useCompression=True)
    tmp.replace(out_path)


def refill_case(
    case: str,
    fixed: Path,
    out_ct: Path,
    out_cbct: Path,
    work_dir: Path,
    fov_mask_dir: Path | None,
    default_pixel: float,
    allow_intensity_fallback: bool,
) -> tuple[str, bool, str]:
    result = work_dir / "result.0.nii.gz"
    if not result.exists() or result.stat().st_size < 100_000:
        return case, False, "missing result.0.nii.gz"
    try:
        work_dir.mkdir(parents=True, exist_ok=True)
        mask_path = work_dir / "fixed_mask_mfov.nii.gz"
        mask, src = load_or_build_fov_mask(
            case, fixed, mask_path, fov_mask_dir, allow_intensity_fallback
        )
        # SynthRAD Task2: FOV-fill deformed CT and CBCT
        write_masked(sitk.ReadImage(str(result)), mask, default_pixel, out_ct)
        cbct_src = out_cbct if out_cbct.exists() else fixed
        write_masked(sitk.ReadImage(str(cbct_src)), mask, default_pixel, out_cbct)
        return case, True, f"refilled CT+CBCT fill={default_pixel:.1f} via {src}"
    except Exception as e:  # noqa: BLE001
        return case, False, str(e)


def register_case(
    case: str,
    fixed: Path,
    moving: Path,
    out_ct: Path,
    out_cbct: Path,
    work_dir: Path,
    threads: int,
    force: bool,
    region: str,
    default_pixel: float,
    fov_mask_dir: Path | None,
    allow_intensity_fallback: bool,
) -> tuple[str, bool, str]:
    out_cbct.parent.mkdir(parents=True, exist_ok=True)
    out_ct.parent.mkdir(parents=True, exist_ok=True)
    work_dir.parent.mkdir(parents=True, exist_ok=True)

    lock_path = work_dir.parent / f".{case}.lock"
    if not try_lock(lock_path):
        return case, True, "skip locked"

    try:
        mask_path = work_dir / "fixed_mask.nii.gz"
        result = work_dir / "result.0.nii.gz"

        # Reuse finished elastix output if present; always re-apply current FOV policy
        if result.exists() and result.stat().st_size > 100_000 and (not force):
            mask, src = load_or_build_fov_mask(
                case, fixed, mask_path, fov_mask_dir, allow_intensity_fallback
            )
            write_masked(sitk.ReadImage(str(result)), mask, default_pixel, out_ct)
            write_masked(sitk.ReadImage(str(fixed)), mask, default_pixel, out_cbct)
            return case, True, f"ok recovered CT+CBCT region={region} via {src}"

        if (not force) and out_ct.exists() and out_ct.stat().st_size > 1_000_000 and out_cbct.exists():
            return case, True, "skip existing"

        if work_dir.exists():
            shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

        mask, src = load_or_build_fov_mask(
            case, fixed, mask_path, fov_mask_dir, allow_intensity_fallback
        )

        template = REGION_PARAM[region]
        param_path = work_dir / "param_def.txt"
        prepare_param(template, param_path, default_pixel)

        cmd = [
            str(ELASTIX),
            "-f",
            str(fixed),
            "-m",
            str(moving),
            "-fMask",
            str(mask_path),
            "-p",
            str(param_path),
            "-out",
            str(work_dir),
            "-threads",
            str(threads),
        ]
        proc = subprocess.run(
            cmd,
            env=env_with_libs(),
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0 or not result.exists() or result.stat().st_size < 100_000:
            tail = (proc.stdout or "")[-1200:] + "\n" + (proc.stderr or "")[-800:]
            return case, False, f"elastix rc={proc.returncode}\n{tail}"

        # SynthRAD Task2: FOV-fill deformed CT and CBCT to -1000
        write_masked(sitk.ReadImage(str(result)), mask, default_pixel, out_ct)
        write_masked(sitk.ReadImage(str(fixed)), mask, default_pixel, out_cbct)
        return case, True, f"ok region={region} CT+CBCT fill={default_pixel:.1f} via {src}"
    except Exception as e:  # noqa: BLE001
        return case, False, str(e)
    finally:
        try:
            lock_path.unlink(missing_ok=True)
        except OSError:
            pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument(
        "--region",
        choices=sorted(REGION_PARAM),
        required=True,
        help="HN=head, TH=thorax/lung, AB=abdomen (SynthRAD param set)",
    )
    ap.add_argument(
        "--fov-mask-dir",
        type=Path,
        default=None,
        help="Dir or dataset root containing M_FOV (fovMasksTr/Ts or labelsTr/Ts)",
    )
    ap.add_argument(
        "--allow-intensity-fallback",
        action="store_true",
        help="If M_FOV missing, fall back to percentile intensity mask (not recommended)",
    )
    ap.add_argument(
        "--refill-only",
        action="store_true",
        help="Do not run elastix; re-apply FOV fill on existing result.0.nii.gz",
    )
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads-per-job", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cases", default="", help="comma-separated case ids; empty = all")
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--default-pixel",
        type=float,
        default=-1000.0,
        help="Background fill HU (SynthRAD Task2 uses -1000)",
    )
    args = ap.parse_args()

    if not args.refill_only and not ELASTIX.exists():
        logger.error(f"Missing elastix: {ELASTIX}")
        return 1
    if not REGION_PARAM[args.region].exists():
        logger.error(f"Missing param: {REGION_PARAM[args.region]}")
        return 1
    if args.fov_mask_dir is None and not args.allow_intensity_fallback:
        logger.error("Provide --fov-mask-dir (M_FOV) or pass --allow-intensity-fallback")
        return 1

    src_cbct = args.src / "CBCT"
    src_ct = args.src / "CT"
    dst_cbct = args.dst / "CBCT"
    dst_ct = args.dst / "CT"
    work_root = args.dst / "_elastix"
    dst_cbct.mkdir(parents=True, exist_ok=True)
    dst_ct.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)

    for lock in work_root.glob(".*.lock"):
        lock.unlink(missing_ok=True)

    cases = sorted(p.name.replace(".nii.gz", "") for p in src_cbct.glob("*.nii.gz"))
    if args.limit > 0:
        cases = cases[: args.limit]
    if args.cases.strip():
        wanted = {c.strip() for c in args.cases.split(",") if c.strip()}
        cases = [c for c in cases if c in wanted]

    mode = "refill-only" if args.refill_only else "register"
    logger.info(
        f"{args.src} -> {args.dst}: {len(cases)} cases | mode={mode} | "
        f"param_def_cbct_{args.region}.txt | fov={args.fov_mask_dir} | "
        f"workers={args.workers} force={args.force} fill={args.default_pixel}"
    )

    if args.refill_only:
        jobs = []
        for case in cases:
            fixed = src_cbct / f"{case}.nii.gz"
            jobs.append(
                (
                    case,
                    fixed,
                    dst_ct / f"{case}.nii.gz",
                    dst_cbct / f"{case}.nii.gz",
                    work_root / case,
                    args.fov_mask_dir,
                    args.default_pixel,
                    args.allow_intensity_fallback,
                )
            )
        ok = fail = 0
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(refill_case, *j): j[0] for j in jobs}
            for i, fut in enumerate(as_completed(futs), 1):
                case, success, msg = fut.result()
                if success:
                    ok += 1
                else:
                    fail += 1
                    msg = f"FAIL: {msg[:240]}"
                (logger.error if not success else logger.info)("[{}/{}] {}: {}", i, len(jobs), case, msg)
        (logger.error if fail else logger.info)(f"Done: ok={ok} fail={fail}")
        if fail == 0:
            _save_registration_vis(args)
        return 0 if fail == 0 else 2

    jobs = []
    for case in cases:
        fixed = src_cbct / f"{case}.nii.gz"
        moving = src_ct / f"{case}.nii.gz"
        if not moving.exists():
            logger.error(f"MISSING CT {case}")
            continue
        jobs.append(
            (
                case,
                fixed,
                moving,
                dst_ct / f"{case}.nii.gz",
                dst_cbct / f"{case}.nii.gz",
                work_root / case,
                args.threads_per_job,
                args.force,
                args.region,
                args.default_pixel,
                args.fov_mask_dir,
                args.allow_intensity_fallback,
            )
        )

    ok = fail = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(register_case, *j): j[0] for j in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            case, success, msg = fut.result()
            if success:
                ok += 1
            else:
                fail += 1
                msg = f"FAIL: {msg[:240]}"
            (logger.error if not success else logger.info)("[{}/{}] {}: {}", i, len(jobs), case, msg)

    (logger.error if fail else logger.info)(f"Done: ok={ok} fail={fail}")
    if fail == 0:
        _save_registration_vis(args)
    return 0 if fail == 0 else 2


def _save_registration_vis(args: argparse.Namespace) -> None:
    from viz_registration import save_registration_vis

    axial_k = 1 if args.region in {"TH", "AB"} else 2
    save_registration_vis(args.src, args.dst, args.fov_mask_dir, axial_k)


if __name__ == "__main__":
    raise SystemExit(main())
