"""Extract 7-class pseudo labels and build the distillation teacher dataset.

TotalSegmentator runs only when use_totalseg is true. --no-totalseg skips it and
reads labels from pseudo_label_dir, or from work_prefix/totalseg/labels7_flat.
Then writes the teacher raw dataset and preprocesses it for nnUNetTrainerSegScratch.
AFP training itself is nnUNetv2_train with nnUNetTrainerMRCT_AFP_scratch300.
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from loguru import logger

CORE = Path(__file__).resolve().parent
REPO = CORE.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from nnunetv2.training.loss.AFP_from_plans import load_config, resolve_cfg, resolve_optional  # noqa: E402


def nnunet_env(cfg: dict) -> dict[str, str]:
    env = os.environ.copy()
    env["nnUNet_raw"] = str(resolve_cfg(cfg, "nnunet_raw"))
    env["nnUNet_preprocessed"] = str(resolve_cfg(cfg, "nnunet_preprocessed"))
    env["nnUNet_results"] = str(resolve_cfg(cfg, "nnunet_results"))
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def run(cmd: list[str], env: dict[str, str] | None = None, cwd: Path | None = None) -> None:
    logger.info("+ {}", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env, cwd=cwd)


def totalseg_python() -> Path:
    py = REPO / "totalseg" / ".venv" / "bin" / "python"
    if not py.is_file():
        raise SystemExit("TotalSegmentator environment is missing. From the repository root: uv sync --project totalseg")
    return py


def totalseg_env() -> dict[str, str]:
    env = os.environ.copy()
    # Keep this fork's nnunetv2 off the path. TotalSegmentator needs the stock package.
    env.pop("PYTHONPATH", None)
    for key in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
        env.pop(key, None)
    return env


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=REPO / "config.yaml")
    ap.add_argument("--no-totalseg", action="store_true", help="Do not run TotalSegmentator")
    args = ap.parse_args()
    cfg = load_config(args.config)
    use_totalseg = bool(cfg.get("use_totalseg", False)) and not args.no_totalseg
    work = resolve_cfg(cfg, "work_prefix")
    raw = resolve_cfg(cfg, "nnunet_raw")
    name = str(cfg["dataset_name"])
    cbct_id = int(cfg["cbct_dataset_id"])
    ct_id = cbct_id + 1
    teacher_id = int(cfg["teacher_dataset_id"])
    if teacher_id in (cbct_id, ct_id):
        raise SystemExit("teacher_dataset_id must differ from the CBCT id and the CBCT id + 1")
    ct_images = raw / f"Dataset{ct_id:03d}_{name}_CT" / "imagesTr"
    if not ct_images.is_dir():
        raise SystemExit(f"missing {ct_images}. Run core/prepare_training_data.py through pair first.")
    labels_out = work / "totalseg" / "labels7_flat"
    supplied = resolve_optional(cfg, "pseudo_label_dir")
    py = sys.executable
    env = nnunet_env(cfg)
    workers = str(int(cfg.get("workers") or 4))

    if use_totalseg:
        if shutil.which("nvidia-smi") is None:
            raise SystemExit("nvidia-smi not found. Stopped before TotalSegmentator.")
        tasks = list(cfg.get("muscle_tasks") or []) if bool(cfg.get("use_hn_muscles", False)) else []
        if not tasks:
            logger.info("head/neck muscle labels off")
        run(
            [
                str(totalseg_python()), str(CORE / "infer_7labels.py"),
                "--input", str(ct_images),
                "--work", str(work / "totalseg"),
                "--labels-out", str(labels_out),
                "--device", str(cfg.get("totalseg_device") or "gpu:0"),
                "--muscle-tasks", ",".join(tasks),
            ],
            totalseg_env(),
            cwd=REPO / "totalseg",
        )
        seg_dir = labels_out
    else:
        seg_dir = supplied or labels_out
        logger.info("TotalSegmentator off. Labels from {}", seg_dir)
        if not seg_dir.is_dir() or not any(seg_dir.glob("*.nii.gz")):
            raise SystemExit(f"missing pseudo labels in {seg_dir}. Set use_totalseg: true or pseudo_label_dir.")

    teacher_raw = raw / f"Dataset{teacher_id:03d}_{name}_CT_7labels"
    cmd = [
        py, str(CORE / "build_teacher_raw.py"),
        "--ct-dir", str(ct_images),
        "--seg-dir", str(seg_dir),
        "--out", str(teacher_raw),
    ]
    if bool(cfg.get("overwrite")):
        cmd.append("--overwrite")
    run(cmd)
    run(
        [
            "nnUNetv2_plan_and_preprocess", "-d", str(teacher_id),
            "-c", "3d_fullres", "--clean", "-np", workers, "-npfp", workers,
        ],
        env,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
