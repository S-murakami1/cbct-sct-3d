"""Build nnUNet CBCT/CT datasets, preprocess them, and link CT targets.

Starts from work_prefix/deformable. Steps: pair, preprocess, link.
Pseudo labels and the distillation teacher are core/extract_pseudo_labels.py.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

from loguru import logger

CORE = Path(__file__).resolve().parent
REPO = CORE.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from nnunetv2.training.loss.AFP_from_plans import load_config, resolve_cfg, resolve_optional  # noqa: E402

STEPS = ("pair", "preprocess", "link")


def nnunet_env(cfg: dict) -> dict[str, str]:
    env = os.environ.copy()
    env["nnUNet_raw"] = str(resolve_cfg(cfg, "nnunet_raw"))
    env["nnUNet_preprocessed"] = str(resolve_cfg(cfg, "nnunet_preprocessed"))
    env["nnUNet_results"] = str(resolve_cfg(cfg, "nnunet_results"))
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    logger.info("+ {}", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=REPO / "config.yaml")
    ap.add_argument("--from", dest="start", default="pair", choices=STEPS)
    ap.add_argument("--to", dest="end", default="link", choices=STEPS)
    args = ap.parse_args()
    if STEPS.index(args.end) < STEPS.index(args.start):
        raise SystemExit("--to is before --from")
    steps = list(STEPS[STEPS.index(args.start) : STEPS.index(args.end) + 1])
    cfg = load_config(args.config)
    py = sys.executable
    env = nnunet_env(cfg)
    work = resolve_cfg(cfg, "work_prefix")
    deformed = work / "deformable"
    raw = resolve_cfg(cfg, "nnunet_raw")
    name = str(cfg["dataset_name"])
    cbct_id = int(cfg["cbct_dataset_id"])
    ct_id = cbct_id + 1
    overwrite = bool(cfg.get("overwrite"))
    workers = str(int(cfg.get("workers") or 4))
    cases = cfg.get("cases") or []
    cases_arg = ",".join(str(c) for c in cases) if isinstance(cases, list) else str(cases)
    test_cases = cfg.get("test_cases") or []
    test_arg = ",".join(str(c) for c in test_cases) if isinstance(test_cases, list) else str(test_cases)
    hu_clip = cfg.get("hu_clip")
    if not (isinstance(hu_clip, (list, tuple)) and len(hu_clip) == 2):
        raise SystemExit("config hu_clip must be [low, high]")
    hu_lo, hu_hi = float(hu_clip[0]), float(hu_clip[1])
    if hu_lo >= hu_hi:
        raise SystemExit(f"config hu_clip low must be below high: {hu_lo}, {hu_hi}")

    if "pair" in steps:
        cmd = [
            py, str(CORE / "prepare_pair.py"),
            "--src", str(deformed),
            "--fov-mask-dir", str(work / "rigid" / "fovMasks"),
            "--raw-root", str(raw),
            "--dataset-name", name,
            "--dataset-id", str(cbct_id),
            "--workers", workers,
            "--split-seed", str(int(cfg.get("split_seed") or 42)),
            "--cases", cases_arg,
            "--test-cases", test_arg,
            "--hu-clip", str(hu_lo), str(hu_hi),
        ]
        for flag, key in (
            ("--exclude-csv", "registration_qc_csv"),
            ("--splits-json", "splits_json"),
        ):
            path = resolve_optional(cfg, key)
            if path is not None:
                cmd += [flag, str(path)]
        if overwrite:
            cmd.append("--overwrite")
        run(cmd)

    if "preprocess" in steps:
        for ds in (cbct_id, ct_id):
            run(
                [
                    "nnUNetv2_plan_and_preprocess", "-d", str(ds),
                    "-c", "3d_fullres", "--clean", "-np", workers, "-npfp", workers,
                ],
                env,
            )
        run(
            ["nnUNetv2_plan_experiment", "-d", str(cbct_id), "-c", "3d_fullres", "-pl", "nnUNetPlannerResUNet"],
            env,
        )
        for ds in (cbct_id, ct_id):
            run(["nnUNetv2_unpack", str(ds), "3d_fullres", "0"], env)
    if "link" in steps:
        run(
            [
                py, str(CORE / "link_translation_targets.py"),
                "--preprocessed", str(resolve_cfg(cfg, "nnunet_preprocessed")),
                "--raw", str(raw),
                "--id-in", str(cbct_id),
                "--id-out", str(ct_id),
                "--name-in", f"{name}_CBCT",
                "--name-out", f"{name}_CT",
            ]
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
