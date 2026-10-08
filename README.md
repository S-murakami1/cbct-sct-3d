# cbct-sct-3d

3D CBCT to synthetic CT. `nnunetv2/` is adapted from [nnUNet_translation](https://github.com/phyrise/nnUNet_translation) (clipped CT normalization, one output channel, AFP trainers). `core/` is registration and dataset preparation. Settings are in `config.yaml` (datasets 93 / 94 / 95, `hu_clip`). Patch size, batch size, epochs, and learning rate stay in the trainer and plans.

## Environment

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
cd cbct-sct-3d
uv python install 3.11
uv sync
uv sync --project totalseg
```

PyTorch wheels from PyPI include CUDA. TotalSegmentator is a second environment because it loads its own `nnunetv2`; one shared environment would import this translation package. Pseudo labels use `totalseg/.venv`. Training uses `uv sync`.

Elastix 5.2.0 (Linux x86-64) is not in git. Parameters are in `core/synthrad2025_configs`.

```bash
bash scripts/fetch_elastix.sh
```

## Data

Run from the repository root. The exports must match `config.yaml`.

Paired NRRD: `data/nrrd/CBCT` and `data/nrrd/CT`, same file name (for example `P001_1.nrrd`).

```bash
export nnUNet_raw="$PWD/raw"
export nnUNet_preprocessed="$PWD/preprocessed"
export nnUNet_results="$PWD/results"
```

```bash
uv run python core/nrrd_to_nifti.py --src data/nrrd --dst data/work/rigid
uv run python core/make_fov_masks.py --cbct-dir data/work/rigid/CBCT --out-dir data/work/rigid/fovMasks
uv run python core/run_elastix_deformable.py --src data/work/rigid --dst data/work/deformable --region HN --fov-mask-dir data/work/rigid/fovMasks --workers 4 --threads-per-job 4
uv run python core/prepare_training_data.py --config config.yaml
```

FOV figures: `data/work/rigid/fovMasks/vis`. Registration figures: `data/work/deformable/vis`. Fixed image is CBCT. `prepare_training_data.py` resumes with `--from` and `--to` (`pair`, `preprocess`, `link`).

`test_cases: [P001_1, P002_1]` writes those cases to `imagesTs`. Empty keeps every case in `imagesTr`.

## MAE training

```bash
uv run nnUNetv2_train 93 3d_fullres 0 -tr nnUNetTrainerMRCT_mae_scratch300 -p nnResUNetPlans
```

## MAE + AFP training

```bash
uv run python core/extract_pseudo_labels.py --config config.yaml
# 7-class CT teacher (dataset 95). AFP reads its features.
uv run nnUNetv2_train 95 3d_fullres 0 -tr nnUNetTrainerSegScratch -p nnUNetPlans
# MAE + AFP training
uv run nnUNetv2_train 93 3d_fullres 0 -tr nnUNetTrainerMRCT_AFP_scratch300 -p nnResUNetPlans
```

## Inference and evaluation

 For AFP, use `-tr nnUNetTrainerMRCT_AFP_scratch300` and `pred_afp`.

```bash
uv run nnUNetv2_predict -i raw/Dataset093_Head_CBCT/imagesTr -o results/pred_mae -d 93 -c 3d_fullres -p nnResUNetPlans -tr nnUNetTrainerMRCT_mae_scratch300 -f 0 --disable_tta --rec mean -chk checkpoint_best.pth
```

```bash
uv run python core/denorm_predictions_to_hu.py -i results/pred_mae -g raw/Dataset094_Head_CT/imagesTr -o results/pred_mae_HU --fingerprint preprocessed/Dataset094_Head_CT/dataset_fingerprint.json
```

MAE, PSNR, and NCC inside the geometric FOV. Score: `metrics_geom.json`. Older intensity scores stay in `metrics_fov.json`.

```bash
uv run python core/eval_mae_body_mask.py --pred-dir results/pred_mae_HU --gt-dir raw/Dataset094_Head_CT/imagesTr --cbct-dir raw/Dataset093_Head_CBCT/imagesTr --mask-dir data/work/rigid/fovMasks --out results/pred_mae_HU/metrics_geom.json
```

Figures: `results/pred_mae_HU/viz`.

```bash
uv run python core/viz_sct_results.py --pred-dir results/pred_mae_HU --cbct-dir raw/Dataset093_Head_CBCT/imagesTr --gt-dir raw/Dataset094_Head_CT/imagesTr --mask-dir data/work/rigid/fovMasks --metrics results/pred_mae_HU/metrics_geom.json --out-dir results/pred_mae_HU/viz
```

## References

1. Longuefosse et al. "[Adapted nnU-Net: A Robust Baseline for Cross-Modality Synthesis and Medical Image Inpainting](https://doi.org/10.1007/978-3-031-73281-2_3)". *SASHIMI*, 2024.

2. Isensee et al. "[nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation](https://doi.org/10.1038/s41592-020-01008-z)". *Nature Methods*, 2021.

3. Longuefosse et al. "[Anatomical feature-prioritized loss for enhanced MR to CT translation](https://doi.org/10.1088/1361-6560/adea07)". *Physics in Medicine & Biology*, 2025.

4. Sequeiro Gonzalez et al. "[Deep Learning-Based Cross-Anatomy CT Synthesis Using Adapted nnResU-Net with Anatomical Feature Prioritized Loss](https://arxiv.org/abs/2509.22394)". *arXiv*, 2025.

5. https://github.com/SynthRAD2025/preprocessing".
