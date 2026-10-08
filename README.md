# cbct-sct-3d

3D CBCT to synthetic CT. This tree is a clone of [nnUNet_translation](https://github.com/phyrise/nnUNet_translation). `nnunetv2/` is the package, `core/` holds registration and dataset scripts, and training is `nnUNetv2_train`.

Paths, dataset ids, FOV, and AFP weights are in `config.yaml`. Patch size, batch size, epochs, and learning rate stay in the trainer and plans. Images are clipped to `hu_clip` in `config.yaml` (`CT_clip`). The checked-in config uses CBCT dataset 93, CT dataset 94, and teacher dataset 95.

## Environment

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
cd cbct-sct-3d
uv python install 3.11
uv sync
uv sync --project totalseg
```

On Linux, the PyTorch wheels from PyPI include CUDA.

TotalSegmentator is a second environment, `totalseg/`. It runs its own segmentation weights through `nnunetv2`. This tree's `nnunetv2` is the translation fork: clipped CT normalization, one output channel, and a different patch overlap. One shared environment makes TotalSegmentator import that fork. Pseudo labels use `totalseg/.venv`, which installs the nnU-Net release TotalSegmentator depends on. Training stays in the environment from `uv sync`.

Elastix is not in git. Registration needs the Linux x86-64 build of elastix 5.2.0. Parameters are already in `core/synthrad2025_configs`.

```bash
bash scripts/fetch_elastix.sh
```

## Environment variables and shared data preparation

Run this from the repository root. The three variables must match `nnunet_raw`, `nnunet_preprocessed`, and `nnunet_results` in `config.yaml`.

Place paired NRRD in `data/nrrd/CBCT` and `data/nrrd/CT`. Each case uses the same file name in both folders, for example `P001_1.nrrd`. Leave `nrrd_src` as `data/nrrd`. Converted images go to `data/work`.

```bash
export nnUNet_raw="$PWD/raw"
export nnUNet_preprocessed="$PWD/preprocessed"
export nnUNet_results="$PWD/results"
```

NRRD to NIfTI, without resampling:

```bash
uv run python core/nrrd_to_nifti.py --src data/nrrd --dst data/work/rigid
```

Geometric FOV masks, CBCT only. Figures go to `data/work/rigid/fovMasks/vis`.

```bash
uv run python core/make_fov_masks.py --cbct-dir data/work/rigid/CBCT --out-dir data/work/rigid/fovMasks
```

Deformable registration. Fixed image is CBCT, moving image is CT. Figures go to `data/work/deformable/vis`.

```bash
uv run python core/run_elastix_deformable.py --src data/work/rigid --dst data/work/deformable --region HN --fov-mask-dir data/work/rigid/fovMasks --workers 4 --threads-per-job 4
```

`prepare_training_data.py` builds the CBCT/CT datasets from `data/work/deformable`, preprocesses them, and links CT targets. Resume it with `--from` and `--to` (`pair`, `preprocess`, `link`).

```bash
uv run python core/prepare_training_data.py --config config.yaml
```

Set test cases in `config.yaml` as `test_cases: [P001_1, P002_1]`. Those cases are written to `imagesTs`. Leave the list empty to keep every case in `imagesTr`.

## MAE training

```bash
uv run nnUNetv2_train 93 3d_fullres 0 -tr nnUNetTrainerMRCT_mae_scratch300 -p nnResUNetPlans
```



## MAE + AFP training

Requires the shared preparation above. `use_totalseg: false` or `--no-totalseg` skips TotalSegmentator and reads `pseudo_label_dir`. `use_hn_muscles: true` paints `head_muscles` and `headneck_muscles` into muscle class 3. An empty `teacher_checkpoint` uses the teacher trained by the first command below. Label figures go to `labels7_flat/vis`.

```bash
uv run python core/extract_pseudo_labels.py --config config.yaml
# Train the 7-class CT segmentation teacher (dataset 95). AFP reads its features.
uv run nnUNetv2_train 95 3d_fullres 0 -tr nnUNetTrainerSegScratch -p nnUNetPlans
# MAE + AFP training
uv run nnUNetv2_train 93 3d_fullres 0 -tr nnUNetTrainerMRCT_AFP_scratch300 -p nnResUNetPlans
```



## Inference and evaluation

The commands below score `imagesTr`. After you fill `test_cases`, change that input to `imagesTs`. For MAE + AFP, use `-tr nnUNetTrainerMRCT_AFP_scratch300` and replace `pred_mae` with `pred_afp`.

Predict sCT from CBCT. The output is still normalized. A smaller `--step_size` (0.3) can reduce patch artifacts. `--rec mean` or `--rec median` selects the overlap reconstruction. `median` is experimental and RAM-intensive.

```bash
uv run nnUNetv2_predict -i raw/Dataset093_Head_CBCT/imagesTr -o results/pred_mae -d 93 -c 3d_fullres -p nnResUNetPlans -tr nnUNetTrainerMRCT_mae_scratch300 -f 0 --disable_tta --rec mean -chk checkpoint_best.pth
```

Convert predictions to HU with the CT dataset fingerprint.

```bash
uv run python core/denorm_predictions_to_hu.py -i results/pred_mae -g raw/Dataset094_Head_CT/imagesTr -o results/pred_mae_HU --fingerprint preprocessed/Dataset094_Head_CT/dataset_fingerprint.json
```

MAE, PSNR, and NCC inside the geometric FOV. `labelsTr` and `labelsTs` are that same mask. The score is `metrics_geom.json`. An existing `metrics_fov.json` is an older intensity-mask score and is left as-is.

```bash
uv run python core/eval_mae_body_mask.py --pred-dir results/pred_mae_HU --gt-dir raw/Dataset094_Head_CT/imagesTr --cbct-dir raw/Dataset093_Head_CBCT/imagesTr --mask-dir data/work/rigid/fovMasks --out results/pred_mae_HU/metrics_geom.json
```

Figures go to `results/pred_mae_HU/viz`. The contour is the geometric FOV.

```bash
uv run python core/viz_sct_results.py --pred-dir results/pred_mae_HU --cbct-dir raw/Dataset093_Head_CBCT/imagesTr --gt-dir raw/Dataset094_Head_CT/imagesTr --mask-dir data/work/rigid/fovMasks --metrics results/pred_mae_HU/metrics_geom.json --out-dir results/pred_mae_HU/viz
```

## References

- **Translation baseline.** Longuefosse et al. (2024). Adapted nnU-Net: A Robust Baseline for Cross-Modality Synthesis and Medical Image Inpainting. SASHIMI, pp. 24–33.
- **nnU-Net.** Isensee et al. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. *Nature Methods*, 18(2), 203–211.
- **AFP loss.** Longuefosse et al. (2025). [Anatomical feature-prioritized loss for enhanced MR to CT translation](https://doi.org/10.1088/1361-6560/adea07). *Physics in Medicine & Biology*, 70(14), 145012.
- **Pseudo-label grouping and distillation.** Sequeiro Gonzalez et al. (2025). [Deep Learning-Based Cross-Anatomy CT Synthesis Using Adapted nnResU-Net with Anatomical Feature Prioritized Loss](https://arxiv.org/abs/2509.22394). arXiv:2509.22394.