# Methods

Paired CBCT and CT are already rigid-aligned. This repository converts them, estimates the CBCT field of view, deforms the CT onto the CBCT, and trains a 3D residual U-Net to translate CBCT into synthetic CT. The translation loss is either image L1 (MAE) or MAE plus an anatomical feature-prioritized (AFP) term.

## Registration

Deformable registration follows stage 2 of the SynthRAD2025 preprocessing pipeline ([SynthRAD2025/preprocessing](https://github.com/SynthRAD2025/preprocessing)). The elastix parameter files in `core/synthrad2025_configs/` are those files:

| Region | File |
|---|---|
| Head and neck | `param_def_cbct_HN.txt` |
| Thorax | `param_def_cbct_TH.txt` |
| Abdomen | `param_def_cbct_AB.txt` |

Elastix is run with the CBCT as the fixed image and the CT as the moving image. This stage does not estimate a new rigid or affine transform. The transform is a B-spline (`BSplineTransform`) optimized over three resolutions (image pyramid schedule 4, 2, 1; grid spacing schedule 4, 2, 1; final grid spacing 10 mm). The metric is the sum of Advanced Mattes mutual information (weight 1), bending-energy penalty (weight 10), and rigidity penalty (weight 0.5). The optimizer is adaptive stochastic gradient descent, with 500, 200, and 100 iterations at the three resolutions. Unspecified voxels are filled with −1000 HU (`DefaultPixelValue`).

The registration mask is the geometric CBCT field of view, not an intensity body mask. It is built only from the CBCT volume: the padding value is the mode of the border voxels, the support is every voxel more than 1 HU away from that value, a morphological closing of about 2 mm is applied, the largest connected component is kept, and each axial slice is hole-filled and replaced by its 2D convex hull. After the warp, voxels outside this mask are set to −1000 HU on both the deformed CT and the CBCT.

## Images and translation labels

CBCT and CT intensities are clipped to `hu_clip` in `config.yaml` (default [−1000, 3000] HU). The nnU-Net channel name is `CT_clip`. Normalization is `CTNormalization_clip`: clip to that same range, then subtract the dataset foreground mean and divide by the dataset foreground standard deviation. At inference, Hounsfield units are recovered with the same CT fingerprint, `HU = z * std + mean`.

The label stored with each translation pair is the geometric CBCT field of view from registration, copied into `labelsTr` and `labelsTs` for both the CBCT and the CT. It is the nnU-Net foreground mask, not a TotalSegmentator label. Earlier datasets stored an intensity mask (voxels above the 10th percentile plus 50 HU); those scores remain in `metrics_fov.json`.

Patch size, batch size, and the residual U-Net topology come from `nnUNetPlannerResUNet` (`nnResUNetPlans`, configuration `3d_fullres`). The network has one output channel. Deep supervision is off. The translation network is trained from scratch for 300 epochs with stochastic gradient descent, an initial learning rate of 1e-3, and 250 iterations per epoch.

## Pseudo labels and the AFP teacher

TotalSegmentator `total` (117 classes) is run on the deformed CT and collapsed to seven labels, following the grouping used with AFP:

| ID | Class |
|---|---|
| 0 | background |
| 1 | organs |
| 2 | cardiac |
| 3 | muscles |
| 4 | bones |
| 5 | ribs |
| 6 | vertebrae |

Skull, hip, humerus, scapula, clavicle, and femur are taken out of the TotalSegmentator muscle group and labeled as bone. Brain and spinal cord from that group are labeled as organs. Head and neck muscle tasks (`head_muscles`, `headneck_muscles`) are not added unless `use_hn_muscles` is set; they would be painted into class 3 only where they do not overwrite another foreground class.

A segmentation U-Net is trained on these labels with standard `nnUNetPlans` (`nnUNetTrainerSegScratch`: 300 epochs, initial learning rate 1e-3, from scratch). Its weights are frozen and used only as the AFP feature extractor. The final segmentation logits are not compared. For a teacher with `S` stages, the compared maps are the `2S − 3` intermediate feature maps.

## Loss

MAE training uses the mean absolute error between the synthetic CT and the CT.

AFP training uses the sum of per-layer mean absolute errors, plus a weighted image MAE. With synthetic CT `x` and CT `y`, frozen teacher features `φ_i`, `λ_AFP = 1`, and `λ_MAE = 0.5`:

```text
L = λ_AFP * Σ_i L1(φ_i(x), φ_i(y)) + λ_MAE * L1(x, y)
```

`L1` is the mean absolute error over the elements of that tensor. The sum is over feature maps only; it is not divided by the number of maps. Both volumes are center-padded to a multiple of `2^(S−1)` before the teacher forward, and the image L1 is computed on those padded tensors. Gradients flow through `φ_i(x)` only. `φ_i(y)` is detached, and the teacher stays in evaluation mode.

This matches the nnU-Net translation implementation used by Sequeiro Gonzalez et al. (layer sum, `λ_MAE = 0.5`). The original AFP paper writes the feature term as an average over layers; this training run does not divide by the layer count.

## Inference and evaluation

Inference is a sliding-window prediction with mirroring test-time augmentation disabled and mean overlap reconstruction. A tile step of 0.3 can be used to reduce patch-boundary artifacts. Predictions are converted back to HU with the CT dataset fingerprint before scoring.

Each case is scored inside the geometric CBCT field of view by MAE, PSNR, and NCC. PSNR is `10*log10(data_range^2 / MSE)` with a data range of 2000 HU. NCC is the Pearson correlation on the masked voxels. The score is written to `metrics_geom.json`. Earlier intensity-mask scores in `metrics_fov.json` are left in place.

## References

- **Translation code and MAE baseline.** Longuefosse et al. (2024). Adapted nnU-Net. SASHIMI, pp. 24–33.
- **Segmentation framework.** Isensee et al. (2021). nnU-Net. *Nature Methods*, 18(2), 203–211.
- **AFP loss.** Longuefosse et al. (2025). Anatomical feature-prioritized loss for enhanced MR to CT translation. *Physics in Medicine & Biology*, 70(14), 145012. https://doi.org/10.1088/1361-6560/adea07. The feature term there is a mean over layers.
- **7-class labels and the layer-sum AFP.** Sequeiro Gonzalez et al. (2025). Deep Learning-Based Cross-Anatomy CT Synthesis Using Adapted nnResU-Net with Anatomical Feature Prioritized Loss. arXiv:2509.22394. https://arxiv.org/abs/2509.22394
- **Elastix parameters.** SynthRAD2025 preprocessing, stage 2. https://github.com/SynthRAD2025/preprocessing
