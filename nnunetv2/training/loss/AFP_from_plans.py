"""AFP + MAE. Teacher architecture comes from that teacher's nnUNet plans.json."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
import pydoc

from nnunetv2.training.loss.unet import PlainConvUNet


def _plans_arch(plans_path: Path) -> tuple[dict, list[str]]:
    import json

    plans = json.loads(plans_path.read_text())
    cfg = plans["configurations"]["3d_fullres"]
    arch = cfg["architecture"]
    kwargs = dict(arch["arch_kwargs"])
    req = list(arch.get("_kw_requires_import") or kwargs.pop("_kw_requires_import", []))
    kwargs.pop("_kw_requires_import", None)
    return kwargs, req


def _import_kwargs(kwargs: dict, required: list[str]) -> dict:
    out = dict(kwargs)
    for key in required:
        if out.get(key) is not None and isinstance(out[key], str):
            located = pydoc.locate(out[key])
            if located is None:
                raise ImportError(f"cannot import plans kwarg {key}={out[key]}")
            out[key] = located
    out["deep_supervision"] = False
    return out


def n_feature_layers(n_stages: int) -> int:
    """Feature maps only. The network also returns final segmentation logits, which are not compared."""
    return 2 * n_stages - 3


class AFPFromPlans(nn.Module):
    def __init__(
        self,
        weights_path: Path,
        plans_path: Path,
        num_classes: int,
        mae_weight: float = 0.5,
        afp_weight: float = 1.0,
        device: torch.device | None = None,
    ):
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device
        kwargs, req = _plans_arch(plans_path)
        kwargs = _import_kwargs(kwargs, req)
        n_stages = int(kwargs["n_stages"])
        self.stages = n_stages - 1
        self.layers = list(range(n_feature_layers(n_stages)))
        model = PlainConvUNet(input_channels=1, num_classes=num_classes, **kwargs)
        if not Path(weights_path).is_file():
            raise FileNotFoundError(f"AFP teacher checkpoint not found: {weights_path}")
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        state = checkpoint.get("state_dict", checkpoint.get("network_weights", checkpoint.get("model_state_dict")))
        model.load_state_dict(state, strict=False)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        self.model = model.to(device=device, dtype=torch.float16)
        self.L1 = nn.L1Loss()
        self.mae_weight = float(mae_weight)
        self.afp_weight = float(afp_weight)
        self.last_afp = torch.zeros(())
        self.last_mae = torch.zeros(())
        print(
            f"AFP teacher {weights_path} plans {plans_path} "
            f"layers={self.layers} mae_weight={self.mae_weight} afp_weight={self.afp_weight}"
        )

    def center_pad_to_multiple_of_2pow(self, x):
        factor = 2 ** self.stages
        shape = x.shape[-3:]
        pad = []
        for s in reversed(shape):
            new = ((s + factor - 1) // factor) * factor
            total = new - s
            pad.extend([total // 2, total - total // 2])
        return F.pad(x, pad, mode="constant", value=0)

    def forward_components(self, x, y):
        src_device = x.device
        x = x.to(self._device, non_blocking=True)
        y = y.to(self._device, non_blocking=True)
        x = self.center_pad_to_multiple_of_2pow(x)
        y = self.center_pad_to_multiple_of_2pow(y)
        emb_x = self.model(x)
        with torch.no_grad():
            emb_y = self.model(y)
        afp_loss = 0
        for i in self.layers:
            afp_loss = afp_loss + self.L1(emb_x[i], emb_y[i].detach())
        weighted_afp = afp_loss * self.afp_weight
        if self.mae_weight > 0.0:
            mae_loss = self.L1(x, y.float()) * self.mae_weight
        else:
            mae_loss = torch.zeros((), device=self._device, dtype=afp_loss.dtype)
        self.last_afp = afp_loss.detach()
        self.last_mae = mae_loss.detach() if torch.is_tensor(mae_loss) else torch.tensor(mae_loss)
        total = (weighted_afp + mae_loss).to(src_device)
        return total, self.last_afp.to(src_device), self.last_mae.to(src_device)

    def forward(self, x, y):
        total, _, _ = self.forward_components(x, y)
        return total


REPO = Path(__file__).resolve().parents[3]


def load_config(path: Path | None = None) -> dict:
    import yaml

    cfg_path = (path or (REPO / "config.yaml")).resolve()
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise SystemExit(f"config must be a mapping: {cfg_path}")
    cfg["_repo"] = REPO
    return cfg


def resolve_cfg(cfg: dict, key: str) -> Path:
    raw = str(cfg.get(key) or "").strip()
    if not raw:
        raise SystemExit(f"config '{key}' is empty")
    path = Path(raw)
    if not path.is_absolute():
        path = cfg["_repo"] / path
    return path.resolve()


def resolve_optional(cfg: dict, key: str) -> Path | None:
    raw = str(cfg.get(key) or "").strip()
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = cfg["_repo"] / path
    return path.resolve()


def teacher_paths_from_config(cfg: dict) -> tuple[Path, Path, int]:
    """Return (checkpoint, plans, num_classes)."""
    results = resolve_cfg(cfg, "nnunet_results")
    name = cfg["dataset_name"]
    tid = int(cfg["teacher_dataset_id"])
    ds = f"Dataset{tid:03d}_{name}_CT_7labels"
    default_ckpt = (
        results
        / ds
        / "nnUNetTrainerSegScratch__nnUNetPlans__3d_fullres"
        / "fold_0"
        / "checkpoint_best.pth"
    )
    raw = str(cfg.get("teacher_checkpoint") or "").strip()
    ckpt = Path(raw) if raw else default_ckpt
    if not ckpt.is_absolute():
        ckpt = (cfg["_repo"] / ckpt).resolve()
    plans = ckpt.parents[1] / "plans.json"
    if not plans.is_file():
        plans = resolve_cfg(cfg, "nnunet_preprocessed") / ds / "nnUNetPlans.json"
    num_classes = 7
    return ckpt, plans, num_classes
