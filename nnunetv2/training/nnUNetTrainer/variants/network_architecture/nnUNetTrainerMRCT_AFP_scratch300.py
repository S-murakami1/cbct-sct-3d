"""300-epoch translation with AFP + MAE.

Teacher checkpoint, plans, and loss weights come from config.yaml at the
repository root. Epochs and learning rate stay on nnUNetTrainerMRCT_mae_scratch300.
"""
from __future__ import annotations

from typing import List

import numpy as np
from torch import autocast

from nnunetv2.training.loss.AFP_from_plans import AFPFromPlans, load_config, teacher_paths_from_config
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerMRCT_mae_scratch300 import (
    nnUNetTrainerMRCT_mae_scratch300,
)
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.helpers import dummy_context


class nnUNetTrainerMRCT_AFP_scratch300(nnUNetTrainerMRCT_mae_scratch300):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device=None,
    ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        cfg = load_config()
        ckpt, plans_path, num_classes = teacher_paths_from_config(cfg)
        self.AFP_loss = AFPFromPlans(
            ckpt,
            plans_path,
            num_classes=num_classes,
            mae_weight=float(cfg["mae_weight"]),
            afp_weight=float(cfg["afp_weight"]),
            device=self.device,
        )
        for key in ("train_afp_losses", "train_mae_losses", "val_afp_losses", "val_mae_losses"):
            self.logger.my_fantastic_logging.setdefault(key, [])

    def _build_loss(self):
        return self.AFP_loss

    def _with_components(self, step_out: dict) -> dict:
        step_out["afp_loss"] = float(self.AFP_loss.last_afp.cpu())
        step_out["mae_loss"] = float(self.AFP_loss.last_mae.cpu())
        return step_out

    def train_step(self, batch: dict) -> dict:
        return self._with_components(super().train_step(batch))

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]
        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(data)
            total, _, _ = self.AFP_loss.forward_components(output, target)
        return self._with_components(
            {"loss": total.detach().cpu().numpy(), "tp_hard": 0, "fp_hard": 0, "fn_hard": 0}
        )

    def _log_components(self, outputs: List[dict], afp_key: str, mae_key: str) -> None:
        c = collate_outputs(outputs)
        self.logger.log(afp_key, float(np.mean(c["afp_loss"])), self.current_epoch)
        self.logger.log(mae_key, float(np.mean(c["mae_loss"])), self.current_epoch)

    def on_train_epoch_end(self, train_outputs: List[dict]):
        super().on_train_epoch_end(train_outputs)
        self._log_components(train_outputs, "train_afp_losses", "train_mae_losses")

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        super().on_validation_epoch_end(val_outputs)
        self._log_components(val_outputs, "val_afp_losses", "val_mae_losses")

    def on_epoch_end(self):
        log = self.logger.my_fantastic_logging
        if log.get("train_afp_losses"):
            self.print_to_log_file(
                "train_afp",
                np.round(log["train_afp_losses"][-1], decimals=4),
                "train_mae",
                np.round(log["train_mae_losses"][-1], decimals=4),
                "val_afp",
                np.round(log["val_afp_losses"][-1], decimals=4),
                "val_mae",
                np.round(log["val_mae_losses"][-1], decimals=4),
            )
        super().on_epoch_end()
