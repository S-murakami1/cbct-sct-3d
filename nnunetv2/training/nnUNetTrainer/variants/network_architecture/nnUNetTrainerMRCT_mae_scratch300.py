"""300-epoch MAE translation. Same trainer as the Head experiments, on top of upstream nnUNetTrainerMRCT_mae.

Output is one channel. Epochs and learning rate are trainer settings, not config.yaml.
"""
from typing import List, Tuple, Union

from torch import nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerMRCT_mae import (
    nnUNetTrainerMRCT_mae,
)


class nnUNetTrainerMRCT_mae_scratch300(nnUNetTrainerMRCT_mae):
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
        self.num_epochs = 300
        self.initial_lr = 1e-3

    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        return nnUNetTrainer.build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            1,
            enable_deep_supervision,
        )
