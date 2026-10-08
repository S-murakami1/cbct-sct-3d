"""Scratch segmentation training: same LR/epochs as SegFT, no pretrained init."""
import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerSegScratch(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(
            plans, configuration, fold, dataset_json, unpack_dataset=unpack_dataset, device=device
        )
        self.initial_lr = 1e-3
        self.num_epochs = 300
