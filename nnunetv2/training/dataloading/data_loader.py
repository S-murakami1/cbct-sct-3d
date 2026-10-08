"""Name kept for TotalSegmentator.

This fork split the loader into data_loader_2d.py and data_loader_3d.py.
TotalSegmentator still imports nnUNetDataLoader from this module when it loads
its custom trainers. Inference does not construct that loader.
"""
from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D as nnUNetDataLoader

__all__ = ["nnUNetDataLoader"]
