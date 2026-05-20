"""
embed_ldm: adapter for running the LDM_correspondences pipeline on EMBED.

Quick start:
    from embed_ldm import EmbedPairBuilder, EmbedPairDataset, MammogramLoader

    builder = EmbedPairBuilder(tables_dir='./tables', dicom_root='./images')
    pairs = builder.build_pairs(min_months_gap=12, max_months_gap=36)

    loader = MammogramLoader(crop_to_breast=True, flip_to_left=True)
    ds = EmbedPairDataset(pairs=pairs, loader=loader, mode='grid')

    batch = ds[0]
    # batch has the schema the LDM_correspondences eval pipeline expects
"""

from .pair_builder import ExamPair, EmbedPairBuilder
from .dicom_loader import MammogramLoader, LoadedMammogram
from .embed_dataset import EmbedPairDataset, make_embed_dataset

__all__ = [
    'ExamPair',
    'EmbedPairBuilder',
    'MammogramLoader',
    'LoadedMammogram',
    'EmbedPairDataset',
    'make_embed_dataset',
]
