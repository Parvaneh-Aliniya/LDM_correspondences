"""
EmbedPairDataset
================

A torch Dataset that yields longitudinal mammogram pairs in the exact dict
schema the LDM_correspondences eval pipeline expects:

    {
        'src_img':     (3, 512, 512) float in [0, 1]
        'trg_img':     (3, 512, 512)
        'src_kps':     (2, max_pts)  float, padded with -1
        'trg_kps':     (2, max_pts)
        'n_pts':       int
        'pckthres':    float    (512.0 in image-threshold mode)
        'idx':         int
        'bool_img_src': (512, 512) bool — breast region mask
        'bool_img_trg': (512, 512) bool
    }

Two modes:

1. KEYPOINT MODE = 'grid'  (default for path A — qualitative)
   Lays a regular grid of keypoints inside the source breast mask. No real
   ground truth — trg_kps is a placeholder copy of src_kps. PCK is meaningless
   here; use --visualize to inspect the predicted correspondences instead.

2. KEYPOINT MODE = 'synthetic_affine'  (for path B — PCK sanity check)
   The target image is the SOURCE image with a known affine transform applied.
   Keypoints in source are projected through the known transform to get
   ground-truth target keypoints. PCK is meaningful and tells you whether
   the SD prior can recover known transformations on mammograms.

3. KEYPOINT MODE = 'manual'  (for path C — real eval, future)
   Reads keypoint pairs from an annotations file. See annotations_path arg.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .pair_builder import ExamPair, EmbedPairBuilder
from .dicom_loader import MammogramLoader, LoadedMammogram


# ---------------------------------------------------------------------------
class EmbedPairDataset(Dataset):
    """
    EMBED longitudinal mammogram pairs, formatted for LDM_correspondences.

    Parameters
    ----------
    pairs : list of ExamPair
        From EmbedPairBuilder.build_pairs().
    loader : MammogramLoader
        For DICOM -> tensor conversion. Pass an instance so you can configure
        crop_to_breast / flip_to_left / etc. consistently.
    mode : {'grid', 'synthetic_affine', 'manual'}
        How to generate keypoints. See module docstring.
    grid_size : int
        For mode='grid': lays out grid_size x grid_size points inside the
        breast mask, keeping only those that fall on breast tissue.
        Default 4 -> up to 16 keypoints per pair.
    max_pts : int
        Pad/truncate keypoints to this many slots. The model's evaluator
        treats slots with x == -1 as padding. Default 40 to match SPair.
    pckthres_mode : {'img', 'bbox'}
        How to set pckthres. 'img' = 512 (image-relative PCK). 'bbox' would
        need a bbox, which we don't have natively for mammograms. Stick with
        'img' unless you supply bboxes.
    synthetic_transform : dict, optional
        For mode='synthetic_affine': spec for the affine. Example:
            {'rot_deg': 5.0, 'tx_px': 10, 'ty_px': 10, 'scale': 1.05}
        Default applies a modest rotation+translation+scale, the kind of
        view-variation you'd expect between exams of the same breast.
    annotations_path : Path, optional
        For mode='manual': path to a JSON or CSV with keypoint pairs.
        (Not implemented in this first cut; raises NotImplementedError.)
    seed : int
        For reproducible synthetic transforms.

    Notes
    -----
    The constructor takes **kwargs to absorb the eval/download.py call
    signature (benchmark=, datapath=, thres=, ...). Most are ignored here
    because we don't need them — but we use `device` if passed.
    """

    def __init__(
        self,
        pairs: Optional[List[ExamPair]] = None,
        loader: Optional[MammogramLoader] = None,
        mode: str = 'grid',
        grid_size: int = 4,
        max_pts: int = 40,
        pckthres_mode: str = 'img',
        synthetic_transform: Optional[dict] = None,
        annotations_path: Optional[Path] = None,
        seed: int = 0,
        # Catch-all for compatibility with download.load_dataset signature:
        **kwargs,
    ):
        super().__init__()
        if pairs is None:
            raise ValueError(
                "EmbedPairDataset needs `pairs` (list of ExamPair). "
                "Build them with EmbedPairBuilder(...).build_pairs(...)."
            )
        if mode not in ('grid', 'synthetic_affine', 'manual'):
            raise ValueError(f"unknown mode: {mode}")
        if mode == 'manual':
            raise NotImplementedError(
                "Manual-annotation mode is path C — not implemented yet."
            )
        if pckthres_mode != 'img':
            raise NotImplementedError(
                "Only pckthres_mode='img' supported; mammograms have no "
                "natural bbox."
            )

        self.pairs = pairs
        self.loader = loader or MammogramLoader()
        self.mode = mode
        self.grid_size = grid_size
        self.max_pts = max_pts
        self.pckthres_mode = pckthres_mode
        self.synthetic_transform = synthetic_transform or {
            'rot_deg': 5.0, 'tx_px': 10.0, 'ty_px': 10.0, 'scale': 1.05,
        }
        self.annotations_path = annotations_path
        self.seed = seed

    # ---------------------------------------------------------------- length
    def __len__(self) -> int:
        return len(self.pairs)

    # -------------------------------------------------------------- main op
    def __getitem__(self, idx: int) -> dict:
        pair = self.pairs[idx]

        # Load source (always real)
        src = self.loader.load(pair.src_path, laterality=pair.laterality)

        if self.mode == 'synthetic_affine':
            # Target = warp of source by known transform. We override pair.trg.
            trg, transform_512 = self._apply_synthetic_transform(src, idx)
            src_kps, trg_kps, n_pts = self._gen_keypoints_synthetic(
                src.mask, transform_512,
            )
        else:
            # mode == 'grid': real target image, fake (placeholder) trg_kps
            trg = self.loader.load(pair.trg_path, laterality=pair.laterality)
            src_kps, trg_kps, n_pts = self._gen_keypoints_grid(src.mask)

        return self._pack_batch(idx, src, trg, src_kps, trg_kps, n_pts)

    # ============================================================ keypoints
    def _gen_keypoints_grid(
        self, src_mask: torch.Tensor,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Lay a grid inside the breast region; trg_kps = src_kps (placeholder).

        The trg_kps are placeholders only — PCK will report whether predictions
        landed near these dummy points, which is meaningless. For real numbers,
        use mode='synthetic_affine'.
        """
        H, W = src_mask.shape
        ys = np.linspace(0, H - 1, self.grid_size + 2)[1:-1]
        xs = np.linspace(0, W - 1, self.grid_size + 2)[1:-1]
        pts = np.array([(x, y) for y in ys for x in xs], dtype=np.float32)

        # Filter to points inside the breast mask
        mask_np = src_mask.numpy()
        keep = np.array(
            [bool(mask_np[int(round(y)), int(round(x))]) for x, y in pts]
        )
        pts = pts[keep]
        # Placeholder target keypoints (will give nonsensical PCK)
        return pts.copy(), pts.copy(), len(pts)

    def _gen_keypoints_synthetic(
        self,
        src_mask: torch.Tensor,
        transform_512: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Grid source keypoints + apply known transform to get ground truth.

        Both src_kps and trg_kps are filtered so that BOTH the source point
        AND its transformed location fall inside the breast (otherwise the
        transformed point lands in black background and is meaningless).
        """
        H, W = src_mask.shape
        ys = np.linspace(0, H - 1, self.grid_size + 2)[1:-1]
        xs = np.linspace(0, W - 1, self.grid_size + 2)[1:-1]
        src_pts = np.array(
            [(x, y) for y in ys for x in xs], dtype=np.float32,
        )

        # Apply transform to get target points (homogeneous coords)
        ones = np.ones((src_pts.shape[0], 1))
        hom = np.concatenate([src_pts, ones], axis=1)         # (N, 3)
        trg_pts = (transform_512 @ hom.T).T[:, :2].astype(np.float32)

        mask_np = src_mask.numpy()

        def _inside(pt):
            x, y = pt
            xi, yi = int(round(x)), int(round(y))
            if not (0 <= xi < W and 0 <= yi < H):
                return False
            return bool(mask_np[yi, xi])

        keep = np.array([
            _inside(s) and _inside(t)
            for s, t in zip(src_pts, trg_pts)
        ])
        src_pts = src_pts[keep]
        trg_pts = trg_pts[keep]
        return src_pts, trg_pts, len(src_pts)

    # ============================================================== transform
    def _apply_synthetic_transform(
        self, src: LoadedMammogram, idx: int,
    ) -> Tuple[LoadedMammogram, np.ndarray]:
        """Build a target image as an affine warp of the source.

        Returns the warped LoadedMammogram and the 3x3 transform in
        512x512-pixel coords that maps source -> target.
        """
        try:
            import cv2
        except ImportError as e:
            raise ImportError(
                "opencv-python required for synthetic affine mode."
            ) from e

        rng = np.random.default_rng(self.seed + idx)
        spec = self.synthetic_transform
        # Allow specifying ranges as (min, max) tuples instead of fixed values
        rot = self._sample(spec.get('rot_deg', 0.0), rng)
        tx = self._sample(spec.get('tx_px', 0.0), rng)
        ty = self._sample(spec.get('ty_px', 0.0), rng)
        sc = self._sample(spec.get('scale', 1.0), rng)

        # Build 3x3 transform: rotate+scale about image center, then translate
        H, W = self.loader.target_size, self.loader.target_size
        cx, cy = W / 2.0, H / 2.0
        cos_a = math.cos(math.radians(rot)) * sc
        sin_a = math.sin(math.radians(rot)) * sc
        # Affine: maps (x, y) -> rotate-scale-around-center + translate
        a = cos_a
        b = -sin_a
        e = sin_a
        f = cos_a
        T = np.array([
            [a, b, (1 - a) * cx - b * cy + tx],
            [e, f, -e * cx + (1 - f) * cy + ty],
            [0, 0, 1],
        ], dtype=np.float64)

        # Warp the source image and mask with this transform
        src_img_np = src.img[0].numpy()  # (H, W) — channel 0 (all 3 identical)
        warped = cv2.warpAffine(
            src_img_np, T[:2], (W, H),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        warped_mask = cv2.warpAffine(
            src.mask.numpy().astype(np.uint8), T[:2], (W, H),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(bool)

        warped_img_t = torch.from_numpy(warped).float()
        warped_img_t = warped_img_t.unsqueeze(0).repeat(3, 1, 1)
        warped_mask_t = torch.from_numpy(warped_mask)

        trg = LoadedMammogram(
            img=warped_img_t,
            mask=warped_mask_t,
            affine=T @ src.affine,   # original-pixel -> warped 512 frame
            orig_shape=src.orig_shape,
            laterality=src.laterality,
        )
        return trg, T

    @staticmethod
    def _sample(value, rng) -> float:
        """Sample a scalar from `value`, which may be a float or (min, max)."""
        if isinstance(value, (tuple, list)) and len(value) == 2:
            lo, hi = value
            return float(rng.uniform(lo, hi))
        return float(value)

    # ================================================================ pack
    def _pack_batch(
        self,
        idx: int,
        src: LoadedMammogram,
        trg: LoadedMammogram,
        src_kps_xy: np.ndarray,
        trg_kps_xy: np.ndarray,
        n_pts: int,
    ) -> dict:
        """Convert per-pair stuff into the dict the LDM eval pipeline expects.

        Keypoints arrive as (N, 2) (x, y) in 512-pixel coords. The model wants
        shape (2, max_pts), x-row then y-row, with -1 padding past n_pts.
        """
        max_pts = self.max_pts

        # Truncate if we somehow generated more than max_pts
        n_pts = int(min(n_pts, max_pts))
        src_kps_xy = src_kps_xy[:n_pts]
        trg_kps_xy = trg_kps_xy[:n_pts]

        def _pad(xy: np.ndarray) -> torch.Tensor:
            out = -1 * torch.ones((2, max_pts), dtype=torch.float32)
            if n_pts > 0:
                out[0, :n_pts] = torch.from_numpy(xy[:, 0])
                out[1, :n_pts] = torch.from_numpy(xy[:, 1])
            return out

        src_kps = _pad(src_kps_xy)
        trg_kps = _pad(trg_kps_xy)

        pckthres = torch.tensor(float(self.loader.target_size))

        return {
            'src_img':       src.img,
            'trg_img':       trg.img,
            'src_kps':       src_kps,
            'trg_kps':       trg_kps,
            'n_pts':         torch.tensor(n_pts),
            'pckthres':      pckthres,
            'idx':           torch.tensor(idx),
            'bool_img_src':  src.mask,
            'bool_img_trg':  trg.mask,
            # Extras for debugging / downstream use (model ignores these):
            'src_imname':    f"{self.pairs[idx].patient_id}_{self.pairs[idx].src_acc}",
            'trg_imname':    f"{self.pairs[idx].patient_id}_{self.pairs[idx].trg_acc}",
            'category':      f"{self.pairs[idx].laterality}_{self.pairs[idx].view}",
            'category_id':   torch.tensor(0),
            'datalen':       len(self.pairs),
        }


# ---------------------------------------------------------------------------
# Factory function matching the eval/download.py call signature
# ---------------------------------------------------------------------------
def make_embed_dataset(
    benchmark: str,
    datapath: str,
    thres: str,
    device: str,
    split: str,
    augmentation: bool,
    feature_size: int,
    sub_class: str = "all",
    item_index: int = -1,
    # EMBED-specific kwargs:
    tables_dir: Optional[str] = None,
    dicom_root: Optional[str] = None,
    pairs_csv: Optional[str] = None,
    mode: str = 'grid',
    **embed_kwargs,
) -> EmbedPairDataset:
    """Build an EmbedPairDataset from the eval/download.py call shape.

    `datapath` is reused as the EMBED root if `tables_dir`/`dicom_root` are
    not set explicitly. By convention we expect:
        datapath/tables/EMBED_OpenData_*.csv
        datapath/images/   (DICOM root)
    """
    root = Path(datapath)
    tables_dir = Path(tables_dir) if tables_dir else (root / 'tables')
    dicom_root = Path(dicom_root) if dicom_root else (root / 'images')

    if pairs_csv:
        # Read prebuilt pair manifest (fast path — skip CSV scan every time)
        import pandas as pd
        df = pd.read_csv(pairs_csv)
        pairs = [
            ExamPair(
                patient_id=str(r['patient_id']),
                laterality=str(r['laterality']),
                view=str(r['view']),
                src_acc=str(r['src_acc']),
                trg_acc=str(r['trg_acc']),
                src_path=Path(r['src_path']),
                trg_path=Path(r['trg_path']),
                src_date=pd.to_datetime(r['src_date']),
                trg_date=pd.to_datetime(r['trg_date']),
                months_gap=float(r['months_gap']),
                src_birads=r.get('src_birads'),
                trg_birads=r.get('trg_birads'),
                src_density=r.get('src_density'),
                trg_density=r.get('trg_density'),
            ) for _, r in df.iterrows()
        ]
    else:
        pairs = EmbedPairBuilder(tables_dir, dicom_root).build_pairs(
            **embed_kwargs.get('build_pairs_kwargs', {})
        )

    if item_index != -1:
        pairs = [pairs[item_index]]

    return EmbedPairDataset(
        pairs=pairs,
        mode=mode,
        **{k: v for k, v in embed_kwargs.items()
           if k not in ('build_pairs_kwargs',)},
    )
