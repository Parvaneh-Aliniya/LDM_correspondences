"""
DICOM -> 512x512 RGB tensor + breast-region mask
================================================

Handles the messy realities of mammogram DICOMs:
- MONOCHROME1 vs MONOCHROME2 photometric interpretation (inverted intensities)
- 12-14 bit pixel data needing window/level
- L vs R laterality (chest wall side convention)
- Huge black background outside the breast
- 2000-5000 px source resized to 512x512 for SD input

Returns:
    img_rgb     : (3, 512, 512) float32 in [0, 1], grayscale replicated to RGB
    breast_mask : (512, 512) bool   - True where the breast tissue is
    affine      : 3x3 transform mapping ORIGINAL pixel coords -> 512x512 coords
                  (so keypoints annotated on the original image can be lifted
                  into the resized space)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Optional dependencies — imported lazily so the module can be inspected
# without pydicom installed.
# ---------------------------------------------------------------------------
def _require_pydicom():
    try:
        import pydicom  # noqa: F401
        return pydicom
    except ImportError as e:
        raise ImportError(
            "pydicom is required to read EMBED DICOMs. "
            "Install with: pip install pydicom"
        ) from e


def _require_cv2():
    try:
        import cv2  # noqa: F401
        return cv2
    except ImportError as e:
        raise ImportError(
            "opencv-python is required for breast-region cropping. "
            "Install with: pip install opencv-python"
        ) from e


# ---------------------------------------------------------------------------
@dataclass
class LoadedMammogram:
    """Result of loading one DICOM file."""
    img: torch.Tensor          # (3, 512, 512) float32 in [0, 1]
    mask: torch.Tensor         # (512, 512) bool — breast region
    affine: np.ndarray         # (3, 3) original -> 512x512 transform
    orig_shape: Tuple[int, int]  # (H, W) of original pixel array
    laterality: str            # 'L' or 'R' (from DICOM or arg)


# ---------------------------------------------------------------------------
class MammogramLoader:
    """
    Loads an EMBED DICOM into a 512x512 RGB tensor for Stable Diffusion.

    Parameters
    ----------
    target_size : int
        Output H = W. Default 512 (what the SD pipeline expects).
    crop_to_breast : bool
        If True, segment the breast region and crop tight before resizing.
        Strongly recommended — without it 60-70% of the 512 px is wasted on
        black background and small features get destroyed.
    flip_to_left : bool
        If True, horizontally flip right-breast images so the chest wall is
        always on the left side of the output. This makes left- and right-
        breast images comparable and matches typical mammography display.
        Note: if you flip, you must also remember to UNFLIP predicted
        coordinates when reporting them in the original image frame.
    apply_clahe : bool
        Contrast-limited adaptive histogram equalization. Helps reveal
        subtle structure. Off by default since it can alter intensities
        meaningfully across images and complicate intensity-based methods.
    window : str
        How to map raw pixel intensities to [0, 1]:
        - 'dicom'   : use VOI LUT from DICOM if present, else percentile
        - 'percentile': clip to (1st, 99th) percentile then linearly stretch
        - 'minmax'  : (deprecated) linear from min..max — sensitive to outliers
    """

    def __init__(
        self,
        target_size: int = 512,
        crop_to_breast: bool = True,
        flip_to_left: bool = True,
        apply_clahe: bool = False,
        window: str = 'percentile',
    ):
        self.target_size = target_size
        self.crop_to_breast = crop_to_breast
        self.flip_to_left = flip_to_left
        self.apply_clahe = apply_clahe
        if window not in ('dicom', 'percentile', 'minmax'):
            raise ValueError(f"unknown window mode: {window}")
        self.window = window

    # ------------------------------------------------------------------ load
    def load(
        self,
        dicom_path: str | Path,
        laterality: Optional[str] = None,
    ) -> LoadedMammogram:
        """
        Load one DICOM and return the preprocessed mammogram.

        Parameters
        ----------
        dicom_path : path to .dcm file
        laterality : 'L' or 'R'. If None, read from DICOM tags. Used to decide
            whether to horizontally flip when flip_to_left=True.
        """
        pydicom = _require_pydicom()

        dicom_path = Path(dicom_path)
        if not dicom_path.is_file():
            raise FileNotFoundError(
                f"DICOM not on disk: {dicom_path}\n"
                f"This usually means the CSV references the full EMBED "
                f"release but only a subset is downloaded locally. Rebuild "
                f"pairs with require_files_exist=True (default), or set "
                f"the env var EMBED_REQUIRE_FILES=1 if running through "
                f"embed_ldm_integration.py."
            )
        ds = pydicom.dcmread(str(dicom_path))
        arr = ds.pixel_array.astype(np.float32)
        orig_shape = arr.shape  # (H, W)

        # ---- 1. Photometric / windowing -----------------------------------
        photometric = getattr(ds, 'PhotometricInterpretation', 'MONOCHROME2')
        if photometric == 'MONOCHROME1':
            # MONOCHROME1: low pixel values = bright. Invert so bright tissue
            # is high value (the MONOCHROME2 convention SD expects).
            arr = arr.max() - arr

        arr = self._window(arr, ds)  # -> [0, 1] float32

        # ---- 2. Laterality (for flip decision) -----------------------------
        if laterality is None:
            laterality = str(getattr(ds, 'ImageLaterality', 'L'))[:1].upper()

        flipped = False
        if self.flip_to_left and laterality == 'R':
            arr = np.fliplr(arr).copy()
            flipped = True

        # ---- 3. Breast-region mask + optional crop -------------------------
        breast_mask_full = self._segment_breast(arr)

        if self.crop_to_breast:
            arr, breast_mask_full, crop_box = self._crop_to_mask(
                arr, breast_mask_full,
            )
        else:
            crop_box = (0, 0, arr.shape[0], arr.shape[1])  # (top, left, h, w)

        # ---- 4. CLAHE ------------------------------------------------------
        if self.apply_clahe:
            cv2 = _require_cv2()
            u8 = (arr * 255).astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            arr = clahe.apply(u8).astype(np.float32) / 255.0

        # ---- 5. Resize to target_size x target_size -----------------------
        cv2 = _require_cv2()
        h, w = arr.shape
        arr_resized = cv2.resize(
            arr, (self.target_size, self.target_size),
            interpolation=cv2.INTER_AREA,
        )
        # Resize mask with nearest-neighbor to keep it boolean
        mask_resized = cv2.resize(
            breast_mask_full.astype(np.uint8),
            (self.target_size, self.target_size),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

        # ---- 6. Build affine: original pixel -> 512 pixel ------------------
        # Compose: (flip if needed) -> (crop) -> (scale)
        affine = self._build_affine(
            orig_shape=orig_shape,
            flipped=flipped,
            crop_box=crop_box,
            out_size=self.target_size,
        )

        # ---- 7. To tensor, replicate to 3 channels ------------------------
        img_t = torch.from_numpy(arr_resized).float()                # (H, W)
        img_t = img_t.unsqueeze(0).repeat(3, 1, 1)                   # (3, H, W)
        mask_t = torch.from_numpy(mask_resized)                      # (H, W) bool

        return LoadedMammogram(
            img=img_t,
            mask=mask_t,
            affine=affine,
            orig_shape=orig_shape,
            laterality=laterality,
        )

    # ----------------------------------------------------------------- helpers
    def _window(self, arr: np.ndarray, ds) -> np.ndarray:
        """Map raw pixel array to [0, 1] float32 according to self.window."""
        if self.window == 'dicom':
            # Try DICOM's VOI LUT machinery
            try:
                from pydicom.pixel_data_handlers.util import apply_voi_lut
                arr_w = apply_voi_lut(arr.astype(np.uint16, copy=False), ds)
                arr_w = arr_w.astype(np.float32)
                lo, hi = arr_w.min(), arr_w.max()
                return (arr_w - lo) / (hi - lo + 1e-8)
            except Exception:
                # Fall through to percentile
                pass
        if self.window == 'minmax':
            lo, hi = arr.min(), arr.max()
            return (arr - lo) / (hi - lo + 1e-8)
        # default / fallback: percentile
        lo, hi = np.percentile(arr, [1.0, 99.0])
        out = np.clip((arr - lo) / (hi - lo + 1e-8), 0.0, 1.0)
        return out.astype(np.float32)

    @staticmethod
    def _segment_breast(arr01: np.ndarray) -> np.ndarray:
        """
        Coarse segmentation of breast tissue vs background.

        Approach: Otsu threshold on a blurred image, then keep the largest
        connected component touching the left edge of the image. Works
        because (a) mammogram background is near-black and tissue is bright,
        (b) the breast always touches one image edge (the chest wall side).

        Assumes the image has already been left-flipped if it was a right
        breast — so chest wall is on the LEFT.
        """
        cv2 = _require_cv2()
        u8 = (arr01 * 255).astype(np.uint8)
        blur = cv2.GaussianBlur(u8, (15, 15), 0)
        _, thresh = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        # Connected components
        num, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, 8)
        if num <= 1:
            return np.ones_like(thresh, dtype=bool)
        # Find largest component that touches the LEFT edge (col 0)
        h, w = thresh.shape
        best_label, best_area = 0, 0
        for lbl in range(1, num):
            area = stats[lbl, cv2.CC_STAT_AREA]
            left = stats[lbl, cv2.CC_STAT_LEFT]
            if left == 0 and area > best_area:
                best_area = area
                best_label = lbl
        if best_label == 0:
            # No component touches left edge — fall back to largest overall
            areas = stats[1:, cv2.CC_STAT_AREA]
            best_label = 1 + int(np.argmax(areas))
        mask = (labels == best_label)
        # Small morphological close to fill internal holes
        mask_u8 = mask.astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
        return mask_u8.astype(bool)

    @staticmethod
    def _crop_to_mask(
        arr: np.ndarray, mask: np.ndarray, pad: int = 8,
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
        """Crop arr & mask to the bounding box of the mask, with `pad` px slack.

        Returns the cropped arr, cropped mask, and (top, left, h, w) box in
        original-image coords.
        """
        ys, xs = np.where(mask)
        if len(ys) == 0:
            return arr, mask, (0, 0, arr.shape[0], arr.shape[1])
        top = max(0, ys.min() - pad)
        bottom = min(arr.shape[0], ys.max() + 1 + pad)
        left = max(0, xs.min() - pad)
        right = min(arr.shape[1], xs.max() + 1 + pad)
        h = bottom - top
        w = right - left
        return (
            arr[top:bottom, left:right],
            mask[top:bottom, left:right],
            (int(top), int(left), int(h), int(w)),
        )

    @staticmethod
    def _build_affine(
        orig_shape: Tuple[int, int],
        flipped: bool,
        crop_box: Tuple[int, int, int, int],
        out_size: int,
    ) -> np.ndarray:
        """Compose flip -> crop -> scale into a 3x3 transform on pixel coords.

        Applies to homogeneous (x, y, 1)^T column vectors:
            x_out = A @ [x_in, y_in, 1]^T

        Useful for later: if you have a keypoint at (x_orig, y_orig) in the
        full-resolution DICOM, you can project it into the 512x512 frame
        without re-deriving the math.
        """
        H, W = orig_shape
        top, left, h_c, w_c = crop_box

        # 1. Flip about x = W/2 if needed
        flip_x = -1 if flipped else 1
        flip_b = (W - 1) if flipped else 0
        T_flip = np.array([
            [flip_x, 0, flip_b],
            [0,      1, 0],
            [0,      0, 1],
        ], dtype=np.float64)

        # 2. Crop: subtract (left, top) — but crop_box was computed on the
        #    POST-FLIP array, so this is in the post-flip frame, fine.
        T_crop = np.array([
            [1, 0, -left],
            [0, 1, -top],
            [0, 0, 1],
        ], dtype=np.float64)

        # 3. Scale: (h_c, w_c) -> (out_size, out_size)
        sx = out_size / w_c if w_c else 1.0
        sy = out_size / h_c if h_c else 1.0
        T_scale = np.array([
            [sx, 0, 0],
            [0, sy, 0],
            [0, 0, 1],
        ], dtype=np.float64)

        return T_scale @ T_crop @ T_flip

    # -------------------------------------------------- public helper for kps
    @staticmethod
    def apply_affine(affine: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
        """Apply a 3x3 affine to (N, 2) (x, y) points. Returns (N, 2)."""
        pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
        ones = np.ones((pts.shape[0], 1))
        hom = np.concatenate([pts, ones], axis=1)         # (N, 3)
        out = (affine @ hom.T).T                          # (N, 3)
        return out[:, :2]