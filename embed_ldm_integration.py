"""
Integration shim: register the EMBED dataset with LDM_correspondences.

Place this file at the top level of your LDM_correspondences checkout (next
to eval/ and utils/) and the embed_ldm/ package alongside it:

    LDM_correspondences/
    ├── eval/
    ├── utils/
    ├── embed_ldm/             <-- the package from this conversation
    ├── embed_ldm_integration.py   <-- this file
    └── ...

Then patch `eval/download.py` with the two-line addition shown in the README,
or simply import this module before launching eval.eval. The shim
monkey-patches download.load_dataset to recognize benchmark='embed'.

Run:
    # Path A: qualitative
    python -m eval.eval --benchmark embed --datapath /path/to/EMBED \
        --visualize --thres img --item_index 0

    # Path B: PCK with synthetic affine
    EMBED_MODE=synthetic_affine python -m eval.eval --benchmark embed \
        --datapath /path/to/EMBED --thres img
"""

import os
from pathlib import Path

from embed_ldm import (
    EmbedPairBuilder, MammogramLoader, EmbedPairDataset, ExamPair,
)


# ---------------------------------------------------------------------------
# Configuration: read from env vars so you don't have to edit eval/eval.py
# ---------------------------------------------------------------------------
#   EMBED_TABLES_DIR    path to folder with EMBED_OpenData_*.csv
#   EMBED_DICOM_ROOT    path to DICOM root (typically EMBED_root/images)
#   EMBED_PAIRS_CSV     (optional) pre-built pair manifest CSV
#   EMBED_MODE          'grid' (default) or 'synthetic_affine'
#   EMBED_VIEWS         comma-separated, e.g. 'CC' or 'CC,MLO'
#   EMBED_MIN_GAP_MO    min months between paired exams, default 6
#   EMBED_MAX_GAP_MO    max months, default 36
#   EMBED_GRID_SIZE     grid side for keypoints, default 4 (-> up to 16 pts)
#   EMBED_FLIP_TO_LEFT  '1' (default) to flip right breasts to left-side orientation
#   EMBED_CLAHE         '1' to apply CLAHE, default off
#   EMBED_REQUIRE_FILES '1' (default) to drop pairs whose DICOMs aren't on
#                       disk. Essential when working with a subset of EMBED
#                       because the CSV references the full release. Set to
#                       '0' only if all images are guaranteed present.
# ---------------------------------------------------------------------------


def _build_embed_dataset(
    benchmark, datapath, thres, device, split,
    augmentation, feature_size, sub_class="all", item_index=-1,
):
    """Build the EMBED dataset to match eval.download.load_dataset signature."""
    tables_dir = os.environ.get('EMBED_TABLES_DIR') or str(Path(datapath) / 'tables')
    dicom_root = os.environ.get('EMBED_DICOM_ROOT') or str(Path(datapath) / 'images')
    pairs_csv = os.environ.get('EMBED_PAIRS_CSV')
    mode = os.environ.get('EMBED_MODE', 'grid')
    views = tuple(os.environ.get('EMBED_VIEWS', 'CC,MLO').split(','))
    min_gap = float(os.environ.get('EMBED_MIN_GAP_MO', '6'))
    max_gap = float(os.environ.get('EMBED_MAX_GAP_MO', '36'))
    grid_size = int(os.environ.get('EMBED_GRID_SIZE', '4'))
    flip_to_left = os.environ.get('EMBED_FLIP_TO_LEFT', '1') == '1'
    apply_clahe = os.environ.get('EMBED_CLAHE', '0') == '1'
    require_files_exist = os.environ.get('EMBED_REQUIRE_FILES', '1') == '1'

    print(f"[embed_ldm] tables={tables_dir} dicoms={dicom_root}")
    print(f"[embed_ldm] mode={mode} views={views} gap={min_gap}-{max_gap} mo "
          f"grid={grid_size} flip_to_left={flip_to_left}")

    loader = MammogramLoader(
        target_size=512,
        crop_to_breast=True,
        flip_to_left=flip_to_left,
        apply_clahe=apply_clahe,
    )

    if pairs_csv:
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
        print(f"[embed_ldm] loaded {len(pairs)} pairs from {pairs_csv}")
    else:
        builder = EmbedPairBuilder(tables_dir, dicom_root)
        pairs = builder.build_pairs(
            views=views,
            min_months_gap=min_gap,
            max_months_gap=max_gap,
            consecutive_only=True,
            require_files_exist=require_files_exist,
        )
        print(f"[embed_ldm] built {len(pairs)} pairs from CSVs")

    if item_index != -1 and 0 <= item_index < len(pairs):
        pairs = [pairs[item_index]]

    if not pairs:
        raise RuntimeError(
            "No EMBED pairs found. Check that tables_dir + dicom_root point "
            "to real files and that views/gap windows are reasonable."
        )

    return EmbedPairDataset(
        pairs=pairs, loader=loader, mode=mode, grid_size=grid_size,
    )


def install_embed_benchmark():
    """Monkey-patch eval/download.py to accept benchmark='embed' (or 'custom'
    when EMBED_ENABLE=1 is set, since the upstream argparse rejects 'embed').
    """
    from eval import download

    original_load = download.load_dataset

    def patched_load_dataset(
        benchmark, datapath, thres, device, split='test',
        augmentation=False, feature_size=16, sub_class="all", item_index=-1,
    ):
        # Accept either an explicit 'embed' benchmark, or 'custom' when the
        # user has set EMBED_ENABLE=1 in the environment. The second path
        # works around the hardcoded argparse choices in eval/eval.py.
        is_embed = (
            benchmark == 'embed'
            or (benchmark == 'custom'
                and os.environ.get('EMBED_ENABLE', '0') == '1')
        )
        if is_embed:
            return _build_embed_dataset(
                benchmark, datapath, thres, device, split,
                augmentation, feature_size, sub_class, item_index,
            )
        return original_load(
            benchmark, datapath, thres, device, split,
            augmentation, feature_size, sub_class, item_index,
        )

    download.load_dataset = patched_load_dataset

    # Also bypass the Google-drive auto-download
    original_download = download.download_dataset

    def patched_download_dataset(datapath, benchmark):
        is_embed = (
            benchmark == 'embed'
            or (benchmark == 'custom'
                and os.environ.get('EMBED_ENABLE', '0') == '1')
        )
        if is_embed:
            return  # nothing to download
        return original_download(datapath, benchmark)

    download.download_dataset = patched_download_dataset
    print("[embed_ldm] registered EMBED loader; trigger with either "
          "--benchmark embed (if argparse accepts it) or "
          "--benchmark custom + EMBED_ENABLE=1")


# Auto-install on import
install_embed_benchmark()
from scripts.auto_commit import commit_with_hf_message

if __name__ == '__main__':
    # ------------------------------------------------------------------
    # Smoke test — runs directly from PyCharm with the green play button.
    # Edit the three paths below for your machine.
    # ------------------------------------------------------------------
    import os

    commit_with_hf_message(commit_msg='Embed Stat')
    # The folder that contains 'tables/' (and conceptually 'images/').
    # Used as a base; the two more specific paths below override the
    # tables and DICOM locations independently.
    embed_root = r"C:\Users\paliniya\embed_dev"

    # Where EMBED_OpenData_clinical.csv and EMBED_OpenData_metadata.csv live.
    os.environ['EMBED_TABLES_DIR'] = r"C:\Users\paliniya\embed_dev\tables"

    # Whatever folder makes (DICOM_ROOT + metadata.csv's anon_dicom_path)
    # resolve to an actual file. Your earlier error showed a doubled
    # 'images\images\' — that means anon_dicom_path already starts with
    # 'images\...', so DICOM_ROOT should be the parent of that 'images'
    # folder, not 'images' itself.
    os.environ['EMBED_DICOM_ROOT'] = r"C:\Users\paliniya\embed_dev"

    # Optional knobs — uncomment to use:
    # os.environ['EMBED_VIEWS'] = 'CC'           # CC only
    # os.environ['EMBED_MIN_GAP_MO'] = '10'      # at least 10 months apart
    # os.environ['EMBED_MAX_GAP_MO'] = '14'      # at most 14 months apart
    # os.environ['EMBED_MODE'] = 'grid'          # 'grid' or 'synthetic_affine'

    ds = _build_embed_dataset(
        'embed', embed_root, 'img', 'cpu', 'test',
        False, 16, item_index=0,
    )
    print(f"\nDataset size: {len(ds)}")
    batch = ds[0]
    for k, v in batch.items():
        if hasattr(v, 'shape'):
            print(f"  {k:15s} shape={tuple(v.shape)} dtype={v.dtype}")
        else:
            print(f"  {k:15s} = {v}")