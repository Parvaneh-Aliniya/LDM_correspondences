"""
EMBED longitudinal pair builder
================================

Reads the EMBED CSV tables and produces a list of (source, target) exam pairs
for the same patient, same laterality, same view position, ordered by date.

This is the EMBED-specific part: the rest of the pipeline (DICOM loading,
512x512 conversion, dataset class) doesn't need to know anything about EMBED's
CSV schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd


@dataclass
class ExamPair:
    """One (source, target) longitudinal pair for one breast & view."""
    patient_id: str          # empi_anon
    laterality: str          # 'L' or 'R'
    view: str                # 'CC' or 'MLO'
    src_acc: str             # acc_anon of source exam (earlier)
    trg_acc: str             # acc_anon of target exam (later)
    src_path: Path           # path to source DICOM
    trg_path: Path           # path to target DICOM
    src_date: pd.Timestamp
    trg_date: pd.Timestamp
    months_gap: float        # months between exams
    src_birads: Optional[str] = None
    trg_birads: Optional[str] = None
    src_density: Optional[float] = None
    trg_density: Optional[float] = None


class EmbedPairBuilder:
    """
    Selects longitudinal exam pairs from EMBED CSV tables.

    Parameters
    ----------
    tables_dir : Path
        Folder with EMBED_OpenData_clinical.csv and EMBED_OpenData_metadata.csv.
    dicom_root : Path
        Root of the EMBED images directory. The metadata table has relative
        paths under this root (column 'anon_dicom_path' or similar).
    """

    def __init__(self, tables_dir: str | Path, dicom_root: str | Path):
        self.tables_dir = Path(tables_dir)
        self.dicom_root = Path(dicom_root)
        self.clinical: Optional[pd.DataFrame] = None
        self.metadata: Optional[pd.DataFrame] = None
        self._load_tables()

    def _load_tables(self) -> None:
        self.clinical = pd.read_csv(
            self.tables_dir / 'EMBED_OpenData_clinical.csv', low_memory=False,
        )
        self.metadata = pd.read_csv(
            self.tables_dir / 'EMBED_OpenData_metadata.csv', low_memory=False,
        )

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _find_col(df: pd.DataFrame, *candidates: str) -> Optional[str]:
        lower = {c.lower(): c for c in df.columns}
        for cand in candidates:
            if cand in df.columns:
                return cand
            if cand.lower() in lower:
                return lower[cand.lower()]
        return None

    # -------------------------------------------------------------- pair logic
    def build_pairs(
        self,
        views: tuple = ('CC', 'MLO'),
        min_months_gap: float = 6.0,
        max_months_gap: float = 60.0,
        consecutive_only: bool = True,
        same_density_only: bool = False,
        require_screening: bool = True,
        max_pairs_per_patient: Optional[int] = None,
        require_files_exist: bool = True,
        verbose: bool = True,
    ) -> List[ExamPair]:
        """
        Select (source, target) exam pairs from the same patient.

        Parameters
        ----------
        views : tuple of str
            Which standard views to include. CC and MLO are the typical
            screening views.
        min_months_gap, max_months_gap : float
            Only keep pairs whose exam dates differ within this window.
            Default 6-60 months covers typical annual/biennial screening
            intervals while excluding same-day duplicate exams.
        consecutive_only : bool
            If True, only pair exam[i] with exam[i+1] for that breast/view.
            If False, pair every exam with every later exam (combinatorial).
        same_density_only : bool
            If True, drop pairs where the per-exam density code differs
            (because the breast composition itself has changed substantially).
        require_screening : bool
            Restrict to exams whose 'desc' contains 'screen'. Diagnostic exams
            often involve compression spot views and aren't comparable.
        max_pairs_per_patient : int or None
            Cap to avoid one patient dominating.
        require_files_exist : bool
            If True (default), drop any pair where either DICOM file isn't
            actually on disk. Essential when working with a subset of EMBED
            because the CSV references the full release. The CSV is filtered
            first (per acc_anon) so the date-ordering is correct relative to
            available exams.
        verbose : bool
            Print a one-line summary of how many pairs were kept / dropped.

        Returns
        -------
        list of ExamPair
        """
        pid_col = self._find_col(self.clinical, 'empi_anon')
        eid_col = self._find_col(self.clinical, 'acc_anon')
        date_col = self._find_col(self.clinical, 'study_date_anon', 'sdate_anon')
        desc_col = self._find_col(self.clinical, 'desc')
        birads_col = self._find_col(self.clinical, 'asses')
        density_col = self._find_col(self.clinical, 'tissueden')

        if not all([pid_col, eid_col, date_col]):
            raise KeyError(
                "Need empi_anon, acc_anon, and study_date_anon in clinical.csv"
            )

        # Metadata gives us image paths, view, laterality
        m_eid = self._find_col(self.metadata, 'acc_anon')
        m_view = self._find_col(self.metadata, 'ViewPosition')
        m_lat = self._find_col(self.metadata, 'ImageLateralityFinal')
        m_path = (self._find_col(self.metadata, 'anon_dicom_path')
                  or self._find_col(self.metadata, 'png_path')
                  or self._find_col(self.metadata, 'image_path'))
        m_imgtype = self._find_col(self.metadata, 'FinalImageType')

        if not all([m_eid, m_view, m_lat, m_path]):
            raise KeyError(
                "Need acc_anon, ViewPosition, ImageLateralityFinal, and a path "
                "column (anon_dicom_path / png_path) in metadata.csv. Got: "
                f"{list(self.metadata.columns)[:20]}"
            )

        # ---- Build one-row-per-(exam, view, side) image index --------------
        meta = self.metadata[[m_eid, m_view, m_lat, m_path]].copy()
        meta.columns = ['acc_anon', 'view', 'lat', 'path']

        # Keep only requested views & valid lat
        meta = meta[meta['view'].isin(views) & meta['lat'].isin(['L', 'R'])]

        # Prefer 2D (not C-view) when both exist
        if m_imgtype:
            it = self.metadata[[m_eid, m_imgtype]].drop_duplicates()
            it.columns = ['acc_anon', 'imgtype']
            meta = meta.merge(it, on='acc_anon', how='left')
            # Sort 2D first, then keep first per (acc, view, lat)
            meta['imgtype_rank'] = meta['imgtype'].map(
                lambda x: 0 if str(x).lower() == '2d' else 1
            )
            meta = meta.sort_values('imgtype_rank')
            meta = meta.drop_duplicates(
                subset=['acc_anon', 'view', 'lat'], keep='first',
            )
            meta = meta.drop(columns=['imgtype_rank', 'imgtype'])
        else:
            meta = meta.drop_duplicates(
                subset=['acc_anon', 'view', 'lat'], keep='first',
            )

        # ---- Optional: drop rows whose DICOM file isn't on disk ------------
        # Done BEFORE pairing so that "consecutive" means consecutive among
        # *available* exams, not among all exams in the CSV.
        n_meta_before = len(meta)
        if require_files_exist:
            # Build full paths and check existence. This is the slow step on
            # large subsets but happens once per build_pairs() call.
            full_paths = meta['path'].apply(
                lambda p: self.dicom_root / str(p)
            )
            exists_mask = full_paths.apply(lambda p: p.is_file())
            n_missing = int((~exists_mask).sum())
            meta = meta[exists_mask].reset_index(drop=True)
            if verbose:
                print(
                    f"[pair_builder] available-files filter: "
                    f"{len(meta):,}/{n_meta_before:,} images on disk "
                    f"({n_missing:,} missing — likely the rest of the EMBED "
                    f"release)."
                )
                if len(meta) == 0:
                    print(
                        "[pair_builder] WARNING: zero images matched. "
                        f"Check that dicom_root='{self.dicom_root}' is "
                        f"correct and that metadata.csv's path column "
                        f"('anon_dicom_path' or similar) is relative to it."
                    )

        # ---- Build one-row-per-exam clinical info --------------------------
        cli_cols = [pid_col, eid_col, date_col]
        if desc_col:
            cli_cols.append(desc_col)
        if density_col:
            cli_cols.append(density_col)
        if birads_col:
            cli_cols.append(birads_col)
        cli = (self.clinical[cli_cols]
               .drop_duplicates(subset=eid_col)
               .copy())
        cli[date_col] = pd.to_datetime(cli[date_col], errors='coerce')
        cli = cli.dropna(subset=[date_col])

        if require_screening and desc_col:
            cli = cli[cli[desc_col].astype(str).str.lower().str.contains(
                'screen', na=False,
            )]

        # ---- Cross & sort ---------------------------------------------------
        joined = meta.merge(
            cli, left_on='acc_anon', right_on=eid_col, how='inner',
        )

        # For each (patient, lat, view) bucket, sort by date and emit pairs
        pairs: List[ExamPair] = []
        group_keys = [pid_col, 'lat', 'view']
        for (pid, lat, view), grp in joined.groupby(group_keys, sort=False):
            grp = grp.sort_values(date_col).reset_index(drop=True)
            if len(grp) < 2:
                continue

            if consecutive_only:
                index_pairs = [(i, i + 1) for i in range(len(grp) - 1)]
            else:
                index_pairs = [
                    (i, j)
                    for i in range(len(grp))
                    for j in range(i + 1, len(grp))
                ]

            n_added_for_patient = 0
            for i, j in index_pairs:
                src, trg = grp.iloc[i], grp.iloc[j]
                gap_months = (trg[date_col] - src[date_col]).days / 30.44
                if not (min_months_gap <= gap_months <= max_months_gap):
                    continue
                if same_density_only and density_col:
                    if pd.notna(src[density_col]) and pd.notna(trg[density_col]):
                        if src[density_col] != trg[density_col]:
                            continue

                pairs.append(ExamPair(
                    patient_id=str(pid),
                    laterality=lat,
                    view=view,
                    src_acc=str(src['acc_anon']),
                    trg_acc=str(trg['acc_anon']),
                    src_path=self.dicom_root / str(src['path']),
                    trg_path=self.dicom_root / str(trg['path']),
                    src_date=src[date_col],
                    trg_date=trg[date_col],
                    months_gap=round(gap_months, 1),
                    src_birads=(str(src[birads_col])
                                if birads_col and pd.notna(src[birads_col])
                                else None),
                    trg_birads=(str(trg[birads_col])
                                if birads_col and pd.notna(trg[birads_col])
                                else None),
                    src_density=(float(src[density_col])
                                 if density_col and pd.notna(src[density_col])
                                 else None),
                    trg_density=(float(trg[density_col])
                                 if density_col and pd.notna(trg[density_col])
                                 else None),
                ))
                n_added_for_patient += 1
                if (max_pairs_per_patient is not None
                        and n_added_for_patient >= max_pairs_per_patient):
                    break

        if verbose:
            n_patients = len({p.patient_id for p in pairs})
            print(
                f"[pair_builder] built {len(pairs):,} pairs across "
                f"{n_patients:,} unique patients "
                f"(views={views}, gap={min_months_gap}-{max_months_gap} mo)"
            )

        return pairs

    @staticmethod
    def to_dataframe(pairs: List[ExamPair]) -> pd.DataFrame:
        """Convert pair list to a tidy DataFrame (for inspection / export)."""
        return pd.DataFrame([{
            'patient_id': p.patient_id,
            'laterality': p.laterality,
            'view': p.view,
            'src_acc': p.src_acc,
            'trg_acc': p.trg_acc,
            'src_path': str(p.src_path),
            'trg_path': str(p.trg_path),
            'src_date': p.src_date,
            'trg_date': p.trg_date,
            'months_gap': p.months_gap,
            'src_birads': p.src_birads,
            'trg_birads': p.trg_birads,
            'src_density': p.src_density,
            'trg_density': p.trg_density,
        } for p in pairs])