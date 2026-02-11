"""Readers for DBiT-seq RNA/tissue-position and ATAC fragment tar bundles."""

from __future__ import annotations

import re
import shutil
import tarfile
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import pandas as pd

if TYPE_CHECKING:
    import anndata as ad

TISSUE_POSITION_COLUMNS = [
    "barcode",
    "in_tissue",
    "array_row",
    "array_col",
    "pxl_row_in_fullres",
    "pxl_col_in_fullres",
]

PAIR_PATTERN = re.compile(
    r"^(?P<sample>.+?)_(?P<kind>RNA_matrix|tissue_positions_list)\.csv\.tar$"
)
ATAC_TAR_PATTERN = re.compile(
    r"^(?P<sample>.+?)_(?P<kind>atac_fragments|fragments)\.tsv\.tar$"
)
ATAC_KIND_PRIORITY = ("atac_fragments", "fragments")


def read_csv_from_tar_local(
    tar_path: str | Path,
    prefer_suffix: str = ".csv.gz",
    **read_csv_kwargs,
) -> pd.DataFrame:
    """Read a CSV (optionally gzipped) from a tar file."""
    tar_path = Path(tar_path)
    with tarfile.open(tar_path, "r") as tar:
        members = [member for member in tar.getmembers() if member.isfile()]
        if not members:
            raise ValueError(f"No files found inside tar: {tar_path}")

        member = next((m for m in members if m.name.endswith(prefer_suffix)), None)
        if member is None:
            member = next((m for m in members if m.name.endswith(".csv")), members[0])

        with tar.extractfile(member) as fh:
            if fh is None:
                raise ValueError(f"Could not extract member '{member.name}' from {tar_path}")
            compression = "gzip" if member.name.endswith(".gz") else None
            return pd.read_csv(fh, compression=compression, **read_csv_kwargs)


def discover_rna_tissue_pairs(
    data_dir: str | Path,
    pattern: re.Pattern[str] = PAIR_PATTERN,
) -> dict[str, dict[str, Path]]:
    """Discover sample pairs keyed by sample_id from a directory of tar files."""
    data_dir = Path(data_dir)
    pairs: dict[str, dict[str, Path]] = {}
    for tar_path in sorted(data_dir.glob("*.tar")):
        match = pattern.match(tar_path.name)
        if match is None:
            continue
        sample_id = match.group("sample")
        kind = match.group("kind")
        pairs.setdefault(sample_id, {})[kind] = tar_path
    return pairs


def discover_atac_fragment_tars(
    data_dir: str | Path,
    sample_ids: Iterable[str] | None = None,
    pattern: re.Pattern[str] = ATAC_TAR_PATTERN,
) -> pd.DataFrame:
    """Discover ATAC fragment tar files and return one best tar per sample.

    ATAC file names can appear as either:
    - ``<sample>_atac_fragments.tsv.tar``
    - ``<sample>_fragments.tsv.tar``
    """
    data_dir = Path(data_dir)
    sample_filter = {str(s) for s in sample_ids} if sample_ids is not None else None

    by_sample: dict[str, dict[str, Path]] = {}
    for tar_path in sorted(data_dir.glob("*.tar")):
        match = pattern.match(tar_path.name)
        if match is None:
            continue
        sample_id = match.group("sample")
        if sample_filter is not None and sample_id not in sample_filter:
            continue
        kind = match.group("kind")
        by_sample.setdefault(sample_id, {})[kind] = tar_path

    rows: list[dict[str, str]] = []
    for sample_id in sorted(by_sample):
        kinds = by_sample[sample_id]
        selected_kind = next((k for k in ATAC_KIND_PRIORITY if k in kinds), None)
        if selected_kind is None:
            continue
        rows.append(
            {
                "sample_id": sample_id,
                "atac_kind": selected_kind,
                "atac_tar": str(kinds[selected_kind]),
            }
        )

    return pd.DataFrame(rows)


def _extract_tar_member_file(
    tar: tarfile.TarFile,
    member: tarfile.TarInfo,
    out_path: Path,
    overwrite: bool = False,
) -> None:
    if out_path.exists() and not overwrite:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    src = tar.extractfile(member)
    if src is None:
        raise ValueError(f"Could not extract member '{member.name}'")
    with src, out_path.open("wb") as dst:
        shutil.copyfileobj(src, dst)


def extract_atac_fragment_archives(
    out_dir: str | Path,
    atac_manifest: pd.DataFrame | None = None,
    data_dir: str | Path | None = None,
    sample_ids: Iterable[str] | None = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Extract fragment files from tar archives and return a local file manifest.

    Either pass ``atac_manifest`` (from :func:`discover_atac_fragment_tars`) or ``data_dir``.
    """
    if atac_manifest is None:
        if data_dir is None:
            raise ValueError("Provide either atac_manifest or data_dir.")
        atac_manifest = discover_atac_fragment_tars(data_dir=data_dir, sample_ids=sample_ids)

    out_dir = Path(out_dir)
    rows: list[dict[str, str]] = []

    for row in atac_manifest.itertuples(index=False):
        sample_id = str(row.sample_id)
        atac_kind = str(row.atac_kind)
        tar_path = Path(str(row.atac_tar))

        with tarfile.open(tar_path, "r") as tar:
            members = [member for member in tar.getmembers() if member.isfile()]
            fragment_member = next(
                (member for member in members if member.name.endswith(".tsv.gz")),
                None,
            )
            if fragment_member is None:
                raise ValueError(f"No .tsv.gz fragment file found inside {tar_path}")

            index_member = next(
                (member for member in members if member.name.endswith(".tsv.gz.tbi")),
                None,
            )

            sample_out_dir = out_dir / sample_id
            fragments_path = sample_out_dir / Path(fragment_member.name).name
            _extract_tar_member_file(
                tar=tar,
                member=fragment_member,
                out_path=fragments_path,
                overwrite=overwrite,
            )

            index_path = None
            if index_member is not None:
                index_path = sample_out_dir / Path(index_member.name).name
                _extract_tar_member_file(
                    tar=tar,
                    member=index_member,
                    out_path=index_path,
                    overwrite=overwrite,
                )

        rows.append(
            {
                "sample_id": sample_id,
                "atac_kind": atac_kind,
                "atac_tar": str(tar_path),
                "fragments_tsv_gz": str(fragments_path),
                "fragments_tbi": str(index_path) if index_path is not None else "",
            }
        )

    return pd.DataFrame(rows)


def import_atac_fragments_with_snap(
    atac_manifest: pd.DataFrame,
    genome,
    whitelist_by_sample: dict[str, Iterable[str]] | None = None,
    sorted_by_barcode: bool = False,
    **import_kwargs,
) -> dict[str, object]:
    """Import extracted fragment files into SnapATAC2 objects.

    ``genome`` can be a SnapATAC2 genome object (for example ``snap.genome.mm10``).
    ``atac_manifest`` should contain ``sample_id`` and ``fragments_tsv_gz`` columns.
    """
    import snapatac2 as snap

    sample_adatas: dict[str, object] = {}
    for row in atac_manifest.itertuples(index=False):
        sample_id = str(row.sample_id)
        fragment_file = str(row.fragments_tsv_gz)
        kwargs = dict(import_kwargs)
        kwargs["sorted_by_barcode"] = sorted_by_barcode

        if whitelist_by_sample is not None and sample_id in whitelist_by_sample:
            kwargs["whitelist"] = list(whitelist_by_sample[sample_id])

        try:
            adata_atac = snap.pp.import_data(
                fragment_file=fragment_file,
                chrom_sizes=genome,
                **kwargs,
            )
        except TypeError:
            # Compatibility with SnapATAC2 versions using `genome=` instead of `chrom_sizes=`.
            adata_atac = snap.pp.import_data(
                fragment_file=fragment_file,
                genome=genome,
                **kwargs,
            )

        if hasattr(adata_atac, "uns"):
            adata_atac.uns["sample_id"] = sample_id
            adata_atac.uns.setdefault("source_paths", {})
            adata_atac.uns["source_paths"]["fragments_tsv_gz"] = fragment_file

        sample_adatas[sample_id] = adata_atac

    return sample_adatas


def _build_sample_adata(
    sample_id: str,
    rna_tar: str | Path,
    tissue_tar: str | Path,
    tissue_position_columns: Iterable[str] = TISSUE_POSITION_COLUMNS,
) -> ad.AnnData:
    import anndata as ad
    from scipy import sparse

    tissue_pos = read_csv_from_tar_local(
        tissue_tar,
        header=None,
        names=list(tissue_position_columns),
    )
    rna_matrix = read_csv_from_tar_local(rna_tar, index_col=0)

    tissue_pos["barcode"] = tissue_pos["barcode"].astype(str)
    obs_df = tissue_pos.drop_duplicates("barcode").set_index("barcode")

    rna_matrix.index = rna_matrix.index.astype(str)
    rna_matrix.columns = rna_matrix.columns.astype(str)
    idx_overlap = int(rna_matrix.index.isin(obs_df.index).sum())
    col_overlap = int(rna_matrix.columns.isin(obs_df.index).sum())
    expr_df = rna_matrix if idx_overlap >= col_overlap else rna_matrix.T
    expr_df = expr_df.loc[~expr_df.index.duplicated(keep="first")]

    common_barcodes = obs_df.index.intersection(expr_df.index)
    if len(common_barcodes) == 0:
        raise ValueError(f"No overlapping barcodes for sample '{sample_id}'")

    obs = obs_df.loc[common_barcodes].copy()
    var = pd.DataFrame(index=expr_df.columns.astype(str))
    X = sparse.csr_matrix(expr_df.loc[common_barcodes].to_numpy())

    adata_sample = ad.AnnData(X=X, obs=obs, var=var)
    adata_sample.obs_names_make_unique()
    adata_sample.var_names_make_unique()
    adata_sample.obsm["spatial"] = adata_sample.obs[
        ["pxl_col_in_fullres", "pxl_row_in_fullres"]
    ].to_numpy()
    adata_sample.uns["n_missing_barcodes"] = int(obs_df.index.difference(expr_df.index).size)
    adata_sample.uns["source_paths"] = {
        "rna_matrix_tar": str(rna_tar),
        "tissue_positions_tar": str(tissue_tar),
    }
    adata_sample.uns["sample_id"] = sample_id
    return adata_sample


def read_dbit_rna_directory(
    data_dir: str | Path,
) -> tuple[ad.AnnData, pd.DataFrame]:
    """Load and concatenate all complete RNA+tissue sample pairs from ``data_dir``.

    Expected files:
    - ``<sample>_RNA_matrix.csv.tar``
    - ``<sample>_tissue_positions_list.csv.tar``
    """
    data_dir = Path(data_dir)
    pairs = discover_rna_tissue_pairs(data_dir)
    import anndata as ad
    import numpy as np

    required = {"RNA_matrix", "tissue_positions_list"}
    sample_ids = sorted(
        sample_id for sample_id, files in pairs.items() if required.issubset(files)
    )
    if not sample_ids:
        raise ValueError(f"No complete sample pairs found in {data_dir}")

    sample_adatas: dict[str, ad.AnnData] = {}
    summary_rows: list[dict[str, int | str]] = []

    for sample_id in sample_ids:
        rna_tar = pairs[sample_id]["RNA_matrix"]
        tissue_tar = pairs[sample_id]["tissue_positions_list"]
        adata_sample = _build_sample_adata(sample_id, rna_tar, tissue_tar)
        sample_adatas[sample_id] = adata_sample
        summary_rows.append(
            {
                "sample_id": sample_id,
                "n_obs": int(adata_sample.n_obs),
                "n_vars": int(adata_sample.n_vars),
                "missing_barcodes": int(adata_sample.uns["n_missing_barcodes"]),
            }
        )

    adata_multi = ad.concat(
        [sample_adatas[sample_id] for sample_id in sample_ids],
        label="sample_id",
        keys=sample_ids,
        join="outer",
        fill_value=0,
        index_unique="-",
    )
    adata_multi.obs["sample_id"] = adata_multi.obs["sample_id"].astype("category")
    adata_multi.obsm["spatial"] = np.asarray(adata_multi.obsm["spatial"])
    adata_multi.uns["source_data_dir"] = str(data_dir)

    summary = pd.DataFrame(summary_rows).sort_values("sample_id").reset_index(drop=True)
    return adata_multi, summary


__all__ = [
    "ATAC_KIND_PRIORITY",
    "ATAC_TAR_PATTERN",
    "PAIR_PATTERN",
    "TISSUE_POSITION_COLUMNS",
    "discover_atac_fragment_tars",
    "discover_rna_tissue_pairs",
    "extract_atac_fragment_archives",
    "import_atac_fragments_with_snap",
    "read_csv_from_tar_local",
    "read_dbit_rna_directory",
]
