# DBiT Analysis

Notebook-first workflows for loading DBiT-seq RNA/ATAC archives, building `AnnData` objects, and running unimodal + multimodal analysis.

## Repository Contents

- `dbit_nature_multisample_workflow.ipynb`: original multisample RNA-focused workflow.
- `rna_atac_multimodal_snapatac2.ipynb`: local RNA+ATAC workflow modeled after the SnapATAC2 modality tutorial, adapted for `data-RNA`.
- `utils/dbit_rna_reader.py`: reusable readers for RNA/tissue tar files and ATAC fragment tar files.
- `utils/plot_spatial_compact_fast.py`: compact spatial plotting utility.
- `utils/aggregate_neighbors_weighted.py`: neighborhood aggregation utility.

## Expected Input Layout

The local reader expects filenames in `data-RNA/` like:

- RNA matrix: `<sample_id>_RNA_matrix.csv.tar`
- Tissue positions: `<sample_id>_tissue_positions_list.csv.tar`
- ATAC fragments (either form):  
  - `<sample_id>_atac_fragments.tsv.tar`  
  - `<sample_id>_fragments.tsv.tar`

Each ATAC tar should contain:

- `*.tsv.gz` fragment file
- optional `*.tsv.gz.tbi` index

## Local Multimodal Workflow (Recommended)

Use `rna_atac_multimodal_snapatac2.ipynb`.

1. Discover local RNA+tissue and ATAC tar files under `data-RNA` (recursive).
2. Checkpoint per-sample RNA h5ad with `write_rna_h5ad_per_sample(...)`.
3. Discover and extract local ATAC tar files with:
   - `discover_atac_fragment_tars(...)`
   - `extract_atac_fragment_archives(...)`
4. Import local ATAC fragments one sample at a time with `import_atac_fragments_to_h5ad_per_sample(...)`.
5. Build one combined `rna` object and one combined `atac` object from per-sample h5ad files.
6. Align by shared barcodes and run joint embedding (`snap.tl.multi_spectral`).

The notebook now uses sample-aware barcodes (`sample_id:barcode`) so RNA and ATAC can be aligned across multiple samples.
It also supports restart-safe checkpointing in `data/checkpoints/` to avoid rerunning everything after kernel crashes.

## Utilities API

`utils/dbit_rna_reader.py` provides:

- `read_dbit_rna_directory(data_dir)`
- `discover_atac_fragment_tars(data_dir, sample_ids=None)`
- `extract_atac_fragment_archives(out_dir, atac_manifest=..., overwrite=False)`
- `write_rna_h5ad_per_sample(data_dir, out_dir, overwrite=False)`
- `import_atac_fragments_to_h5ad_per_sample(out_dir, genome=..., overwrite=False)`
- `import_atac_fragments_with_snap(atac_manifest, genome=..., whitelist_by_sample=...)`

## Running

Open and run one of:

- `rna_atac_multimodal_snapatac2.ipynb` (local multimodal)
- `dbit_nature_multisample_workflow.ipynb` (legacy RNA workflow)

Recommended Python packages:

- `python`
- `pandas`
- `numpy`
- `scipy`
- `anndata`
- `scanpy`
- `snapatac2`
- `matplotlib`

## Notes

- Raw archives and large intermediates are git-ignored.
- For local ATAC import, set the correct genome in the notebook (for example `snap.genome.mm10` or `snap.genome.hg38`).
- If you changed reader code in-session and imports look stale, restart the kernel and rerun from the first cell.
