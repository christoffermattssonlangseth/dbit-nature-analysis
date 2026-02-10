# DBiT Analysis

Notebook-driven workflow for loading DBiT tar archives, building `AnnData`, preprocessing with Scanpy, and plotting compact spatial maps.

## Repository Contents

- `dbit_nature_multisample_workflow.ipynb`: main analysis notebook
- `utils/spatial_utils.py`: compact spatial plotting utility
- `.gitignore`: ignores large data artifacts (`csv`, `tsv`, `h5ad`, archives, etc.)

## Expected Input Layout

The notebook expects sample pairs named like:

- `<sample_id>_RNA_matrix.csv.tar`
- `<sample_id>_tissue_positions_list.csv.tar`

Example folder used in the notebook:

`/Users/christoffer/portal_client/DBIT-nature-data`

## Workflow Summary

1. Discover and pair all sample tar files in the input folder.
2. Build one `AnnData` per sample from RNA matrix + tissue positions.
3. Concatenate all samples into `adata_multi`.
4. Run preprocessing on `adata_multi_pp`:
   - QC metrics
   - filtering
   - normalization + `log1p`
   - HVG selection
   - PCA / neighbors / UMAP
   - Leiden clustering
5. Plot spatial clusters per sample using `plot_spatial_compact_fast`.

## Running

Open and run:

`dbit_nature_multisample_workflow.ipynb`

Recommended environment should include at least:

- `python`
- `pandas`
- `anndata`
- `scanpy`
- `scipy`
- `matplotlib`

## Notes

- Large raw data files are intentionally ignored by git.
- If needed, save processed objects with:
  - `adata_multi.write_h5ad(...)`
  - `adata_multi_pp.write_h5ad(...)`
