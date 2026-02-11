"""
MANA (Metric-Aware Neighborhood Aggregation) utilities.

This package provides tools for distance-weighted neighborhood aggregation
and spatial visualization of spatial transcriptomics data.
"""

from .aggregate_neighbors_weighted import aggregate_neighbors_weighted
from .dbit_rna_reader import (
    discover_atac_fragment_tars,
    extract_atac_fragment_archives,
    import_atac_fragments_with_snap,
    read_dbit_rna_directory,
)
from .plot_spatial_compact_fast import plot_spatial_compact_fast

__all__ = [
    'aggregate_neighbors_weighted',
    'discover_atac_fragment_tars',
    'extract_atac_fragment_archives',
    'import_atac_fragments_with_snap',
    'read_dbit_rna_directory',
    'plot_spatial_compact_fast',
]

__version__ = '0.1.0'
