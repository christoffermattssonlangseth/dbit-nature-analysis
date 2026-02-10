"""
MANA (Metric-Aware Neighborhood Aggregation) utilities.

This package provides tools for distance-weighted neighborhood aggregation
and spatial visualization of spatial transcriptomics data.
"""

from .aggregate_neighbors_weighted import aggregate_neighbors_weighted
from .plot_spatial_compact_fast import plot_spatial_compact_fast

__all__ = [
    'aggregate_neighbors_weighted',
    'plot_spatial_compact_fast',
]

__version__ = '0.1.0'
