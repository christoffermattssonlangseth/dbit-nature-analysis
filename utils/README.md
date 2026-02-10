# MANA Utils

Core implementation of metric-aware neighborhood aggregation and visualization functions.

## Modules

### aggregate_neighbors_weighted.py

**Main function:** `aggregate_neighbors_weighted()`

Performs distance-weighted neighborhood aggregation with two-level weighting:
1. Hop-level decay: `hop_decay^h` for neighbors at hop distance h
2. Distance-based weighting: Exponential, inverse, or gaussian kernels

**Key Parameters:**
- `n_layers` (int): Number of neighborhood layers (recommended: 3-4)
- `hop_decay` (float): Weight decay per hop (recommended: 0.2)
- `distance_kernel` (str): 'exponential', 'inverse', 'gaussian', or 'none'
- `aggregations` (str/list): 'mean', 'median', 'sum', 'max', 'var', 'std'
- `use_rep` (str): Input representation key (e.g., 'X_scVI')

**Output:**
Stores aggregated features in `adata.obsm[out_key]`

### plot_spatial_compact_fast.py

**Main function:** `plot_spatial_compact_fast()`

Fast, publication-quality spatial plotting with multi-panel layouts.

**Key Features:**
- Categorical and continuous coloring
- Gene expression visualization (from `var_names`)
- Category highlighting with custom transparency
- Automatic metadata display (region, course)
- Ordered categorical support
- Shared or per-panel color scales

**Key Parameters:**
- `color` (str): obs column name or gene name
- `groupby` (str): Column to group panels by (e.g., 'sample_id')
- `cols` (int): Number of columns in grid layout
- `highlight` (list): Categories to highlight (others greyed)
- `grey_alpha` (float): Alpha for non-highlighted categories
- `shared_scale` (bool): Use same vmin/vmax across all panels

## Usage

### Basic Import

```python
# From notebooks
import sys
sys.path.insert(0, '../utils')
from aggregate_neighbors_weighted import aggregate_neighbors_weighted
from plot_spatial_compact_fast import plot_spatial_compact_fast
```

### Package Import

```python
# If using as installed package
from utils import aggregate_neighbors_weighted, plot_spatial_compact_fast
```

## Example Workflow

```python
import scanpy as sc
import squidpy as sq
from utils import aggregate_neighbors_weighted, plot_spatial_compact_fast

# Load data
adata = sc.read_h5ad('data.h5ad')

# Build spatial graph
sq.gr.spatial_neighbors(adata, coord_type='generic', delaunay=True)

# Aggregate features
aggregate_neighbors_weighted(
    adata,
    n_layers=3,
    hop_decay=0.2,
    aggregations='mean',
    use_rep='X_scVI',
    distance_kernel='exponential'
)

# Cluster
sc.pp.neighbors(adata, use_rep='X_weighted_agg_mean', n_neighbors=15)
sc.tl.leiden(adata, resolution=0.5, key_added='leiden_weighted')

# Visualize - categorical
plot_spatial_compact_fast(
    adata,
    color='leiden_weighted',
    groupby='sample_id',
    cols=3,
    height=8
)

# Visualize - gene expression
plot_spatial_compact_fast(
    adata,
    color='CD8A',  # gene name
    groupby='sample_id',
    cmap_name='viridis',
    shared_scale=True
)

# Highlight specific clusters
plot_spatial_compact_fast(
    adata,
    color='leiden_weighted',
    groupby='sample_id',
    highlight=['0', '3', '5'],
    grey_alpha=0.1
)
```

## Distance Kernels

### Exponential (Recommended)
- Formula: `weight = exp(-d/scale)`
- Use case: Diffusion-like processes, gradual transitions
- Best for: Tumor microenvironments, inflammation zones

### Inverse
- Formula: `weight = 1 / (1 + d/scale)`
- Use case: Simpler distance decay
- Best for: General spatial context

### Gaussian
- Formula: `weight = exp(-d²/(2*scale²))`
- Use case: Sharp boundaries, localized effects
- Best for: Discrete anatomical structures

### None
- Formula: `weight = 1` (uniform within hop)
- Use case: Hop-based only (like CellCharter)

## Aggregation Methods

- **mean**: Standard weighted average (recommended starting point)
- **median**: Robust to outliers, uses weighted quantiles
- **sum**: Emphasizes total neighborhood activity
- **max**: Highlights strongest signals
- **var/std**: Measures neighborhood heterogeneity
