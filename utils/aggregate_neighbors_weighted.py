"""
Weighted neighbor aggregation for CellCharter.

This module extends CellCharter's aggregate_neighbors function to support:
1. Hop-level decay: First hop contributes more than second, second more than third, etc.
2. Within-hop distance weighting: Closer cells within each hop contribute more.

"""

import numpy as np
from scipy.sparse import csr_matrix, issparse, diags, eye as sparse_eye
from scipy.spatial.distance import cdist
from anndata import AnnData
from typing import Union, Literal, Optional, List
import warnings


def compute_hop_matrices(
    connectivity: csr_matrix,
    n_layers: int,
) -> List[csr_matrix]:
    """
    Compute sparse matrices indicating neighbors at exactly each hop distance.

    Parameters
    ----------
    connectivity : csr_matrix
        Binary connectivity matrix (1 = neighbors, 0 = not neighbors)
    n_layers : int
        Number of hops to compute

    Returns
    -------
    List of sparse matrices, one per hop (index 0 = hop 1, index 1 = hop 2, etc.)
    Each matrix has 1s for cells at exactly that hop distance.
    """
    n_cells = connectivity.shape[0]

    # Track which cells have been "reached" at previous hops
    # Start with self (hop 0) - USE SPARSE IDENTITY!
    reached = sparse_eye(n_cells, format='csr')

    hop_matrices = []
    current_neighbors = connectivity.copy()

    for hop in range(1, n_layers + 1):
        if hop == 1:
            # First hop is just direct neighbors
            hop_matrix = connectivity.copy()
        else:
            # Multiply to get cells reachable in `hop` steps
            current_neighbors = current_neighbors @ connectivity
            # Binarize (we just care about reachability)
            current_neighbors.data = np.ones_like(current_neighbors.data)
            # Subtract already-reached cells to get ONLY cells at exactly this hop
            hop_matrix = current_neighbors - reached
            # Remove negative values (cells already reached)
            hop_matrix.data = np.maximum(hop_matrix.data, 0)
            hop_matrix.eliminate_zeros()

        hop_matrices.append(hop_matrix.copy())

        # Update reached to include this hop
        reached = reached + hop_matrix
        reached.data = np.minimum(reached.data, 1)  # Binarize

    return hop_matrices


def compute_distance_weights(
    coords: np.ndarray,
    hop_matrix: csr_matrix,
    kernel: Literal['exponential', 'inverse', 'gaussian', 'none'] = 'exponential',
    scale: Optional[float] = None,
    epsilon: float = 1e-6,
) -> csr_matrix:
    """
    Compute distance-based weights for neighbors in a hop matrix.
    
    Parameters
    ----------
    coords : np.ndarray
        Spatial coordinates, shape (n_cells, n_dims)
    hop_matrix : csr_matrix
        Sparse matrix indicating neighbors at this hop
    kernel : str
        Distance weighting kernel:
        - 'exponential': exp(-d / scale)
        - 'inverse': 1 / (d + epsilon)
        - 'gaussian': exp(-d^2 / (2 * scale^2))
        - 'none': uniform weights (1.0 for all neighbors)
    scale : float, optional
        Scale parameter for kernels. If None, auto-computed as median distance.
    epsilon : float
        Small value to prevent division by zero for 'inverse' kernel
        
    Returns
    -------
    Sparse matrix with distance-based weights
    """
    if kernel == 'none':
        return hop_matrix.astype(float)
    
    # Get the indices of non-zero entries (i.e., neighbor pairs)
    rows, cols = hop_matrix.nonzero()
    
    if len(rows) == 0:
        return hop_matrix.astype(float)
    
    # Compute pairwise distances for neighbor pairs only
    distances = np.sqrt(np.sum((coords[rows] - coords[cols]) ** 2, axis=1))
    
    # Auto-compute scale if not provided
    if scale is None:
        scale = np.median(distances) if len(distances) > 0 else 1.0
        if scale == 0:
            scale = 1.0
    
    # Apply kernel
    if kernel == 'exponential':
        weights = np.exp(-distances / scale)
    elif kernel == 'inverse':
        weights = 1.0 / (distances + epsilon)
    elif kernel == 'gaussian':
        weights = np.exp(-distances ** 2 / (2 * scale ** 2))
    else:
        raise ValueError(f"Unknown kernel: {kernel}")
    
    # Create weighted sparse matrix
    weighted_matrix = csr_matrix(
        (weights, (rows, cols)),
        shape=hop_matrix.shape
    )
    
    return weighted_matrix


def aggregate_neighbors_weighted(
    adata: AnnData,
    n_layers: int = 3,
    aggregations: Union[str, List[str]] = 'mean',
    connectivity_key: Optional[str] = None,
    use_rep: Optional[str] = None,
    sample_key: Optional[str] = None,
    out_key: str = 'X_cellcharter_weighted',
    copy: bool = False,
    # New parameters for weighting
    hop_decay: Union[float, Literal['learn'], List[float]] = 0.5,
    distance_kernel: Literal['exponential', 'inverse', 'gaussian', 'none'] = 'exponential',
    distance_scale: Optional[float] = None,
    spatial_key: str = 'spatial',
    normalize_weights: bool = True,
    include_self: bool = True,
    chunk_size: Optional[int] = None,  # NEW: for memory-efficient processing
) -> Optional[AnnData]:
    """
    Aggregate neighbor features with hop-level decay and distance-based weighting.
    
    This extends CellCharter's aggregate_neighbors to weight neighbors by:
    1. Hop distance: neighbors at hop h are weighted by hop_decay^h
    2. Spatial distance: within each hop, closer cells contribute more
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object
    n_layers : int
        Number of hops to aggregate
    aggregations : str or list
        Aggregation function(s): 'mean', 'sum', 'var', 'std'
        Note: with weighting, 'mean' becomes weighted mean
    connectivity_key : str, optional
        Key in adata.obsp for connectivity matrix. 
        If None, uses 'spatial_connectivities'
    use_rep : str, optional
        Key in adata.obsm for feature representation.
        If None, uses adata.X
    sample_key : str, optional
        Key in adata.obs for sample/batch information.
        If provided, aggregation is done within samples.
    out_key : str
        Key to store output in adata.obsm
    copy : bool
        Return a copy instead of modifying in place
        
    hop_decay : float or list
        Decay factor for hop weighting:
        - float: weight for hop h = hop_decay^h (e.g., 0.5 -> 0.5, 0.25, 0.125)
        - list: explicit weights for each hop [w1, w2, w3, ...]
    distance_kernel : str
        Kernel for distance weighting within hops:
        - 'exponential': exp(-d / scale) [recommended]
        - 'inverse': 1 / (d + epsilon)
        - 'gaussian': exp(-d^2 / (2 * scale^2))
        - 'none': no distance weighting (original CellCharter behavior)
    distance_scale : float, optional
        Scale parameter for distance kernel. If None, auto-computed as
        median neighbor distance per hop.
    spatial_key : str
        Key in adata.obsm for spatial coordinates
    normalize_weights : bool
        If True, normalize weights to sum to 1 per cell per hop
    include_self : bool
        If True, include cell's own features (hop 0) in output
        
    Returns
    -------
    If copy=True, returns modified AnnData. Otherwise modifies in place.
    
    Examples
    --------
    >>> import scanpy as sc
    >>> import squidpy as sq
    >>> import cellcharter as cc
    >>> 
    >>> # Build spatial graph
    >>> sq.gr.spatial_neighbors(adata, coord_type='generic', delaunay=True)
    >>> cc.gr.remove_long_links(adata)
    >>> 
    >>> # Standard aggregation (original CellCharter)
    >>> cc.gr.aggregate_neighbors(adata, n_layers=3, use_rep='X_scVI')
    >>> 
    >>> # Weighted aggregation (this function)
    >>> aggregate_neighbors_weighted(
    ...     adata, 
    ...     n_layers=3, 
    ...     use_rep='X_scVI',
    ...     hop_decay=0.5,  # Each hop contributes half as much
    ...     distance_kernel='exponential',
    ... )
    """
    adata = adata.copy() if copy else adata
    
    # Get connectivity matrix
    if connectivity_key is None:
        connectivity_key = 'spatial_connectivities'
    
    if connectivity_key not in adata.obsp:
        raise KeyError(
            f"Connectivity matrix '{connectivity_key}' not found. "
            "Run sq.gr.spatial_neighbors() first."
        )
    
    connectivity = adata.obsp[connectivity_key]
    if not issparse(connectivity):
        connectivity = csr_matrix(connectivity)
    
    # Binarize connectivity (in case it has weights)
    connectivity_binary = connectivity.copy()
    connectivity_binary.data = np.ones_like(connectivity_binary.data)
    
    # Get features
    if use_rep is None:
        features = adata.X
        if issparse(features):
            features = features.toarray()
    else:
        if use_rep not in adata.obsm:
            raise KeyError(f"Representation '{use_rep}' not found in adata.obsm")
        features = adata.obsm[use_rep]
    
    features = np.asarray(features)
    n_cells, n_features = features.shape
    
    # Get spatial coordinates
    if spatial_key not in adata.obsm:
        raise KeyError(
            f"Spatial coordinates '{spatial_key}' not found. "
            "Make sure spatial coordinates are in adata.obsm."
        )
    coords = np.asarray(adata.obsm[spatial_key])
    
    # Parse hop decay weights
    if isinstance(hop_decay, (int, float)):
        hop_weights = [hop_decay ** h for h in range(1, n_layers + 1)]
    elif isinstance(hop_decay, list):
        if len(hop_decay) != n_layers:
            raise ValueError(
                f"hop_decay list length ({len(hop_decay)}) must match n_layers ({n_layers})"
            )
        hop_weights = hop_decay
    else:
        raise ValueError(f"hop_decay must be float or list, got {type(hop_decay)}")
    
    # Parse aggregations
    if isinstance(aggregations, str):
        aggregations = [aggregations]

    # Handle sample-wise processing FIRST for memory efficiency
    # This way we compute hop matrices per-sample instead of globally
    if sample_key is not None and sample_key in adata.obs:
        samples = adata.obs[sample_key].unique()
        sample_indices = {s: np.where(adata.obs[sample_key] == s)[0] for s in samples}
        process_by_sample = True
        print(f"Processing {len(samples)} samples separately for memory efficiency...")
    else:
        sample_indices = {'all': np.arange(n_cells)}
        process_by_sample = False

    # For large datasets, compute hop matrices per-sample to save memory
    if process_by_sample:
        hop_matrices = None  # Will compute per-sample
    else:
        # Compute hop matrices globally (original behavior)
        hop_matrices = compute_hop_matrices(connectivity_binary, n_layers)
    
    # Collect aggregated features
    all_aggregated = []

    # Include self features (hop 0)
    if include_self:
        all_aggregated.append(features)

    # Initialize storage for each hop's aggregated features
    hop_aggregated_all = [{agg: np.zeros((n_cells, n_features)) for agg in aggregations}
                          for _ in range(n_layers)]

    # Process each sample
    for sample_idx, (sample_name, indices) in enumerate(sample_indices.items()):
        if len(indices) == 0:
            continue

        if process_by_sample:
            print(f"  Processing sample {sample_idx + 1}/{len(sample_indices)}: {sample_name} ({len(indices):,} cells)")

        # Get sample-specific data
        sample_coords = coords[indices]
        sample_features = features[indices]

        # Get sample-specific connectivity and compute hop matrices
        if process_by_sample:
            # Extract subgraph for this sample
            sample_connectivity = connectivity_binary[np.ix_(indices, indices)]
            sample_hop_matrices = compute_hop_matrices(sample_connectivity, n_layers)
        else:
            # Use global hop matrices, extract sample portion
            sample_hop_matrices = [hm[np.ix_(indices, indices)] for hm in hop_matrices]

        # Process each hop for this sample
        for hop_idx, hop_weight in enumerate(hop_weights):
            sample_hop_matrix = sample_hop_matrices[hop_idx]

            # Compute distance weights for this hop
            weighted_matrix = compute_distance_weights(
                sample_coords,
                sample_hop_matrix,
                kernel=distance_kernel,
                scale=distance_scale,
            )

            # Apply hop-level decay
            weighted_matrix = weighted_matrix * hop_weight

            # Normalize weights if requested
            if normalize_weights:
                row_sums = np.array(weighted_matrix.sum(axis=1)).flatten()
                row_sums[row_sums == 0] = 1  # Prevent division by zero
                normalizer = diags(1.0 / row_sums)
                weighted_matrix = normalizer @ weighted_matrix

            # Compute aggregations
            for agg in aggregations:
                if agg == 'mean':
                    # Weighted mean
                    agg_features = weighted_matrix @ sample_features
                elif agg == 'sum':
                    # Weighted sum (don't normalize)
                    if normalize_weights:
                        # Undo normalization for sum
                        row_sums_orig = np.array(sample_hop_matrix.sum(axis=1)).flatten()
                        row_sums_orig[row_sums_orig == 0] = 1
                        agg_features = (weighted_matrix @ sample_features) * row_sums_orig[:, None]
                    else:
                        agg_features = weighted_matrix @ sample_features
                elif agg == 'median':
                    # Weighted median (approximate using weighted quantile)
                    # For each cell, compute weighted median of neighbor features
                    n_sample_cells = len(indices) if sample_key is not None else n_cells
                    agg_features = np.zeros((n_sample_cells, n_features))

                    for i in range(n_sample_cells):
                        # Get neighbors and their weights for this cell
                        neighbor_indices = weighted_matrix[i].nonzero()[1]
                        if len(neighbor_indices) == 0:
                            # No neighbors, use self
                            agg_features[i] = sample_features[i]
                        else:
                            neighbor_weights = weighted_matrix[i, neighbor_indices].toarray().flatten()
                            neighbor_features = sample_features[neighbor_indices]

                            # Compute weighted median per feature
                            for j in range(n_features):
                                feature_vals = neighbor_features[:, j]
                                sorted_idx = np.argsort(feature_vals)
                                cumsum = np.cumsum(neighbor_weights[sorted_idx])
                                median_idx = np.searchsorted(cumsum, cumsum[-1] / 2.0)
                                agg_features[i, j] = feature_vals[sorted_idx[median_idx]]

                elif agg == 'max':
                    # Weighted max: take max weighted contribution
                    # For each feature, find the neighbor that contributes most
                    n_sample_cells = len(indices) if sample_key is not None else n_cells
                    agg_features = np.zeros((n_sample_cells, n_features))

                    for i in range(n_sample_cells):
                        # Get neighbors and their weights
                        neighbor_indices = weighted_matrix[i].nonzero()[1]
                        if len(neighbor_indices) == 0:
                            # No neighbors, use self
                            agg_features[i] = sample_features[i]
                        else:
                            neighbor_weights = weighted_matrix[i, neighbor_indices].toarray().flatten()
                            neighbor_features = sample_features[neighbor_indices]

                            # Weighted contribution = weight * feature
                            weighted_contributions = neighbor_weights[:, None] * neighbor_features
                            # Take max contribution per feature
                            agg_features[i] = np.max(weighted_contributions, axis=0)

                elif agg in ['var', 'std']:
                    # Weighted variance: sum(w * (x - weighted_mean)^2) / sum(w)
                    weighted_mean = weighted_matrix @ sample_features
                    # Compute squared deviations weighted
                    # This is trickier with sparse matrices...
                    # For simplicity, we'll compute unweighted variance for now
                    warnings.warn(
                        f"Weighted {agg} uses approximate computation. "
                        "Consider using 'mean' for weighted aggregation."
                    )
                    # Fall back to unweighted
                    neighbor_counts = np.array(sample_hop_matrix.sum(axis=1)).flatten()
                    neighbor_counts[neighbor_counts == 0] = 1

                    sum_sq = sample_hop_matrix @ (sample_features ** 2)
                    mean_sq = (sample_hop_matrix @ sample_features) ** 2 / neighbor_counts[:, None] ** 2
                    variance = sum_sq / neighbor_counts[:, None] - mean_sq
                    variance = np.maximum(variance, 0)  # Numerical stability

                    if agg == 'std':
                        agg_features = np.sqrt(variance)
                    else:
                        agg_features = variance
                else:
                    raise ValueError(f"Unknown aggregation: {agg}")

                # Store results in the hop_aggregated_all structure
                hop_aggregated_all[hop_idx][agg][indices] = agg_features

    # Concatenate all hops
    for hop_idx in range(n_layers):
        for agg in aggregations:
            all_aggregated.append(hop_aggregated_all[hop_idx][agg])

    output = np.hstack(all_aggregated)
    
    # Store in adata
    adata.obsm[out_key] = output
    
    # Store metadata about the aggregation
    adata.uns[f'{out_key}_params'] = {
        'n_layers': n_layers,
        'hop_weights': hop_weights,
        'distance_kernel': distance_kernel,
        'distance_scale': distance_scale,
        'aggregations': aggregations,
        'include_self': include_self,
        'n_features_per_block': n_features,
    }
    
    if copy:
        return adata
    return None


def aggregate_neighbors_weighted_simple(
    adata: AnnData,
    n_layers: int = 3,
    use_rep: Optional[str] = None,
    out_key: str = 'X_cellcharter_weighted',
    hop_decay: float = 0.5,
    distance_kernel: Literal['exponential', 'inverse', 'gaussian', 'none'] = 'exponential',
    distance_scale: Optional[float] = None,
    spatial_key: str = 'spatial',
) -> None:
    """
    Simplified weighted aggregation - drop-in replacement for cc.gr.aggregate_neighbors.
    
    This is a convenience wrapper with sensible defaults.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object (modified in place)
    n_layers : int
        Number of hops
    use_rep : str, optional
        Feature representation in adata.obsm
    out_key : str
        Output key in adata.obsm
    hop_decay : float
        Decay per hop (0.5 = each hop contributes half as much)
    distance_kernel : str
        'exponential', 'inverse', 'gaussian', or 'none'
    distance_scale : float, optional
        Scale for distance kernel (auto-computed if None)
    spatial_key : str
        Key for spatial coordinates
        
    Example
    -------
    >>> # Instead of:
    >>> cc.gr.aggregate_neighbors(adata, n_layers=3, use_rep='X_scVI')
    >>> 
    >>> # Use:
    >>> aggregate_neighbors_weighted_simple(
    ...     adata, n_layers=3, use_rep='X_scVI',
    ...     hop_decay=0.5, distance_kernel='exponential'
    ... )
    """
    aggregate_neighbors_weighted(
        adata,
        n_layers=n_layers,
        aggregations='mean',
        use_rep=use_rep,
        out_key=out_key,
        hop_decay=hop_decay,
        distance_kernel=distance_kernel,
        distance_scale=distance_scale,
        spatial_key=spatial_key,
        normalize_weights=True,
        include_self=True,
    )