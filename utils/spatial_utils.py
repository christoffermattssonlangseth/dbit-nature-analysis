"""Compact spatial plotting utilities for AnnData objects.

Adapted from `baloMS/utils/spatial_utils.py` for reuse in this repository.
"""

from __future__ import annotations

from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd
from anndata import AnnData
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from pandas.api.types import is_categorical_dtype, is_numeric_dtype
from scipy.sparse import issparse


def _default_categorical_palette(n_colors: int) -> list[str]:
    """Return a deterministic categorical palette with at least n colors."""
    pools = []
    for cmap_name in ("tab20", "tab20b", "tab20c"):
        cmap = plt.get_cmap(cmap_name)
        pools.extend([mcolors.to_hex(cmap(i)) for i in range(cmap.N)])
    if n_colors <= len(pools):
        return pools[:n_colors]
    reps = int(np.ceil(n_colors / len(pools)))
    return (pools * reps)[:n_colors]


def plot_spatial_compact_fast(
    ad: AnnData,
    color: str = "leiden",
    groupby: str = "sample_id",
    spot_size: float = 8,
    cols: int = 3,
    height: float = 8,
    legend_col_width: float = 1.2,
    palette: Optional[Union[dict, list, str]] = None,
    rasterized: bool = True,
    invert_y: bool = True,
    dpi: int = 120,
    highlight: Optional[Union[str, Iterable[str]]] = None,
    group_order: Optional[Iterable[str]] = None,
    background: str = "white",
    grey_alpha: float = 0.2,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap_name: str = "viridis",
    shared_scale: bool = False,
):
    """Plot compact faceted spatial maps (one panel per `groupby` value).

    `color` can be either:
    - `ad.obs` column (categorical or numeric), or
    - gene name in `ad.var_names` (continuous expression).
    """
    if "spatial" not in ad.obsm:
        raise ValueError("ad.obsm['spatial'] not found.")
    if groupby not in ad.obs.columns:
        raise KeyError(f"groupby '{groupby}' not found in ad.obs.")

    fig_face = background
    ax_face = background
    text_color = "white" if background in ("black", "#000000", "k") else "black"

    if color in ad.obs.columns:
        color_source = "obs"
        col_series = ad.obs[color]
    elif color in ad.var_names:
        color_source = "var"
        col_series = None
    else:
        raise KeyError(f"'{color}' not found in ad.obs.columns or ad.var_names.")

    coords = np.asarray(ad.obsm["spatial"])[:, :2]

    if color_source == "var":
        is_continuous = True
    elif is_categorical_dtype(col_series):
        is_continuous = False
    else:
        is_continuous = is_numeric_dtype(col_series)

    if is_continuous:
        if color_source == "obs":
            vals = pd.to_numeric(col_series, errors="coerce").to_numpy()
        else:
            gene_idx = ad.var_names.get_loc(color)
            x = ad.X[:, gene_idx]
            vals = x.toarray().ravel() if issparse(x) else np.asarray(x).ravel()

        cmap = plt.get_cmap(cmap_name) if palette is None else (
            plt.get_cmap(palette) if isinstance(palette, str) else palette
        )

        if shared_scale:
            full_vals = vals
            finite_full = np.isfinite(full_vals)
            if finite_full.sum() == 0:
                raise ValueError(f"All values for '{color}' are NaN or non-finite.")
            vmin_use = float(np.min(full_vals[finite_full]))
            vmax_use = float(np.max(full_vals[finite_full]))
        else:
            finite_mask = np.isfinite(vals)
            if finite_mask.sum() == 0:
                raise ValueError(f"All values for '{color}' are NaN or non-finite.")
            vmin_use = float(np.min(vals[finite_mask]))
            vmax_use = float(np.max(vals[finite_mask]))

        if vmin is not None:
            vmin_use = float(vmin)
        if vmax is not None:
            vmax_use = float(vmax)
        if vmin_use == vmax_use:
            vmin_use -= 1.0
            vmax_use += 1.0

        norm = mcolors.Normalize(vmin=vmin_use, vmax=vmax_use)
        colors_arr = np.zeros((vals.size, 4), dtype=float)
        finite_mask = np.isfinite(vals)
        colors_arr[finite_mask] = cmap(norm(vals[finite_mask]))
        colors_arr[~finite_mask] = (0, 0, 0, 0)

        ad.uns[f"{color}_continuous"] = {
            "vmin": float(vmin_use),
            "vmax": float(vmax_use),
            "cmap": cmap.name if hasattr(cmap, "name") else str(cmap),
        }

        cat_names = None
        col_list = None
    else:
        cats = col_series.cat.remove_unused_categories() if is_categorical_dtype(col_series) else col_series.astype("category")
        cat_names = cats.cat.categories
        cat_codes = cats.cat.codes.to_numpy()

        if isinstance(palette, dict):
            col_list = [palette[c] for c in cat_names]
        elif isinstance(palette, (list, tuple)):
            if len(palette) < len(cat_names):
                raise ValueError("Palette shorter than number of categories.")
            col_list = list(palette)[: len(cat_names)]
        elif f"{color}_colors" in ad.uns and len(ad.uns[f"{color}_colors"]) == len(cat_names):
            col_list = list(ad.uns[f"{color}_colors"])
        else:
            col_list = _default_categorical_palette(len(cat_names))

        ad.uns[f"{color}_colors"] = col_list

        rgba = np.array([mcolors.to_rgba(c) for c in col_list], dtype=float)
        colors_arr = np.empty((cat_codes.size, 4), dtype=float)
        colors_arr[cat_codes >= 0] = rgba[cat_codes[cat_codes >= 0]]
        colors_arr[cat_codes < 0] = (0, 0, 0, 0)

        if highlight is not None:
            if not isinstance(highlight, (list, tuple, set, np.ndarray)):
                highlight = [highlight]
            keep = {str(v) for v in highlight}
            cat_name_str = np.array([str(c) for c in cat_names])
            keep_cat_mask = np.isin(cat_name_str, list(keep))

            grey_rgba = (0.8, 0.8, 0.8, float(grey_alpha))
            valid = cat_codes >= 0
            keep_flag = np.zeros_like(cat_codes, dtype=bool)
            keep_flag[valid] = keep_cat_mask[cat_codes[valid]]

            colors_arr[valid & ~keep_flag] = grey_rgba
            col_list = [
                col_list[k] if keep_cat_mask[k] else mcolors.to_hex(grey_rgba)
                for k in range(len(cat_names))
            ]

    gser = ad.obs[groupby]
    if group_order is not None:
        group_order = [str(g) for g in group_order]
        present = set(gser.astype(str))
        uniq_groups = [g for g in group_order if g in present]
    elif is_categorical_dtype(gser) and gser.cat.ordered:
        cats = list(gser.cat.categories)
        present = set(gser.astype(str))
        uniq_groups = [str(c) for c in cats if str(c) in present]
    else:
        uniq_groups = sorted(gser.astype(str).unique())

    gvals = gser.astype(str).to_numpy()
    gid_to_idx = {g: i for i, g in enumerate(uniq_groups)}
    gcodes = np.array([gid_to_idx.get(g, -1) for g in gvals], dtype=int)
    group_indices = [np.flatnonzero(gcodes == gi) for gi in range(len(uniq_groups))]

    n = len(uniq_groups)
    rows = int(np.ceil(n / cols))
    panel_w = height * cols * 0.6 / rows
    fig_w = panel_w + legend_col_width

    plt.ioff()
    fig = plt.figure(figsize=(fig_w, height), dpi=dpi, constrained_layout=False)
    fig.patch.set_facecolor(fig_face)

    gs = GridSpec(
        rows,
        cols + 1,
        figure=fig,
        width_ratios=[1] * cols + [legend_col_width / (fig_w - legend_col_width)],
        wspace=0.02,
        hspace=0.02,
    )

    for i, group_val in enumerate(uniq_groups):
        r, c = divmod(i, cols)
        ax = fig.add_subplot(gs[r, c])
        ax.set_facecolor(ax_face)

        idx = group_indices[i]
        if idx.size:
            xy = coords[idx]
            y = -xy[:, 1] if invert_y else xy[:, 1]
            ax.scatter(
                xy[:, 0],
                y,
                c=colors_arr[idx],
                s=spot_size,
                marker="o",
                linewidths=0,
                rasterized=rasterized,
            )

        ax.set_title(str(group_val), fontsize=6, pad=2, color=text_color)
        ax.set_aspect("equal")
        if invert_y:
            ax.invert_yaxis()
        ax.set_axis_off()

    for j in range(n, rows * cols):
        r, c = divmod(j, cols)
        ax = fig.add_subplot(gs[r, c])
        ax.set_facecolor(ax_face)
        ax.axis("off")

    ax_leg = fig.add_subplot(gs[:, -1])
    ax_leg.set_facecolor(ax_face)
    ax_leg.axis("off")

    if is_continuous:
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=ax_leg)
        cbar.set_label(color, rotation=90, color=text_color)
        cbar.ax.yaxis.set_tick_params(color=text_color)
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=text_color)
    else:
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=col_list[k],
                markersize=7,
                label=str(cat),
            )
            for k, cat in enumerate(cat_names)
        ]
        leg = ax_leg.legend(
            handles=handles,
            title=color,
            frameon=False,
            loc="center left",
            labelcolor=text_color,
            title_fontsize=10,
        )
        leg.get_title().set_color(text_color)
        for text in leg.get_texts():
            text.set_color(text_color)

    fig.subplots_adjust(left=0.01, right=0.98, top=0.98, bottom=0.02, wspace=0.02, hspace=0.02)
    plt.ion()
    plt.show()
    return fig
