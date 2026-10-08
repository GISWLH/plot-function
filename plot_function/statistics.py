"""Spatial summaries with explicit weighting and significance decisions."""

import numpy as np
import xarray as xr


def field_weights(data, weights=None):
    """Return aligned weights. NaN weights exclude cells; negative/inf weights fail."""
    if weights is None:
        return xr.ones_like(data, dtype=float)
    if isinstance(weights, str):
        if weights != "coslat":
            raise ValueError("weights must be None, 'coslat', or an aligned DataArray.")
        return np.cos(np.deg2rad(data.lat)).clip(min=0).broadcast_like(data)
    if not isinstance(weights, xr.DataArray) or set(weights.dims) != set(data.dims):
        raise ValueError(
            "Explicit weights must be a DataArray with the field's lat/lon dimensions."
        )
    try:
        _, weights = xr.align(data, weights, join="exact")
    except ValueError as exc:
        raise ValueError("Weight coordinates must exactly match the normalized field.") from exc
    if bool(((weights < 0) | np.isinf(weights)).any()):
        raise ValueError("Weights cannot be negative or infinite.")
    return weights.transpose(*data.dims).fillna(0)


def spatial_profile(data, *, coordinate="lat", statistic="mean", spread="std", weights=None):
    """Summarize rows/columns; bands describe spatial spread, never sampling uncertainty.

    ``coordinate='lat'`` retains latitude (a zonal profile); ``'lon'`` retains
    longitude. Median and interquartile bands currently require equal cell weights.
    """
    if coordinate not in ("lat", "lon"):
        raise ValueError("coordinate must be 'lat' or 'lon'.")
    if statistic not in ("mean", "median") or spread not in ("std", "iqr", None):
        raise ValueError("Choose statistic='mean'/'median' and spread='std'/'iqr'/None.")
    if weights is not None and (statistic == "median" or spread == "iqr"):
        raise ValueError("Median and IQR profiles require equal cell weights (weights=None).")
    other = "lon" if coordinate == "lat" else "lat"
    w = field_weights(data, weights).where(np.isfinite(data), 0)
    total = w.sum(other)
    center = (data.fillna(0) * w).sum(other) / total.where(total > 0)
    if statistic == "median":
        center = data.where(w > 0).median(other, skipna=True)
    if spread == "std":
        # Population spatial spread about the weighted mean, not standard error.
        mean = (data.fillna(0) * w).sum(other) / total.where(total > 0)
        std = np.sqrt((((data - mean) ** 2).fillna(0) * w).sum(other) / total.where(total > 0))
        lower, upper = center - std, center + std
    elif spread == "iqr":
        q = data.where(w > 0).quantile([0.25, 0.75], dim=other, skipna=True)
        lower, upper = q.sel(quantile=0.25, drop=True), q.sel(quantile=0.75, drop=True)
    else:
        lower = upper = center
    return xr.Dataset(
        {
            "center": center,
            "lower": lower,
            "upper": upper,
            "count": (w > 0).sum(other),
            "weight_sum": total,
        },
        attrs={
            "statistic": statistic,
            "spread": spread or "none",
            "weighting": "equal cells"
            if weights is None
            else (weights if isinstance(weights, str) else "explicit"),
        },
    )


def distribution_values(data, weights=None):
    w = field_weights(data, weights).values.ravel()
    values = data.values.ravel()
    valid = np.isfinite(values) & (w > 0)
    if not valid.any():
        raise ValueError("No finite, positive-weight cells are available for the distribution.")
    return values[valid], w[valid]


def histogram(data, *, bins=20, weights=None, density=True):
    """Weighted histogram; density integrates to one over the supplied bin range."""
    values, w = distribution_values(data, weights)
    if np.isscalar(bins):
        if not isinstance(bins, (int, np.integer)) or bins < 1:
            raise ValueError("bins must be a positive integer or increasing finite edges.")
    else:
        bins = np.asarray(bins, dtype=float)
        if (
            bins.ndim != 1
            or len(bins) < 2
            or not np.isfinite(bins).all()
            or np.any(np.diff(bins) <= 0)
        ):
            raise ValueError("Bin edges must be finite and strictly increasing.")
    mass, edges = np.histogram(values, bins=bins, weights=w)
    if not mass.sum() > 0:
        raise ValueError("The requested histogram bins contain no positive-weight observations.")
    height = mass / (mass.sum() * np.diff(edges)) if density else mass
    return xr.Dataset(
        {
            "height": ("bin", height),
            "mass": ("bin", mass),
            "left": ("bin", edges[:-1]),
            "right": ("bin", edges[1:]),
        },
        coords={"bin": (edges[:-1] + edges[1:]) / 2},
        attrs={
            "density": bool(density),
            "sample_count": len(values),
            "mean": float(np.average(values, weights=w)),
            "weighting": "equal cells"
            if weights is None
            else (weights if isinstance(weights, str) else "explicit"),
        },
    )


def significance_mask(pvalues, *, alpha=0.05, correction="none", valid=None):
    """Threshold p-values, optionally controlling BH FDR within the valid family.

    NaNs are excluded. All other p-values must lie in [0, 1]. FDR decisions are
    computed before any visual thinning. Users must choose an appropriate test
    and family; BH's assumptions are not guaranteed for arbitrary spatial dependence.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between 0 and 1.")
    if correction not in ("none", "fdr_bh"):
        raise ValueError("correction must be 'none' or 'fdr_bh'.")
    if bool((np.isinf(pvalues) | (pvalues < 0) | (pvalues > 1)).any()):
        raise ValueError("P-values must be in [0, 1] or NaN.")
    if valid is not None:
        pvalues, valid = xr.align(pvalues, valid, join="exact")
        pvalues = pvalues.where(valid)
    values = pvalues.values[np.isfinite(pvalues.values)]
    cutoff = alpha
    if correction == "fdr_bh":
        ordered = np.sort(values)
        passing = ordered <= alpha * np.arange(1, len(ordered) + 1) / max(len(ordered), 1)
        cutoff = float(ordered[passing][-1]) if passing.any() else -1.0
    mask = np.isfinite(pvalues) & (pvalues <= cutoff)
    mask.attrs = {
        "alpha": alpha,
        "correction": correction,
        "tested_cells": len(values),
        "significant_cells": int(mask.sum()),
        "critical_p": float(cutoff) if cutoff >= 0 else np.nan,
    }
    return mask
