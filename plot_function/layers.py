"""Composable map layers; all statistical outputs remain available to the caller."""

from dataclasses import dataclass, field

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import numpy as np
import xarray as xr
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from .data import _select_variable, open_field
from .legacy import _cyclic_if_global
from .options import Distribution, MapFeatures, Profile, Significance
from .statistics import (
    distribution_values,
    field_weights,
    histogram,
    significance_mask,
    spatial_profile,
)


@dataclass
class LayerResult:
    axes: object
    statistics: object = None
    artists: list = field(default_factory=list)


def _scope(result, weights=None):
    data = result.data
    if isinstance(weights, xr.DataArray):
        weights = field_weights(data, weights)
    if result.extent is not None:
        west, east, south, north = result.extent
        data = data.sel(lon=slice(west, east), lat=slice(south, north))
        if isinstance(weights, xr.DataArray):
            weights = weights.sel(lon=data.lon, lat=data.lat)
    if data.size == 0 or not bool(data.notnull().any()):
        raise ValueError("The map extent contains no finite grid-cell centers for summaries.")
    return data, weights


def _style_axes(ax, result):
    ink = "#163438" if result.theme == "light" else "#edf5ef"
    muted = "#647b80" if result.theme == "light" else "#a3bec2"
    for name, spine in ax.spines.items():
        spine.set_visible(name in ("left", "bottom"))
        spine.set_color(muted)
        spine.set_linewidth(0.55)
    ax.set_facecolor("white" if result.theme == "light" else "#10252e")
    ax.tick_params(labelsize=7, colors=muted, length=2, width=0.5)
    ax.xaxis.label.set_color(ink)
    ax.yaxis.label.set_color(ink)
    return ink, muted


def add_profile(result, options=None, **kwargs):
    spec = options if options is not None else Profile(**kwargs)
    if not isinstance(spec, Profile):
        raise TypeError("Profile options must be a Profile instance.")
    if spec.position not in ("right", "top") or spec.style not in ("line", "band", "bars"):
        raise ValueError("Profiles use position='right'/'top' and style='line'/'band'/'bars'.")
    if spec.position in result.profiles:
        raise ValueError(f"A {spec.position} profile already exists on this map.")
    if not np.isfinite([spec.width, spec.pad]).all() or spec.width <= 0 or spec.pad < 0:
        raise ValueError("Profile width must be positive and pad nonnegative.")
    if spec.reference is not None and not np.isfinite(spec.reference):
        raise ValueError("Profile reference must be finite.")
    data, weights = _scope(result, spec.weights)
    right = spec.position == "right"
    coordinate = "lat" if right else "lon"
    spread = spec.spread if spec.style == "band" else None
    stats = spatial_profile(
        data, coordinate=coordinate, statistic=spec.statistic, spread=spread, weights=weights
    )
    if not bool(np.isfinite(stats.center).any()):
        raise ValueError("The profile has no finite positive-weight samples.")
    bounds = [1 + spec.pad, 0, spec.width, 1] if right else [0, 1 + spec.pad, 1, spec.width]
    ax = result.axes.inset_axes(bounds, transform=result.axes.transAxes, zorder=6)
    ink, muted = _style_axes(ax, result)
    coord, center = stats[coordinate].values, stats.center.values
    artists = []
    if spec.style == "band" and spread is not None:
        fill = ax.fill_betweenx if right else ax.fill_between
        artists.append(
            fill(coord, stats.lower, stats.upper, color=spec.color, alpha=0.16, linewidth=0)
        )
    if spec.style == "bars":
        spacing = float(np.min(np.diff(coord))) * 0.75 if len(coord) > 1 else 0.6
        if right:
            artists.extend(ax.barh(coord, center, height=spacing, color=spec.color, alpha=0.75))
        else:
            artists.extend(ax.bar(coord, center, width=spacing, color=spec.color, alpha=0.75))
    else:
        artists.extend(
            ax.plot(center, coord, color=spec.color, lw=1.35)
            if right
            else ax.plot(coord, center, color=spec.color, lw=1.35)
        )
    if spec.reference is not None:
        reference = ax.axvline if right else ax.axhline
        artists.append(reference(spec.reference, color=muted, lw=0.65, ls="--"))
    if result.extent is not None:
        limits = result.extent[2:] if right else result.extent[:2]
    else:
        limits = [float(coord.min()), float(coord.max())]
    if limits[0] == limits[1]:
        limits = [limits[0] - 0.5, limits[1] + 0.5]
    unit = data.attrs.get("units", "")
    if right:
        ax.set_ylim(limits)
        ax.yaxis.tick_right()
        ax.spines["right"].set_visible(True)
        ax.spines["left"].set_visible(False)
        ax.set_xlabel(unit, fontsize=8)
        ax.yaxis.set_label_position("right")
        ax.set_ylabel("Latitude / °", fontsize=8, labelpad=5)
        ax.xaxis.set_major_locator(MaxNLocator(3))
    else:
        ax.set_xlim(limits)
        ax.set_ylabel(unit, fontsize=8)
        ax.tick_params(axis="x", labelbottom=False)
        ax.yaxis.set_major_locator(MaxNLocator(3))
        if result._title is not None:
            result._title = result.axes.set_title(
                result._title.get_text(),
                loc="left",
                fontproperties=result._title.get_fontproperties(),
                color=result._title.get_color(),
                y=1 + spec.pad + spec.width + 0.07,
                pad=28 if result._subtitle is not None else 14,
            )
        if result._subtitle is not None:
            result._subtitle.set_y(1 + spec.pad + spec.width + 0.07)
    band_label = {"std": " ±1 spatial SD", "iqr": " · spatial IQR", None: ""}[spread]
    title = spec.label if spec.label is not None else spec.statistic.capitalize() + band_label
    ax.set_title(title, loc="left", fontsize=8, color=ink, pad=7)
    layer = LayerResult(ax, stats, artists)
    result.profiles[spec.position] = layer
    return layer


def add_distribution(result, options=None, **kwargs):
    spec = options if options is not None else Distribution(**kwargs)
    if not isinstance(spec, Distribution):
        raise TypeError("Distribution options must be a Distribution instance.")
    if result.distribution is not None:
        raise ValueError("This map already has a distribution inset.")
    if spec.style not in ("bars", "step", "line", "ecdf"):
        raise ValueError("Distribution style must be 'bars', 'step', 'line', or 'ecdf'.")
    if len(spec.bounds) != 4 or not np.isfinite(spec.bounds).all():
        raise ValueError("Inset bounds must be four finite axes fractions.")
    x, y, w, h = spec.bounds
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > 1 or y + h > 1:
        raise ValueError("Inset bounds must lie within the map axes with positive width/height.")
    data, weights = _scope(result, spec.weights)
    values, sample_weights = distribution_values(data, weights)
    if spec.style == "ecdf":
        order = np.argsort(values)
        xx, yy = values[order], np.cumsum(sample_weights[order]) / sample_weights.sum()
        stats = xr.Dataset(
            {"value": ("sample", xx), "cumulative": ("sample", yy)},
            attrs={
                "mean": float(np.average(values, weights=sample_weights)),
                "sample_count": len(values),
            },
        )
    else:
        stats = histogram(data, bins=spec.bins, weights=weights, density=spec.density)
    ax = result.axes.inset_axes(spec.bounds, transform=result.axes.transAxes, zorder=8)
    ink, muted = _style_axes(ax, result)
    # The solid white panel separates the data summary from the map beneath it.
    ax.patch.set_alpha(0.96)
    artists = []
    color = spec.color if spec.color != "map" else "#267f83"
    if spec.style == "bars":
        colors = (
            result.artist.cmap(result.artist.norm(stats.bin.values))
            if spec.color == "map"
            else color
        )
        artists.extend(
            ax.bar(
                stats.bin,
                stats.height,
                width=(stats.right - stats.left) * 0.88,
                color=colors,
                edgecolor="none",
                alpha=0.9,
            )
        )
    elif spec.style == "step":
        edges = np.r_[stats.left.values, stats.right.values[-1]]
        artists.append(ax.stairs(stats.height, edges, color=color, lw=1.2, fill=False))
    elif spec.style == "line":
        artists.extend(ax.plot(stats.bin, stats.height, color=color, lw=1.35))
        artists.append(ax.fill_between(stats.bin, stats.height, color=color, alpha=0.10))
    else:
        artists.extend(
            ax.step(stats.value, stats["cumulative"], where="post", color=color, lw=1.35)
        )
    if spec.show_mean:
        artists.append(ax.axvline(stats.attrs["mean"], color=ink, lw=0.75, ls=(0, (3, 2))))
        ax.text(
            0.98,
            0.90,
            f"Mean {stats.attrs['mean']:.2g}",
            ha="right",
            va="top",
            transform=ax.transAxes,
            fontsize=6.5,
            color=ink,
        )
    weighted = spec.weights is not None
    label = (
        "Cumulative fraction"
        if spec.style == "ecdf"
        else ("Density" if spec.density else ("Weight" if weighted else "Cell count"))
    )
    ax.set_title(
        spec.label or ("Weighted distribution" if weighted else "Cell distribution"),
        loc="left",
        fontsize=8,
        color=ink,
        pad=5,
    )
    ax.set_xlabel(data.attrs.get("units", ""), fontsize=7, labelpad=2)
    ax.set_ylabel(label, fontsize=7, labelpad=3)
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.tick_params(labelsize=6.5)
    ax.set_ylim(bottom=0)
    if spec.style == "ecdf":
        ax.set_ylim(0, 1.04)
    layer = LayerResult(ax, stats, artists)
    result.distribution = layer
    return layer


def add_significance(result, options=None, **kwargs):
    spec = options if options is not None else Significance(**kwargs)
    if not isinstance(spec, Significance):
        raise TypeError("Significance options must be a Significance instance.")
    if spec.kind not in ("pvalue", "mask") or spec.style not in ("stipple", "hatch", "contour"):
        raise ValueError("Use kind='pvalue'/'mask' and style='stipple'/'hatch'/'contour'.")
    if not isinstance(spec.stride, (int, np.integer)) or spec.stride < 1:
        raise ValueError("Significance stride must be a positive integer.")
    if spec.kind == "mask" and spec.correction != "none":
        raise ValueError("FDR correction requires p-values, not a precomputed mask.")

    def load_overlay(source):
        selected = _select_variable(source, spec.variable)
        if spec.kind == "mask" and selected.dtype == bool:
            selected = selected.astype(float)
        return open_field(
            selected, sel=spec.sel, isel=spec.isel, latitude=spec.latitude, longitude=spec.longitude
        )

    if isinstance(spec.data, (xr.DataArray, xr.Dataset)):
        overlay = load_overlay(spec.data)
    else:
        with xr.open_dataset(spec.data) as source:
            overlay = load_overlay(source)
    try:
        _, overlay = xr.align(result.data, overlay, join="exact")
    except ValueError as exc:
        raise ValueError(
            "Significance coordinates must exactly match the map; no interpolation is performed."
        ) from exc
    data, _ = _scope(result)
    overlay = overlay.sel(lat=data.lat, lon=data.lon)
    if spec.kind == "pvalue":
        mask = significance_mask(
            overlay, alpha=spec.alpha, correction=spec.correction, valid=np.isfinite(data)
        )
        label = (
            f"BH FDR q={spec.alpha:g}"
            if spec.correction == "fdr_bh"
            else f"p ≤ {spec.alpha:g} (unadjusted)"
        )
    else:
        if bool((overlay.notnull() & (overlay != 0) & (overlay != 1)).any()):
            raise ValueError("A significance mask must contain only 0, 1, or NaN.")
        mask = (overlay == 1) & np.isfinite(data)
        mask.attrs = {"significant_cells": int(mask.sum()), "kind": "supplied mask"}
        label = "Supplied significance mask"
    artists = []
    ax = result.axes
    if spec.style != "stipple" and min(mask.sizes.values()) < 2:
        raise ValueError("Hatching and significance contours need at least two rows and columns.")
    if spec.style == "stipple":
        shown = mask.isel(lat=slice(None, None, spec.stride), lon=slice(None, None, spec.stride))
        lon, lat = np.meshgrid(shown.lon, shown.lat)
        if bool(shown.any()):
            artists.append(
                ax.scatter(
                    lon[shown.values],
                    lat[shown.values],
                    s=spec.size,
                    color=spec.color,
                    linewidths=0,
                    alpha=0.8,
                    transform=ccrs.PlateCarree(),
                    zorder=5,
                )
            )
        handle = Line2D([], [], ls="none", marker=".", color=spec.color, markersize=4)
    elif spec.style == "hatch":
        if bool(mask.any()):
            field = _cyclic_if_global(mask.astype(float))
            with mpl.rc_context({"hatch.color": spec.color, "hatch.linewidth": 0.45}):
                contour = ax.contourf(
                    field.lon,
                    field.lat,
                    field,
                    levels=[0.5, 1.5],
                    colors="none",
                    hatches=[spec.hatch],
                    transform=ccrs.PlateCarree(),
                    zorder=5,
                )
            collections = [contour] if hasattr(contour, "set_edgecolor") else contour.collections
            for collection in collections:
                collection.set_edgecolor(spec.color)
                collection.set_linewidth(0)
                if hasattr(collection, "set_hatch_linewidth"):
                    collection.set_hatch_linewidth(0.45)
            artists.append(contour)
        handle = Patch(facecolor="none", edgecolor=spec.color, hatch=spec.hatch, linewidth=0.5)
    else:
        if bool(mask.any()) and not bool(mask.all()):
            field = _cyclic_if_global(mask.astype(float))
            artists.append(
                ax.contour(
                    field.lon,
                    field.lat,
                    field,
                    levels=[0.5],
                    colors=[spec.color],
                    linewidths=0.85,
                    transform=ccrs.PlateCarree(),
                    zorder=5,
                )
            )
        handle = Line2D([], [], color=spec.color, linewidth=0.85)
    if spec.legend:
        legend = ax.legend(
            [handle],
            [spec.label or label],
            loc="upper right",
            fontsize=7,
            frameon=True,
            facecolor="white",
            edgecolor="0.85",
            framealpha=0.96,
        )
        legend.set_zorder(10)
        artists.append(legend)
    layer = LayerResult(ax, mask, artists)
    result.significance.append(layer)
    return layer


def add_features(result, options=None, **kwargs):
    spec = options if options is not None else MapFeatures(**kwargs)
    if not isinstance(spec, MapFeatures):
        raise TypeError("Feature options must be a MapFeatures instance.")
    if spec.resolution not in ("110m", "50m", "10m"):
        raise ValueError("Feature resolution must be '110m', '50m', or '10m'.")
    if not np.isfinite(spec.linewidth) or spec.linewidth <= 0:
        raise ValueError("Feature linewidth must be positive and finite.")
    styles = {
        "land": (cfeature.LAND, dict(facecolor=spec.land_color, edgecolor="none", zorder=0)),
        "ocean": (cfeature.OCEAN, dict(facecolor=spec.ocean_color, edgecolor="none", zorder=0)),
        "lakes": (
            cfeature.LAKES,
            dict(
                facecolor=spec.lake_color,
                edgecolor=spec.river_color,
                linewidth=spec.linewidth * 0.6,
                zorder=3,
            ),
        ),
        "rivers": (
            cfeature.RIVERS,
            dict(facecolor="none", edgecolor=spec.river_color, linewidth=spec.linewidth, zorder=3),
        ),
        "borders": (
            cfeature.BORDERS,
            dict(
                facecolor="none",
                edgecolor=spec.border_color,
                linewidth=spec.linewidth,
                linestyle=(0, (3, 2)),
                zorder=3,
            ),
        ),
    }
    artists = [
        result.axes.add_feature(feature.with_scale(spec.resolution), **style)
        for name, (feature, style) in styles.items()
        if getattr(spec, name)
    ]
    result.feature_artists.extend(artists)
    return artists
