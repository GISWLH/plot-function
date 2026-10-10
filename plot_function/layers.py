"""Composable map layers; all statistical outputs remain available to the caller."""

from dataclasses import dataclass, field

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import numpy as np
import xarray as xr
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator

from . import journal as pj
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
    """Journal inset styling: transparent background, thin left/bottom spines."""
    light = result.theme == "light"
    ink = pj.INK if light else "#edf5ef"
    muted = pj.MUTED if light else "#a3bec2"
    for name, spine in ax.spines.items():
        spine.set_visible(name in ("left", "bottom"))
        spine.set_color(ink)
        spine.set_linewidth(0.5)
    ax.patch.set_alpha(0)
    ax.tick_params(labelsize=6, colors=ink, length=2, width=0.5, pad=1.5)
    ax.xaxis.label.set_color(ink)
    ax.yaxis.label.set_color(ink)
    return ink, muted


def _attach_draw_hook(child, sync):
    original = child.draw

    def draw(renderer):
        sync()
        return original(renderer)

    child.draw = draw
    sync()


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
    map_ax = result.axes
    bounds = [1 + spec.pad, 0, spec.width, 1] if right else [0, 1 + spec.pad, 1, spec.width]
    ax = map_ax.inset_axes(bounds, transform=map_ax.transAxes, zorder=6)
    ink, muted = _style_axes(ax, result)
    coord, center = stats[coordinate].values, stats.center.values
    # Push geographic coordinates through the map projection so that, e.g., 40°N on
    # the profile is level with 40°N on the map, for Robinson/Equal Earth as well.
    if result.extent is not None:
        west, east, south, north = result.extent
        ref = (west + east) / 2 if right else (south + north) / 2
    else:
        ref = None if right else 0.0
    try:
        pos = (
            pj._proj_coord(map_ax, lat=coord, ref=ref)
            if right
            else pj._proj_coord(map_ax, lon=coord, ref=ref)
        )
        aligned = bool(np.isfinite(pos).all())
    except Exception:  # pragma: no cover - exotic projections
        aligned = False
    if not aligned:
        pos = coord
    artists = []
    if spec.style == "band" and spread is not None:
        fill = ax.fill_betweenx if right else ax.fill_between
        artists.append(
            fill(pos, stats.lower, stats.upper, color=spec.color, alpha=0.22, linewidth=0)
        )
    if spec.style == "bars":
        spacing = float(np.min(np.abs(np.diff(pos)))) * 0.8 if len(pos) > 1 else 0.6
        if right:
            artists.extend(ax.barh(pos, center, height=spacing, color=spec.color, alpha=0.8))
        else:
            artists.extend(ax.bar(pos, center, width=spacing, color=spec.color, alpha=0.8))
    else:
        artists.extend(
            ax.plot(center, pos, color=spec.color, lw=1.0)
            if right
            else ax.plot(pos, center, color=spec.color, lw=1.0)
        )
    if spec.reference is not None:
        reference = ax.axvline if right else ax.axhline
        artists.append(reference(spec.reference, color=ink, lw=0.5, ls=(0, (2.5, 2))))
    unit = data.attrs.get("units", "")
    if aligned:
        if result.extent is not None:
            lo, hi = result.extent[2:] if right else result.extent[:2]
        else:
            lo, hi = (-90, 90) if right else (-180, 180)
        span = hi - lo
        step = next((v for v in (5, 10, 15, 20, 30, 60) if span / v <= 6), 60)
        ticks = np.arange(np.ceil(lo / step) * step, hi + 1e-9, step)
        if right:
            ticks = ticks[np.abs(ticks) < 90]
        tick_pos = (
            pj._proj_coord(map_ax, lat=ticks, ref=ref)
            if right
            else pj._proj_coord(map_ax, lon=ticks, ref=ref)
        )
        fmt = pj.format_lat if right else pj.format_lon
        axis = ax.yaxis if right else ax.xaxis
        axis.set_major_locator(FixedLocator(tick_pos))
        axis.set_major_formatter(
            FuncFormatter(lambda v, _p: fmt(ticks[int(np.argmin(np.abs(tick_pos - v)))]))
        )
        for t in tick_pos:
            (ax.axhline if right else ax.axvline)(t, color=pj.GRID, lw=0.35, ls=(0, (3, 3)), zorder=0)

        def sync():
            if right:
                ax.set_ylim(map_ax.get_ylim())
            else:
                ax.set_xlim(map_ax.get_xlim())

        _attach_draw_hook(ax, sync)
    else:
        limits = (
            (result.extent[2:] if right else result.extent[:2])
            if result.extent is not None
            else [float(coord.min()), float(coord.max())]
        )
        if limits[0] == limits[1]:
            limits = [limits[0] - 0.5, limits[1] + 0.5]
        (ax.set_ylim if right else ax.set_xlim)(limits)
    band_label = {"std": " ± 1 s.d.", "iqr": " (IQR)", None: ""}[spread]
    title = spec.label if spec.label is not None else spec.statistic.capitalize() + band_label
    if right:
        ax.yaxis.tick_right()
        ax.spines["right"].set_visible(True)
        ax.spines["left"].set_visible(False)
        # The description goes under the axis, where it cannot collide with titles.
        ax.set_xlabel(f"{title}\n({unit})" if unit else title, fontsize=6.5)
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.xaxis.set_major_formatter(pj.clean_formatter())
    else:
        ax.set_ylabel(unit, fontsize=7)
        ax.tick_params(axis="x", labelbottom=False, bottom=False)
        ax.yaxis.set_major_locator(MaxNLocator(3))
        if result._title is not None:
            result._title = result.axes.set_title(
                result._title.get_text(),
                loc="left",
                fontproperties=result._title.get_fontproperties(),
                color=result._title.get_color(),
                y=1 + spec.pad + spec.width + 0.08,
                pad=14 if result._subtitle is not None else 5,
            )
        if result._subtitle is not None:
            result._subtitle.set_y(1 + spec.pad + spec.width + 0.08)
        ax.set_title(title, loc="left", fontsize=6.5, color=ink, pad=2, fontweight="normal")
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
    if spec.background is not None:
        ax.patch.set_facecolor(spec.background)
        ax.patch.set_alpha(spec.background_alpha)
    artists = []
    color = spec.color if spec.color != "map" else "#3b6f8f"
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
                width=(stats.right - stats.left),
                color=colors,
                edgecolor="white" if result.theme == "light" else "none",
                linewidth=0.3,
            )
        )
        edges = np.r_[stats.left.values, stats.right.values[-1]]
        artists.append(ax.stairs(stats.height, edges, color=ink, lw=0.6, fill=False))
    elif spec.style == "step":
        edges = np.r_[stats.left.values, stats.right.values[-1]]
        artists.append(ax.stairs(stats.height, edges, color=color, lw=0.9, fill=False))
    elif spec.style == "line":
        artists.extend(ax.plot(stats.bin, stats.height, color=color, lw=1.0))
        artists.append(ax.fill_between(stats.bin, stats.height, color=color, alpha=0.15, lw=0))
    else:
        artists.extend(
            ax.step(stats.value, stats["cumulative"], where="post", color=color, lw=1.0)
        )
    if spec.show_mean:
        mean = stats.attrs["mean"]
        artists.append(
            ax.plot(
                [mean], [0], marker="^", ms=4, color=ink, clip_on=False, zorder=5,
                transform=ax.get_xaxis_transform(),
            )[0]
        )
        ax.text(
            mean, 1.0, f"mean {mean:.3g}", transform=ax.get_xaxis_transform(), ha="center",
            va="bottom", fontsize=5.5, color=ink,
        )
    weighted = spec.weights is not None
    label = (
        "CDF"
        if spec.style == "ecdf"
        else ("Density" if spec.density else ("Weight" if weighted else "Cells"))
    )
    if spec.label:
        ax.set_title(spec.label, loc="left", fontsize=6.5, color=ink, pad=8, fontweight="normal")
    ax.set_xlabel(data.attrs.get("units", ""), fontsize=6.5, labelpad=1)
    ax.set_ylabel(label, fontsize=6.5, labelpad=2)
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.xaxis.set_major_formatter(pj.clean_formatter())
    ax.yaxis.set_major_formatter(pj.clean_formatter())
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
            loc="lower right",
            fontsize=6.5,
            frameon=True,
            facecolor="white",
            edgecolor="none",
            framealpha=0.85,
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
