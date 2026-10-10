"""A small public API around the original xarray → Cartopy plotting approach."""

from dataclasses import dataclass, field
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from . import journal as pj
from .data import Source, open_field
from .legacy import one_map_flat
from .options import MapFeatures, Profile

# "light" is the publication theme: white paper, near-black ink, light dashed grid.
_THEMES = {
    "light": {"paper": "#ffffff", "ink": pj.INK, "muted": pj.MUTED, "water": "#ffffff"},
    "dark": {"paper": "#10252e", "ink": "#edf5ef", "muted": "#a3bec2", "water": "#16333e"},
}
_PROJECTIONS = {
    "robinson": ccrs.Robinson,
    "platecarree": ccrs.PlateCarree,
    "equalearth": ccrs.EqualEarth,
    "mollweide": ccrs.Mollweide,
}


@dataclass
class MapResult:
    """The figure, map axes, mappable, colorbar, and normalized data for further editing."""

    figure: mpl.figure.Figure
    axes: object
    artist: object
    colorbar: object
    data: xr.DataArray
    theme: str = "light"
    extent: object = None
    profiles: dict = field(default_factory=dict)
    distribution: object = None
    significance: list = field(default_factory=list)
    feature_artists: list = field(default_factory=list)
    _title: object = field(default=None, repr=False)
    _subtitle: object = field(default=None, repr=False)

    def add_profile(self, options=None, **kwargs):
        """Add a right/top summary and return its axes, statistics, and artists."""
        from .layers import add_profile

        return add_profile(self, options, **kwargs)

    def add_distribution(self, options=None, **kwargs):
        """Add a distribution inset and return its computed statistics."""
        from .layers import add_distribution

        return add_distribution(self, options, **kwargs)

    def add_significance(self, options=None, **kwargs):
        """Overlay supplied p-values or a significance mask, without interpolation."""
        from .layers import add_significance

        return add_significance(self, options, **kwargs)

    def add_features(self, options=None, **kwargs):
        """Add optional Natural Earth layers (may download uncached data)."""
        from .layers import add_features

        return add_features(self, options, **kwargs)

    def save(self, path: str | Path, *, dpi: int = 300, **kwargs) -> Path:
        """Save PNG, PDF, SVG, or another Matplotlib format; create parent directories."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        options = {
            "dpi": dpi,
            "bbox_inches": "tight",
            "pad_inches": 0.03,
            "facecolor": self.figure.get_facecolor(),
        }
        options.update(kwargs)
        with pj.journal_style():
            self.figure.savefig(path, **options)
        return path


def plot_map(
    source: Source,
    variable: str | None = None,
    *,
    sel=None,
    isel=None,
    reduce=None,
    statistic="mean",
    latitude=None,
    longitude=None,
    scale=1,
    offset=0,
    units=None,
    projection="robinson",
    extent=None,
    cmap="viridis",
    levels=None,
    vmin=None,
    vmax=None,
    title=None,
    subtitle=None,
    label=None,
    theme="light",
    coastlines=True,
    gridlines=True,
    colorbar=True,
    plotfunc="pcolormesh",
    profiles=None,
    distribution=None,
    significance=None,
    features=None,
    panel_label=None,
    ax=None,
    figsize=(7.2, 4.2),
    output=None,
    dpi=300,
) -> MapResult:
    """Plot a NetCDF field with one function call; return editable Matplotlib objects.

    Use ``isel={'time': 0}`` for a time slice or ``reduce='time'`` for a temporal
    mean. ``extent=(west, east, south, north)`` focuses on a region in degrees.
    Choose ``coastlines=False`` for a completely offline, NetCDF-only plot.
    Coastlines otherwise use Cartopy's cached/downloaded Natural Earth 110m data.
    A supplied Cartopy ``ax`` owns the projection; ``projection`` is then ignored.

    Figures use the journal typography of :mod:`plot_function.journal` (Arial /
    Helvetica-like sans serif, 7–9 pt, thin 0.5 pt lines) and are exported at
    300 dpi by default; pass ``dpi=600`` for print.
    """
    if theme not in _THEMES:
        raise ValueError(f"Unknown theme {theme!r}; choose 'light' or 'dark'.")
    if plotfunc not in ("pcolormesh", "contourf"):
        raise ValueError("plotfunc must be 'pcolormesh' or 'contourf'.")
    profiles = [profiles] if isinstance(profiles, Profile) else list(profiles or [])
    if any(not isinstance(profile, Profile) for profile in profiles):
        raise TypeError("profiles must be a Profile or a sequence of Profile instances.")
    if features is not None and not isinstance(features, MapFeatures):
        raise TypeError("features must be a MapFeatures instance.")
    if extent is not None:
        if len(extent) != 4 or not np.isfinite(extent).all():
            raise ValueError("extent must contain four finite values: west, east, south, north.")
        west, east, south, north = extent
        if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
            raise ValueError(
                "extent requires -180 <= west < east <= 180 and -90 <= south < north <= 90."
            )
    if ax is None:
        if isinstance(projection, str):
            if projection not in _PROJECTIONS:
                raise ValueError(f"Unknown projection {projection!r}; choose {list(_PROJECTIONS)}.")
            projection = _PROJECTIONS[projection]()
        if not isinstance(projection, ccrs.Projection):
            raise ValueError("projection must be a supported name or a Cartopy projection.")
    elif not hasattr(ax, "projection"):
        raise ValueError("ax must be a Cartopy GeoAxes, created with a map projection.")
    data = open_field(
        source,
        variable,
        sel=sel,
        isel=isel,
        reduce=reduce,
        statistic=statistic,
        latitude=latitude,
        longitude=longitude,
        scale=scale,
        offset=offset,
        units=units,
    )
    colors = _THEMES[theme]
    owns_figure = ax is None
    with pj.journal_style():
        if owns_figure:
            figure, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": projection})
            figure.patch.set_facecolor(colors["paper"])
            figure.subplots_adjust(
                left=0.06,
                right=0.80 if any(p.position == "right" for p in profiles) else 0.97,
                top=0.72 if any(p.position == "top" for p in profiles) else 0.88,
                bottom=0.13,
            )
        else:
            figure = ax.figure
        try:
            ax.set_facecolor(colors["water"])
            artist = one_map_flat(
                data,
                ax,
                levels=levels,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                add_coastlines=False,
                colorbar=False,
                plotfunc=plotfunc,
            )
            if extent is not None:
                ax.set_extent(extent, crs=ccrs.PlateCarree())
            if coastlines:
                ax.coastlines(
                    resolution=features.resolution if features else "110m",
                    color=colors["ink"],
                    linewidth=0.35,
                )
            ax.spines["geo"].set_edgecolor(colors["ink"] if theme == "light" else colors["muted"])
            ax.spines["geo"].set_linewidth(0.5)
            if gridlines:
                _journal_grid(ax, extent, colors, theme)
            heading = title if title is not None else data.attrs.get("long_name", data.name or "")
            title_artist = ax.set_title(
                heading,
                loc="left",
                fontsize=9 if owns_figure else 8,
                fontweight="bold",
                color=colors["ink"],
                pad=14 if subtitle else 5,
            )
            subtitle_artist = None
            if subtitle:
                subtitle_artist = ax.text(
                    0,
                    1.012,
                    subtitle,
                    transform=ax.transAxes,
                    fontsize=7,
                    color=colors["muted"],
                    ha="left",
                    va="bottom",
                )
            cbar = None
            if colorbar:
                cbar = figure.colorbar(
                    artist,
                    ax=ax,
                    orientation="horizontal",
                    pad=0.06 if extent is None else 0.09,
                    fraction=0.04,
                    shrink=0.6,
                    aspect=38,
                    extendfrac=0.04,
                )
                cbar.set_label(
                    label if label is not None else data.attrs.get("units", ""),
                    color=colors["ink"],
                    fontsize=7.5,
                    labelpad=2,
                )
                cbar.ax.tick_params(
                    colors=colors["ink"], labelsize=6.5, length=2, width=0.5, pad=1.5
                )
                cbar.outline.set_linewidth(0.5)
                cbar.outline.set_edgecolor(colors["ink"])
                cbar.ax.minorticks_off()
                cbar.ax.xaxis.set_major_formatter(pj.clean_formatter())
            result = MapResult(
                figure,
                ax,
                artist,
                cbar,
                data,
                theme=theme,
                extent=extent,
                _title=title_artist,
                _subtitle=subtitle_artist,
            )
            if features is not None:
                result.add_features(features)
            if significance is not None:
                result.add_significance(significance)
            for profile in profiles:
                result.add_profile(profile)
            if distribution is not None:
                result.add_distribution(distribution)
            if panel_label is not None:
                label_style = dict(
                    fontsize=10, fontweight="bold", color=colors["ink"], zorder=11, ha="right"
                )
                if heading:
                    # Nature style: bold lowercase letter immediately left of the title.
                    ax.annotate(
                        panel_label, xy=(0, 0), xycoords=result._title, xytext=(-5, 0),
                        textcoords="offset points", va="bottom", **label_style,
                    )
                else:
                    ax.text(
                        -0.01, 1.01, panel_label, transform=ax.transAxes, va="bottom",
                        **label_style,
                    )
            if output is not None:
                result.save(output, dpi=dpi)
            return result
        except Exception:
            if owns_figure:
                plt.close(figure)
            raise


def _journal_grid(ax, extent, colors, theme):
    """Light dashed graticule; outward degree ticks on rectangular regional maps."""
    rectangular = isinstance(ax.projection, (ccrs.PlateCarree, ccrs.Mercator))
    if extent is not None:
        west, east, south, north = extent
        xstep = _nice_step(east - west)
        ystep = _nice_step(north - south)
        xticks = np.arange(np.ceil(west / xstep) * xstep, east + 1e-9, xstep)
        yticks = np.arange(np.ceil(south / ystep) * ystep, north + 1e-9, ystep)
    else:
        xticks, yticks = np.arange(-180, 181, 60), np.arange(-60, 61, 30)
    pj.geo_ticks(
        ax,
        xticks=xticks,
        yticks=yticks,
        labels=extent is not None or rectangular,
        grid_color=pj.GRID if theme == "light" else colors["muted"],
    )
    # Axes created outside journal_style() have DejaVu tick labels; restyle them.
    ax.tick_params(labelfontfamily=pj.resolved_font())
    if theme != "light":
        ax.tick_params(colors=colors["muted"], labelcolor=colors["muted"])


def _nice_step(span):
    for step in (1, 2, 5, 10, 20, 30, 60):
        if span / step <= 5:
            return step
    return 60
