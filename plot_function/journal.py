"""Publication ("journal") styling helpers that work with any Matplotlib/Cartopy axes.

Every helper is functional: it receives the axes it should decorate and returns
the artists or axes it created, so it composes with ``plot_map`` *and* with
hand-built figures.  Nothing here mutates global ``rcParams``; wrap your figure
code in :func:`journal_style` to apply the typography only while drawing.

Typical use::

    import plot_function.journal as pj

    with pj.journal_style():
        fig = plt.figure(figsize=pj.figsize("double", 0.5))
        ax = fig.add_subplot(projection=ccrs.PlateCarree())
        cmap, norm = pj.discrete_cmap("BrBG", np.arange(-10, 11, 2.5), extend="both")
        mesh = ax.pcolormesh(lon, lat, field, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
        pj.geo_ticks(ax, xticks=range(-180, 181, 60), yticks=range(-40, 81, 20))
        pj.add_stippling(ax, lon, lat, low_agreement)
        pj.add_colorbar(mesh, ax, label="ΔBGWS [ppts]", title="Blue water regime")
        pj.add_latitude_profile(ax, lat, zonal_mean, lower=q25, upper=q75)
        pj.add_panel_label(ax, "a")
        pj.save_figure(fig, "figure1.png", dpi=600)
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullLocator

__all__ = [
    "INK",
    "MUTED",
    "GRID",
    "LAND",
    "JOURNAL_RC",
    "journal_style",
    "figsize",
    "discrete_cmap",
    "clean_formatter",
    "add_colorbar",
    "geo_ticks",
    "format_lat",
    "format_lon",
    "add_land",
    "add_stippling",
    "add_hatching",
    "hatch_patch",
    "add_inset_bars",
    "add_inset_histogram",
    "add_latitude_profile",
    "add_longitude_profile",
    "add_panel_label",
    "add_size_legend",
    "save_figure",
    "cell_area_km2",
    "marginal_totals",
    "add_lat_lon_marginals",
    "ternary_colors",
    "add_ternary_legend",
    "add_inset_density",
    "plot_rgb",
]

INK = "#1a1a1a"
MUTED = "#555555"
GRID = "#b4b4b4"
LAND = "#fbf6e4"  # cream land (Nature-style site maps)
SANS = [
    "Arial",
    "Helvetica",
    "Helvetica Neue",
    "Liberation Sans",
    "Nimbus Sans",
    "TeX Gyre Heros",
    "DejaVu Sans",
]

#: rcParams for 7–9 pt figures sized for a 89 mm (single) or 183 mm (double) column.
JOURNAL_RC = {
    # An explicit family list (not the generic "sans-serif") is stored on every Text
    # object, so lazily created tick labels keep Arial/Helvetica after the context ends.
    "font.family": SANS,
    "font.sans-serif": SANS,
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.titleweight": "bold",
    "axes.labelsize": 7,
    "axes.linewidth": 0.5,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "axes.labelpad": 2.0,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "xtick.major.pad": 1.8,
    "ytick.major.pad": 1.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.fontsize": 6.5,
    "legend.frameon": False,
    "legend.handlelength": 1.4,
    "legend.handletextpad": 0.5,
    "legend.borderaxespad": 0.3,
    "lines.linewidth": 0.9,
    "patch.linewidth": 0.5,
    "hatch.linewidth": 0.4,
    "hatch.color": INK,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
    "pdf.fonttype": 42,  # editable text in Illustrator / Inkscape
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "axes.unicode_minus": True,
    "mathtext.default": "regular",
}

_MM = 1 / 25.4
_WIDTHS = {"single": 89 * _MM, "onehalf": 120 * _MM, "double": 183 * _MM}


def installed_fonts():
    """The installed subset of the Arial/Helvetica fallback chain (DejaVu Sans last)."""
    from matplotlib import font_manager

    names = {f.name for f in font_manager.fontManager.ttflist}
    found = [n for n in SANS if n in names]
    return found if found and found[-1] == "DejaVu Sans" else found + ["DejaVu Sans"]


def resolved_font():
    """Name of the first installed font in the Arial/Helvetica fallback chain."""
    return installed_fonts()[0]


def journal_style(**overrides):
    """Context manager applying :data:`JOURNAL_RC` (plus overrides) temporarily."""
    rc = dict(JOURNAL_RC)
    rc["font.family"] = rc["font.sans-serif"] = installed_fonts()
    rc.update(overrides)
    return mpl.rc_context(rc)


def figsize(width="double", ratio=0.55):
    """Figure size in inches for a journal column: ``'single'``, ``'onehalf'``, ``'double'``
    or a width in millimetres; ``ratio`` is height / width."""
    w = _WIDTHS[width] if isinstance(width, str) else float(width) * _MM
    return (w, w * ratio)


# --------------------------------------------------------------------------- colour


def discrete_cmap(cmap, levels, *, extend="both", under=None, over=None, bad="none"):
    """Return ``(cmap, norm)`` with one colour per interval and distinct extend colours.

    ``levels`` are the interval boundaries.  With ``extend='both'`` the colormap is
    sampled at ``len(levels) + 1`` evenly spaced points so the triangles at the
    ends of the colourbar get the darkest colours, exactly as in printed atlases.
    A list of colours can be given instead of a colormap name.
    """
    levels = np.asarray(levels, dtype=float)
    if levels.ndim != 1 or len(levels) < 2 or np.any(np.diff(levels) <= 0):
        raise ValueError("levels must be at least two strictly increasing values.")
    if extend not in ("neither", "min", "max", "both"):
        raise ValueError("extend must be 'neither', 'min', 'max', or 'both'.")
    n_int = len(levels) - 1
    n_extra = {"neither": 0, "min": 1, "max": 1, "both": 2}[extend]
    if isinstance(cmap, (list, tuple)):
        colors = list(cmap)
        if len(colors) != n_int + n_extra:
            raise ValueError(f"Expected {n_int + n_extra} colours for {n_int} intervals.")
    else:
        base = mpl.colormaps[cmap] if isinstance(cmap, str) else cmap
        colors = list(base(np.linspace(0, 1, n_int + n_extra)))
    lo = 1 if extend in ("min", "both") else 0
    inner = colors[lo : lo + n_int]
    extremes = {"bad": bad}
    if extend in ("min", "both"):
        extremes["under"] = under if under is not None else colors[0]
    if extend in ("max", "both"):
        extremes["over"] = over if over is not None else colors[-1]
    out = ListedColormap(inner, name=f"{getattr(cmap, 'name', cmap)}_discrete")
    out = out.with_extremes(**extremes)
    norm = BoundaryNorm(levels, n_int, extend="neither")
    out.colorbar_extend = extend
    return out, norm


def add_colorbar(
    mappable,
    ax,
    *,
    label=None,
    title=None,
    orientation="horizontal",
    extend=None,
    ticks=None,
    bounds=None,
    size=0.035,
    pad=0.12,
    length=0.6,
    anchor=0.5,
    labelsize=None,
    outline=True,
    tick_every=1,
):
    """Slim colourbar with triangle ends, an optional bold title and a label.

    The colourbar is placed relative to ``ax`` (axes fraction) unless explicit
    figure-fraction ``bounds=[x, y, w, h]`` are given.  ``title`` is drawn above a
    horizontal bar (bold, like "Historical blue water regime") and ``label`` below.
    """
    fig = ax.figure
    horizontal = orientation == "horizontal"
    if bounds is not None:
        cax = fig.add_axes(bounds)
    elif horizontal:
        pad = pad + (0.07 if title else 0.0)
        cax = ax.inset_axes([anchor - length / 2, -pad - size, length, size], transform=ax.transAxes)
    else:
        cax = ax.inset_axes([1 + pad, 0.5 - length / 2, size, length], transform=ax.transAxes)
    if extend is None:
        extend = getattr(getattr(mappable, "cmap", None), "colorbar_extend", "neither")
        extend = extend if extend in ("neither", "min", "max", "both") else "neither"
    cbar = fig.colorbar(
        mappable,
        cax=cax,
        orientation=orientation,
        extend=extend,
        extendfrac=0.05 if horizontal else 0.04,
        drawedges=False,
    )
    norm = getattr(mappable, "norm", None)
    if ticks is None and isinstance(norm, BoundaryNorm):
        ticks = norm.boundaries[::tick_every]
    if ticks is not None:
        cbar.set_ticks(ticks)
        cbar.ax.minorticks_off()
        fmt = _tick_formatter(np.asarray(ticks, dtype=float))
        (cbar.ax.xaxis if horizontal else cbar.ax.yaxis).set_major_formatter(fmt)
    cbar.outline.set_visible(outline)
    cbar.outline.set_linewidth(0.5)
    cbar.outline.set_edgecolor(INK)
    size_pt = labelsize or mpl.rcParams["xtick.labelsize"]
    cbar.ax.tick_params(labelsize=size_pt, length=2.0, width=0.5, pad=1.5, color=INK)
    for patch in getattr(cbar, "_extend_patches", []):
        patch.set_edgecolor(INK)
        patch.set_linewidth(0.5)
    if label:
        cbar.set_label(label, fontsize=mpl.rcParams["axes.labelsize"] + 0.5, labelpad=1.5)
    if title:
        if horizontal:
            cax.set_title(title, fontsize=mpl.rcParams["axes.titlesize"], fontweight="bold", pad=3)
        else:
            cax.set_title(title, fontsize=mpl.rcParams["axes.labelsize"], pad=4, loc="left")
    return cbar


def clean_formatter():
    """``%g`` tick labels with a true minus sign and no ``-0``."""
    return FuncFormatter(lambda v, _: "0" if abs(v) < 1e-12 else f"{v:g}".replace("-", "\u2212"))


def _tick_formatter(ticks):
    finite = ticks[np.isfinite(ticks)]
    decimals = 0
    for d in range(0, 4):
        if np.allclose(finite, np.round(finite, d)):
            decimals = d
            break
    else:
        decimals = 3
    return FuncFormatter(
        lambda v, _: "0" if abs(v) < 1e-12 else f"{v:.{decimals}f}".replace("-", "\u2212")
    )


# --------------------------------------------------------------------------- geography


def format_lat(value, _=None):
    if np.isclose(value, 0):
        return "0°"
    return f"{abs(value):g}°{'N' if value > 0 else 'S'}"


def format_lon(value, _=None):
    value = (value + 180) % 360 - 180 if abs(value) > 180 else value
    if np.isclose(abs(value), 180):
        return "180°"
    if np.isclose(value, 0):
        return "0°"
    return f"{abs(value):g}°{'E' if value > 0 else 'W'}"


def geo_ticks(
    ax,
    *,
    xticks=None,
    yticks=None,
    gridlines=True,
    grid_color=GRID,
    grid_style=(0, (3, 3)),
    grid_width=0.35,
    labels=True,
    lat_labels=True,
    lon_labels=True,
    frame=True,
    degree_style="hemisphere",
):
    """Clean ``60°E`` / ``40°N`` ticks and light dashed gridlines on a GeoAxes.

    Rectangular projections (PlateCarree, Mercator) get real outward ticks on
    the left/bottom frame; other projections fall back to Cartopy gridline labels.
    """
    proj = ax.projection
    xticks = list(xticks) if xticks is not None else None
    yticks = list(yticks) if yticks is not None else None
    rectangular = isinstance(proj, (ccrs.PlateCarree, ccrs.Mercator))
    artists = []
    if rectangular and labels:
        if xticks is not None and lon_labels:
            ax.set_xticks(xticks, crs=ccrs.PlateCarree())
            ax.xaxis.set_major_formatter(
                FuncFormatter(format_lon if degree_style == "hemisphere" else _signed_lon)
            )
        if yticks is not None and lat_labels:
            ax.set_yticks(yticks, crs=ccrs.PlateCarree())
            ax.yaxis.set_major_formatter(
                FuncFormatter(format_lat if degree_style == "hemisphere" else _signed_lat)
            )
        ax.tick_params(
            which="major", length=2.5, width=0.5, labelsize=mpl.rcParams["xtick.labelsize"] + 0.5
        )
    if gridlines:
        grid = ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=(labels and not rectangular),
            xlocs=xticks,
            ylocs=yticks,
            linewidth=grid_width,
            color=grid_color,
            linestyle=grid_style,
            alpha=0.9,
            zorder=0.6,
        )
        if labels and not rectangular:
            grid.top_labels = grid.right_labels = False
            grid.left_labels = lat_labels
            grid.bottom_labels = lon_labels
            grid.rotate_labels = False
            grid.x_inline = grid.y_inline = False
            grid.xpadding = grid.ypadding = 3
            style = {"size": mpl.rcParams["xtick.labelsize"], "color": INK}
            grid.xlabel_style = grid.ylabel_style = style
        artists.append(grid)
    spine = ax.spines.get("geo")
    if spine is not None:
        spine.set_visible(frame)
        spine.set_linewidth(0.5)
        spine.set_edgecolor(INK if rectangular else "#7a7a7a")
    return artists


def add_land(
    ax,
    *,
    color=LAND,
    edgecolor="#8c8c8c",
    linewidth=0.25,
    borders=True,
    coast_color=INK,
    coast_width=0.3,
    resolution="110m",
    zorder=0.5,
):
    """Cream land, thin grey country borders and a dark coastline (Natural Earth)."""
    import cartopy.feature as cfeature

    artists = [
        ax.add_feature(
            cfeature.LAND.with_scale(resolution), facecolor=color, edgecolor="none", zorder=zorder
        )
    ]
    if borders:
        artists.append(
            ax.add_feature(
                cfeature.BORDERS.with_scale(resolution),
                facecolor="none",
                edgecolor=edgecolor,
                linewidth=linewidth,
                zorder=zorder + 0.1,
            )
        )
    if coast_width:
        artists.append(
            ax.add_feature(
                cfeature.COASTLINE.with_scale(resolution),
                facecolor="none",
                edgecolor=coast_color,
                linewidth=coast_width,
                zorder=zorder + 0.2,
            )
        )
    return artists


# --------------------------------------------------------------------------- significance


def _grid(lon, lat, mask):
    lon, lat, mask = np.asarray(lon), np.asarray(lat), np.asarray(mask)
    if lon.ndim == 1 and lat.ndim == 1:
        lon, lat = np.meshgrid(lon, lat)
    if mask.shape != lon.shape:
        raise ValueError(f"mask shape {mask.shape} must match the lat/lon grid {lon.shape}.")
    return lon, lat, mask.astype(bool) & np.isfinite(lon) & np.isfinite(lat)


def add_stippling(
    ax,
    lon,
    lat,
    mask,
    *,
    stride=2,
    size=1.2,
    color=INK,
    marker="o",
    alpha=1.0,
    offset=True,
    transform=None,
    zorder=4,
):
    """Black dots where ``mask`` is True (e.g. low ensemble agreement or p < 0.05).

    ``stride`` thins the dot lattice for readability; ``offset`` staggers every
    other row (a hexagonal pattern looks much calmer than a square lattice).
    Thinning only changes the display, never the underlying decision.
    """
    lon2, lat2, m = _grid(lon, lat, mask)
    keep = np.zeros_like(m)
    for k, row in enumerate(range(0, m.shape[0], stride)):
        start = stride // 2 if (offset and stride > 1 and k % 2) else 0
        keep[row, start::stride] = True
    show = m & keep
    if not show.any():
        return None
    return ax.scatter(
        lon2[show],
        lat2[show],
        s=size,
        c=color,
        marker=marker,
        linewidths=0,
        alpha=alpha,
        transform=transform or ccrs.PlateCarree(),
        zorder=zorder,
        rasterized=False,
    )


def add_hatching(
    ax, lon, lat, mask, *, hatch="....", color=INK, linewidth=0.4, transform=None, zorder=4
):
    """Hatch (``'....'``, ``'///'``, ``'xxx'``) the region where ``mask`` is True."""
    lon2, lat2, m = _grid(lon, lat, mask)
    if not m.any():
        return None
    with mpl.rc_context({"hatch.color": color, "hatch.linewidth": linewidth}):
        cs = ax.contourf(
            lon2,
            lat2,
            m.astype(float),
            levels=[0.5, 1.5],
            colors="none",
            hatches=[hatch],
            transform=transform or ccrs.PlateCarree(),
            zorder=zorder,
        )
    for coll in [cs] if hasattr(cs, "set_edgecolor") else cs.collections:
        coll.set_edgecolor(color)
        coll.set_linewidth(0)
        if hasattr(coll, "set_hatch_linewidth"):
            coll.set_hatch_linewidth(linewidth)
    return cs


def hatch_patch(hatch="....", *, color=INK, facecolor="white", edgecolor=INK, linewidth=0.5):
    """A legend handle for a hatched region (use with ``ax.legend``)."""
    patch = Patch(facecolor=facecolor, edgecolor=edgecolor, hatch=hatch, linewidth=linewidth)
    patch._hatch_color = mpl.colors.to_rgba(color)
    return patch


# --------------------------------------------------------------------------- insets


def _halo(ax, width=2.2):
    """White halo behind inset tick labels and axis labels, so they stay legible on maps."""
    from matplotlib import patheffects

    effect = [patheffects.withStroke(linewidth=width, foreground="white")]
    original = ax.draw

    def draw(renderer):
        texts = [ax.xaxis.label, ax.yaxis.label]
        texts += ax.get_xticklabels() + ax.get_yticklabels()
        for text in texts:
            text.set_path_effects(effect)
        return original(renderer)

    ax.draw = draw
    return ax


def _clean_inset(ax, frame):
    if frame == "box":
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
            spine.set_color(INK)
        ax.patch.set_facecolor("white")
        ax.patch.set_alpha(0.92)
    else:  # "open": transparent background, left + bottom spines only
        for name, spine in ax.spines.items():
            spine.set_visible(name in ("left", "bottom"))
            spine.set_linewidth(0.5)
            spine.set_color(INK)
        ax.patch.set_alpha(0)
    ax.tick_params(length=2.0, width=0.5, pad=1.5)
    _halo(ax)
    return ax


def _inset(ax, bounds, zorder):
    child = ax.inset_axes(bounds, transform=ax.transAxes, zorder=zorder)
    # GeoAxes children inherit nothing from the projection; they are plain axes.
    return child


def add_inset_bars(
    ax,
    labels,
    values,
    *,
    colors=None,
    hatched=None,
    hatch="....",
    hatch_label="Low agreement",
    bounds=(0.07, 0.10, 0.2, 0.36),
    ylabel=None,
    frame="box",
    edgecolor=INK,
    width=0.72,
    ylim=None,
    rotation=0,
    zorder=8,
    backdrop=True,
):
    """Category bar chart inset (e.g. land area per regime) for the lower-left corner.

    ``hatched`` gives, per bar, the part of ``values`` that is uncertain (e.g. low
    ensemble agreement).  It is stacked *on top* of the confident part with dot
    hatching and separated by a black line, as in Nature/Science regime maps.
    """
    labels = list(labels)
    values = np.asarray(values, dtype=float)
    child = _inset(ax, list(bounds), zorder)
    _clean_inset(child, frame)
    x = np.arange(len(values))
    colors = colors if colors is not None else ["#7f7f7f"] * len(values)
    artists = []
    if hatched is None:
        artists.append(
            child.bar(x, values, width, color=colors, edgecolor=edgecolor, linewidth=0.5)
        )
    else:
        hatched = np.asarray(hatched, dtype=float)
        solid = values - hatched
        artists.append(child.bar(x, solid, width, color=colors, edgecolor="none"))
        with mpl.rc_context({"hatch.color": INK, "hatch.linewidth": 0.6}):
            artists.append(
                child.bar(
                    x,
                    hatched,
                    width,
                    bottom=solid,
                    color=colors,
                    edgecolor=INK,
                    linewidth=0,
                    hatch=hatch,
                )
            )
        for xi, s, v in zip(x, solid, values):
            child.plot([xi - width / 2, xi + width / 2], [s, s], color=INK, lw=0.6, zorder=3)
            child.add_patch(
                mpl.patches.Rectangle(
                    (xi - width / 2, 0), width, v, fill=False, edgecolor=edgecolor, lw=0.5, zorder=3
                )
            )
        if hatch_label:
            child.legend(
                [hatch_patch(hatch, facecolor="white")],
                [hatch_label],
                loc="upper right",
                bbox_to_anchor=(1.02, 1.04),
                handlelength=1.6,
                handleheight=0.9,
                borderpad=0.2,
                fontsize=mpl.rcParams["legend.fontsize"],
            )
    child.set_xticks(x)
    child.set_xticklabels(labels, rotation=rotation, ha="right" if rotation else "center")
    child.tick_params(axis="x", length=0)
    child.set_xlim(-0.6, len(values) - 0.4)
    top = ylim[1] if ylim else float(np.nanmax(values)) * (1.38 if hatch_label else 1.12)
    child.set_ylim(ylim[0] if ylim else 0, top)
    child.yaxis.set_major_locator(MaxNLocator(4, integer=True))
    if ylabel:
        child.set_ylabel(ylabel)
    if backdrop and frame == "box":
        child.patch.set_alpha(0.92)
    return child


def add_inset_histogram(
    ax,
    groups,
    *,
    bins=20,
    colors=None,
    labels=None,
    total=True,
    total_label="All",
    stacked=True,
    log=False,
    cumulative=None,
    cumulative_color="#e8161b",
    cumulative_label=None,
    cumulative_reverse=True,
    bounds=(0.09, 0.08, 0.22, 0.3),
    xlabel=None,
    ylabel="Count",
    frame="open",
    zorder=8,
    alpha=0.55,
    mean_line=False,
    legend=False,
):
    """Histogram inset with optional stacked groups, a black total outline and a
    cumulative curve on a second, outward-offset y axis (the Nature "site map" look).

    ``groups`` is an array or a dict ``{label: values}``.  ``cumulative`` may be
    ``True`` (cumulative count) or an array of weights the same length as the
    concatenated values (e.g. emissions), plotted as a red line on the outer axis.
    ``log=True`` uses logarithmic bins and axis.
    Returns ``(hist_axes, cumulative_axes_or_None)``.
    """
    if not isinstance(groups, dict):
        groups = {total_label: np.asarray(groups, dtype=float)}
        total = False if len(groups) == 1 and not stacked else total
    names = list(groups)
    arrays = [np.asarray(groups[n], dtype=float).ravel() for n in names]
    arrays = [a[np.isfinite(a)] for a in arrays]
    allv = np.concatenate(arrays)
    if allv.size == 0:
        raise ValueError("The histogram has no finite values.")
    if np.isscalar(bins):
        if log:
            pos = allv[allv > 0]
            edges = np.logspace(np.log10(pos.min()), np.log10(pos.max()), int(bins) + 1)
        else:
            edges = np.histogram_bin_edges(allv, bins=int(bins))
    else:
        edges = np.asarray(bins, dtype=float)
    colors = colors or ["#f7a541", "#b56ccf", "#4f9fd6", "#61a85c"][: len(names)]
    child = _inset(ax, list(bounds), zorder)
    _clean_inset(child, frame)
    single = len(names) == 1
    child.hist(
        arrays if not single else arrays[0],
        bins=edges,
        stacked=stacked and not single,
        color=colors if not single else colors[0],
        alpha=alpha,
        edgecolor="none",
        label=labels or names,
    )
    if total or single:
        counts, _ = np.histogram(allv, bins=edges)
        child.stairs(counts, edges, color=INK, lw=0.8, label=total_label if total else None)
    if log:
        child.set_xscale("log")
    if mean_line:
        child.axvline(np.mean(allv), color=INK, lw=0.6, ls=(0, (2, 1.5)))
    child.set_xlim(edges[0], edges[-1])
    child.set_ylim(bottom=0)
    child.yaxis.set_major_locator(MaxNLocator(4, integer=True))
    child.xaxis.set_major_locator(MaxNLocator(4) if not log else mpl.ticker.LogLocator())
    if xlabel:
        child.set_xlabel(xlabel)
    if ylabel:
        child.set_ylabel(ylabel)
    twin = None
    if cumulative is not None and cumulative is not False:
        w = np.ones_like(allv) if cumulative is True else np.asarray(cumulative, dtype=float)
        order = np.argsort(allv)
        xs, ws = allv[order], w[order]
        if cumulative_reverse:  # amount from sources >= x (Nature style)
            ys = np.cumsum(ws[::-1])[::-1]
        else:
            ys = np.cumsum(ws)
        twin = child.twinx()
        _halo(twin)
        twin.patch.set_alpha(0)
        twin.plot(xs, ys, color=cumulative_color, lw=0.9, zorder=5)
        for name, spine in twin.spines.items():
            spine.set_visible(name == "left")
        twin.spines["left"].set_position(("outward", 26))
        twin.spines["left"].set_linewidth(0.5)
        twin.yaxis.set_ticks_position("left")
        twin.yaxis.set_label_position("left")
        twin.tick_params(length=2.0, width=0.5, pad=1.5)
        twin.set_ylim(bottom=0)
        twin.yaxis.set_major_locator(MaxNLocator(4))
        if cumulative_label:
            twin.set_ylabel(cumulative_label)
        twin.set_zorder(child.get_zorder() + 0.1)
    if legend:
        child.legend(loc="upper right", fontsize=mpl.rcParams["legend.fontsize"] - 0.5)
    return child, twin


# --------------------------------------------------------------------------- marginal profiles


def _proj_coord(ax, lat=None, lon=None, ref=None):
    """Map latitudes (or longitudes) onto the projected y (or x) axis of ``ax``."""
    proj = ax.projection
    geo = ccrs.PlateCarree()
    if lat is not None:
        lat = np.asarray(lat, dtype=float)
        ref = ref if ref is not None else getattr(proj, "proj4_params", {}).get("lon_0", 0.0)
        pts = proj.transform_points(geo, np.full_like(lat, ref), lat)
        return pts[:, 1]
    lon = np.asarray(lon, dtype=float)
    ref = 0.0 if ref is None else ref
    pts = proj.transform_points(geo, lon, np.full_like(lon, ref))
    return pts[:, 0]


def add_latitude_profile(
    ax,
    lat,
    center,
    *,
    lower=None,
    upper=None,
    members=None,
    outer=None,
    width=0.13,
    pad=0.03,
    color="#2a6f97",
    band_alpha=0.22,
    outer_alpha=0.10,
    linewidth=1.0,
    xlabel=None,
    xlim=None,
    reference=0.0,
    yticks=None,
    ticklabels="right",
    gridlines=True,
    title=None,
    ref_lon=None,
    zorder=6,
    extra=None,
):
    """Right-hand marginal panel: zonal-mean line with a shaded spread, *aligned* with
    the map's latitudes for any projection.

    The y axis of the panel lives in the map's projected coordinates (latitudes are
    pushed through ``ax.projection``), and its limits are linked to the map
    ``ylim`` at draw time, so 40°N on the profile sits exactly level with 40°N on
    the map — on Robinson and Equal Earth too.

    ``lower``/``upper`` shade an inner band (e.g. IQR or ±1 SD); ``outer`` =
    ``(low, high)`` adds a lighter outer band (e.g. ensemble min–max).
    ``members`` (n_members × n_lat) draws thin individual lines instead.
    ``extra`` is a list of ``(center, dict(**line_kwargs))`` for further lines.
    """
    lat = np.asarray(lat, dtype=float)
    center = np.asarray(center, dtype=float)
    child = ax.inset_axes([1 + pad, 0, width, 1], transform=ax.transAxes, zorder=zorder)
    y = _proj_coord(ax, lat=lat, ref=ref_lon)
    if outer is not None:
        child.fill_betweenx(
            y, outer[0], outer[1], color=color, alpha=outer_alpha, lw=0, zorder=1
        )
    if members is not None:
        for m in np.asarray(members, dtype=float):
            child.plot(m, y, color=color, lw=0.3, alpha=0.35, zorder=2)
    if lower is not None and upper is not None:
        child.fill_betweenx(y, lower, upper, color=color, alpha=band_alpha, lw=0, zorder=2)
    for line, kw in extra or []:
        child.plot(np.asarray(line, dtype=float), y, **{"lw": 0.8, "zorder": 3, **kw})
    child.plot(center, y, color=color, lw=linewidth, zorder=4, solid_capstyle="round")
    if reference is not None:
        child.axvline(reference, color=INK, lw=0.5, ls=(0, (2.5, 2)), zorder=1.5)
    for name, spine in child.spines.items():
        spine.set_visible(name in ("bottom", ticklabels if ticklabels in ("left", "right") else ""))
        spine.set_linewidth(0.5)
        spine.set_color(INK)
    child.patch.set_alpha(0)
    if yticks is None:
        span = np.nanmax(lat) - np.nanmin(lat)
        step = 30 if span > 120 else (20 if span > 60 else (10 if span > 25 else 5))
        yticks = np.arange(np.ceil(np.nanmin(lat) / step) * step, np.nanmax(lat) + 1e-9, step)
    yticks = np.asarray(yticks, dtype=float)
    ypos = _proj_coord(ax, lat=yticks, ref=ref_lon)
    child.yaxis.set_major_locator(FixedLocator(ypos))

    def _lat_label(value, _pos=None):
        i = int(np.argmin(np.abs(ypos - value)))
        return format_lat(yticks[i])

    child.yaxis.set_major_formatter(FuncFormatter(_lat_label))
    if ticklabels == "right":
        child.yaxis.tick_right()
    elif ticklabels in (None, False, "none"):
        child.tick_params(axis="y", left=False, labelleft=False)
    child.yaxis.set_minor_locator(NullLocator())
    if gridlines:
        for t in child.get_yticks():
            child.axhline(t, color=GRID, lw=0.35, ls=(0, (3, 3)), zorder=0)
    child.xaxis.set_major_locator(MaxNLocator(3, symmetric=reference == 0))
    child.tick_params(length=2.0, width=0.5, pad=1.5)
    if xlim is not None:
        child.set_xlim(xlim)
    if xlabel:
        child.set_xlabel(xlabel)
    if title:
        child.set_title(title, fontsize=mpl.rcParams["axes.labelsize"], fontweight="normal", pad=3)

    _orig_draw = child.draw

    def draw(renderer):
        # Keep the profile's latitude axis locked to the map, even after set_extent.
        child.set_ylim(ax.get_ylim())
        return _orig_draw(renderer)

    child.set_ylim(ax.get_ylim())
    child.draw = draw
    return child


def add_longitude_profile(
    ax,
    lon,
    center,
    *,
    lower=None,
    upper=None,
    height=0.16,
    pad=0.02,
    color="#2a6f97",
    band_alpha=0.22,
    linewidth=1.0,
    ylabel=None,
    reference=None,
    ref_lat=0.0,
    zorder=6,
):
    """Top marginal panel (meridional mean vs longitude), aligned with the map x axis."""
    lon = np.asarray(lon, dtype=float)
    center = np.asarray(center, dtype=float)
    child = ax.inset_axes([0, 1 + pad, 1, height], transform=ax.transAxes, zorder=zorder)
    x = _proj_coord(ax, lon=lon, ref=ref_lat)
    if lower is not None and upper is not None:
        child.fill_between(x, lower, upper, color=color, alpha=band_alpha, lw=0)
    child.plot(x, center, color=color, lw=linewidth)
    if reference is not None:
        child.axhline(reference, color=INK, lw=0.5, ls=(0, (2.5, 2)))
    for name, spine in child.spines.items():
        spine.set_visible(name in ("left", "bottom"))
        spine.set_linewidth(0.5)
    child.patch.set_alpha(0)
    child.tick_params(axis="x", bottom=False, labelbottom=False)
    child.yaxis.set_major_locator(MaxNLocator(3))
    child.tick_params(length=2.0, width=0.5, pad=1.5)
    if ylabel:
        child.set_ylabel(ylabel)
    _orig_draw = child.draw

    def draw(renderer):
        child.set_xlim(ax.get_xlim())
        return _orig_draw(renderer)

    child.set_xlim(ax.get_xlim())
    child.draw = draw
    return child


# --------------------------------------------------------------------------- labels & export


def add_panel_label(ax, label, *, style="({})", x=0.008, y=0.985, size=None, inside=True, **kw):
    """Bold panel label such as ``(a)`` (``style='{}'`` gives a bare ``a``)."""
    text = style.format(label)
    opts = dict(
        transform=ax.transAxes,
        fontsize=size or mpl.rcParams["axes.titlesize"] + 1,
        fontweight="bold",
        va="top" if inside else "bottom",
        ha="left",
        color=INK,
        zorder=20,
    )
    opts.update(kw)
    return ax.text(x, y if inside else 1.01, text, **opts)


def add_size_legend(
    ax,
    values,
    labels,
    *,
    scale,
    marker="o",
    facecolor="none",
    edgecolor=INK,
    linewidth=0.8,
    title=None,
    **legend_kw,
):
    """Legend for markers sized by value; ``scale(value) -> marker area (pt²)``."""
    handles = [
        Line2D(
            [],
            [],
            ls="none",
            marker=marker,
            markersize=np.sqrt(scale(v)),
            markerfacecolor=facecolor,
            markeredgecolor=edgecolor,
            markeredgewidth=linewidth,
        )
        for v in values
    ]
    opts = dict(frameon=False, title=title, labelspacing=0.9, borderpad=0.2, handletextpad=0.4)
    opts.update(legend_kw)
    leg = ax.legend(handles, labels, **opts)
    if title:
        leg.get_title().set_fontsize(mpl.rcParams["legend.fontsize"] + 0.5)
    return leg


def save_figure(fig, path, *, dpi=600, formats=None, **kwargs):
    """Save at print resolution; ``formats=('png', 'pdf')`` writes several files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    opts = {"dpi": dpi, "bbox_inches": "tight", "pad_inches": 0.02, "facecolor": "white"}
    opts.update(kwargs)
    paths = []
    for fmt in formats or [path.suffix.lstrip(".") or "png"]:
        target = path.with_suffix("." + fmt)
        fig.savefig(target, **opts)
        paths.append(target)
    return paths[0] if len(paths) == 1 else paths



# --------------------------------------------------------------------------- Pekel-style marginals


def cell_area_km2(lon, lat):
    """Area (km²) of each cell of a regular lon/lat grid, shape ``(len(lat), len(lon))``."""
    lon, lat = np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)
    radius = 6371.0088
    dlon = np.deg2rad(np.abs(np.gradient(lon))) if lon.size > 1 else np.array([np.deg2rad(1.0)])
    half = np.abs(np.gradient(lat)) / 2 if lat.size > 1 else np.array([0.5])
    top = np.deg2rad(np.clip(lat + half, -90, 90))
    bottom = np.deg2rad(np.clip(lat - half, -90, 90))
    band = radius**2 * np.abs(np.sin(top) - np.sin(bottom))
    return band[:, None] * dlon[None, :]


def marginal_totals(field, lon, lat, *, area=True, how="sum", scale=1.0):
    """Collapse a ``(lat, lon)`` field to a latitude profile and a longitude profile.

    With ``area=True`` the field is treated as a cell *fraction* (0–1) and multiplied by
    the cell area in km², so ``how='sum'`` gives the area per latitude/longitude row,
    e.g. water area as in Pekel et al. (2016).  ``scale`` divides the result (``1e3``
    for 10³ km²).  Returns ``(by_lat, by_lon)``.
    """
    field = np.asarray(field, dtype=float)
    values = field * cell_area_km2(lon, lat) if area else field
    reducer = {"sum": np.nansum, "mean": np.nanmean}[how]
    with np.errstate(all="ignore"):
        return reducer(values, axis=1) / scale, reducer(values, axis=0) / scale


def _signed_lat(value, _=None):
    return "0°" if np.isclose(value, 0) else f"{value:g}°".replace("-", "\u2212")


def _signed_lon(value, _=None):
    return "0°" if np.isclose(value, 0) else f"{value:g}°".replace("-", "\u2212")


def add_lat_lon_marginals(
    ax,
    *,
    lat=None,
    lat_series=None,
    lon=None,
    lon_series=None,
    fill=(),
    colors=None,
    fill_color="#dadada",
    fill_edge="#a6a6a6",
    linewidth=0.9,
    right_width=0.22,
    right_pad=0.06,
    bottom_height=0.42,
    bottom_pad=0.17,
    lat_ticks=None,
    degree_style="signed",
    right_xlabel=None,
    bottom_ylabel=None,
    latitude_label="Latitude",
    right_xlim=None,
    bottom_ylim=None,
    legend=True,
    legend_labels=None,
    zero_line=False,
    zorder=6,
):
    """Nature-style marginal panels: a latitude profile on the right *and* a longitude
    profile below the map, both locked to the map's projected axes.

    ``lat_series`` / ``lon_series`` are dicts ``{label: values}`` aligned with ``lat`` /
    ``lon``.  Labels listed in ``fill`` are drawn as a grey filled envelope (e.g.
    "Maximum water extent"); the others as coloured lines (``colors={label: c}``).
    The bottom panel carries its y axis on the right and a legend to its right, as
    in Pekel et al. (2016, *Nature*).  ``zero_line=True`` adds a 0 reference for
    signed series such as gains/losses.  Returns ``(right_axes, bottom_axes)``; either
    is ``None`` if its series are not given.
    """
    colors = dict(colors or {})
    palette = ["#1f4e9c", "#7cc6e8", "#2ca02c", "#a1123a", "#c8e04a", "#e8a7bd"]
    fmt_lat = _signed_lat if degree_style == "signed" else format_lat
    right = bottom = None
    handles = {}

    def style(label, i):
        return colors.get(label, palette[i % len(palette)])

    if lat_series:
        lat = np.asarray(lat, dtype=float)
        right = ax.inset_axes([1 + right_pad, 0, right_width, 1], transform=ax.transAxes,
                              zorder=zorder)
        y = _proj_coord(ax, lat=lat)
        for i, (label, values) in enumerate(lat_series.items()):
            values = np.asarray(values, dtype=float)
            if label in fill:
                handles[label] = right.fill_betweenx(
                    y, 0, values, facecolor=fill_color, edgecolor=fill_edge, lw=0.6, zorder=1
                )
            else:
                handles[label] = right.plot(values, y, color=style(label, i), lw=linewidth,
                                            zorder=3, solid_joinstyle="round")[0]
        if zero_line:
            right.axvline(0, color="#c0832f", lw=0.6, zorder=2)
        for name, spine in right.spines.items():
            spine.set_visible(name in ("left", "bottom"))
            spine.set_linewidth(0.5)
        right.patch.set_alpha(0)
        ticks = np.asarray(
            lat_ticks if lat_ticks is not None else np.arange(-60, 61, 30), dtype=float
        )
        ticks = ticks[(ticks >= np.nanmin(lat) - 1) & (ticks <= np.nanmax(lat) + 1)]
        pos = _proj_coord(ax, lat=ticks)
        right.yaxis.set_major_locator(FixedLocator(pos))
        right.yaxis.set_major_formatter(
            FuncFormatter(lambda v, _p: fmt_lat(ticks[int(np.argmin(np.abs(pos - v)))]))
        )
        right.yaxis.set_minor_locator(NullLocator())
        if latitude_label:
            right.set_ylabel(latitude_label)
        right.xaxis.set_major_locator(MaxNLocator(4))
        right.xaxis.set_major_formatter(clean_formatter())
        if right_xlim is not None:
            right.set_xlim(right_xlim)
        elif not zero_line:
            right.set_xlim(left=0)
        if right_xlabel:
            right.set_xlabel(right_xlabel)
        right.tick_params(length=2.0, width=0.5, pad=1.5)
        _lock(right, ax, "y")

    if lon_series:
        lon = np.asarray(lon, dtype=float)
        bottom = ax.inset_axes([0, -bottom_pad - bottom_height, 1, bottom_height],
                               transform=ax.transAxes, zorder=zorder)
        x = _proj_coord(ax, lon=lon)
        for i, (label, values) in enumerate(lon_series.items()):
            values = np.asarray(values, dtype=float)
            if label in fill:
                handles.setdefault(label, None)
                handles[label] = bottom.fill_between(
                    x, 0, values, facecolor=fill_color, edgecolor=fill_edge, lw=0.6, zorder=1
                )
            else:
                handles[label] = bottom.plot(x, values, color=style(label, i), lw=linewidth,
                                             zorder=3, solid_joinstyle="round")[0]
        if zero_line:
            bottom.axhline(0, color="#c0832f", lw=0.6, zorder=2)
        for name, spine in bottom.spines.items():
            spine.set_visible(name in ("right", "bottom"))
            spine.set_linewidth(0.5)
        bottom.patch.set_alpha(0)
        bottom.yaxis.tick_right()
        bottom.yaxis.set_label_position("right")
        bottom.tick_params(axis="x", bottom=False, labelbottom=False)
        bottom.yaxis.set_major_locator(MaxNLocator(4, symmetric=zero_line))
        bottom.yaxis.set_major_formatter(clean_formatter())
        if bottom_ylim is not None:
            bottom.set_ylim(bottom_ylim)
        elif not zero_line:
            bottom.set_ylim(bottom=0)
        if bottom_ylabel:
            bottom.set_ylabel(bottom_ylabel)
        bottom.tick_params(length=2.0, width=0.5, pad=1.5)
        _lock(bottom, ax, "x")
        if legend:
            order = [k for k in (legend_labels or handles) if handles.get(k) is not None]
            bottom.legend(
                [handles[k] for k in order], order, loc="center left",
                bbox_to_anchor=(1.0 + right_pad + 0.035, 0.5),
                bbox_transform=bottom.transAxes, frameon=False, handlelength=1.6,
                borderaxespad=0,
            )
    return right, bottom


def _lock(child, parent, which):
    """Keep ``child``'s x or y limits identical to the map's, now and at every draw."""
    original = child.draw

    def sync():
        if which == "y":
            child.set_ylim(parent.get_ylim())
        else:
            child.set_xlim(parent.get_xlim())

    def draw(renderer):
        sync()
        return original(renderer)

    child.draw = draw
    sync()


# --------------------------------------------------------------------------- ternary colours

TERNARY_CORNERS = ("#ff00ff", "#ffff00", "#00ffff")  # magenta, yellow, cyan (subtractive)


def ternary_colors(a, b, c, *, ranges=None, corners=TERNARY_CORNERS, quantiles=(0.02, 0.98)):
    """Map three components to RGBA by barycentric mixing of three corner colours.

    Each component is first rescaled to [0, 1] over its ``range`` (``(lo, hi)``; by
    default the ``quantiles`` of its finite values), then the three rescaled values are
    normalised to proportions that weight the corner colours.  A pixel dominated by
    ``a`` takes ``corners[0]``; balanced pixels tend to grey.  Cells where any component
    is not finite are transparent.  Returns ``(rgba, ranges)``.
    """
    stack = np.stack([np.asarray(v, dtype=float) for v in (a, b, c)])
    valid = np.isfinite(stack).all(axis=0)
    if ranges is None:
        ranges = [tuple(np.nanquantile(v[valid], quantiles)) if valid.any() else (0, 1)
                  for v in stack]
    scaled = np.empty_like(stack)
    for i, (lo, hi) in enumerate(ranges):
        scaled[i] = np.clip((stack[i] - lo) / ((hi - lo) or 1.0), 0, 1)
    total = scaled.sum(axis=0)
    with np.errstate(all="ignore"):
        weights = np.where(total > 0, scaled / total, 1 / 3)
    rgb_corners = np.array([mpl.colors.to_rgb(c) for c in corners])  # (3, 3)
    rgb = np.tensordot(np.moveaxis(weights, 0, -1), rgb_corners, axes=1)
    rgba = np.concatenate([np.clip(rgb, 0, 1), valid[..., None].astype(float)], axis=-1)
    return rgba, [tuple(map(float, r)) for r in ranges]


def plot_rgb(ax, lon, lat, rgba, *, transform=None, zorder=1, **kwargs):
    """Draw an ``(lat, lon, 4)`` RGBA array on a GeoAxes (regular grid, cell centres)."""
    lon, lat = np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)
    dx = abs(lon[1] - lon[0]) / 2 if lon.size > 1 else 0.5
    dy = abs(lat[1] - lat[0]) / 2 if lat.size > 1 else 0.5
    if lat[0] > lat[-1]:
        lat, rgba = lat[::-1], rgba[::-1]
    extent = [lon.min() - dx, lon.max() + dx, lat.min() - dy, lat.max() + dy]
    return ax.imshow(rgba, origin="lower", extent=extent, transform=transform or ccrs.PlateCarree(),
                     interpolation="nearest", zorder=zorder, **kwargs)


def add_ternary_legend(
    ax,
    labels,
    *,
    corners=TERNARY_CORNERS,
    bounds=(0.0, 0.0, 0.2, 0.3),
    resolution=240,
    fontsize=None,
    zorder=9,
    outline=False,
):
    """Triangular colour key for :func:`ternary_colors`, with labels along the edges.

    Corner order matches the components: ``corners[0]`` bottom-right, ``corners[1]``
    bottom-left, ``corners[2]`` top.  ``labels`` = (bottom edge, left edge, right edge),
    e.g. ``("lower (14–30%)", "middle (9–20%)", "upper (50–74%)")``; edge labels are
    rotated to run parallel to their edge.  Returns the inset axes.
    """
    child = ax.inset_axes(list(bounds), transform=ax.transAxes, zorder=zorder)
    h = np.sqrt(3) / 2
    xs = np.linspace(0, 1, resolution)
    ys = np.linspace(0, h, int(resolution * h))
    xx, yy = np.meshgrid(xs, ys)
    # barycentric weights for vertices A=(1,0) [corner 0], B=(0,0) [corner 1], C=(.5,h) [2]
    wc = yy / h
    wa = xx - 0.5 * wc
    wb = 1 - wa - wc
    weights = np.stack([wa, wb, wc], axis=-1)
    inside = (weights >= -1e-9).all(axis=-1)
    rgb_corners = np.array([mpl.colors.to_rgb(c) for c in corners])
    rgb = np.clip(np.clip(weights, 0, 1) @ rgb_corners, 0, 1)
    rgba = np.concatenate([rgb, inside[..., None].astype(float)], axis=-1)
    child.imshow(rgba, origin="lower", extent=[0, 1, 0, h], interpolation="bilinear")
    if outline:
        child.plot([0, 1, 0.5, 0], [0, 0, h, 0], color=INK, lw=0.5)
    child.set_xlim(-0.08, 1.08)
    child.set_ylim(-0.12, h + 0.04)
    child.set_aspect("equal")
    child.axis("off")
    size = fontsize or mpl.rcParams["axes.labelsize"] + 0.5
    bottom_label, left_label, right_label = labels
    halo = _halo_effect()
    child.text(0.5, -0.05, bottom_label, ha="center", va="top", fontsize=size,
               path_effects=halo)
    child.text(0.25 - 0.06, h / 2 + 0.035, left_label, ha="center", va="center", rotation=60,
               rotation_mode="anchor", fontsize=size, path_effects=halo)
    child.text(0.75 + 0.06, h / 2 + 0.035, right_label, ha="center", va="center", rotation=-60,
               rotation_mode="anchor", fontsize=size, path_effects=halo)
    return child


def _halo_effect(width=2.2):
    from matplotlib import patheffects

    return [patheffects.withStroke(linewidth=width, foreground="white")]


def add_inset_density(
    ax,
    groups,
    *,
    colors=None,
    bins=40,
    bounds=(0.0, 0.6, 0.2, 0.3),
    xlabel=None,
    ylabel="Density",
    alpha=0.55,
    legend=True,
    xticks=None,
    zorder=8,
):
    """Overlapping filled density histograms (e.g. the three ternary components)."""
    names = list(groups)
    arrays = [np.asarray(groups[n], dtype=float).ravel() for n in names]
    arrays = [v[np.isfinite(v)] for v in arrays]
    edges = np.histogram_bin_edges(np.concatenate(arrays), bins=bins)
    colors = colors or ["#f05a5a", "#5aaa50", "#6a6af5"][: len(names)]
    child = _inset(ax, list(bounds), zorder)
    _clean_inset(child, "open")
    for v, c, n in zip(arrays, colors, names):
        dens, _ = np.histogram(v, bins=edges, density=True)
        child.stairs(dens, edges, fill=True, color=c, alpha=alpha, label=n, lw=0)
    span = edges[-1] - edges[0]
    child.set_xlim(edges[0] - 0.02 * span, edges[-1] + 0.02 * span)
    child.set_ylim(0, child.get_ylim()[1] * 1.08)
    child.set_yticks([])
    for name in ("left",):
        child.spines[name].set_visible(True)
    if xticks is not None:
        child.set_xticks(xticks)
    else:
        child.xaxis.set_major_locator(MaxNLocator(5, integer=True))
    if xlabel:
        child.set_xlabel(xlabel)
    if ylabel:
        child.set_ylabel(ylabel)
    if legend:
        child.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0), handlelength=1.4,
                     handleheight=0.9, fontsize=mpl.rcParams["legend.fontsize"])
    return child


@contextlib.contextmanager
def figure(width="double", ratio=0.55, **subplot_kw):
    """Convenience: ``with journal.figure('double', .5) as fig:`` in journal style."""
    with journal_style():
        fig = plt.figure(figsize=figsize(width, ratio), **subplot_kw)
        yield fig
