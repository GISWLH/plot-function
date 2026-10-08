"""Original plotting helpers, retained for existing notebooks and advanced layouts."""

from pathlib import Path as FilePath

import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.hatch
import matplotlib.pyplot as plt
import mplotutils as mpu
import numpy as np
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

__all__ = [
    "one_map_flat",
    "mask_ocean",
    "coastlines",
    "hatch_map",
    "one_map",
    "at_warming_level_one",
    "one_map_region",
    "one_map_global_line",
    "one_map_china",
    "add_china",
    "add_dashline",
    "sub_china_map",
    "add_sta",
]


def _cyclic_if_global(da):
    """Close a regular global longitude seam without wrapping regional grids."""
    for name in ("lon", "longitude", "x"):
        if name not in da.coords or da[name].ndim != 1 or da[name].size < 2:
            continue
        values = da[name].values
        steps = np.diff(values)
        if np.allclose(steps, steps[0]) and np.isclose(
            abs(values[-1] - values[0]) + abs(steps[0]), 360
        ):
            return mpu.cyclic_dataarray(da, coord=name)
    return da


def one_map_flat(
    da,
    ax,
    levels=None,
    mask_ocean=False,
    ocean_kws=None,
    add_coastlines=True,
    coastline_kws=None,
    add_land=False,
    land_kws=None,
    colorbar=False,
    plotfunc="pcolormesh",
    **kwargs,
):
    """plot 2D (=flat) DataArray on a cartopy GeoAxes

    Parameters
    ----------
    da : DataArray
        DataArray to plot.
    ax : cartopy.GeoAxes
        GeoAxes to plot da on.
    levels : int or list-like object, optional
        Split the colormap (cmap) into discrete color intervals.
    mask_ocean : bool, default: False
        If true adds the ocean feature.
    ocean_kws : dict, default: None
        Arguments passed to ``ax.add_feature(OCEAN)``.
    add_coastlines : bool, default: True
        If true, plots coastlines. See coastline_kws.
    coastline_kws : dict, default: None
        Arguments passed to ``ax.coastlines()``.
    add_land : bool, default: False
        If true adds the land feature. See land_kws.
    land_kws : dict, default: None
        Arguments passed to ``ax.add_feature(LAND)``.
    plotfunc : {"pcolormesh", "contourf"}, default: "pcolormesh"
        Which plot function to use
    **kwargs : keyword arguments
        Further keyword arguments passed to the plotting function.

    Returns
    -------
    h : handle (artist)
    The same type of primitive artist that the wrapped matplotlib
    function returns
    """

    # ploting options
    opt = dict(
        transform=ccrs.PlateCarree(),
        add_colorbar=False,
        rasterized=True,
        extend="both",
        levels=levels,
        add_labels=False,
    )
    # allow to override the defaults
    opt.update(kwargs)

    if land_kws is None:
        land_kws = dict(fc="0.8", ec="none")

    if add_land:
        ax.add_feature(cfeature.LAND, **land_kws)

    if "contour" in plotfunc:
        opt.pop("rasterized", None)
        da = _cyclic_if_global(da)
        plotfunc = getattr(da.plot, plotfunc)
    elif plotfunc == "pcolormesh":
        plotfunc = getattr(da.plot, plotfunc)
    else:
        raise ValueError(f"Unknown plotfunc: {plotfunc}")

    h = plotfunc(ax=ax, **opt)

    if mask_ocean:
        ocean_kws = {} if ocean_kws is None else ocean_kws
        _mask_ocean(ax, **ocean_kws)

    if coastline_kws is None:
        coastline_kws = dict()

    if add_coastlines:
        coastlines(ax, **coastline_kws)

    # make the spines a bit finer
    s = ax.spines["geo"]
    s.set_lw(0.5)
    s.set_color("0.5")

    if colorbar:
        factor = 1
        colorbar_opt = dict(
            mappable=h,
            ax1=ax,
            size=0.05,  # height
            shrink=0.05 * factor,  # width
            orientation="horizontal",
            pad=0.16,  # interval
        )
        cbar = mpu.colorbar(**colorbar_opt)
        #        cbar.set_label('C', labelpad=1, size=9)
        cbar.ax.tick_params(labelsize=9)

    ax.set_global()

    return h


def mask_ocean(ax, facecolor="w", zorder=1.1, lw=0, **kwargs):
    """plot the ocean feature on a cartopy GeoAxes

    Parameters
    ----------
    ax : cartopy.GeoAxes
        GeoAxes to plot the ocean.
    facecolor : matplotlib color, default: "w"
        Color the plot the ocean in.
    zorder : float, default: 1.2
        Zorder of the ocean mask. Slightly more than 1 so it's higher than a normal
        artist.
    lw : float, default: 0
        With of the edge. Set to 0 to avoid overlaps with the land and coastlines.
    **kwargs : keyword arguments
        Additional keyword arguments to be passed to ax.add_feature.

    """
    NEF = cfeature.NaturalEarthFeature
    OCEAN = NEF(
        "physical",
        "ocean",
        "110m",
    )
    ax.add_feature(OCEAN, facecolor=facecolor, zorder=zorder, lw=lw, **kwargs)


# to use in one_map_flat so the name does not get shadowed
_mask_ocean = mask_ocean


def coastlines(ax, color="0.1", lw=1, zorder=1.2, **kwargs):
    """plot coastlines on a cartopy GeoAxes

    Parameters
    ----------
    ax : cartopy.GeoAxes
        GeoAxes to plot the coastlines.
    color : matplotlib color, default: "0.1"
        Color the plot the coastlines.
    lw : float, default: 0
        With of the edge. Set to 0 to avoid overlaps with the land and coastlines.
    zorder : float, default: 1.2
        Zorder of the ocean mask - slightly more than the ocean.
    **kwargs : keyword arguments
        Additional keyword arguments to be passed to ax.add_feature.
    """
    return ax.coastlines(color=color, lw=lw, zorder=zorder, **kwargs)


def hatch_map(ax, da, hatch, label, invert=False, linewidth=0.25, color="0.1"):
    """add hatch pattern to a cartopy map

    Parameters
    ----------
    ax : matplotlib.axes
        Axes to draw the hatch on.
    da : xr.DataArray
        DataArray with the hatch information. Data of value 1 is hatched.
    hatch : str
        Hatch pattern.
    label : str
        label for a legend entry
    invert : bool, default: False
        If True hatches 0 values instead.
    linewidth : float, default: 0.25
        Default thickness of the hatching.
    color : matplotlib color, default: "0.1"
        Color of the hatch lines.

    Returns
    -------
    legend_handle : handle for the legend entry
    """

    # dummpy patch for the legend entry
    legend_handle = mpl.patches.Patch(
        facecolor="none",
        ec=color,
        lw=linewidth,
        hatch=hatch,
        label=label,
    )

    mn = da.min().item()
    mx = da.max().item()
    if mx > 1 or mn < 0:
        raise ValueError(f"Expected da in 0..1, got {mn}..{mx}")

    # ~ does only work for bool
    if invert:
        da = np.abs(da - 1)

    # Check after inversion: an all-zero mask may intentionally hatch everywhere.
    if not bool((da >= 0.95).any()):
        return legend_handle

    da = _cyclic_if_global(da)

    # plot "True"
    levels = [0.95, 1.05]
    hatches = [hatch, ""]

    with mpl.rc_context({"hatch.linewidth": linewidth, "hatch.color": color}):
        contours = da.plot.contourf(
            ax=ax,
            levels=levels,
            hatches=hatches,
            colors="none",
            extend="neither",
            transform=ccrs.PlateCarree(),
            add_colorbar=False,
            add_labels=False,
        )
    # Store styling on the artist so deferred drawing does not depend on global rcParams.
    collections = [contours] if hasattr(contours, "set_edgecolor") else contours.collections
    for collection in collections:
        collection.set_edgecolor(color)
        collection.set_linewidth(0)
        if hasattr(collection, "set_hatch_linewidth"):
            collection.set_hatch_linewidth(linewidth)

    return legend_handle


def one_map(
    da,
    ax,
    average=None,
    dim=None,
    levels=None,
    mask_ocean=False,
    ocean_kws=None,
    skipna=None,
    add_coastlines=True,
    coastline_kws=None,
    hatch_data=None,
    add_land=False,
    land_kws=None,
    plotfunc="pcolormesh",
    colorbar=False,
    getmean=False,
    **kwargs,
):
    """flatten and plot a 3D DataArray on a cartopy GeoAxes, maybe add simple hatch

    Parameters
    ----------
    da : DataArray
        DataArray to plot.
    ax : cartopy.GeoAxes
        GeoAxes to plot da on.
    average : str
        Function to reduce da with (along dim), e.g. "mean", "median".
    dim : str, default: None
        Dimension to reduce da over when getmean=True.
    levels : int or list-like object, optional
        Split the colormap (cmap) into discrete color intervals.
    mask_ocean : bool, default: False
        If true adds the ocean feature.
    ocean_kws : dict, default: None
        Arguments passed to ``ax.add_feature(OCEAN)``.
    skipna : bool, optional
        If True, skip missing values (as marked by NaN). By default, only
        skips missing values for float dtypes
    add_coastlines : bool, default: True
        If true, plots coastlines. See coastline_kws.
    coastline_kws : dict, default: None
        Arguments passed to ``ax.coastlines()``.
    hatch_data : DataArray, default: None
        Agreement mask in 0..1; its inverse is hatched.
    add_land : bool, default: False
        If true adds the land feature. See land_kws.
    land_kws : dict, default: None
        Arguments passed to ``ax.add_feature(LAND)``.
    plotfunc : {"pcolormesh", "contourf"}, default: "pcolormesh"
        Which plot function to use
    getmean : bool, default: False
        Reduce the field with average along dim before plotting.
    **kwargs : keyword arguments
        Further keyword arguments passed to the plotting function.

    Returns
    -------
    h : handle (artist)
        The same type of primitive artist that the wrapped matplotlib
        function returns
    legend_handle
        Handle of the legend (or None):
    """

    # reduce da with the choosen function
    d = da
    if getmean and ((dim is not None) and (average is not None)):
        if average not in {"mean", "median", "min", "max", "sum", "std"}:
            raise ValueError(f"Unsupported reduction: {average}")
        d = getattr(da, average)(dim, skipna=skipna, keep_attrs=True)

    if getmean and ((dim is None) or (average is None)):
        raise ValueError("Can only get mean value when average and dim is specific")

    h = one_map_flat(
        d,
        ax,
        levels=levels,
        mask_ocean=mask_ocean,
        ocean_kws=ocean_kws,
        add_coastlines=add_coastlines,
        coastline_kws=coastline_kws,
        add_land=add_land,
        land_kws=land_kws,
        plotfunc=plotfunc,
        **kwargs,
    )

    if colorbar:
        factor = 1
        colorbar_opt = dict(
            mappable=h,
            ax1=ax,
            size=0.05,  # height
            shrink=0.05 * factor,  # width
            orientation="horizontal",
            pad=0.1,  # interval
        )
        cbar = mpu.colorbar(**colorbar_opt)
        cbar.set_label(da.attrs.get("units", ""), labelpad=1, size=9)
        cbar.ax.tick_params(labelsize=9)

    legend_handle = None
    if hatch_data is not None:
        legend_handle = hatch_map(
            ax,
            hatch_data,
            6 * "/",
            label="Lack of model agreement",
            invert=True,
            linewidth=0.25,
            color="0.1",
        )

    return h, legend_handle


def at_warming_level_one(
    at_warming_c,
    unit,
    title,
    levels,
    average,
    mask_ocean=False,
    colorbar=True,
    ocean_kws=None,
    skipna=None,
    hatch_data=None,
    add_legend=False,
    plotfunc="pcolormesh",
    colorbar_kwargs=None,
    legend_kwargs=None,
    getmean=True,
    **kwargs,
):
    """
    plot at three warming levels: flatten and plot a 3D DataArray on a cartopy GeoAxes,
    maybe add simple hatch

    Parameters
    ----------
    at_warming_c : list of DataArray
        List of three DataArray objects at warming levels to plot.
    unit : str
        Unit of the data. Added as label to the colorbar.
    title : str
        Suptitle of the figure. If average is not "mean" it is added to the title.
    levels : int or list-like object, optional
        Split the colormap (cmap) into discrete color intervals.
    average : str
        Function to reduce da with (along dim), e.g. "mean", "median".
    mask_ocean : bool, default: False
        If true adds the ocean feature.
    colorbar : bool, default: True
        If to add a colorbar to the figure.
    ocean_kws : dict, default: None
        Arguments passed to ``ax.add_feature(OCEAN)``.
    skipna : bool, optional
        If True, skip missing values (as marked by NaN). By default, only
        skips missing values for float dtypes
    hatch_data : DataArray, default: None
        Agreement mask in 0..1; its inverse is hatched.
    add_legend : bool, default: False
        If a legend should be added.
    plotfunc : {"pcolormesh", "contourf"}, default: "pcolormesh"
        Which plot function to use
    colorbar_kwargs : keyword arguments for the colorbar
        Additional keyword arguments passed on to mpu.colorbar
    legend_kwargs : keyword arguments for the legend
        Additional keyword arguments passed on to ax.legend.
    **kwargs : keyword arguments
        Further keyword arguments passed to the plotting function.

    Returns
    -------
    cbar : handle (artist)
        Colorbar handle.
    """

    if len(at_warming_c) != 3:
        raise ValueError("at_warming_c must contain exactly three DataArrays.")
    if add_legend and (not colorbar or hatch_data is None):
        raise ValueError("A legend requires colorbar=True and hatch_data.")

    if average != "mean":
        title += f" – {average}"

    f, axes = plt.subplots(1, 3, subplot_kw=dict(projection=ccrs.Robinson()))
    axes = axes.flatten()

    if colorbar_kwargs is None:
        colorbar_kwargs = dict()

    if legend_kwargs is None:
        legend_kwargs = dict()

    for i in range(3):
        h, legend_handle = one_map(
            da=at_warming_c[i],
            ax=axes[i],
            average=average,
            levels=levels,
            mask_ocean=mask_ocean,
            ocean_kws=ocean_kws,
            skipna=skipna,
            hatch_data=hatch_data[i] if isinstance(hatch_data, (list, tuple)) else hatch_data,
            plotfunc=plotfunc,
            getmean=getmean,
            **kwargs,
        )

    for ax in axes:
        ax.set_global()

    if colorbar:
        factor = 0.66 if add_legend else 1
        ax2 = axes[1] if add_legend else axes[2]

        colorbar_opt = dict(
            mappable=h,
            ax1=axes[0],
            ax2=ax2,
            size=0.15,
            shrink=0.25 * factor,
            orientation="horizontal",
            pad=0.1,
        )
        colorbar_opt.update(colorbar_kwargs)
        cbar = mpu.colorbar(**colorbar_opt)

        cbar.set_label(unit, labelpad=1, size=9)
        cbar.ax.tick_params(labelsize=9)

    if add_legend and (not colorbar or hatch_data is None):
        raise ValueError("Can only add legend when colorbar and hatch_data is True")

    if add_legend:
        # add a text legend entry - the non-hatched regions show high agreement
        h0 = mpl.patches.Patch(facecolor="0.6", label="High model agreement")

        legend_opt = dict(
            handlelength=2.6,
            handleheight=1.3,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.45),
            fontsize=8.5,
            borderaxespad=0,
            frameon=True,
            ncol=1,
        )

        legend_opt.update(legend_kwargs)

        axes[2].legend(handles=[h0, legend_handle], **legend_opt)

    axes[0].set_title("At 1.5°C global warming", fontsize=9, pad=4)
    axes[1].set_title("At 2.0°C global warming", fontsize=9, pad=4)
    axes[2].set_title("At 4.0°C global warming", fontsize=9, pad=4)

    axes[0].set_title("(a)", fontsize=9, pad=4, loc="left")
    axes[1].set_title("(b)", fontsize=9, pad=4, loc="left")
    axes[2].set_title("(c)", fontsize=9, pad=4, loc="left")

    # axes[0].set_title("Tglob anomaly +1.5 °C", fontsize=9, pad=2)
    # axes[1].set_title("Tglob anomaly +2.0 °C", fontsize=9, pad=2)
    # axes[2].set_title("Tglob anomaly +4.0 °C", fontsize=9, pad=2)

    side = 0.01
    subplots_adjust_opt = dict(wspace=0.025, left=side, right=1 - side)
    if colorbar:
        subplots_adjust_opt.update({"bottom": 0.3, "top": 0.82})
    else:
        subplots_adjust_opt.update({"bottom": 0.08, "top": 0.77})

    f.suptitle(title, fontsize=9, y=0.975)
    plt.subplots_adjust(**subplots_adjust_opt)
    mpu.set_map_layout(axes, width=18)

    f.canvas.draw()

    if colorbar:
        return cbar


def one_map_region(
    da,
    ax,
    levels=None,
    mask_ocean=False,
    ocean_kws=None,
    add_coastlines=True,
    coastline_kws=None,
    add_land=False,
    land_kws=None,
    add_gridlines=False,
    colorbar=False,
    plotfunc="pcolormesh",
    extents=None,
    interval=None,
    add_river=False,
    add_lake=False,
    add_stock=False,
    **kwargs,
):
    """plot 2D (=flat) DataArray on a cartopy GeoAxes

    Parameters
    ----------
    da : DataArray
        DataArray to plot.
    ax : cartopy.GeoAxes
        GeoAxes to plot da on.
    levels : int or list-like object, optional
        Split the colormap (cmap) into discrete color intervals.
    mask_ocean : bool, default: False
        If true adds the ocean feature.
    ocean_kws : dict, default: None
        Arguments passed to ``ax.add_feature(OCEAN)``.
    add_coastlines : bool, default: True
        If true, plots coastlines. See coastline_kws.
    coastline_kws : dict, default: None
        Arguments passed to ``ax.coastlines()``.
    add_land : bool, default: False
        If true adds the land feature. See land_kws.
    land_kws : dict, default: None
        Arguments passed to ``ax.add_feature(LAND)``.
    add_stock: bool, default: False
        If true add the stock image
    plotfunc : {"pcolormesh", "contourf"}, default: "pcolormesh"
        Which plot function to use
    add_gridlines : bool, default: False
        If None or true plots gridlines
    extents: list, default: None
        The region specific in the map follows: '[lonMin, lonMax, LatMin, LatMax]'
    interval: List, default: '[30, 60]'
        The intervals in the map follows: '[lonInterval, latInterval]'
    **kwargs : keyword arguments
        Further keyword arguments passed to the plotting function.

    Returns
    -------
    h : handle (artist)
    The same type of primitive artist that the wrapped matplotlib
    function returns
    """

    # ploting options
    opt = dict(
        transform=ccrs.PlateCarree(),
        add_colorbar=False,
        rasterized=True,
        extend="both",
        levels=levels,
        add_labels=False,
    )
    # allow to override the defaults
    opt.update(kwargs)

    if land_kws is None:
        land_kws = dict(fc="0.8", ec="none")

    if add_land:
        ax.add_feature(cfeature.LAND, **land_kws)
    if add_river:
        ax.add_feature(cfeature.RIVERS, lw=0.25)  #####添加河流######
    if add_lake:
        ax.add_feature(cfeature.LAKES)  ######添加湖泊#####
    if add_stock:
        ax.stock_img()

    if "contour" in plotfunc:
        opt.pop("rasterized", None)
        da = _cyclic_if_global(da)
        plotfunc = getattr(da.plot, plotfunc)
    elif plotfunc == "pcolormesh":
        plotfunc = getattr(da.plot, plotfunc)
    else:
        raise ValueError(f"Unknown plotfunc: {plotfunc}")

    h = plotfunc(ax=ax, **opt)

    if mask_ocean:
        ocean_kws = {} if ocean_kws is None else ocean_kws
        _mask_ocean(ax, **ocean_kws)

    if coastline_kws is None:
        coastline_kws = dict()

    if add_coastlines:
        coastlines(ax, **coastline_kws)

    # make the spines a bit finer
    s = ax.spines["geo"]
    s.set_lw(0.5)
    s.set_color("0.5")
    if extents is not None:
        ax.set_extent(extents, crs=ccrs.PlateCarree())

    # add the x y ticks
    if interval is not None:
        if extents is None:
            raise ValueError("extents is required when interval is provided.")
        if len(interval) != 2 or min(interval) <= 0:
            raise ValueError("interval must contain positive longitude and latitude steps.")
        ax.set_xticks(np.arange(extents[0], extents[1] + interval[0], interval[0]))
        ax.set_xticks(
            np.arange(extents[0], extents[1] + interval[0] / 2, interval[0] / 2), minor=True
        )
        ax.set_yticks(np.arange(extents[2], extents[3] + interval[1], interval[1]))
        ax.set_yticks(
            np.arange(extents[2], extents[3] + interval[1] / 2, interval[1] / 2), minor=True
        )
        ax.xaxis.set_major_formatter(LongitudeFormatter())
        ax.yaxis.set_major_formatter(LatitudeFormatter())

    # add the gridlines
    if add_gridlines:
        grid_options = {}
        if interval is not None:
            grid_options = dict(
                xlocs=np.arange(extents[0], extents[1] + interval[0] / 2, interval[0] / 2),
                ylocs=np.arange(extents[2], extents[3] + interval[1] / 2, interval[1] / 2),
            )
        ax.gridlines(linestyle="--", **grid_options)

    if colorbar:
        factor = 1
        colorbar_opt = dict(
            mappable=h,
            ax1=ax,
            size=0.05,  # height
            shrink=0.05 * factor,  # width
            orientation="horizontal",
            pad=0.16,  # interval
        )
        cbar = mpu.colorbar(**colorbar_opt)
        #        cbar.set_label('C', labelpad=1, size=9)
        cbar.ax.tick_params(labelsize=9)

    return h


def one_map_global_line(
    da,
    ax,
    levels=None,
    mask_ocean=False,
    ocean_kws=None,
    add_coastlines=True,
    coastline_kws=None,
    add_land=False,
    land_kws=None,
    colorbar=False,
    plotfunc="pcolormesh",
    **kwargs,
):
    """plot 2D (=flat) DataArray on a cartopy GeoAxes

    Parameters
    ----------
    da : DataArray
        DataArray to plot.
    ax : cartopy.GeoAxes
        GeoAxes to plot da on.
    levels : int or list-like object, optional
        Split the colormap (cmap) into discrete color intervals.
    mask_ocean : bool, default: False
        If true adds the ocean feature.
    ocean_kws : dict, default: None
        Arguments passed to ``ax.add_feature(OCEAN)``.
    add_coastlines : bool, default: True
        If true, plots coastlines. See coastline_kws.
    coastline_kws : dict, default: None
        Arguments passed to ``ax.coastlines()``.
    add_land : bool, default: False
        If true adds the land feature. See land_kws.
    land_kws : dict, default: None
        Arguments passed to ``ax.add_feature(LAND)``.
    plotfunc : {"pcolormesh", "contourf"}, default: "pcolormesh"
        Which plot function to use
    add_gridlines : bool, default: False
        If None or true plots gridlines
    **kwargs : keyword arguments
        Further keyword arguments passed to the plotting function.

    Returns
    -------
    h : handle (artist)
    The same type of primitive artist that the wrapped matplotlib
    function returns
    """

    # ploting options
    opt = dict(
        transform=ccrs.PlateCarree(),
        add_colorbar=False,
        rasterized=True,
        extend="both",
        levels=levels,
        add_labels=False,
    )
    # allow to override the defaults
    opt.update(kwargs)

    if land_kws is None:
        land_kws = dict(fc="0.8", ec="none")

    if add_land:
        ax.add_feature(cfeature.LAND, **land_kws)

    if "contour" in plotfunc:
        opt.pop("rasterized", None)
        da = _cyclic_if_global(da)
        plotfunc = getattr(da.plot, plotfunc)
    elif plotfunc == "pcolormesh":
        plotfunc = getattr(da.plot, plotfunc)
    else:
        raise ValueError(f"Unknown plotfunc: {plotfunc}")

    h = plotfunc(ax=ax, **opt)

    if mask_ocean:
        ocean_kws = {} if ocean_kws is None else ocean_kws
        _mask_ocean(ax, **ocean_kws)

    if coastline_kws is None:
        coastline_kws = dict()

    if add_coastlines:
        coastlines(ax, **coastline_kws)

    # make the spines a bit finer
    s = ax.spines["geo"]
    s.set_lw(0.5)
    s.set_color("0.5")

    # add the gridlines
    ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=1,
        color="gray",
        alpha=0.5,
        linestyle="--",
    )

    if colorbar:
        factor = 1
        colorbar_opt = dict(
            mappable=h,
            ax1=ax,
            size=0.05,  # height
            shrink=0.05 * factor,  # width
            orientation="horizontal",
            pad=0.16,  # interval
        )
        cbar = mpu.colorbar(**colorbar_opt)
        #        cbar.set_label('C', labelpad=1, size=9)
        cbar.ax.tick_params(labelsize=9)

    return h


def one_map_china(
    da,
    ax,
    levels=None,
    mask_ocean=False,
    ocean_kws=None,
    add_coastlines=True,
    coastline_kws=None,
    add_land=False,
    land_kws=None,
    add_gridlines=False,
    colorbar=False,
    plotfunc="pcolormesh",
    add_river=False,
    add_lake=False,
    add_stock=False,
    **kwargs,
):
    """plot 2D (=flat) DataArray on a cartopy GeoAxes

    Parameters
    ----------
    da : DataArray
        DataArray to plot.
    ax : cartopy.GeoAxes
        GeoAxes to plot da on.
    levels : int or list-like object, optional
        Split the colormap (cmap) into discrete color intervals.
    mask_ocean : bool, default: False
        If true adds the ocean feature.
    ocean_kws : dict, default: None
        Arguments passed to ``ax.add_feature(OCEAN)``.
    add_coastlines : bool, default: True
        If true, plots coastlines. See coastline_kws.
    coastline_kws : dict, default: None
        Arguments passed to ``ax.coastlines()``.
    add_land : bool, default: False
        If true adds the land feature. See land_kws.
    land_kws : dict, default: None
        Arguments passed to ``ax.add_feature(LAND)``.
    add_stock: bool, default: False
        If true add the stock image
    plotfunc : {"pcolormesh", "contourf"}, default: "pcolormesh"
        Which plot function to use
    add_gridlines : bool, default: False
        If None or true plots gridlines
    **kwargs : keyword arguments
        Further keyword arguments passed to the plotting function.

    Returns
    -------
    h : handle (artist)
    The same type of primitive artist that the wrapped matplotlib
    function returns
    """

    # ploting options
    opt = dict(
        transform=ccrs.PlateCarree(),
        add_colorbar=False,
        rasterized=True,
        extend="both",
        levels=levels,
        add_labels=False,
    )
    # allow to override the defaults
    opt.update(kwargs)

    if land_kws is None:
        land_kws = dict(fc="0.8", ec="none")

    if add_land:
        ax.add_feature(cfeature.LAND, **land_kws)
    if add_river:
        ax.add_feature(cfeature.RIVERS, lw=0.25)  #####添加河流######
    if add_lake:
        ax.add_feature(cfeature.LAKES)  ######添加湖泊#####
    if add_stock:
        ax.stock_img()

    if "contour" in plotfunc:
        opt.pop("rasterized", None)
        da = _cyclic_if_global(da)
        plotfunc = getattr(da.plot, plotfunc)
    elif plotfunc == "pcolormesh":
        plotfunc = getattr(da.plot, plotfunc)
    else:
        raise ValueError(f"Unknown plotfunc: {plotfunc}")

    h = plotfunc(ax=ax, **opt)
    add_dashline(ax, ec="black", linewidth=1)
    add_china(ax, ec="black", fc="None", linewidth=1)
    #    h = add_china(ax=ax)

    if mask_ocean:
        ocean_kws = {} if ocean_kws is None else ocean_kws
        _mask_ocean(ax, **ocean_kws)

    if coastline_kws is None:
        coastline_kws = dict()

    if add_coastlines:
        coastlines(ax, **coastline_kws)

    # make the spines a bit finer
    s = ax.spines["geo"]
    s.set_lw(0.5)
    s.set_color("0.5")

    ax.set_extent([70, 140, 15, 55], crs=ccrs.PlateCarree())

    # add the x y ticks

    # ax.set_xticks(np.arange(70, 140 + 20, 20))
    # ax.set_xticks(np.arange(70, 140 + 10, 10), minor=True)
    # ax.set_yticks(np.arange(15, 55 + 20, 20))
    # ax.set_yticks(np.arange(15, 55 + 10, 10), minor=True)
    # ax.xaxis.set_major_formatter(LongitudeFormatter())
    # ax.yaxis.set_major_formatter(LatitudeFormatter())
    if colorbar:
        factor = 1
        colorbar_opt = dict(
            mappable=h,
            ax1=ax,
            size=0.05,  # height
            shrink=0.05 * factor,  # width
            orientation="horizontal",
            pad=0.16,  # interval
        )
        cbar = mpu.colorbar(**colorbar_opt)
        #        cbar.set_label('C', labelpad=1, size=9)
        cbar.ax.tick_params(labelsize=9)

    # add the gridlines
    if add_gridlines:
        ax.gridlines(
            crs=ccrs.PlateCarree(),
            linestyle="--",
            xlocs=np.arange(70, 140 + 10, 10),
            draw_labels=True,
            ylocs=np.arange(15, 55 + 10, 10),
            y_inline=False,
            x_inline=False,
        )

    return h


def _boundary_file(name, shapefile):
    path = (
        FilePath(shapefile)
        if shapefile is not None
        else FilePath(__file__).resolve().parents[1] / "data" / name
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"Boundary file not found: {path}. Legacy boundaries need the repository's "
            "data directory or an explicit shapefile= path. The NetCDF API needs neither."
        )
    return path


def add_china(ax, shapefile=None, **kwargs):
    """
    Plot the Chinese province map shapefile.

    Parameters
    ----------
    ax : targate GeoAxes
    **kwargs
        Parameter when plot shapefile e.g. linewidth, edgecolor and facecolor etc.
    """
    proj = ccrs.PlateCarree()
    reader = shpreader.Reader(_boundary_file("china.shp", shapefile))
    provinces = list(reader.geometries())
    ax.add_geometries(provinces, proj, **kwargs)
    reader.close()


def add_dashline(ax, shapefile=None, **kwargs):
    """
    Plot the Chinese dashline map shapefile.

    Parameters
    ----------
    ax : targate GeoAxes
    **kwargs
        Parameter when plot dashline e.g. linewidth, edgecolor and facecolor etc.
    """
    proj = ccrs.PlateCarree()
    reader = shpreader.Reader(_boundary_file("dashline.shp", shapefile))
    provinces = list(reader.geometries())
    ax.add_geometries(provinces, proj, **kwargs)
    reader.close()


def sub_china_map(
    da,
    ax,
    levels=None,
    add_coastlines=True,
    coastline_kws=None,
    add_land=False,
    land_kws=None,
    add_stock=False,
    plotfunc="pcolormesh",
    **kwargs,
):
    """plot 2D (=flat) DataArray on a cartopy GeoAxes

    Parameters
    ----------
    da : DataArray
        DataArray to plot.
    ax : cartopy.GeoAxes
        GeoAxes to plot da on.
    levels : int or list-like object, optional
        Split the colormap (cmap) into discrete color intervals.
    add_coastlines : bool, default: True
        If true, plots coastlines. See coastline_kws.
    coastline_kws : dict, default: None
        Arguments passed to ``ax.coastlines()``.
    add_land : bool, default: False
        If true adds the land feature. See land_kws.
    land_kws : dict, default: None
        Arguments passed to ``ax.add_feature(LAND)``.
    add_stock: bool, default: False
        If true add the stock image
    plotfunc : {"pcolormesh", "contourf"}, default: "pcolormesh"
        Which plot function to use
    **kwargs : keyword arguments
        Further keyword arguments passed to the plotting function.

    Returns
    -------
    h : handle (artist)
    The same type of primitive artist that the wrapped matplotlib
    function returns
    """

    # ploting options
    opt = dict(
        transform=ccrs.PlateCarree(),
        add_colorbar=False,
        rasterized=True,
        extend="both",
        levels=levels,
        add_labels=False,
    )
    # allow to override the defaults
    opt.update(kwargs)

    if land_kws is None:
        land_kws = dict(fc="0.8", ec="none")

    if add_land:
        ax.add_feature(cfeature.LAND, **land_kws)

    if "contour" in plotfunc:
        opt.pop("rasterized", None)
        da = _cyclic_if_global(da)
        plotfunc = getattr(da.plot, plotfunc)
    elif plotfunc == "pcolormesh":
        plotfunc = getattr(da.plot, plotfunc)
    else:
        raise ValueError(f"Unknown plotfunc: {plotfunc}")

    h = plotfunc(ax=ax, **opt)

    if coastline_kws is None:
        coastline_kws = dict()

    if add_coastlines:
        coastlines(ax, **coastline_kws)
    if add_stock:
        ax.stock_img()

    # make the spines a bit finer
    s = ax.spines["geo"]
    s.set_lw(0.5)
    s.set_color("0.5")

    ax.set_extent([104.5, 125, 0, 26], crs=ccrs.PlateCarree())
    ax.gridlines(
        draw_labels=False,
        x_inline=False,
        y_inline=False,
        linewidth=0.1,
        color="gray",
        alpha=0.8,
        linestyle="--",
    )
    add_dashline(ax, ec="black", linewidth=1)
    add_china(ax, ec="black", fc="None", linewidth=1)

    return h


def add_sta(ax, da, xlim, lat, **kwargs):
    """Plot a latitude profile of the zonal mean and ±1 standard deviation.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes for the profile.
    da : xarray.DataArray
        Two-dimensional geographic field in either dimension order.
    xlim : pair of numbers
        Horizontal limits in the field's units.
    lat : str
        Name of the latitude dimension, typically ``lat`` or ``y``.
    **kwargs
        Keyword arguments passed to the mean line's ``ax.plot`` call.
    """

    ax.patch.set_alpha(0)
    if da.ndim != 2 or lat not in da.dims:
        raise ValueError("Expected a 2D DataArray with the specified latitude dimension.")
    other = next(dim for dim in da.dims if dim != lat)
    MED = da.mean(other, skipna=True).values
    STD = da.std(other, skipna=True).values
    ydata = da[lat].values

    ax.tick_params(
        top=False,
        bottom=False,
        left=False,
        right=False,
        labeltop=False,
        labelbottom=True,
        labelleft=False,
        labelright=False,
    )
    # Plots the results for the main area, with the other areas extremely close or at 0.
    line_options = dict(c="#1f77b4", linewidth=0.75, label="Mean", zorder=10)
    line_options.update(kwargs)
    ax.plot(MED, ydata, **line_options)
    ax.fill_betweenx(
        ydata,
        MED + STD,
        MED - STD,
        alpha=0.12,
        linewidth=0,
        color="#1f77b4",
        label="1 std",
        zorder=10,
    )
    # ax2.yaxis.set_major_formatter(LatitudeFormatter())
    ax.set_ylim([-60, 90])
    ax.set_xlim(xlim)

    ax.set_yticks(np.arange(-60, 91, 30))
    ax.spines["top"].set_color("#b0b0b0")
    ax.spines["right"].set_color("#b0b0b0")
    ax.spines["bottom"].set_color("#b0b0b0")
    ax.spines["left"].set_color("#b0b0b0")
