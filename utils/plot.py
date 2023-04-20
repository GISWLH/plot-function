import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.hatch
import matplotlib.pyplot as plt
import mplotutils as mpu
import numpy as np
from matplotlib.path import Path

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
    add_coastlines : bool, default: None
        If None or true plots coastlines. See coastline_kws.
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
    )
    # allow to override the defaults
    opt.update(kwargs)

    if land_kws is None:
        land_kws = dict(fc="0.8", ec="none")

    if add_land:
        ax.add_feature(cfeature.LAND, **land_kws)

    if "contour" in plotfunc:
        opt.pop("rasterized", None)
        da = mpu.cyclic_dataarray(da)
        plotfunc = getattr(da.plot, plotfunc)
    elif plotfunc == "pcolormesh":
        plotfunc = getattr(da.plot, plotfunc)
    else:
        raise ValueError(f"unkown plotfunc: {plotfunc}")

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
    ax.coastlines(color=color, lw=lw, zorder=zorder, *kwargs)
    
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
    ax.coastlines(color=color, lw=lw, zorder=zorder, *kwargs)

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
        raise ValueError("Expected da in 0..1, got {mn}..{mx}")

    # contourf has trouble if no gridcell is True
    if da.sum() == 0:
        return legend_handle

    # ~ does only work for bool
    if invert:
        da = np.abs(da - 1)

    da = mpu.cyclic_dataarray(da)

    # plot "True"
    levels = [0.95, 1.05]
    hatches = [hatch, ""]

    mpl.rcParams["hatch.linewidth"] = linewidth
    mpl.rcParams["hatch.color"] = color

    # unfortunately cannot set options via context manager
    # with mpl.rc_context({"hatch.linewidth": linewidth, "hatch.color": color}):
    da.plot.contourf(
        ax=ax,
        levels=levels,
        hatches=hatches,
        colors="none",
        extend="neither",
        transform=ccrs.PlateCarree(),
        add_colorbar=False,
    )

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
    colorbar = False,
    getmean = False,
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
    dim : str, default: "mod_ens"
        Dimension to reduce da over.
    levels : int or list-like object, optional
        Split the colormap (cmap) into discrete color intervals.
    mask_ocean : bool, default: False
        If true adds the ocean feature.
    ocean_kws : dict, default: None
        Arguments passed to ``ax.add_feature(OCEAN)``.
    skipna : bool, optional
        If True, skip missing values (as marked by NaN). By default, only
        skips missing values for float dtypes
    add_coastlines : bool, default: None
        If None or true plots coastlines. See coastline_kws.
    coastline_kws : dict, default: None
        Arguments passed to ``ax.coastlines()``.
    hatch_simple : float, default: None
        If not None determines hatching on the fraction of models with the same sign.
        hatch_simple must be in 0..1.
    add_land : bool, default: False
        If true adds the land feature. See land_kws.
    land_kws : dict, default: None
        Arguments passed to ``ax.add_feature(LAND)``.
    plotfunc : {"pcolormesh", "contourf"}, default: "pcolormesh"
        Which plot function to use
    add_n_models : bool, default: True
        If True adds to number of models in the top right of the map. May only work for
        the Robinson projection.
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
        d = getattr(da, average)(dim, skipna=skipna)
        
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
            size=0.05, #height
            shrink=0.05 * factor, #width
            orientation="horizontal",
            pad=0.1, #interval
        )
        cbar = mpu.colorbar(**colorbar_opt)
        cbar.set_label('C', labelpad=1, size=9)
        cbar.ax.tick_params(labelsize=9)
        
    if hatch_data is not None:
        h = hatch_map(
            ax,
            hatch_data,
            6 * "/",
            label="Lack of model agreement",
            invert=True,
            linewidth=0.25,
            color="0.1",
        )
        
    return h, None

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
    hatch_simple : float, default: None
        If not None determines hatching on the fraction of models with the same sign.
        hatch_simple must be in 0..1.
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
            hatch_data=hatch_data,
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
        cbar = mpu.colorbar(mappable=h, ax1=axes[0], ax2=ax2, size=0.15, shrink=0.25 * factor, orientation="horizontal", pad=0.1)

        cbar.set_label(unit, labelpad=1, size=9)
        cbar.ax.tick_params(labelsize=9)

    if add_legend and (not colorbar or hatch_data is None):
        raise ValueError("Can only add legend when colorbar and hatch_data is True")

    if add_legend:
        # add a text legend entry - the non-hatched regions show high agreement
        h0 = text_legend(ax, "Colour", "High model agreement", size=7)

        legend_opt = dict(
            handlelength=2.6,
            handleheight=1.3,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.45),
            fontsize=8.5,
            borderaxespad=0,
            frameon=True,
            handler_map={mpl.text.Text: TextHandler()},
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