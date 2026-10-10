"""Journal-grade figure recipes built with ``plot_function.journal``.

Every figure below is reproducible from this script alone.  Figures 1, 2 and 4
use **synthetic example data** (seeded random fields / sites, clearly labelled in
the figures); figure 3 uses the bundled ERA5 1978 monthly 2 m temperature file.

    python examples/journal_figures.py                 # writes docs/gallery/*.png
    python examples/journal_figures.py --output out/   # elsewhere
    python examples/journal_figures.py --dpi 600 --pdf # print-ready
"""

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cartopy.crs as ccrs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

import plot_function.journal as pj  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PC = ccrs.PlateCarree()


# ----------------------------------------------------------------------------- data helpers


def land_mask(lon, lat):
    """Boolean land mask on a lon/lat grid from Natural Earth 110m land (cached by Cartopy)."""
    import cartopy.io.shapereader as shpreader
    import shapely
    from shapely.ops import unary_union

    geoms = list(shpreader.Reader(shpreader.natural_earth("110m", "physical", "land")).geometries())
    land = unary_union(geoms)
    shapely.prepare(land)
    xx, yy = np.meshgrid(lon, lat)
    return shapely.contains_xy(land, xx, yy)


def smooth_field(rng, lon, lat, n=60, scale=(8, 22)):
    """Sum of random Gaussian blobs: a smooth synthetic field in roughly [-1, 1]."""
    xx, yy = np.meshgrid(lon, lat)
    out = np.zeros_like(xx, dtype=float)
    for _ in range(n):
        x0 = rng.uniform(np.min(lon) - 10, np.max(lon) + 10)
        y0 = rng.uniform(max(np.min(lat), -60) - 5, min(np.max(lat), 80) + 5)
        sx, sy = rng.uniform(*scale), rng.uniform(*scale) * 0.7
        dx = (xx - x0 + 180) % 360 - 180
        out += rng.normal() * np.exp(-((dx / sx) ** 2) - ((yy - y0) / sy) ** 2)
    return out / np.nanmax(np.abs(out))


def footnote(fig, text, y=-0.01):
    fig.text(0.01, y, text, fontsize=5.5, color=pj.MUTED, ha="left", va="top")


# ----------------------------------------------------------------------------- figure 1


def fig_regimes(out, dpi, pdf):
    """Two-regime diverging map + stippling + stacked inset bars + aligned zonal profile."""
    rng = np.random.default_rng(42)
    lon = np.arange(-179.5, 180, 1.0)
    lat = np.arange(-59.5, 90, 1.0)
    land = land_mask(lon, lat)
    land &= lat[:, None] > -58
    n_mem = 12
    signal = 12 * smooth_field(rng, lon, lat, n=160, scale=(5, 16))
    shared = 5 * smooth_field(rng, lon, lat, n=200, scale=(2, 6))  # small-scale texture
    members = np.stack(
        [signal + shared + 9 * smooth_field(rng, lon, lat, n=90, scale=(6, 20))
         for _ in range(n_mem)]
    )
    members[:, ~land] = np.nan
    delta = np.nanmean(members, axis=0)
    agree = (np.sign(members) == np.sign(delta)).sum(0)
    low_agree = land & (agree < 8)
    regime = smooth_field(rng, lon, lat, n=50, scale=(15, 35)) > 0  # True: blue, False: green
    blue = np.where(land & regime, delta, np.nan)
    green = np.where(land & ~regime, -delta, np.nan)

    levels = np.arange(-10, 10.1, 2.5)
    brown_blue = ["#8c6d46", "#a68a62", "#cdb994", "#ece2c8", "#fbf6e6",
                  "#e9f3fb", "#b6d5f0", "#7cb2e6", "#3f86d9", "#2a6fbf"]
    green_purple = ["#1f7a1f", "#3d9a3d", "#77b96f", "#b9dbb0", "#eef7ea",
                    "#e7e0ef", "#c8b5da", "#a68cc4", "#8566ad", "#6a4b95"]
    cmap_b, norm = pj.discrete_cmap(brown_blue, levels, extend="both")
    cmap_g, _ = pj.discrete_cmap(green_purple, levels, extend="both")

    with pj.journal_style():
        fig = plt.figure(figsize=pj.figsize("double", 0.49))
        ax = fig.add_axes([0.065, 0.255, 0.79, 0.72], projection=PC)
        ax.set_extent([-180, 180, -60, 90], crs=PC)
        ax.coastlines("110m", lw=0.35, color=pj.INK, zorder=3)
        ax.pcolormesh(lon, lat, blue, cmap=cmap_b, norm=norm, transform=PC, zorder=1)
        mg = ax.pcolormesh(lon, lat, green, cmap=cmap_g, norm=norm, transform=PC, zorder=1)
        mb = ax.collections[-2]
        pj.geo_ticks(ax, xticks=range(-180, 181, 60), yticks=range(-40, 81, 20))
        pj.add_stippling(ax, lon, lat, low_agree, stride=3, size=0.9)
        pj.add_panel_label(ax, "a")

        # lower-left inset: land area per regime, hatched part = low agreement
        area = np.cos(np.deg2rad(lat))[:, None] * np.ones_like(delta)
        tot = np.nansum(area[land])
        cats = {
            "Greener": land & ~regime & (green < 0),
            "Less\ngreen": land & ~regime & (green >= 0),
            "Less\nblue": land & regime & (blue < 0),
            "Bluer": land & regime & (blue >= 0),
        }
        vals = [100 * area[m].sum() / tot for m in cats.values()]
        low = [100 * area[m & low_agree].sum() / tot for m in cats.values()]
        pj.add_inset_bars(
            ax,
            list(cats),
            vals,
            hatched=low,
            colors=["#4f9a4f", "#a68cc4", "#c9b58f", "#5ea3e6"],
            bounds=(0.075, 0.13, 0.22, 0.36),
            ylabel="Land area [%]",
        )

        # right marginal: zonal mean with ensemble IQR and min-max envelope
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            zonal = np.nanmean(members, axis=2)  # (member, lat)
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            q25, q50, q75 = np.nanpercentile(zonal, [25, 50, 75], axis=0)
            mn, mx = np.nanmin(zonal, 0), np.nanmax(zonal, 0)
        ok = np.isfinite(q50)
        pj.add_latitude_profile(
            ax, lat[ok], q50[ok], lower=q25[ok], upper=q75[ok], outer=(mn[ok], mx[ok]),
            width=0.115, pad=0.03, color="#2b5d8c", xlabel="Zonal mean\nΔBGWS [ppts]",
            yticks=range(-40, 81, 20), xlim=(-8, 8),
        )

        # legend for stippling (lower right, inside map)
        ax.legend(
            [pj.hatch_patch("....", facecolor="white")],
            ["Low ensemble\nsign agreement (<8/12)"],
            loc="lower right", bbox_to_anchor=(0.995, 0.01), frameon=True, fancybox=False,
            edgecolor="none", framealpha=0.9, handlelength=2.0, handleheight=1.4,
            prop={"weight": "bold", "size": 6.5},
        )
        pj.add_colorbar(mb, ax, title="Historical blue water regime", label="ΔBGWS [ppts]",
                        bounds=[0.10, 0.11, 0.30, 0.03], tick_every=2)
        pj.add_colorbar(mg, ax, title="Historical green water regime", label="ΔBGWS [ppts]",
                        bounds=[0.52, 0.11, 0.30, 0.03], tick_every=2)
        footnote(fig, "Synthetic example data (seeded random fields, 12 pseudo-members); "
                      "layout after Nature-style blue/green-water regime maps.", y=0.0)
        path = pj.save_figure(fig, out / "fig1_regimes.png", dpi=dpi)
        if pdf:
            pj.save_figure(fig, out / "fig1_regimes.pdf")
        plt.close(fig)
    return path


# ----------------------------------------------------------------------------- figure 2


def fig_sites(out, dpi, pdf):
    """Cream-land site map, sized markers, log histogram + cumulative inset, legends below."""
    rng = np.random.default_rng(7)
    hubs = np.array([[-95, 37], [-75, 41], [10, 47], [32, 30], [77, 24], [105, 30],
                     [-47, -18], [28, -26], [140, -33], [3, 10], [120, 35], [55, 30],
                     [-99, 19], [78, 28], [110, -7], [-58, -34], [36, 0], [-2, 52],
                     [67, 33], [-77, 4], [44, 40], [126, 37], [100, 14]])

    def sites(n, spread):
        idx = rng.integers(0, len(hubs), n)
        pts = hubs[idx] + rng.normal(0, spread, (n, 2))
        return pts[:, 0], np.clip(pts[:, 1], -50, 70)

    groups = {
        "managed": (*sites(108, 6), rng.lognormal(0.3, 1.0, 108)),
        "dumping": (*sites(43, 4), rng.lognormal(0.9, 0.8, 43)),
        "urban": (*sites(46, 5), rng.lognormal(1.2, 0.6, 46)),
    }
    scale = lambda q: 9 + 22 * np.log10(1 + q)  # marker area in pt^2  # noqa: E731
    orange, purple = "#f28e1c", "#9b4fc2"

    with pj.journal_style():
        fig = plt.figure(figsize=pj.figsize("double", 0.5))
        ax = fig.add_axes([0.0, 0.22, 1.0, 0.77], projection=ccrs.Robinson())
        ax.set_extent([-180, 180, -57, 84], crs=PC)
        pj.add_land(ax, borders=True, coast_width=0.3)
        ax.spines["geo"].set_visible(False)
        x, y, q = groups["urban"]
        ax.scatter(x, y, s=scale(q) * 1.3, marker="D", facecolor="#a9a9a9", edgecolor="#6f6f6f",
                   lw=0.6, transform=PC, zorder=5)
        x, y, q = groups["managed"]
        ax.scatter(x, y, s=scale(q), marker="^", facecolor="#ffd9a8", edgecolor=orange, lw=0.8,
                   transform=PC, zorder=6)
        x, y, q = groups["dumping"]
        ax.scatter(x, y, s=scale(q), marker="o", facecolor="#e2c4f0", edgecolor=purple, lw=0.8,
                   transform=PC, zorder=6)

        rates = np.r_[groups["managed"][2], groups["dumping"][2]]
        pj.add_inset_histogram(
            ax,
            {"Managed landfills": groups["managed"][2], "Dumping sites": groups["dumping"][2]},
            bins=18, log=True, colors=["#fbb260", "#b37fd6"], alpha=0.85,
            total_label="All waste disposal sites",
            cumulative=rates, cumulative_label="Cumulative emissions\n(t h$^{-1}$)",
            bounds=(0.10, 0.11, 0.165, 0.38),
            xlabel="Methane emission rate (t h$^{-1}$)", ylabel="Number of sites",
        )
        # legends below the map (three tidy blocks)
        lax = fig.add_axes([0.0, 0.0, 1.0, 0.2])
        lax.axis("off")
        h1 = [Patch(facecolor="none", edgecolor=pj.INK, lw=0.8),
              Patch(facecolor="#fbb260", alpha=0.85), Patch(facecolor="#b37fd6", alpha=0.85)]
        leg1 = lax.legend(h1, ["All waste disposal sites", "Managed landfills", "Dumping sites"],
                          loc="upper left", bbox_to_anchor=(0.06, 1.0))
        lax.add_artist(leg1)
        mk = dict(ls="none", markersize=6.5, markeredgewidth=0.8)
        h2 = [Line2D([], [], marker="D", markerfacecolor="#a9a9a9", markeredgecolor="#6f6f6f", **mk),
              Line2D([], [], marker="^", markerfacecolor="#ffd9a8", markeredgecolor=orange, **mk),
              Line2D([], [], marker="o", markerfacecolor="#e2c4f0", markeredgecolor=purple, **mk)]
        lax.legend(h2, [f"Urban plume clusters (N = {len(groups['urban'][2])})",
                               f"Managed landfills (N = {len(groups['managed'][2])})",
                               f"Dumping sites (N = {len(groups['dumping'][2])})"],
                          loc="upper left", bbox_to_anchor=(0.38, 1.0))
        sax = fig.add_axes([0.74, 0.0, 0.25, 0.2])
        sax.axis("off")
        pj.add_size_legend(sax, [1, 10], ["1 t h$^{-1}$", "10 t h$^{-1}$"], scale=scale,
                           marker="o", edgecolor=purple, title="Site emissions",
                           loc="upper left", ncol=2, columnspacing=1.2)
        pj.add_panel_label(ax, "b", x=0.0, y=1.0)
        footnote(fig, "Synthetic example data (seeded random sites and emission rates); "
                      "layout after Nature-style point-source maps.", y=0.05)
        path = pj.save_figure(fig, out / "fig2_sites.png", dpi=dpi)
        if pdf:
            pj.save_figure(fig, out / "fig2_sites.pdf")
        plt.close(fig)
    return path


# ----------------------------------------------------------------------------- figure 3


def load_era5():
    with xr.open_dataset(ROOT / "data/ERA5temp_1978_monthly.nc") as ds:
        t = ds.t2m.isel(latitude=slice(None, None, 4), longitude=slice(None, None, 4)).load()
    t = (t - 273.15).rename(latitude="lat", longitude="lon")
    t = t.assign_coords(lon=((t.lon + 180) % 360) - 180).sortby("lon").sortby("lat")
    return t


def fig_era5(out, dpi, pdf):
    """Real ERA5 1978: July anomaly from the annual mean, Robinson, aligned zonal profile."""
    t = load_era5()
    anom = t.isel(time=6) - t.mean("time")
    monthly_anom = t - t.mean("time")
    levels = np.arange(-20, 20.1, 4)
    cmap, norm = pj.discrete_cmap("RdBu_r", levels, extend="both")
    with pj.journal_style():
        fig = plt.figure(figsize=pj.figsize("double", 0.53))
        ax = fig.add_axes([0.04, 0.21, 0.75, 0.76], projection=ccrs.Robinson())
        ax.set_global()
        m = ax.pcolormesh(anom.lon, anom.lat, anom, cmap=cmap, norm=norm, transform=PC,
                          rasterized=True)
        ax.coastlines("110m", lw=0.35, color=pj.INK)
        # latitude labels live on the aligned profile panel, so the map only labels longitude
        pj.geo_ticks(ax, xticks=range(-180, 181, 60), yticks=range(-60, 61, 30), lat_labels=False)
        pj.add_panel_label(ax, "c", x=0.0, y=1.0)
        w = np.cos(np.deg2rad(anom.lat))
        # zonal mean of the July anomaly and the spread of all 12 months' zonal means
        zm = anom.mean("lon")
        q = monthly_anom.mean("lon").quantile([0.25, 0.75], "time")
        mm = monthly_anom.mean("lon")
        pj.add_latitude_profile(
            ax, anom.lat, zm, lower=q.sel(quantile=0.25), upper=q.sel(quantile=0.75),
            outer=(mm.min("time"), mm.max("time")), color="#b2182b", width=0.16, pad=0.02,
            xlabel="Zonal mean (°C)", yticks=range(-60, 61, 30), xlim=(-22, 22),
            title="Jul (line) · monthly IQR / range",
        )
        vals = anom.values.ravel()
        okv = np.isfinite(vals)
        child, _ = pj.add_inset_histogram(
            ax, vals[okv], bins=np.arange(-24, 24.1, 2), colors=["#9e9e9e"], total=False,
            stacked=False, bounds=(0.035, 0.06, 0.17, 0.25), xlabel="Anomaly (°C)",
            ylabel="Cells", alpha=1.0,
        )
        # colour the histogram bars like the map
        for patch in child.patches:
            if not isinstance(patch, Rectangle):
                continue
            xc = patch.get_x() + patch.get_width() / 2
            patch.set_facecolor(cmap(norm(xc)))
        area_warm = float((w * (anom > 0)).sum() / (w * anom.notnull()).sum())
        child.text(1.0, 1.02, f"{100 * area_warm:.0f}% of area warmer", transform=child.transAxes,
                   ha="right", va="bottom", fontsize=6)
        pj.add_colorbar(m, ax, label="July minus annual-mean 2 m temperature (°C)",
                        bounds=[0.205, 0.095, 0.42, 0.026])
        footnote(fig, "Data: ERA5 monthly 2 m temperature, 1978 (bundled file, 1° display grid). "
                      "Profile band: inter-month IQR; light band: inter-month range.", y=0.01)
        path = pj.save_figure(fig, out / "fig3_era5_profile.png", dpi=dpi)
        if pdf:
            pj.save_figure(fig, out / "fig3_era5_profile.pdf")
        plt.close(fig)
    return path


# ----------------------------------------------------------------------------- figure 4


def fig_regional(out, dpi, pdf):
    """Two regional panels sharing one discrete scale: stippling vs hatching (synthetic p)."""
    rng = np.random.default_rng(2026)
    lon = np.arange(70.25, 140, 0.5)
    lat = np.arange(15.25, 55, 0.5)
    xx, yy = np.meshgrid(lon, lat)
    trend = (1.2 * np.exp(-((xx - 110) / 14) ** 2 - ((yy - 33) / 8) ** 2)
             - 0.9 * np.exp(-((xx - 88) / 9) ** 2 - ((yy - 42) / 6) ** 2)
             + 0.45 * smooth_field(rng, lon, lat, n=160, scale=(2, 6))
             + 0.35 * smooth_field(rng, lon, lat, n=60, scale=(6, 14)))
    se = 0.35 + 0.1 * rng.random(trend.shape)
    from math import erfc
    p = np.vectorize(lambda z: erfc(abs(z) / np.sqrt(2)))(trend / se)
    land = land_mask(lon, lat)
    trend[~land] = np.nan
    levels = np.arange(-1.5, 1.51, 0.3)
    cmap, norm = pj.discrete_cmap("BrBG", levels, extend="both")
    with pj.journal_style():
        fig = plt.figure(figsize=pj.figsize("double", 0.36))
        axes = [fig.add_axes([0.045 + 0.505 * i, 0.25, 0.355, 0.72], projection=PC)
                for i in range(2)]
        for i, (ax, style) in enumerate(zip(axes, ["stipple", "hatch"])):
            ax.set_extent([70, 140, 15, 55], crs=PC)
            m = ax.pcolormesh(lon, lat, trend, cmap=cmap, norm=norm, transform=PC)
            pj.add_land(ax, color="none", borders=True, coast_width=0.35, resolution="50m",
                        zorder=2)
            pj.geo_ticks(ax, xticks=range(80, 131, 20), yticks=range(20, 51, 10),
                         lat_labels=(i == 0))  # panel e shares latitude with d's profile
            sig = (p < 0.05) & land
            if style == "stipple":
                pj.add_stippling(ax, lon, lat, sig, stride=3, size=0.7)
                handle = Line2D([], [], ls="none", marker="o", ms=1.6, color=pj.INK)
                label = "p < 0.05 (dots)"
            else:
                pj.add_hatching(ax, lon, lat, sig, hatch="////", linewidth=0.35)
                handle = pj.hatch_patch("////", facecolor="white")
                label = "p < 0.05 (hatch)"
            ax.legend([handle], [label], loc="lower right", frameon=True, edgecolor="none",
                      framealpha=0.85)
            zm = np.nanmean(trend, axis=1)
            sd = np.nanstd(trend, axis=1)
            prof = pj.add_latitude_profile(ax, lat, zm, lower=zm - sd, upper=zm + sd, width=0.17,
                                    pad=0.045, color="#01665e", xlabel="Mean ± SD",
                                    yticks=range(20, 51, 10), xlim=(-1.6, 1.6))
            prof.set_xticks([-1, 0, 1])
            pj.add_panel_label(ax, "de"[i], x=0.01, y=0.98)
        pj.add_colorbar(m, axes[0], label="Trend (mm d$^{-1}$ decade$^{-1}$)",
                        bounds=[0.3, 0.10, 0.4, 0.04])
        footnote(fig, "Synthetic example data (seeded trend field and z-test p-values).", y=0.0)
        path = pj.save_figure(fig, out / "fig4_regional_significance.png", dpi=dpi)
        if pdf:
            pj.save_figure(fig, out / "fig4_regional_significance.pdf")
        plt.close(fig)
    return path


# ----------------------------------------------------------------------------- figure 5


def _speckle(rng, shape, power):
    """Sparse, pixel-scale texture in [0, 1] (lakes / small water bodies)."""
    return rng.random(shape) ** power


def fig_water(out, dpi, pdf):
    """Pekel-style map with a right latitude profile and a bottom longitude profile."""
    rng = np.random.default_rng(1984)
    lon = np.arange(-179.75, 180, 0.5)
    lat = np.arange(-55.75, 84, 0.5)
    land = land_mask(lon, lat)
    xx, yy = np.meshgrid(lon, lat)
    boreal = np.exp(-((yy - 61) / 7.5) ** 2) * (
        np.exp(-((xx + 95) / 30) ** 2) + 0.8 * np.exp(-((xx - 70) / 55) ** 2)
        + 0.6 * np.exp(-((xx - 25) / 12) ** 2)
    )
    tropics = 0.55 * np.exp(-((yy + 3) / 6) ** 2) * (
        np.exp(-((xx + 60) / 12) ** 2) + np.exp(-((xx - 22) / 8) ** 2)
    )
    lakes = 1.4 * np.exp(-((xx - 33) / 1.3) ** 2 - ((yy + 1) / 1.3) ** 2)  # one big lake
    monsoon = 0.35 * np.exp(-((yy - 24) / 6) ** 2) * np.exp(-((xx - 88) / 14) ** 2)
    patch = np.clip(smooth_field(rng, lon, lat, 400, (1, 4)) + 0.35, 0, None)  # lake districts
    base = (boreal + tropics + monsoon) * patch * 1.6 + lakes
    base = np.clip(base, 0, None) * land
    permanent = np.clip(base * (0.1 + 1.6 * _speckle(rng, base.shape, 2.5)), 0, 1)
    seasonal = np.clip((0.12 + 0.4 * tropics + 0.5 * monsoon) * land
                       * _speckle(rng, base.shape, 6) * (base > 0.02), 0, 1)
    maximum = np.clip(permanent + seasonal + 0.25 * base * rng.random(base.shape), 0, 1)
    # change layers (gains positive, losses negative), synthetic
    new_perm = permanent * 0.18 * _speckle(rng, base.shape, 4) * (1 + 3 * np.exp(-((yy - 45) / 8) ** 2))
    lost_perm = permanent * 0.12 * _speckle(rng, base.shape, 4)
    lost_perm += 0.5 * np.exp(-((xx - 59.5) / 1.3) ** 2 - ((yy - 45) / 1.1) ** 2) * land * _speckle(rng, base.shape, 1.5)  # a drying lake
    new_seas = seasonal * 0.5 * _speckle(rng, base.shape, 3) + 0.05 * base * rng.random(base.shape)
    lost_seas = seasonal * 0.35 * _speckle(rng, base.shape, 3)
    net = new_perm + new_seas - lost_perm - lost_seas

    k = 1e3  # 10^3 km^2
    by = {name: pj.marginal_totals(v, lon, lat, scale=k) for name, v in {
        "max": maximum, "perm": permanent, "seas": seasonal, "np": new_perm, "lp": lost_perm,
        "ns": new_seas, "ls": lost_seas}.items()}

    occ = np.where(land & (maximum > 0.01), permanent, np.nan)
    occ_cmap, occ_norm = pj.discrete_cmap(
        ["#e3eefa", "#b3cdef", "#7ea6e3", "#4a78d6", "#1f46c4", "#0c1f9e"],
        [0.02, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0], extend="neither")
    chg = np.where(land & (np.abs(net) > 0.01), net, np.nan)
    chg_cmap, chg_norm = pj.discrete_cmap(
        ["#a1123a", "#d7708f", "#f2c4d1", "#f5f5f5", "#d8ec9f", "#9bcc4b", "#2e9a3c"],
        [-0.3, -0.1, -0.03, -0.005, 0.005, 0.03, 0.1, 0.3], extend="neither")

    grey = "#d4d4d4"
    with pj.journal_style():
        fig = plt.figure(figsize=pj.figsize("double", 1.0))
        panels = [fig.add_axes([0.06, 0.635, 0.70, 0.33], projection=PC),
                  fig.add_axes([0.06, 0.155, 0.70, 0.33], projection=PC)]
        for i, ax in enumerate(panels):
            ax.set_extent([-180, 180, -56, 84], crs=PC)
            ax.set_facecolor(grey)  # ocean and no-data
            pj.add_land(ax, color="white", borders=False, coast_width=0, zorder=0.5)
            if i == 0:  # real Natural Earth rivers give the water map its texture
                import cartopy.feature as cfeature

                ax.add_feature(cfeature.RIVERS.with_scale("50m"), edgecolor="#3355cc",
                               facecolor="none", linewidth=0.3, zorder=1.5)
                ax.add_feature(cfeature.LAKES.with_scale("50m"), edgecolor="none",
                               facecolor="#1f3fb0", zorder=1.5)
            data, cmap, norm = (occ, occ_cmap, occ_norm) if i == 0 else (chg, chg_cmap, chg_norm)
            ax.pcolormesh(lon, lat, data, cmap=cmap, norm=norm, transform=PC, zorder=1,
                          rasterized=True)
            pj.geo_ticks(ax, xticks=range(-180, 181, 30), yticks=range(-30, 61, 30),
                         lat_labels=False, gridlines=False, degree_style="signed")
            ax.set_xlabel("Longitude", labelpad=1)
            ax.spines["geo"].set_visible(False)
            pj.add_panel_label(ax, "ab"[i], style="{}", x=0.006, y=0.985, size=10)
        pj.add_lat_lon_marginals(
            panels[0], lat=lat, lon=lon,
            lat_series={"Maximum\nwater extent": by["max"][0], "Permanent": by["perm"][0],
                        "Seasonal": by["seas"][0]},
            lon_series={"Maximum\nwater extent": by["max"][1], "Permanent": by["perm"][1],
                        "Seasonal": by["seas"][1]},
            fill=("Maximum\nwater extent",),
            colors={"Permanent": "#1f4e9c", "Seasonal": "#7cc6e8"},
            right_xlabel="Area (10$^3$ km$^2$)", bottom_ylabel="Area\n(10$^3$ km$^2$)",
            lat_ticks=[-30, 0, 30, 60],
        )
        pj.add_lat_lon_marginals(
            panels[1], lat=lat, lon=lon,
            lat_series={"New permanent": by["np"][0], "Lost permanent": -by["lp"][0],
                        "New seasonal": by["ns"][0], "Lost seasonal": -by["ls"][0]},
            lon_series={"New permanent": by["np"][1], "Lost permanent": -by["lp"][1],
                        "New seasonal": by["ns"][1], "Lost seasonal": -by["ls"][1]},
            colors={"New permanent": "#2e9a3c", "Lost permanent": "#8b1538",
                    "New seasonal": "#b7d63a", "Lost seasonal": "#e9a9bf"},
            zero_line=True, right_xlabel="Area (10$^3$ km$^2$)",
            bottom_ylabel="Area\n(10$^3$ km$^2$)", lat_ticks=[-30, 0, 30, 60],
        )
        footnote(fig, "Synthetic example data (seeded water fractions on Natural Earth land); "
                      "layout after Pekel et al. (2016, Nature).", y=-0.012)
        path = pj.save_figure(fig, out / "fig5_water_marginals.png", dpi=dpi)
        if pdf:
            pj.save_figure(fig, out / "fig5_water_marginals.pdf")
        plt.close(fig)
    return path


# ----------------------------------------------------------------------------- figure 6


def fig_ternary(out, dpi, pdf):
    """Three-component (ternary) RGB map with density insets and a triangle colour key."""
    rng = np.random.default_rng(314)
    lon = np.arange(-179.75, 180, 0.5)
    lat = np.arange(-55.75, 84, 0.5)
    land = land_mask(lon, lat)
    xx, yy = np.meshgrid(lon, lat)
    ranges = [(14, 30), (9, 20), (50, 74)]

    def uniform_field(n, scale, bias):
        """Smooth field rank-transformed to [0, 1] plus pixel noise (texture)."""
        f = smooth_field(rng, lon, lat, n, scale) + bias + 0.15 * rng.standard_normal(xx.shape)
        z = (f - f.mean()) / f.std()
        return 1.0 / (1.0 + np.exp(-0.9 * z))  # smooth, unimodal, no clipping spikes

    tropical = np.exp(-(yy / 13) ** 2)
    boreal = np.exp(-((yy - 52) / 10) ** 2)
    u_lower = uniform_field(140, (4, 14), 1.6 * tropical)
    u_middle = uniform_field(140, (4, 14), 0.9 * np.exp(-((yy - 30) / 10) ** 2) + 0.6 * boreal)
    u_upper = uniform_field(140, (4, 14), 1.2 * np.exp(-((np.abs(yy) - 25) / 9) ** 2) + boreal)
    lower, middle, upper = (lo + u * (hi - lo)
                            for u, (lo, hi) in zip((u_lower, u_middle, u_upper), ranges))
    # no data in deserts, ice and high latitudes (grey), as in the reference layout
    dry = np.exp(-((yy - 23) / 8) ** 2) * (np.abs(xx - 20) < 60) + np.exp(-((yy + 25) / 6) ** 2) * (
        np.abs(xx - 20) < 30)
    vegetated = land & (yy < 60) & (yy > -50) & (dry < 0.55)
    vegetated &= smooth_field(rng, lon, lat, 80, (6, 18)) > -0.5
    for v in (lower, middle, upper):
        v[~vegetated] = np.nan
    rgba, used = pj.ternary_colors(lower, middle, upper)  # 2–98 % quantile stretch

    with pj.journal_style():
        fig = plt.figure(figsize=pj.figsize("double", 0.41))
        ax = fig.add_axes([0.13, 0.02, 0.87, 0.97], projection=PC)
        ax.set_extent([-180, 180, -56, 84], crs=PC)
        pj.add_land(ax, color="#d3d3d3", borders=True, edgecolor="white", linewidth=0.6,
                    coast_width=0, zorder=0.5)
        ax.spines["geo"].set_visible(False)
        pj.plot_rgb(ax, lon, lat, rgba, zorder=1)
        import cartopy.feature as cfeature

        ax.add_feature(cfeature.BORDERS.with_scale("110m"), edgecolor="white", linewidth=0.6,
                       facecolor="none", zorder=2)
        ax.text(0.03, 0.97, "a", transform=ax.transAxes, fontsize=13, fontweight="bold",
                va="top", path_effects=pj._halo_effect(2.5))
        dens = pj.add_inset_density(
            ax, {"lower": lower, "middle": middle, "upper": upper}, bins=60,
            bounds=(-0.105, 0.5, 0.2, 0.3), xticks=[0, 15, 30, 45, 60, 75],
            xlabel="Accumulated\nfeature importance [%]",
        )
        dens.text(-0.2, 1.12, "b", transform=dens.transAxes, fontsize=13, fontweight="bold",
                  va="top")
        tern = pj.add_ternary_legend(
            ax, tuple(f"{n} ({lo:.0f}–{hi:.0f}%)" for n, (lo, hi) in
                      zip(("lower", "middle", "upper"), used)),
            bounds=(-0.135, -0.02, 0.22, 0.36),
        )
        tern.text(0.0, 1.0, "c", transform=tern.transAxes, fontsize=13, fontweight="bold",
                  va="top")
        footnote(fig, "Synthetic example data (three seeded components); "
                      "ternary layout after Nature-style feature-importance maps.", y=-0.01)
        path = pj.save_figure(fig, out / "fig6_ternary.png", dpi=dpi)
        if pdf:
            pj.save_figure(fig, out / "fig6_ternary.pdf")
        plt.close(fig)
    return path


FIGURES = {
    "regimes": fig_regimes,
    "sites": fig_sites,
    "era5": fig_era5,
    "regional": fig_regional,
    "water": fig_water,
    "ternary": fig_ternary,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/gallery")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--pdf", action="store_true", help="also write vector PDFs")
    parser.add_argument("--only", choices=list(FIGURES), nargs="*")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name in args.only or FIGURES:
        print("wrote", FIGURES[name](args.output, args.dpi, args.pdf))


if __name__ == "__main__":
    main()
