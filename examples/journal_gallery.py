"""Publication-style compositions with reproducible statistics and white backgrounds.

Real ERA5 fields illustrate descriptive summaries. Significance is demonstrated
only with a separately labeled synthetic experiment and known-variance z-tests.
"""

import argparse
import math
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib

matplotlib.use("Agg")
import matplotlib.lines
import matplotlib.patches
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import xarray as xr

from plot_function import Distribution, MapFeatures, Profile, plot_map

ROOT = Path(__file__).resolve().parents[1]
INK, MUTED, TEAL = "#1a1a1a", "#555555", "#2b5d8c"


def footer(fig, text):
    from plot_function.journal import resolved_font

    fig.text(0.01, 0.005, text, fontsize=5.5, color=MUTED, ha="left", va="bottom",
             fontfamily=resolved_font())


def _smooth(field, sigma_cells):
    """Gaussian smoothing over the last two axes via FFT (numpy only)."""
    ny, nx = field.shape[-2:]
    ky = np.fft.fftfreq(ny)[:, None]
    kx = np.fft.fftfreq(nx)[None, :]
    kernel = np.exp(-2 * (np.pi * sigma_cells) ** 2 * (kx**2 + ky**2))
    return np.real(np.fft.ifft2(np.fft.fft2(field) * kernel))


def synthetic_experiment(path):
    """Seeded synthetic ensemble: a known signal plus spatially correlated noise.

    Each grid cell is tested with a two-sided z-test (known sigma), so the p-values
    are exact; the Benjamini–Hochberg FDR correction is applied in the figure.
    """
    rng = np.random.default_rng(20261009)
    lat, lon = np.arange(0, 60.01, 0.5), np.arange(80, 150.01, 0.5)
    xx, yy = np.meshgrid(lon, lat)
    signal = (
        1.1 * np.exp(-(((xx - 108) / 14) ** 2) - ((yy - 38) / 9) ** 2)
        + 0.55 * np.exp(-(((xx - 125) / 7) ** 2) - ((yy - 47) / 5) ** 2)
        - 0.9 * np.exp(-(((xx - 136) / 8) ** 2) - ((yy - 18) / 7) ** 2)
        - 0.35 * np.exp(-(((xx - 92) / 6) ** 2) - ((yy - 12) / 5) ** 2)
    )
    n, sigma = 40, 2.0
    noise = _smooth(rng.standard_normal((n, len(lat), len(lon))), 4.0)
    noise *= sigma / noise.std(axis=(1, 2), keepdims=True)  # marginal s.d. = sigma
    members = 1.6 * signal + noise
    response = members.mean(axis=0)
    z = np.abs(response) / (sigma / np.sqrt(n))
    p = np.fromiter((math.erfc(v / np.sqrt(2)) for v in z.ravel()), float).reshape(z.shape)
    ds = xr.Dataset(
        {
            "response": (("lat", "lon"), response, {"units": "a.u."}),
            "p_value": (
                ("lat", "lon"),
                p,
                {"test": "two-sided z-test, known sigma=2, null mean=0"},
            ),
        },
        coords={"lat": lat, "lon": lon},
        attrs={
            "description": "Synthetic ensemble with spatially correlated noise, "
                           "not observations or projections",
            "seed": 20261009,
            "realizations": n,
            "known_sigma": sigma,
        },
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)


def bh_threshold(p, q=0.05):
    """Benjamini–Hochberg p-value threshold (0 if nothing is rejected)."""
    p = np.sort(np.asarray(p, float)[np.isfinite(p)])
    m = p.size
    below = p <= q * np.arange(1, m + 1) / m
    return float(p[below].max()) if below.any() else 0.0


def _land_mask(lon, lat):
    try:
        import cartopy.io.shapereader as shpreader
        import shapely
        from shapely.ops import unary_union

        land = unary_union(list(shpreader.Reader(
            shpreader.natural_earth("110m", "physical", "land")).geometries()))
        shapely.prepare(land)
        xx, yy = np.meshgrid(np.where(lon > 180, lon - 360, lon), lat)
        return shapely.contains_xy(land, xx, yy)
    except Exception:  # offline: no land/ocean split
        return None


def _letter(fig, ax, letter, title, dx=0.0, dy=0.012):
    import plot_function.journal as pj

    box = ax.get_position()
    fig.text(box.x0 + dx, box.y1 + dy, letter, fontsize=10, fontweight="bold", va="bottom",
             ha="left", fontfamily=pj.resolved_font())
    fig.text(box.x0 + dx + 0.022, box.y1 + dy + 0.001, title, fontsize=7.5, va="bottom",
             ha="left", fontfamily=pj.resolved_font())


def _area_hist(child, values, weights, levels, cmap, norm):
    """Area-weighted histogram whose bars take the colour of their map class."""
    v, w = values.ravel(), weights.ravel()
    ok = np.isfinite(v)
    edges = np.concatenate([[levels[0] - (levels[1] - levels[0])], levels,
                            [levels[-1] + (levels[-1] - levels[-2])]])
    counts, _ = np.histogram(np.clip(v[ok], edges[0], edges[-1]), bins=edges, weights=w[ok])
    counts = 100 * counts / counts.sum()
    mids = 0.5 * (edges[:-1] + edges[1:])
    child.bar(mids, counts, width=np.diff(edges) * 0.9, color=cmap(norm(mids)),
              edgecolor="#1a1a1a", linewidth=0.35)
    return counts


def global_figure(data, output, *, coastlines=True):
    """Two-row Robinson figure: annual mean and July−January contrast with aligned
    zonal profiles (land vs ocean), an area-weighted class histogram and colourbars."""
    import plot_function.journal as pj
    from cartopy.util import add_cyclic_point

    lat = data.latitude.values
    lon = data.longitude.values
    annual = data.mean("time").values
    jan, jul = data.isel(time=0).values, data.isel(time=6).values
    contrast = jul - jan
    w2d = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], annual.shape)
    land = _land_mask(lon, lat)

    robinson = ccrs.Robinson()
    fields = [
        dict(values=annual, levels=np.arange(-30, 31, 5), cmap="RdYlBu_r",
             label="2 m air temperature (°C)",
             title="Annual-mean 2 m air temperature (ERA5, 1978)"),
        dict(values=contrast, levels=np.array([-40, -30, -20, -10, -5, -2, 2, 5, 10, 20, 30, 40]),
             cmap="RdBu_r", label="July − January temperature (°C)",
             title="Seasonal contrast, July minus January"),
    ]
    with pj.journal_style():
        fig = plt.figure(figsize=pj.figsize("double", 0.9))
        for row, spec in enumerate(fields):
            y0 = 0.575 - row * 0.49
            ax = fig.add_axes([0.035, y0, 0.625, 0.385], projection=robinson)
            ax.set_global()
            ax.spines["geo"].set_linewidth(0.6)
            cmap, norm = pj.discrete_cmap(spec["cmap"], spec["levels"])
            z, lon_c = add_cyclic_point(spec["values"], coord=lon)
            cs = ax.contourf(lon_c, lat, z, levels=spec["levels"], cmap=cmap, norm=norm,
                             extend="both", transform=ccrs.PlateCarree(), zorder=1)
            if coastlines:
                ax.coastlines(resolution="110m", linewidth=0.35, color="#1a1a1a", zorder=3)
            gl = ax.gridlines(color="#9a9a9a", linewidth=0.3, linestyle=(0, (3, 3)),
                              xlocs=range(-180, 181, 60), ylocs=range(-60, 61, 30), zorder=2)
            gl.draw_labels = False
            pj.add_colorbar(cs, ax, label=spec["label"], length=0.62, size=0.03, pad=0.05)
            _letter(fig, ax, "ab"[row], spec["title"], dy=0.0)

            # zonal statistics: land and ocean separately (cos-lat weights cancel per row)
            series = []
            if land is not None:
                vl = np.where(land, spec["values"], np.nan)
                vo = np.where(~land, spec["values"], np.nan)
                with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
                    __import__("warnings").simplefilter("ignore")
                    series = [(np.nanmean(vl, axis=1), dict(color="#a6611a", lw=0.9,
                                                            label="Land")),
                              (np.nanmean(vo, axis=1), dict(color="#2c7fb8", lw=0.9,
                                                            label="Ocean"))]
            mean = spec["values"].mean(axis=1)
            sd = spec["values"].std(axis=1)
            prof = pj.add_latitude_profile(
                ax, lat, mean, lower=mean - sd, upper=mean + sd, color="#1a1a1a",
                band_alpha=0.12, linewidth=1.1, width=0.3, pad=0.035, extra=series,
                reference=0 if row == 1 else None, yticks=[-60, -30, 0, 30, 60],
                xlabel=("Zonal mean (°C)"),
            )
            if row == 0:
                prof.axvline(0, color="#1a1a1a", lw=0.4, ls=(0, (2.5, 2)), zorder=1)
            prof.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
            handles = [matplotlib.lines.Line2D([], [], color="#1a1a1a", lw=1.1),
                       matplotlib.patches.Patch(color="#1a1a1a", alpha=0.12, lw=0)]
            names = ["All longitudes", "± 1 s.d."]
            if series:
                handles += [matplotlib.lines.Line2D([], [], **{k: v for k, v in kw.items()
                                                               if k != "label"})
                            for _, kw in series]
                names += [kw["label"] for _, kw in series]
            if row == 0:
                prof.legend(handles, names, loc="upper left", bbox_to_anchor=(1.18, 1.0),
                            handlelength=1.3, borderaxespad=0, labelspacing=0.35)

            # area-weighted share of the globe in each colour class (lower-left corner)
            ins = ax.inset_axes([-0.01, 0.0, 0.2, 0.24], zorder=8)
            _area_hist(ins, spec["values"], w2d, spec["levels"], cmap, norm)
            for name, sp in ins.spines.items():
                sp.set_visible(name in ("left", "bottom"))
                sp.set_linewidth(0.5)
            ins.patch.set_alpha(0)
            ins.tick_params(length=2, width=0.5, pad=1, labelsize=5.5)
            ins.set_ylabel("Area (%)", fontsize=5.8, labelpad=1)
            lv = spec["levels"]
            ins.set_xticks([lv[0], 0, lv[-1]])
            ins.xaxis.set_major_formatter(pj.clean_formatter())
            ins.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3, integer=True))
            gmean = float(np.sum(spec["values"] * w2d) / np.sum(w2d))
            ins.axvline(gmean, color="#1a1a1a", lw=0.6, ls=(0, (2, 1.5)))
            ins.text(gmean, ins.get_ylim()[1] * 1.02, f"mean {gmean:.1f} °C",
                     fontsize=5.3, ha="center", va="bottom", color="#1a1a1a")
        footer(fig, "ERA5 monthly means, 1978 (0.5° display grid). Profiles: zonal mean over "
                    "all longitudes (± 1 s.d. across longitudes) and over land / ocean "
                    "(Natural Earth 110m). Insets: cos-latitude-weighted area per colour class.")
        pj.save_figure(fig, output / "journal-global.png", dpi=300)
        plt.close(fig)


def significance_figure(path, output, *, coastlines=True):
    """2×2 figure: the same BH-FDR decision drawn three ways plus the BH diagnostic."""
    import plot_function.journal as pj

    with xr.open_dataset(path) as ds:
        lon, lat = ds.lon.values, ds.lat.values
        resp, p = ds.response.values, ds.p_value.values
    q = 0.05
    p_star = bh_threshold(p, q)
    sig = p <= p_star
    raw = p < 0.05
    levels = np.round(np.arange(-1.6, 1.61, 0.2), 1)
    cmap, norm = pj.discrete_cmap("RdBu_r", levels)
    PC = ccrs.PlateCarree()
    with pj.journal_style():
        fig = plt.figure(figsize=pj.figsize("double", 0.86))
        boxes = [[0.06, 0.55, 0.36, 0.385], [0.53, 0.55, 0.36, 0.385],
                 [0.06, 0.085, 0.36, 0.385]]
        titles = ["Stippling: BH-FDR significant", "Hatching: BH-FDR significant",
                  "Outline: FDR (solid) vs uncorrected (dashed)"]
        axes = []
        for k, (box, title) in enumerate(zip(boxes, titles)):
            ax = fig.add_axes(box, projection=PC)
            ax.set_extent([80, 150, 0, 60], crs=PC)
            cf = ax.contourf(lon, lat, resp, levels=levels, cmap=cmap, norm=norm,
                             extend="both", transform=PC, zorder=1)
            if coastlines:
                ax.coastlines(resolution="50m", linewidth=0.4, color="#1a1a1a", zorder=3)
            pj.geo_ticks(ax, xticks=range(80, 151, 20), yticks=range(0, 61, 20),
                         gridlines=False)
            if k == 0:
                pj.add_stippling(ax, lon, lat, sig, stride=2, size=1.1)
                handle = matplotlib.lines.Line2D([], [], ls="", marker="o", ms=1.6,
                                                 color="#1a1a1a")
                label = f"$p \\leq p^*$ = {p_star:.4f}"
            elif k == 1:
                pj.add_hatching(ax, lon, lat, sig, hatch="////", linewidth=0.45)
                handle, label = pj.hatch_patch("////"), f"$p \\leq p^*$ = {p_star:.4f}"
            else:
                ax.contour(lon, lat, raw.astype(float), levels=[0.5], colors="#555555",
                           linewidths=0.6, linestyles=[(0, (2.5, 1.5))], transform=PC, zorder=4)
                ax.contour(lon, lat, sig.astype(float), levels=[0.5], colors="#1a1a1a",
                           linewidths=0.9, transform=PC, zorder=5)
                handle = [matplotlib.lines.Line2D([], [], color="#1a1a1a", lw=0.9),
                          matplotlib.lines.Line2D([], [], color="#555555", lw=0.6,
                                                  ls=(0, (2.5, 1.5)))]
                label = [f"BH-FDR, q = {q}", "uncorrected p < 0.05"]
            handles = handle if isinstance(handle, list) else [handle]
            labels = label if isinstance(label, list) else [label]
            leg = ax.legend(handles, labels, loc="lower left", frameon=True, fancybox=False,
                            framealpha=0.92, edgecolor="none", borderpad=0.35,
                            handlelength=1.5, handleheight=0.9, fontsize=6)
            leg.set_zorder(9)
            _letter(fig, ax, "abc"[k], title)
            axes.append(ax)
        pj.add_colorbar(cf, axes[1], orientation="vertical", label="Ensemble-mean response (a.u.)",
                        size=0.035, pad=0.04, length=0.92, tick_every=2)

        # d: Benjamini–Hochberg diagnostic
        ax = fig.add_axes([0.6, 0.085, 0.29, 0.385])
        ps = np.sort(p.ravel())
        m = ps.size
        rank = np.arange(1, m + 1) / m
        ax.plot(rank, ps, color="#1a1a1a", lw=1.0, label="Sorted p-values", zorder=3)
        ax.plot(rank, q * rank, color="#d6604d", lw=0.9, label=f"BH line  q·k/m (q = {q})")
        ax.axhline(0.05, color="#555555", lw=0.6, ls=(0, (2.5, 1.5)),
                   label="Uncorrected 0.05")
        k_fdr = int((ps <= p_star).sum())
        ax.axvspan(0, k_fdr / m, color="#4393c3", alpha=0.14, lw=0,
                   label=f"Rejected (FDR): {100 * k_fdr / m:.0f}% of cells")
        ax.set_yscale("log")
        ax.set_ylim(1e-6, 1.5)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Rank / number of tests (k / m)")
        ax.set_ylabel("p-value")
        ax.legend(loc="lower right", handlelength=1.6, borderaxespad=0.3)
        for name, sp in ax.spines.items():
            sp.set_visible(name in ("left", "bottom"))
        _letter(fig, ax, "d", f"Benjamini–Hochberg: {raw.sum() - sig.sum()} cells "
                              "lose significance")
        footer(fig, f"Synthetic ensemble: 40 realizations, spatially correlated noise, "
                    f"two-sided z-test (known σ), BH FDR q = {q} over {m} cells. "
                    "Panels a–c show the identical decision.")
        pj.save_figure(fig, output / "journal-significance.png", dpi=300)
        plt.close(fig)


def gallery(output, *, coastlines=True, map_details=True):
    output.mkdir(parents=True, exist_ok=True)
    with xr.open_dataset(ROOT / "data/ERA5temp_1978_monthly.nc") as ds:
        fine = ds.t2m.isel(latitude=slice(None, None, 2), longitude=slice(None, None, 2)).load()
    fine = fine - 273.15
    global_figure(fine, output, coastlines=coastlines)
    data = fine.isel(latitude=slice(None, None, 2), longitude=slice(None, None, 2))
    data.attrs = {"units": "°C", "long_name": "2 m air temperature"}

    regional = plot_map(
        data,
        isel={"time": 6},
        extent=[92, 142, 8, 53],
        projection=ccrs.PlateCarree(),
        cmap="RdYlBu_r",
        levels=np.arange(-10, 37, 3),
        title="East Asia, July 1978",
        subtitle="ERA5 2 m air temperature · summaries within the displayed bounds",
        label="Temperature (°C)",
        figsize=(5.2, 5.0),
        coastlines=coastlines,
        profiles=[
            Profile(style="band", spread="std", color=TEAL, label="Zonal mean ± 1 s.d."),
            Profile(
                position="top",
                style="band",
                statistic="median",
                spread="iqr",
                color="#a6611a",
                width=0.16,
                label="Meridional median (IQR)",
            ),
        ],
        distribution=Distribution(
            style="step", bins=20, weights="coslat", bounds=(0.06, 0.07, 0.3, 0.2)
        ),
        features=MapFeatures(
            resolution="50m", rivers=map_details, lakes=map_details, borders=map_details,
            river_color="#4a90b8", border_color="#8c8c8c",
        ),
        panel_label="b",
    )
    regional.save(output / "journal-regional.png", dpi=300)
    plt.close(regional.figure)

    synthetic = ROOT / "examples/output/synthetic-significance.nc"
    synthetic_experiment(synthetic)
    significance_figure(synthetic, output, coastlines=coastlines)
    print(f"Rendered three journal compositions to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/assets")
    parser.add_argument(
        "--offline", action="store_true", help="Disable all downloaded map features"
    )
    args = parser.parse_args()
    gallery(args.output, coastlines=not args.offline, map_details=not args.offline)
