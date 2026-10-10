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
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from plot_function import Distribution, MapFeatures, Profile, Significance, plot_map

ROOT = Path(__file__).resolve().parents[1]
INK, MUTED, TEAL = "#1a1a1a", "#555555", "#2b5d8c"


def footer(fig, text):
    from plot_function.journal import resolved_font

    fig.text(0.01, 0.005, text, fontsize=5.5, color=MUTED, ha="left", va="bottom",
             fontfamily=resolved_font())


def synthetic_experiment(path):
    rng = np.random.default_rng(20261009)
    lat, lon = np.linspace(0, 60, 61), np.linspace(80, 150, 71)
    xx, yy = np.meshgrid(lon, lat)
    signal = 1.5 * np.exp(-(((xx - 112) / 16) ** 2) - ((yy - 38) / 12) ** 2) - 1.2 * np.exp(
        -(((xx - 137) / 9) ** 2) - ((yy - 17) / 10) ** 2
    )
    n, sigma = 40, 2.0
    members = signal + rng.normal(0, sigma, (n, len(lat), len(lon)))
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
            "description": "Synthetic independent Gaussian experiment, not observations or projections",
            "seed": 20261009,
            "realizations": n,
            "known_sigma": sigma,
        },
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)


def gallery(output, *, coastlines=True, map_details=True):
    output.mkdir(parents=True, exist_ok=True)
    with xr.open_dataset(ROOT / "data/ERA5temp_1978_monthly.nc") as ds:
        data = ds.t2m.isel(latitude=slice(None, None, 4), longitude=slice(None, None, 4)).load()
    data = data - 273.15
    data.attrs = {"units": "°C", "long_name": "2 m air temperature"}

    global_map = plot_map(
        data,
        reduce="time",
        cmap="RdYlBu_r",
        levels=np.arange(-40, 41, 5),
        title="Annual-mean 2 m air temperature",
        subtitle="ERA5 1978 · mean of 12 monthly means · 1° display grid",
        label="Temperature (°C)",
        figsize=(7.2, 4.0),
        coastlines=coastlines,
        profiles=Profile(style="band", spread="std", reference=0, width=0.15,
                         label="Zonal mean ± 1 s.d."),
        distribution=Distribution(
            style="bars",
            weights="coslat",
            color="map",
            bounds=(0.03, 0.06, 0.17, 0.22),
        ),
        panel_label="a",
    )
    footer(global_map.figure, "Band: ±1 spatial s.d. across longitudes (not a confidence "
                              "interval). Inset: cos-latitude-weighted density.")
    global_map.save(output / "journal-global.png", dpi=300)
    plt.close(global_map.figure)

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
    import plot_function.journal as pj

    with pj.journal_style():
        fig, axes = plt.subplots(
            1, 3, figsize=(7.2, 2.7), subplot_kw={"projection": ccrs.PlateCarree()},
            facecolor="white",
        )
    fig.subplots_adjust(left=0.06, right=0.9, bottom=0.3, top=0.9, wspace=0.55)
    for ax, style, profile_style, dist_style, letter in zip(
        axes,
        ["stipple", "hatch", "contour"],
        ["band", "line", "bars"],
        ["bars", "line", "ecdf"],
        ["a", "b", "c"],
    ):
        result = plot_map(
            synthetic,
            "response",
            ax=ax,
            extent=[80, 150, 0, 60],
            cmap="RdBu_r",
            levels=np.linspace(-1.8, 1.8, 13),
            title={"stipple": "Stippling", "hatch": "Hatching", "contour": "Contour"}[style],
            coastlines=False,
            colorbar=False,
            significance=Significance(
                synthetic,
                variable="p_value",
                style=style,
                correction="fdr_bh",
                stride=2,
                size=2.2,
                hatch="///",
                legend=False,
            ),
            profiles=Profile(
                style=profile_style,
                width=0.2,
                reference=0,
                label="Mean" if profile_style != "band" else "Mean ± s.d.",
            ),
            distribution=Distribution(
                style=dist_style, bins=14, bounds=(0.1, 0.16, 0.36, 0.22), show_mean=False
            ),
        )
        ax.annotate(letter, xy=(0, 0), xycoords=ax._left_title, xytext=(-4, 0),
                    textcoords="offset points", fontsize=10, fontweight="bold", ha="right",
                    fontfamily=pj.resolved_font())
        if letter != "a":  # latitude is labelled once, on panel a and on each profile
            ax.tick_params(labelleft=False)
    with pj.journal_style():
        pj.add_colorbar(result.artist, axes[1], label="Synthetic response (arbitrary units)",
                        bounds=[0.3, 0.13, 0.4, 0.03], tick_every=2)
    footer(fig, "Synthetic experiment: 40 realizations, two-sided z-test (known σ), "
                "BH FDR q = 0.05. Identical decisions in all panels.")
    pj.save_figure(fig, output / "journal-significance.png", dpi=300)
    plt.close(fig)
    print(f"Rendered three journal compositions to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/assets")
    parser.add_argument(
        "--offline", action="store_true", help="Disable all downloaded map features"
    )
    args = parser.parse_args()
    gallery(args.output, coastlines=not args.offline, map_details=not args.offline)
