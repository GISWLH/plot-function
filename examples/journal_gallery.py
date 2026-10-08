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
INK, MUTED, TEAL = "#163438", "#647b80", "#267f83"


def footer(fig, text):
    fig.text(0.055, 0.025, text, fontsize=8, color=MUTED)
    fig.text(0.95, 0.025, "plot-function / research atlas", ha="right", fontsize=8, color=INK)


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
        title="Temperature, in space and in distribution",
        subtitle="ERA5 / 1978 · Mean of 12 monthly means · 1° display grid",
        figsize=(13, 7.5),
        coastlines=coastlines,
        profiles=Profile(style="band", spread="std", reference=0, width=0.23),
        distribution=Distribution(
            style="bars",
            weights="coslat",
            color="map",
            bounds=(0.035, 0.07, 0.26, 0.25),
            label="Cos-lat weighted density",
        ),
        panel_label="a",
    )
    global_map.figure.set_facecolor("white")
    footer(
        global_map.figure,
        "SPATIAL SUMMARY    /    Ribbon: ±1 spatial SD, not a confidence interval",
    )
    global_map.save(output / "journal-global.png", dpi=180)
    plt.close(global_map.figure)

    regional = plot_map(
        data,
        isel={"time": 6},
        extent=[92, 142, 8, 53],
        projection=ccrs.PlateCarree(),
        cmap="RdYlBu_r",
        levels=np.arange(-10, 37, 2),
        title="East Asia / climate in context",
        subtitle="ERA5 / July 1978 · Regional summaries within the displayed bounds",
        figsize=(10, 9),
        coastlines=coastlines,
        profiles=[
            Profile(style="line", color=TEAL),
            Profile(
                position="top",
                style="band",
                statistic="median",
                spread="iqr",
                color="#ad7843",
                width=0.18,
            ),
        ],
        distribution=Distribution(
            style="step", bins=20, weights="coslat", bounds=(0.045, 0.09, 0.30, 0.24)
        ),
        features=MapFeatures(
            resolution="50m", rivers=map_details, lakes=map_details, borders=map_details
        ),
        panel_label="b",
    )
    regional.figure.set_facecolor("white")
    footer(
        regional.figure,
        "REGIONAL ATLAS    /    Top: median + spatial IQR    ·    Right: zonal mean",
    )
    regional.save(output / "journal-regional.png", dpi=180)
    plt.close(regional.figure)

    synthetic = ROOT / "examples/output/synthetic-significance.nc"
    synthetic_experiment(synthetic)
    fig, axes = plt.subplots(
        1, 3, figsize=(16, 6.0), subplot_kw={"projection": ccrs.PlateCarree()}, facecolor="white"
    )
    fig.subplots_adjust(left=0.04, right=0.94, bottom=0.24, top=0.77, wspace=0.54)
    fig.text(
        0.04,
        0.94,
        "One statistical decision. Three visual languages.",
        fontsize=23,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.04,
        0.865,
        "SYNTHETIC EXPERIMENT  /  40 independent realizations · Two-sided z-test, known σ · BH FDR q = 0.05",
        fontsize=10,
        color=MUTED,
    )
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
            title=style.capitalize(),
            coastlines=False,
            colorbar=False,
            panel_label=letter,
            significance=Significance(
                synthetic,
                variable="p_value",
                style=style,
                correction="fdr_bh",
                stride=2,
                size=2.8,
                hatch="....",
                legend=False,
            ),
            profiles=Profile(
                style=profile_style,
                width=0.19,
                reference=0,
                label="Mean" if profile_style != "band" else "Mean ± SD",
            ),
            distribution=Distribution(
                style=dist_style, bins=14, bounds=(0.075, 0.065, 0.41, 0.25), show_mean=False
            ),
        )
    bar = fig.colorbar(
        result.artist, cax=fig.add_axes([0.32, 0.14, 0.36, 0.022]), orientation="horizontal"
    )
    bar.set_label("Synthetic response / arbitrary units", fontsize=9, color=INK)
    bar.ax.tick_params(labelsize=8, colors=MUTED, length=2)
    bar.outline.set_visible(False)
    footer(
        fig,
        "METHOD STUDY    /    Identical p-values and FDR family in all panels · No observed climate significance is implied",
    )
    fig.savefig(
        output / "journal-significance.png", dpi=180, bbox_inches="tight", facecolor="white"
    )
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
