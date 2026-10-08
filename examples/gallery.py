"""Render the README gallery from the bundled ERA5 NetCDF file.

All panels use every fourth grid point (1° display spacing). Monthly averages
and differences are explicit; these are illustrations of 1978, not trends.
"""

import argparse
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from plot_function import open_field, plot_map

ROOT = Path(__file__).resolve().parents[1]
INK, MINT, PAPER = "#10252e", "#78bda8", "#f8faf9"


def footer(fig, text, dark=False):
    fig.text(0.055, 0.025, text, fontsize=7.5, color="#a3bec2" if dark else "#647b80")
    fig.text(
        0.945,
        0.025,
        "plot-function",
        ha="right",
        fontsize=8,
        fontweight="bold",
        color=MINT if dark else "#163438",
    )


def gallery(source, output, coastlines=True):
    output.mkdir(parents=True, exist_ok=True)
    with xr.open_dataset(source) as ds:
        temperature = (
            ds["t2m"].isel(latitude=slice(None, None, 4), longitude=slice(None, None, 4)).load()
        )
    if temperature.sizes["time"] != 12:
        raise ValueError("This gallery expects the bundled 12-month 1978 ERA5 file.")
    temperature = temperature - 273.15
    temperature.attrs = {"units": "°C", "long_name": "2 m air temperature"}
    common = dict(coastlines=coastlines, cmap="RdYlBu_r", levels=np.arange(-40, 41, 5))

    annual = plot_map(
        temperature,
        reduce="time",
        title="A year of surface temperature",
        subtitle="ERA5 / 1978 · Arithmetic mean of 12 monthly means",
        projection="robinson",
        **common,
    )
    footer(
        annual.figure, "01 / GLOBAL VIEW     ·     1° display spacing     ·     Temperature in °C"
    )
    annual.save(output / "global-temperature.png")
    plt.close(annual.figure)

    fig, axes = plt.subplots(
        1, 2, figsize=(12, 5.2), subplot_kw={"projection": ccrs.Robinson()}, facecolor=INK
    )
    fig.subplots_adjust(left=0.04, right=0.96, top=0.78, bottom=0.22, wspace=0.08)
    fig.text(0.04, 0.93, "The rhythm of the seasons", color=PAPER, fontsize=23, fontweight="bold")
    fig.text(
        0.04,
        0.86,
        "ERA5 / 1978 · Two monthly means, one shared temperature scale",
        color=MINT,
        fontsize=10,
    )
    for ax, index, title in zip(axes, [0, 6], ["January", "July"]):
        panel = plot_map(
            temperature,
            isel={"time": index},
            ax=ax,
            title=title,
            theme="dark",
            colorbar=False,
            **common,
        )
    bar = fig.colorbar(
        panel.artist, cax=fig.add_axes([0.29, 0.16, 0.42, 0.026]), orientation="horizontal"
    )
    bar.set_label("2 m air temperature / °C", color=PAPER, fontsize=9)
    bar.ax.tick_params(colors=PAPER, labelsize=8, length=2)
    bar.outline.set_visible(False)
    footer(fig, "02 / SEASONAL COMPARISON     ·     1° display spacing", dark=True)
    fig.savefig(output / "seasons.png", dpi=180, bbox_inches="tight", facecolor=INK)
    plt.close(fig)

    january = open_field(temperature, isel={"time": 0})
    july = open_field(temperature, isel={"time": 6})
    difference = july - january
    difference.attrs = {"units": "°C", "long_name": "July minus January"}
    contrast = plot_map(
        difference,
        projection="equalearth",
        title="Two seasons. A world of contrast.",
        subtitle="ERA5 / 1978 · July minus January · Seasonal difference, not a climate trend",
        cmap="PuOr_r",
        levels=np.arange(-40, 41, 5),
        coastlines=coastlines,
        label="Temperature difference / °C",
    )
    footer(
        contrast.figure, "03 / SEASONAL DIFFERENCE     ·     Symmetric, zero-centered color scale"
    )
    contrast.save(output / "seasonal-contrast.png")
    plt.close(contrast.figure)

    regional = plot_map(
        july,
        projection=ccrs.LambertConformal(central_longitude=115, central_latitude=30),
        extent=[90, 145, 5, 55],
        title="A closer look at East Asia",
        subtitle="ERA5 / July 1978 · Monthly mean 2 m air temperature",
        figsize=(8, 7),
        plotfunc="contourf",
        **common,
    )
    crop = july.sel(lon=slice(85, 150), lat=slice(0, 60))
    lines = regional.axes.contour(
        crop.lon,
        crop.lat,
        crop,
        levels=[0, 10, 20, 30],
        transform=ccrs.PlateCarree(),
        colors="#29454a",
        linewidths=0.55,
        alpha=0.65,
    )
    regional.axes.clabel(lines, inline=True, fontsize=7, fmt="%d°")
    footer(
        regional.figure,
        "04 / REGIONAL DETAIL     ·     1° display spacing     ·     Isotherms in °C",
    )
    regional.save(output / "east-asia.png")
    plt.close(regional.figure)
    print(f"Rendered four gallery figures to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/ERA5temp_1978_monthly.nc")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/assets")
    parser.add_argument(
        "--no-coastlines", action="store_true", help="Avoid Natural Earth downloads"
    )
    args = parser.parse_args()
    gallery(args.input, args.output, coastlines=not args.no_coastlines)


if __name__ == "__main__":
    main()
