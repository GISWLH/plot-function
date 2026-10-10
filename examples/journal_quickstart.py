"""Ten-line journal figure: discrete map, stippling, inset bars, aligned zonal profile.

Uses an analytic **example field** (no downloads except Natural Earth coastlines):
    python examples/journal_quickstart.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cartopy.crs as ccrs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import plot_function.journal as pj  # noqa: E402


def main(coastlines=True):
    lon, lat = np.arange(-179.5, 180, 1.0), np.arange(-59.5, 90, 1.0)
    field = (8 * np.sin(np.deg2rad(2 * lon))[None, :] * np.cos(np.deg2rad(lat))[:, None]
             + 4 * np.sin(np.deg2rad(3 * lat))[:, None])  # example data
    low_agreement = np.abs(field) < 1.5

    with pj.journal_style():
        fig = plt.figure(figsize=pj.figsize("double", 0.45))
        ax = fig.add_axes([0.06, 0.24, 0.78, 0.72], projection=ccrs.PlateCarree())
        ax.set_extent([-180, 180, -60, 90], crs=ccrs.PlateCarree())
        cmap, norm = pj.discrete_cmap("BrBG", np.arange(-10, 10.1, 2.5), extend="both")
        mesh = ax.pcolormesh(lon, lat, field, cmap=cmap, norm=norm,
                             transform=ccrs.PlateCarree())
        if coastlines:
            ax.coastlines(lw=0.3)
        pj.geo_ticks(ax, xticks=range(-180, 181, 60), yticks=range(-40, 81, 20))
        pj.add_stippling(ax, lon, lat, low_agreement, stride=3)
        pj.add_inset_bars(ax, ["Drier", "Wetter"], [42, 58], hatched=[9, 12],
                          colors=["#a6611a", "#018571"], ylabel="Area [%]")
        pj.add_latitude_profile(ax, lat, field.mean(1), lower=np.percentile(field, 25, 1),
                                upper=np.percentile(field, 75, 1), xlabel="Zonal mean")
        pj.add_colorbar(mesh, ax, title="Example regime", label="Δ [units]")
        pj.add_panel_label(ax, "a")
        out = Path(__file__).resolve().parent / "output" / "journal_quickstart.png"
        pj.save_figure(fig, out, dpi=300)
    print(f"Created {out}")


if __name__ == "__main__":
    main()
