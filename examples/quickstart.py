"""Create a tiny synthetic NetCDF file and plot it, entirely offline.

Run after `python -m pip install -e .`:
    python examples/quickstart.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from plot_function import plot_map


def main():
    out = Path(__file__).resolve().parent / "output"
    out.mkdir(exist_ok=True)
    lat = np.linspace(-89, 89, 90)
    lon = np.arange(0, 360, 2)
    longitude, latitude = np.meshgrid(lon, lat)
    temperature = 28 * np.cos(np.deg2rad(latitude)) - 8 + 5 * np.sin(np.deg2rad(longitude))
    ds = xr.Dataset(
        {
            "temperature": (
                ("latitude", "longitude"),
                temperature,
                {"units": "°C", "long_name": "Synthetic temperature"},
            )
        },
        coords={"latitude": lat, "longitude": lon},
        attrs={"description": "Analytic demonstration field; not observations or reanalysis."},
    )
    path = out / "synthetic.nc"
    ds.to_netcdf(path)
    result = plot_map(
        path,
        cmap="RdYlBu_r",
        coastlines=False,
        subtitle="Synthetic field · offline NetCDF-only example",
        output=out / "quickstart.png",
    )
    plt.close(result.figure)
    print(f"Created {path} and {out / 'quickstart.png'}")


if __name__ == "__main__":
    main()
