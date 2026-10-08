import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def dataset():
    return xr.Dataset(
        {
            "temperature": (
                ("time", "latitude", "longitude"),
                np.arange(24, dtype=float).reshape(2, 3, 4) + 273.15,
                {"units": "K", "long_name": "Air temperature"},
            )
        },
        coords={
            "time": ["2020-01", "2020-07"],
            "latitude": [60, 0, -60],
            "longitude": [0, 90, 180, 270],
        },
    )


@pytest.fixture
def ncfile(dataset, tmp_path):
    path = tmp_path / "climate.nc"
    dataset.to_netcdf(path)
    return path
