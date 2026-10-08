import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest

from plot_function import open_field, plot_map
from plot_function import legacy
from utils import plot


@pytest.mark.parametrize("projection", ["robinson", "platecarree", "equalearth", "mollweide"])
def test_map_renders_offline(ncfile, tmp_path, projection):
    before = mpl.rcParams.copy()
    result = plot_map(
        ncfile, reduce="time", projection=projection, coastlines=False, output=tmp_path / "map.png"
    )
    assert (tmp_path / "map.png").read_bytes().startswith(b"\x89PNG")
    assert (tmp_path / "map.png").stat().st_size > 5000
    assert result.artist.get_array().size >= result.data.size
    assert result.colorbar.ax.get_xlabel() == "K"
    assert mpl.rcParams == before


def test_supplied_axes_extent_contours_and_export(dataset, tmp_path):
    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
    result = plot_map(
        dataset,
        reduce="time",
        ax=ax,
        extent=[-90, 90, -50, 50],
        colorbar=False,
        coastlines=False,
        plotfunc="contourf",
        theme="dark",
    )
    assert result.figure is fig
    assert result.colorbar is None
    np.testing.assert_allclose(ax.get_extent(), [-90, 90, -50, 50], atol=1e-6)
    result.save(tmp_path / "nested" / "map.svg")
    assert "<svg" in (tmp_path / "nested" / "map.svg").read_text()


@pytest.mark.parametrize(
    "options",
    [
        {"theme": "unknown"},
        {"projection": "unknown"},
        {"extent": [10, -10, 0, 30]},
        {"extent": [1, 2, 3]},
        {"plotfunc": "imshow"},
    ],
)
def test_bad_map_options(dataset, options):
    with pytest.raises(ValueError):
        plot_map(dataset, reduce="time", coastlines=False, **options)


def test_legacy_imports_and_coastline_options(monkeypatch):
    assert plot.one_map is legacy.one_map
    _, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
    received = {}
    monkeypatch.setattr(ax, "coastlines", lambda **kwargs: received.update(kwargs))
    plot.coastlines(ax, resolution="50m", alpha=0.3)
    assert received["resolution"] == "50m"
    assert received["alpha"] == 0.3


def test_hatch_inversion_and_mappable_return(dataset):
    da = open_field(dataset, reduce="time")
    _, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
    before = mpl.rcParams.copy()
    artist, legend = plot.one_map(da, ax, hatch_data=da * 0, add_coastlines=False)
    assert hasattr(artist, "get_array")
    assert isinstance(legend, mpl.patches.Patch)
    assert len(ax.collections) >= 2  # the all-zero mask inverted to all-hatched
    ax.figure.canvas.draw()
    assert mpl.rcParams == before
    with pytest.raises(ValueError, match="2.0"):
        plot.hatch_map(ax, da * 0 + 2, "/", "invalid")


def test_region_extent_without_interval_and_gridlines(dataset):
    da = open_field(dataset, reduce="time")
    _, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
    plot.one_map_region(da, ax, extents=[0, 90, -30, 30], add_gridlines=True, add_coastlines=False)
    np.testing.assert_allclose(ax.get_extent(), [0, 90, -30, 30])
    ax.figure.canvas.draw()


def test_warming_panels_legend_and_custom_colorbar(dataset):
    da = open_field(dataset, reduce="time")
    cbar = plot.at_warming_level_one(
        [da] * 3,
        "K",
        "Regression test",
        levels=np.linspace(270, 300, 7),
        average="mean",
        getmean=False,
        add_coastlines=False,
        hatch_data=da > 285,
        add_legend=True,
        colorbar_kwargs={"pad": 0.2},
        legend_kwargs={"fontsize": 6},
    )
    assert cbar.ax.get_xlabel() == "K"
    assert any(ax.get_legend() is not None for ax in cbar.ax.figure.axes)


def test_regional_contours_do_not_wrap():
    import xarray as xr

    da = xr.DataArray(
        np.ones((2, 3)), dims=("lat", "lon"), coords={"lat": [20, 30], "lon": [100, 110, 120]}
    )
    assert legacy._cyclic_if_global(da).shape == (2, 3)


def test_profile_reduces_correct_dimension(dataset):
    da = open_field(dataset, reduce="time").transpose("lon", "lat")
    _, ax = plt.subplots()
    plot.add_sta(ax, da, [270, 310], "lat")
    np.testing.assert_allclose(ax.lines[0].get_xdata(), da.mean("lon"))
    assert ax.lines[0].get_label() == "Mean"
