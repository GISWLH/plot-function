import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from plot_function import Distribution, MapFeatures, Profile, Significance, open_field, plot_map


def make_map(dataset, **kwargs):
    return plot_map(dataset, reduce="time", coastlines=False, projection="platecarree", **kwargs)


@pytest.mark.parametrize("style", ["line", "band", "bars"])
def test_profile_styles_and_scoped_statistics(dataset, style):
    result = make_map(
        dataset,
        extent=[-90, 90, -65, 65],
        profiles=[Profile(style=style), Profile(position="top", style=style)],
    )
    expected = result.data.sel(lon=slice(-90, 90)).mean("lon")
    xr.testing.assert_allclose(
        result.profiles["right"].statistics.center, expected.rename("center")
    )
    result.figure.canvas.draw()
    assert len(result.axes.child_axes) == 2
    with pytest.raises(ValueError, match="already exists"):
        result.add_profile()


@pytest.mark.parametrize("style", ["bars", "step", "line", "ecdf"])
def test_distribution_styles(dataset, style, tmp_path):
    before = mpl.rcParams.copy()
    result = make_map(dataset, distribution=Distribution(style=style, weights="coslat"))
    stats = result.distribution.statistics
    assert stats.attrs["sample_count"] == result.data.size
    if style == "ecdf":
        assert stats["cumulative"][-1] == pytest.approx(1)
    result.save(tmp_path / f"{style}.png")
    assert (tmp_path / f"{style}.png").stat().st_size > 5000
    assert mpl.rcParams == before


@pytest.mark.parametrize("style", ["stipple", "hatch", "contour"])
def test_significance_styles_and_actual_decisions(dataset, style):
    da = open_field(dataset, reduce="time")
    p = xr.where(da > 288, 0.001, 0.8)
    result = make_map(
        dataset, significance=Significance(p, style=style, stride=1, correction="fdr_bh")
    )
    mask = result.significance[0].statistics
    np.testing.assert_array_equal(mask, p < 0.05)
    assert mask.attrs["tested_cells"] == da.size
    result.figure.canvas.draw()


def test_significance_alignment_and_display_thinning(dataset):
    result = make_map(dataset)
    p = xr.full_like(result.data, 0.001)
    layer = result.add_significance(data=p, stride=2)
    assert int(layer.statistics.sum()) == p.size
    assert len(layer.artists[0].get_offsets()) == 4
    with pytest.raises(ValueError, match="exactly match"):
        result.add_significance(data=p.assign_coords(lat=p.lat + 1))


def test_precomputed_mask_is_not_treated_as_pvalues(dataset):
    result = make_map(dataset)
    layer = result.add_significance(data=result.data > 288, kind="mask")
    assert int(layer.statistics.sum()) == int((result.data > 288).sum())
    with pytest.raises(ValueError, match="p-values"):
        result.add_significance(data=result.data > 288, kind="mask", correction="fdr_bh")


def test_boolean_mask_from_netcdf(dataset, tmp_path):
    result = make_map(dataset)
    path = tmp_path / "mask.nc"
    mask = (result.data > 288).rename("significant")
    mask.to_netcdf(path)
    layer = result.add_significance(data=path, variable="significant", kind="mask")
    np.testing.assert_array_equal(layer.statistics, mask)


def test_features_are_explicit_and_can_be_tested_without_downloads(dataset, monkeypatch):
    result = make_map(dataset)
    names = []
    monkeypatch.setattr(result.axes, "add_feature", lambda f, **kw: names.append((f.name, f.scale)))
    result.add_features(MapFeatures(rivers=True, lakes=True, borders=True, resolution="50m"))
    assert names == [
        ("lakes", "50m"),
        ("rivers_lake_centerlines", "50m"),
        ("admin_0_boundary_lines_land", "50m"),
    ]


def test_custom_axes_posthoc_layers(dataset, tmp_path):
    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.Robinson()})
    result = make_map(dataset, ax=ax, title="A map", subtitle="A subtitle")
    result.add_profile(position="top", style="line")
    result.add_profile(position="right", style="band")
    result.add_distribution(style="bars", color="map")
    result.save(tmp_path / "composition.svg")
    assert result.figure is fig
    assert (tmp_path / "composition.svg").stat().st_size > 1000
