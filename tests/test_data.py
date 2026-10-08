import numpy as np
import pytest
import xarray as xr

from plot_function import open_field


def test_netcdf_selection_normalization_and_units(ncfile):
    data = open_field(ncfile, isel={"time": 0}, offset=-273.15, units="°C")
    assert data.dims == ("lat", "lon")
    np.testing.assert_array_equal(data.lon, [-180, -90, 0, 90])
    np.testing.assert_array_equal(data.lat, [-60, 0, 60])
    np.testing.assert_allclose(data.values, [[10, 11, 8, 9], [6, 7, 4, 5], [2, 3, 0, 1]])
    assert data.attrs["units"] == "°C"
    assert data.attrs["long_name"] == "Air temperature"


def test_reduction_and_source_not_modified(dataset):
    original = dataset.copy(deep=True)
    result = open_field(dataset, reduce="time")
    assert result.sel(lat=60, lon=0).item() == pytest.approx(279.15)
    xr.testing.assert_identical(dataset, original)


def test_label_selection(dataset):
    result = open_field(dataset, sel={"time": "2020-07"})
    assert result.sel(lat=60, lon=0).item() == pytest.approx(285.15)


def test_cf_coordinates_and_nonstandard_dimension_order():
    da = xr.DataArray(
        np.arange(6).reshape(3, 2),
        dims=("i", "j"),
        coords={
            "north": ("j", [20, 30], {"standard_name": "latitude"}),
            "east": ("i", [100, 110, 120], {"units": "degrees_east"}),
        },
    )
    result = open_field(da)
    np.testing.assert_array_equal(result, da.values.T)
    assert result.dims == ("lat", "lon")


def test_explicit_coordinate_names():
    da = xr.DataArray(np.ones((2, 3)), dims=("y", "x"), coords={"y": [20, 30], "x": [0, 1, 2]})
    with pytest.raises(ValueError, match="Cannot uniquely identify"):
        open_field(da)
    assert open_field(da, latitude="y", longitude="x").shape == (2, 3)


def test_global_duplicate_seam():
    da = xr.DataArray(
        np.ones((2, 5)), dims=("lat", "lon"), coords={"lat": [0, 30], "lon": [0, 90, 180, 270, 360]}
    )
    assert open_field(da).shape == (2, 4)


def test_duplicate_interior_longitude_is_rejected(dataset):
    dataset = dataset.assign_coords(longitude=[0, 90, 90, 180])
    with pytest.raises(ValueError, match="duplicate"):
        open_field(dataset, reduce="time")


def test_antimeridian_region_does_not_silently_fill_the_longitude_gap(dataset):
    dataset = dataset.assign_coords(longitude=[170, 175, 180, 185])
    with pytest.raises(ValueError, match="antimeridian"):
        open_field(dataset, reduce="time")


@pytest.mark.parametrize(
    "options,message",
    [
        ({}, "extra dimensions"),
        ({"variable": "missing"}, "Unknown variable"),
        ({"reduce": "latitude"}, "preserve latitude"),
        ({"reduce": "level"}, "not found"),
        ({"reduce": "time", "statistic": "var"}, "Unknown statistic"),
        ({"isel": {"time": 0}, "sel": {"time": "2020-01"}}, "both sel and isel"),
        ({"isel": {"time": 0, "latitude": 0}}, "two-dimensional"),
        ({"reduce": "time", "offset": np.inf}, "finite numbers"),
    ],
)
def test_actionable_errors(dataset, options, message):
    with pytest.raises(ValueError, match=message):
        open_field(dataset, **options)


def test_ambiguous_variable(dataset):
    dataset["other"] = dataset.temperature * 2
    with pytest.raises(ValueError, match="Choose variable"):
        open_field(dataset, reduce="time")


def test_curvilinear_grid_rejected():
    da = xr.DataArray(
        np.ones((2, 3)),
        dims=("y", "x"),
        coords={"lat": (("y", "x"), np.ones((2, 3))), "lon": (("y", "x"), np.ones((2, 3)))},
    )
    with pytest.raises(ValueError, match="rectilinear"):
        open_field(da)


def test_all_nan_and_missing_sum(dataset):
    dataset.temperature[:] = np.nan
    for statistic in ["mean", "sum"]:
        with pytest.raises(ValueError, match="no finite data"):
            open_field(dataset, reduce="time", statistic=statistic)


def test_singleton_extra_dimension_and_nan_mask(dataset):
    da = dataset.temperature.isel(time=[0]).copy()
    da.values[0, 0, 0] = np.inf
    result = open_field(da)
    assert result.ndim == 2
    assert np.isnan(result.sel(lat=60, lon=0))


def test_result_survives_source_file_close(ncfile):
    result = open_field(ncfile, reduce="time")
    ncfile.unlink()
    assert np.isfinite(result.values).all()
