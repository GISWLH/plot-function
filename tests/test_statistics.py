import numpy as np
import pytest
import xarray as xr

from plot_function import histogram, significance_mask, spatial_profile
from plot_function.statistics import distribution_values


def field():
    return xr.DataArray(
        [[1.0, 3.0, np.nan], [5.0, 7.0, 9.0]],
        dims=("lat", "lon"),
        coords={"lat": [0.0, 60.0], "lon": [-30.0, 0.0, 30.0]},
    )


def test_profile_mean_spread_and_missing_counts():
    stats = spatial_profile(field())
    np.testing.assert_allclose(stats.center, [2, 7])
    np.testing.assert_allclose(stats.upper - stats.center, [1, np.sqrt(8 / 3)])
    np.testing.assert_array_equal(stats["count"], [2, 3])
    median = spatial_profile(field(), statistic="median", spread="iqr")
    np.testing.assert_allclose(median.lower, [1.5, 6])
    np.testing.assert_allclose(median.upper, [2.5, 8])


def test_coslat_weights_applied_across_latitude():
    stats = spatial_profile(field(), coordinate="lon", weights="coslat")
    np.testing.assert_allclose(stats.center, [7 / 3, 13 / 3, 9])
    with pytest.raises(ValueError, match="equal cell"):
        spatial_profile(field(), statistic="median", weights="coslat")


def test_weighted_density_integral_and_excluded_values():
    stats = histogram(field(), bins=[0, 4, 10], weights="coslat")
    np.testing.assert_allclose(stats.mass, [2, 1.5])
    assert float((stats.height * (stats.right - stats.left)).sum()) == pytest.approx(1)
    assert stats.attrs["sample_count"] == 5
    weights = xr.ones_like(field())
    weights[0, 0] = 0
    assert len(distribution_values(field(), weights)[0]) == 4


@pytest.mark.parametrize("bins", [0, -1, [0, 2, 1], [np.nan, 1], [100, 200]])
def test_bad_bins(bins):
    with pytest.raises(ValueError):
        histogram(field(), bins=bins)


def test_weight_alignment_and_negative_values():
    weights = xr.ones_like(field()).assign_coords(lon=[0, 10, 20])
    with pytest.raises(ValueError, match="exactly match"):
        histogram(field(), weights=weights)
    with pytest.raises(ValueError, match="negative"):
        histogram(field(), weights=-xr.ones_like(field()))


def test_bh_fdr_known_decisions_and_family():
    p = xr.DataArray([0.001, 0.01, 0.03, 0.2, np.nan], dims="cell")
    mask = significance_mask(p, alpha=0.05, correction="fdr_bh")
    np.testing.assert_array_equal(mask, [True, True, True, False, False])
    assert mask.attrs["tested_cells"] == 4
    assert mask.attrs["critical_p"] == 0.03
    assert significance_mask(p, alpha=0.01, correction="fdr_bh").sum() == 1
    assert significance_mask(p, valid=p > 0.005).attrs["tested_cells"] == 3
    assert not significance_mask(p * np.nan, correction="fdr_bh").any()


@pytest.mark.parametrize("values", [[-0.1, 0.5], [0.2, np.inf], [0.3, 1.1]])
def test_bad_pvalues(values):
    with pytest.raises(ValueError, match="P-values"):
        significance_mask(xr.DataArray(values))
