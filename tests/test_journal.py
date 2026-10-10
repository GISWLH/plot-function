import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import BoundaryNorm

import plot_function.journal as pj


def test_style_is_temporary_and_uses_sans_list():
    before = mpl.rcParams.copy()
    with pj.journal_style():
        assert mpl.rcParams["font.size"] == 7
        assert mpl.rcParams["pdf.fonttype"] == 42
        assert mpl.rcParams["font.family"][-1] == "DejaVu Sans"
    assert mpl.rcParams == before


def test_figsize_columns():
    w, h = pj.figsize("single", 0.5)
    assert w == pytest.approx(89 / 25.4)
    assert h == pytest.approx(w / 2)
    assert pj.figsize(120)[0] == pytest.approx(120 / 25.4)


@pytest.mark.parametrize("extend,n_colors", [("both", 4), ("min", 4), ("neither", 4)])
def test_discrete_cmap_extends(extend, n_colors):
    cmap, norm = pj.discrete_cmap("BrBG", [-2, -1, 0, 1, 2], extend=extend)
    assert cmap.N == n_colors
    assert isinstance(norm, BoundaryNorm)
    assert cmap.colorbar_extend == extend
    with pytest.raises(ValueError):
        pj.discrete_cmap("BrBG", [0, 0, 1])


def test_discrete_cmap_from_colour_list():
    colors = ["#000000", "#111111", "#222222", "#333333"]
    cmap, _ = pj.discrete_cmap(colors, [0, 1, 2], extend="both")
    assert mpl.colors.to_hex(cmap.get_under()) == "#000000"
    assert mpl.colors.to_hex(cmap.get_over()) == "#333333"


def test_formatters():
    assert pj.format_lat(40) == "40°N"
    assert pj.format_lat(-20) == "20°S"
    assert pj.format_lat(0) == "0°"
    assert pj.format_lon(-120) == "120°W"
    assert pj.format_lon(180) == "180°"


@pytest.fixture
def geo():
    lon = np.arange(-179, 180, 2.0)
    lat = np.arange(-59, 90, 2.0)
    field = np.cos(np.deg2rad(lat))[:, None] * np.sin(np.deg2rad(lon))[None, :] * 10
    return lon, lat, field


@pytest.mark.parametrize("projection", [ccrs.PlateCarree(), ccrs.Robinson()])
def test_full_composition_offline(geo, projection, tmp_path):
    lon, lat, field = geo
    with pj.journal_style():
        fig = plt.figure(figsize=pj.figsize("double", 0.5))
        ax = fig.add_axes([0.05, 0.25, 0.8, 0.7], projection=projection)
        ax.set_global()
        cmap, norm = pj.discrete_cmap("BrBG", np.arange(-10, 11, 2.5))
        mesh = ax.pcolormesh(lon, lat, field, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
        pj.geo_ticks(ax, xticks=range(-180, 181, 60), yticks=range(-60, 61, 30))
        dots = pj.add_stippling(ax, lon, lat, np.abs(field) < 2, stride=2)
        assert dots is not None and len(dots.get_offsets()) > 0
        assert pj.add_hatching(ax, lon, lat, field > 8) is not None
        assert pj.add_stippling(ax, lon, lat, np.zeros_like(field, bool)) is None
        bars = pj.add_inset_bars(ax, ["A", "B"], [30, 20], hatched=[10, 5])
        hist, twin = pj.add_inset_histogram(
            ax, {"x": np.abs(field).ravel() + 0.1, "y": np.ones(5)}, log=True, cumulative=True
        )
        assert twin is not None
        profile = pj.add_latitude_profile(
            ax, lat, field.mean(1), lower=field.min(1), upper=field.max(1)
        )
        top = pj.add_longitude_profile(ax, lon, field.mean(0))
        cbar = pj.add_colorbar(mesh, ax, label="Δ", title="Regime")
        assert cbar.extend == "both"
        pj.add_panel_label(ax, "a")
        out = pj.save_figure(fig, tmp_path / "fig.png", dpi=120)
    assert out.stat().st_size > 10_000
    # The profile's latitude axis is locked to the map's projected y-range.
    assert profile.get_ylim() == pytest.approx(ax.get_ylim())
    assert top.get_xlim() == pytest.approx(ax.get_xlim())
    assert bars in ax.child_axes and hist in ax.child_axes
    plt.close(fig)


def test_profile_alignment_matches_projection(geo):
    lon, lat, field = geo
    fig = plt.figure()
    ax = fig.add_subplot(projection=ccrs.Robinson())
    ax.set_global()
    child = pj.add_latitude_profile(ax, lat, field.mean(1), yticks=[-30, 0, 30, 60])
    expected = ccrs.Robinson().transform_point(0, 60, ccrs.PlateCarree())[1]
    assert np.min(np.abs(np.asarray(child.get_yticks()) - expected)) < 1e-6
    fig.canvas.draw()
    labels = [t.get_text() for t in child.get_yticklabels()]
    assert "60°N" in labels and "30°S" in labels


def test_size_legend_and_bad_mask(geo):
    lon, lat, field = geo
    fig, ax = plt.subplots()
    leg = pj.add_size_legend(ax, [1, 10], ["1", "10"], scale=lambda v: 10 * v)
    assert len(leg.legend_handles) == 2
    gfig = plt.figure()
    gax = gfig.add_subplot(projection=ccrs.PlateCarree())
    with pytest.raises(ValueError, match="shape"):
        pj.add_stippling(gax, lon, lat, np.ones((3, 3), bool))


def test_cell_area_sums_to_earth_surface():
    lon = np.arange(-179.5, 180, 1.0)
    lat = np.arange(-89.5, 90, 1.0)
    area = pj.cell_area_km2(lon, lat)
    assert area.shape == (lat.size, lon.size)
    assert area.sum() == pytest.approx(4 * np.pi * 6371.0088**2, rel=1e-3)


def test_marginal_totals_area_weighted():
    lon = np.arange(-179.5, 180, 1.0)
    lat = np.arange(-89.5, 90, 1.0)
    field = np.ones((lat.size, lon.size))
    by_lat, by_lon = pj.marginal_totals(field, lon, lat)
    assert by_lat.shape == lat.shape and by_lon.shape == lon.shape
    assert by_lat.sum() == pytest.approx(by_lon.sum())
    assert by_lat[90] > by_lat[0]  # equatorial rows hold more area than polar rows
    mean_lat, _ = pj.marginal_totals(field, lon, lat, area=False, how="mean")
    assert np.allclose(mean_lat, 1.0)


def test_lat_lon_marginals_are_aligned_with_map():
    lon = np.arange(-179.5, 180, 1.0)
    lat = np.arange(-55.5, 84, 1.0)
    field = np.random.default_rng(0).random((lat.size, lon.size))
    by_lat, by_lon = pj.marginal_totals(field, lon, lat)
    with pj.journal_style():
        fig = plt.figure(figsize=pj.figsize("double", 0.6))
        ax = fig.add_axes([0.06, 0.4, 0.7, 0.5], projection=ccrs.PlateCarree())
        ax.set_extent([-180, 180, -56, 84], crs=ccrs.PlateCarree())
        right, bottom = pj.add_lat_lon_marginals(
            ax, lat=lat, lat_series={"max": by_lat, "half": by_lat / 2}, lon=lon,
            lon_series={"max": by_lon, "half": by_lon / 2}, fill=("max",),
        )
        fig.canvas.draw()
        assert right.get_ylim() == pytest.approx(ax.get_ylim())
        assert bottom.get_xlim() == pytest.approx(ax.get_xlim())
        r, b, m = right.get_position(), bottom.get_position(), ax.get_position()
        assert r.y0 == pytest.approx(m.y0) and r.y1 == pytest.approx(m.y1)
        assert b.x0 == pytest.approx(m.x0) and b.x1 == pytest.approx(m.x1)
        plt.close(fig)


def test_ternary_colors_corners_and_missing():
    a = np.array([1.0, 0.0, 0.0, np.nan])
    b = np.array([0.0, 1.0, 0.0, 0.5])
    c = np.array([0.0, 0.0, 1.0, 0.5])
    rgba, ranges = pj.ternary_colors(a, b, c, ranges=[(0, 1)] * 3)
    assert rgba.shape == (4, 4)
    for i, corner in enumerate(pj.TERNARY_CORNERS):
        assert np.allclose(rgba[i, :3], mpl.colors.to_rgb(corner))
    assert rgba[3, 3] == 0  # missing values are transparent
    assert rgba.min() >= 0 and rgba.max() <= 1
    assert ranges == [(0.0, 1.0)] * 3


def test_ternary_legend_and_density_inset():
    rng = np.random.default_rng(1)
    lon = np.arange(-179.5, 180, 2.0)
    lat = np.arange(-59.5, 60, 2.0)
    comps = [rng.random((lat.size, lon.size)) for _ in range(3)]
    rgba, used = pj.ternary_colors(*comps)
    with pj.journal_style():
        fig = plt.figure(figsize=pj.figsize("double", 0.45))
        ax = fig.add_axes([0.1, 0.05, 0.85, 0.9], projection=ccrs.PlateCarree())
        img = pj.plot_rgb(ax, lon, lat, rgba)
        assert img.get_array().shape[:2] == (lat.size, lon.size)
        tern = pj.add_ternary_legend(ax, ("x", "y", "z"))
        assert len(tern.texts) == 3
        dens = pj.add_inset_density(ax, {"x": comps[0], "y": comps[1], "z": comps[2]})
        assert len(dens.patches) > 0 and dens.get_legend() is not None
        fig.canvas.draw()
        plt.close(fig)
