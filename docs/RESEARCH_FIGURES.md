# Research figure composition

[English](../README.md) · [简体中文](../README.zh-CN.md) · [日本語](../README.ja.md)

The map remains the primary visual. Statistical summaries use thin teal lines,
translucent spread bands, and restrained axes. Insets sit on a light panel;
geographic details and significance marks use separate visual styles. All new
gallery compositions use white backgrounds. No additional Python dependency is
required beyond the core installation.

## One call or incremental composition

```python
from plot_function import plot_map, Profile, Distribution, MapFeatures

result = plot_map(
    "data/ERA5temp_1978_monthly.nc", "t2m",
    isel={"time": 6}, offset=-273.15, units="°C",
    projection="platecarree", extent=[92, 142, 8, 53],
    profiles=Profile(style="band"),
    distribution=Distribution(style="bars", weights="coslat"),
    features=MapFeatures(rivers=True, lakes=True, resolution="50m"),
    panel_label="a", title="July temperature", figsize=(12, 8),
)
# Add the other profile after drawing the initial map.
profile = result.add_profile(position="top", style="line", weights="coslat")
print(profile.statistics["center"])
result.save("figure.pdf")
```

Declare layers in `plot_map` to reserve room around a newly created figure.
Alternatively, `add_profile`, `add_distribution`, `add_significance`, and
`add_features` work on an existing `MapResult`. They accept a configuration
object or keyword arguments. Each configuration is reusable. Only one profile
per position and one distribution inset are permitted per result.

Passing a custom Cartopy `ax=` leaves figure layout to the caller. Profile axes
are anchored relative to the map and follow its position as it changes. Exports
include outside axes through `bbox_inches='tight'`.

## Marginal profiles

`Profile(position='right')` averages along longitude and retains latitude.
`position='top'` averages along latitude and retains longitude.

| Option | Default | Alternatives / behavior |
| :--- | :--- | :--- |
| `position` | `right` | `top` |
| `style` | `band` | `line`, `bars` |
| `statistic` | `mean` | `median` |
| `spread` | `std` | `iqr`, `None`; used only for `style='band'` |
| `weights` | `None` | `coslat` or an aligned 2D DataArray |
| `color` | `#267f83` | Any Matplotlib color |
| `width`, `pad` | `0.20`, `0.045` | Fractions of map width (right) or height (top) |
| `reference` | `None` | Draw a reference line, e.g. `0` |
| `label` | `None` | Override the automatically generated profile heading |

A standard-deviation band is the center ± one population spatial SD. An IQR band
uses the 25th and 75th percentiles. These are not uncertainty estimates, standard
errors, or confidence intervals. A median profile and IQR currently require
`weights=None`; weighted quantiles are not silently approximated.

Profiles use their own geographic degree axes. On Plate Carrée these correspond
directly to the rectangular map coordinates. On Robinson, Lambert, and other
curved projections they are descriptive geographic profiles, not shared axes in
the projected coordinate system. Choose Plate Carrée for exact visual alignment.

## Distribution inset

```python
from plot_function import Distribution

# Replace the distribution= option in the initial plot call with any of these:
bars = Distribution(style="bars", bins=20, weights="coslat", color="map")
steps = Distribution(style="step", bins=[-40, -20, 0, 10, 20, 30, 40])
line = Distribution(style="line", bins=24, color="#a56c42")
cumulative = Distribution(style="ecdf", show_mean=False)
```

| Option | Default | Meaning |
| :--- | :--- | :--- |
| `style` | `bars` | `bars`, `step`, `line`, `ecdf` |
| `bins` | `18` | Positive integer or strictly increasing edges; unused for ECDF |
| `density` | `True` | Normalize histogram area to one; unused for ECDF |
| `weights` | `None` | Equal cells, `coslat`, or explicit aligned weights |
| `bounds` | `(0.045, 0.075, 0.27, 0.25)` | `(left, bottom, width, height)` as map-axes fractions |
| `color` | `#267f83` | `map` colors bars according to the map's colormap/normalization |
| `show_mean` | `True` | Add a dashed weighted-mean line and numerical label |
| `label` | `None` | Custom inset heading |

`line` draws a frequency polygon through histogram bin centers. It is not KDE
and introduces no bandwidth or smoothness assumption. ECDF is the normalized
cumulative weight of sorted observations. With `density=False`, unweighted
histograms show cell counts; weighted ones show the sum of weights.

For explicit edges, density normalizes only the observations inside those edges.
The mean marker describes all finite positive-weight cells in the summary scope,
including cells outside the supplied bin range. A bin range with no weight is an
error rather than an empty or invalid density plot.

## Scope, weights, and missing values

All summaries and FDR decisions use grid-cell centers within the explicit
`extent` rectangle, inclusively; without `extent`, they use the whole field.
This is a coordinate rectangle, not a polygon mask inferred from coastlines or
from a curved map projection. Land/ocean drawing layers do not mask statistics.
Apply any desired scientific mask to the DataArray first.

- Default weights give each finite cell equal weight.
- `coslat` is cosine latitude. It approximates area weights on evenly spaced
  geographic grids. At a fixed latitude the common weight cancels in a zonal mean.
- For irregular grids, supply actual cell areas as a two-dimensional DataArray on
  the **full normalized map grid**. Weights are cropped alongside the data.
- Coordinates must match exactly. No interpolation or implicit broadcasting of
  custom one-dimensional weights is performed.
- NaN weights and zero weights exclude cells. Negative or infinite weights fail.
  Data NaNs and infinities are excluded from all summaries.

`spatial_profile` and `histogram` are also public, non-plotting helpers for an
already normalized `(lat, lon)` field. Returned xarray datasets expose centers,
band limits, valid counts, weight sums, bin edges, bin heights, and bin masses.

```python
zonal = result.profiles["right"].statistics
zonal.to_netcdf("zonal-statistics.nc")
# ECDF uses dictionary access because Dataset also has a cumulative() method:
# cumulative_values = result.distribution.statistics["cumulative"]
```

## Significance: test first, draw second

```python
from plot_function import Significance

# This file is generated by examples/journal_gallery.py.
path = "examples/output/synthetic-significance.nc"
result = plot_map(
    path, "response", coastlines=False, cmap="RdBu_r", vmin=-1.8, vmax=1.8,
    significance=Significance(path, variable="p_value", style="stipple",
                              alpha=0.05, correction="fdr_bh"),
    title="Synthetic experiment", output="synthetic-significance.png",
)
print(result.significance[0].statistics.attrs)
```

| Option | Default | Meaning |
| :--- | :--- | :--- |
| `data` | Required | NetCDF path, Dataset, or DataArray containing p-values/mask |
| `variable`, `sel`, `isel` | `None` | Select the significance field independently of the map |
| `latitude`, `longitude` | `None` | Optional significance coordinate names |
| `kind` | `pvalue` | `mask` for an externally determined 0/1 or boolean decision |
| `alpha` | `0.05` | Unadjusted threshold or BH target FDR |
| `correction` | `none` | `fdr_bh` for Benjamini–Hochberg step-up correction |
| `style` | `stipple` | `hatch`, `contour` |
| `stride` | `3` | Visual row/column thinning for stippling only; never alters the test family |
| `size` | `3.0` | Stipple marker area in points squared |
| `hatch` | `....` | Matplotlib hatch pattern |
| `color` | `#243e44` | Marker, hatch, or contour color |
| `label`, `legend` | `None`, `True` | Override the description or omit the legend |

Both the field and significance layer undergo the same coordinate normalization,
then must align exactly. Their selected grids must be identical; the library does
not interpolate p-values. Missing map values cannot be marked significant.
P-values must be finite within [0, 1] or missing. An all-missing source field is
rejected by the loader. Precomputed masks must be 0/1/NaN (or boolean arrays, including NetCDF fields)
and cannot request FDR correction.

Unadjusted decisions use `p <= alpha`. For BH, finite p-values in the selected
family are sorted and compared with `alpha * rank / m`; all values at or below the
largest passing p-value are selected. If nothing passes, no cell is selected.
The returned decision DataArray records `tested_cells`, `significant_cells`,
`alpha`, `correction`, and `critical_p` (NaN if no BH threshold passes).

The test, null hypothesis, sampling structure, dependence assumptions, and
multiple-testing family remain scientific choices made by the caller. BH's
standard guarantees require independence or appropriate positive dependence;
arbitrary spatial dependence is not automatically accounted for. Neither spatial
SD nor a low p-value measures model agreement or effect size.

Contours draw the boundary of a binary decision at level 0.5, not a p-value
isoline. A uniformly significant/insignificant field has no internal boundary.
Hatching and contours need at least two displayed rows and columns. Stippling can
be visually thinned with `stride`; retained statistics always include all tested
cells. `significance_mask` is available separately for numerical use without a map.

## Cartographic context

```python
from plot_function import MapFeatures

features = MapFeatures(
    resolution="50m", rivers=True, lakes=True, borders=True,
    land=True, ocean=True, river_color="#518eab", linewidth=0.5,
)
```

Land/ocean fills sit behind the data, while rivers, lake polygons, and boundary
lines sit above it. These are map context, not new quantitative data. Options also
include `lake_color`, `border_color`, `land_color`, and `ocean_color`. Selected
resolution also controls coastlines in `plot_map`. Natural Earth supports 110m,
50m, and 10m scales; higher detail costs more memory and render time.

First use can download uncached shapefiles through Cartopy. For a fully offline
composition, use `coastlines=False` and leave all feature flags off. Neither the
user's data file nor the repository is modified by these downloads.

## Reproduce the design study

```bash
python examples/journal_gallery.py
python examples/journal_gallery.py --offline --output examples/output/journal
```

The two ERA5 figures use the same bundled data and display sampling as the basic
gallery. The significance figure instead uses a seeded synthetic field: 40
independent Gaussian realizations with known σ=2, a spatially varying true mean,
and a two-sided z-test of zero mean. P-values use the normal survival function
via `erfc`; no inference is made from the one-year ERA5 record. The generated
NetCDF stores the mean response and p-values. All three styles share the exact
same BH family and decisions.
