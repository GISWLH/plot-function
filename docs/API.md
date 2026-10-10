# API reference

[English](../README.md) · [简体中文](../README.zh-CN.md) · [日本語](../README.ja.md)

## `open_field(source, variable=None, **options)`

Read a geographic field from a NetCDF path, an `xarray.Dataset`, or an
`xarray.DataArray`. Return an in-memory, two-dimensional `DataArray` with
dimensions `(lat, lon)`. Caller-owned datasets are not closed or modified.

| Option | Default | Meaning |
| :--- | :--- | :--- |
| `variable` | `None` | Select a data variable. Automatic only when one candidate exists; CF coordinate bounds are excluded. |
| `sel` | `None` | xarray label selection, e.g. `{"level": 850}`. Python slices are supported. |
| `isel` | `None` | xarray integer selection, e.g. `{"time": 0}`. Python slices are supported. |
| `reduce` | `None` | One or several nonspatial dimensions to aggregate. |
| `statistic` | `"mean"` | `mean`, `median`, `min`, `max`, `sum`, or `std`. Missing values are skipped; all-missing sums remain missing. |
| `latitude`, `longitude` | `None` | Explicit coordinate names; otherwise common names or CF metadata are used. |
| `scale`, `offset` | `1`, `0` | Apply `value * scale + offset` after reduction. |
| `units` | `None` | Override the label after a deliberate conversion. Does not perform a conversion on its own. |

Processing order: select variable → find coordinates → label/integer selection →
reduction → squeeze nonspatial singleton dimensions → normalize coordinates →
apply scale/offset → mask nonfinite data. Source unit attributes are preserved
unless `units` is supplied. Latitude/longitude remain in degrees.

Examples from the bundled dataset:

```python
from plot_function import open_field

path = "data/ERA5temp_1978_monthly.nc"
january = open_field(path, "t2m", isel={"time": 0}, offset=-273.15, units="°C")
annual = open_field(path, "t2m", reduce="time", offset=-273.15, units="°C")
```

The library deliberately does not choose a time, vertical layer, or ensemble
member on your behalf. Reductions use ordinary, unweighted xarray statistics
(`std` uses the population convention, `ddof=0`). For a day-weighted annual
temperature average, compute the scientific transformation before plotting:

```python
import xarray as xr
from plot_function import plot_map

with xr.open_dataset("data/ERA5temp_1978_monthly.nc") as ds:
    monthly = ds.t2m
    weighted = monthly.weighted(ds.time.dt.days_in_month).mean("time").load()
weighted.attrs = {"long_name": "Day-weighted 1978 temperature", "units": "K"}
result = plot_map(weighted, offset=-273.15, units="°C")
```

Apply any transformations whose ordering matters (for example, unit conversion
before summation or a nonlinear derived variable) in xarray first.

## `plot_map(source, variable=None, **options)`

Accepts all `open_field` options, plus:

| Option | Default | Meaning |
| :--- | :--- | :--- |
| `projection` | `"robinson"` | `robinson`, `platecarree`, `equalearth`, `mollweide`, or a Cartopy projection object. |
| `extent` | `None` | `[west, east, south, north]` in degrees; bounds must be ordered within −180…180 and −90…90. `None` shows the world. |
| `cmap` | `"viridis"` | Matplotlib colormap name or object. |
| `levels` | `None` | Contour/color boundaries; useful for a fixed discrete scale. |
| `vmin`, `vmax` | `None` | Explicit color limits. Otherwise xarray chooses them. |
| `title` | `None` | Defaults to the variable's `long_name`, then its name. |
| `subtitle` | `None` | Optional explanatory line below the title. |
| `label` | `None` | Colorbar label; defaults to the field's units. |
| `theme` | `"light"` | `light` or `dark`. Does not change global Matplotlib settings. |
| `coastlines` | `True` | Natural Earth 110m coastlines. Disable to avoid all basemap downloads. |
| `gridlines` | `True` | Geographic graticule; regional views include labels. |
| `colorbar` | `True` | Add a horizontal colorbar. Disable for shared custom colorbars. |
| `plotfunc` | `"pcolormesh"` | `pcolormesh` or `contourf`. |
| `ax` | `None` | An existing Cartopy GeoAxes; overrides `projection`. Its figure background remains caller-controlled. |
| `figsize` | `(7.2, 4.2)` | Inches (≈183 mm double column), used only when creating a new figure. |
| `output` | `None` | Optional path to save immediately. Parent directories are created. |
| `dpi` | `300` | Resolution for immediate export; use 600 for print. |

Regional extents control the viewport; they do not crop, mask, or resample the
returned data. For very large arrays, select/crop the data before plotting.

### `MapResult`

| Attribute / method | Description |
| :--- | :--- |
| `.figure` | Matplotlib Figure. |
| `.axes` | Cartopy GeoAxes; add text, contours, annotations, or features. |
| `.artist` | The plotted mappable, usable for a shared colorbar. |
| `.colorbar` | Colorbar or `None`. |
| `.data` | The normalized 2D field, including explicitly converted values. |
| `.save(path, dpi=300, **kwargs)` | Save with a tight bounding box and the figure's background. Additional options go to `Figure.savefig`. Returns a `Path`. |

```python
result.axes.text(0.02, 0.02, "ERA5 · 1978", transform=result.axes.transAxes)
result.save("figures/temperature.svg")
```

`pcolormesh` is rasterized inside vector exports to keep file sizes manageable;
labels, coastlines, and other vector artists stay editable. Figures are not
automatically closed. In loops, use `matplotlib.pyplot.close(result.figure)`.

## Offline operation

The core workflow has no runtime network requirement when `coastlines=False`.
With coastlines enabled, Cartopy downloads Natural Earth 110m coastline data on
first use unless cached. In restricted environments, populate Cartopy's standard
data cache from the official Natural Earth source or disable coastlines. No API
key is required. The optional historical Salem workflow has its own sample-data
cache and may download data when first imported.

## Troubleshooting

| Message / symptom | What to do |
| :--- | :--- |
| `Choose variable=` | Inspect the file and select the intended numeric field. |
| `Select or reduce extra dimensions` | Choose time/level/member values or explicitly aggregate them. |
| `Cannot uniquely identify latitude/longitude` | Check CF metadata, or provide coordinate names explicitly. Never label projected metres as degrees. |
| `Only rectilinear grids...` | Regrid 2D geographic coordinates or unstructured data before plotting. |
| `no finite data` | Check the selected slice, missing-value metadata, and prior processing. |
| Natural Earth download fails | Use `coastlines=False` / `--no-coastlines`, or prepare the cache. |
| Non-Latin title glyphs are missing | Install a suitable font and set Matplotlib's `font.family` / `font.sans-serif` before plotting. |
| CLI cannot be found | Activate the installation environment, or use `python -m plot_function`. |

See the translated READMEs for the full getting-started workflow.

## Statistical and geographic layers (0.3)

`plot_map` additionally accepts `profiles` (one `Profile` or a list),
`distribution=Distribution(...)`, `significance=Significance(...)`,
`features=MapFeatures(...)`, and `panel_label='a'`. All default to `None`,
preserving the original API's behavior.

`MapResult` now exposes `.profiles` keyed by `right`/`top`, `.distribution`,
`.significance` (a list), and `.feature_artists`. Statistical layers return a
`LayerResult` with `.axes`, `.statistics`, and `.artists`. Incremental methods
`add_profile`, `add_distribution`, `add_significance`, and `add_features` accept
an option object or its keyword arguments. See [Research figures](RESEARCH_FIGURES.md)
for styles, weights, scope, algorithms, defaults, and examples.

## Data contract

- **Rectilinear geographic grids:** latitude and longitude must each be one-dimensional, finite, and contain at least two distinct values. CF `standard_name` / geographic `units` and common `lat`/`lon` or `latitude`/`longitude` names are recognized.
- **Explicit scientific choices:** multiple fields require `variable=`; nonspatial dimensions with more than one value require selection or reduction. Reductions are unweighted and skip missing values. Supported statistics: `mean`, `median`, `min`, `max`, `sum`, `std`.
- **Coordinate normalization:** latitude is sorted, longitude is wrapped to `[-180, 180)` and sorted; a redundant global 0°/360° endpoint is removed. Other duplicates are rejected.
- **Missing values:** xarray decodes NetCDF fill values; nonfinite values are masked. An entirely missing field produces an actionable error.
- **Bounded scope:** curvilinear grids, unstructured meshes, projected x/y grids in metres, and antimeridian-crossing regional grids/extents need preprocessing. The API does not reproject source grids or infer units, vertical levels, time weights, or area weights.
- **Memory:** the selected/reduced 2D field is loaded before the file is closed. For large workflows, preprocess with xarray and pass a prepared `DataArray`.

## `plot_function.journal`

Publication helpers that decorate *any* Matplotlib/Cartopy axes. They never
modify global `rcParams`; wrap figure code in `journal_style()`. Gallery:
[`examples/journal_figures.py`](../examples/journal_figures.py); minimal example:
[`examples/journal_quickstart.py`](../examples/journal_quickstart.py).

| Function | Key arguments | Returns |
| :--- | :--- | :--- |
| `journal_style(**overrides)` | any rcParams | context manager (Arial → Helvetica → Liberation Sans → Nimbus Sans → DejaVu Sans, 7 pt base, 0.5 pt lines, `pdf.fonttype=42`) |
| `figsize(width="double", ratio=0.55)` | `"single"` (89 mm), `"onehalf"` (120 mm), `"double"` (183 mm) or mm | `(w, h)` inches |
| `discrete_cmap(cmap, levels, extend="both")` | colormap name/object or explicit colour list (`len(levels)-1` + number of extend ends) | `(ListedColormap, BoundaryNorm)` |
| `add_colorbar(mappable, ax, label=, title=, orientation=, bounds=, tick_every=)` | `bounds=[x, y, w, h]` in figure fraction, or placement relative to `ax` | `Colorbar` |
| `geo_ticks(ax, xticks=, yticks=, lat_labels=, lon_labels=, gridlines=)` | degree tick positions | gridliner list |
| `add_land(ax, color=LAND, borders=True, resolution="110m")` | Natural Earth (downloads once) | feature artists |
| `add_stippling(ax, lon, lat, mask, stride=2, size=1.2, offset=True)` | 1-D or 2-D lon/lat, boolean mask | `PathCollection` or `None` |
| `add_hatching(ax, lon, lat, mask, hatch="....")` | | contour set or `None` |
| `hatch_patch(hatch)` | | legend handle |
| `add_inset_bars(ax, labels, values, hatched=, colors=, bounds=, frame="box")` | `hatched` = uncertain part of each bar | inset axes |
| `add_inset_histogram(ax, groups, bins=, log=, cumulative=, bounds=, frame="open")` | array or `{label: values}`; `cumulative=True` or weights | `(hist_axes, twin_axes or None)` |
| `add_latitude_profile(ax, lat, center, lower=, upper=, outer=, members=, width=, pad=)` | values per latitude | profile axes, y axis locked to the map |
| `add_longitude_profile(ax, lon, center, lower=, upper=)` | values per longitude | profile axes above the map |
| `add_panel_label(ax, "a", style="({})")` | | `Text` |
| `add_size_legend(ax, values, labels, scale=)` | `scale(value) -> marker area (pt²)` | `Legend` |
| `save_figure(fig, path, dpi=600, formats=None)` | e.g. `formats=("png", "pdf")` | path(s) |

**Profile alignment.** Profile coordinates are transformed with
`ax.projection.transform_points` at the central longitude (or the centre of
the extent), and the profile's limits are synchronised with the map's at draw
time. On pseudo-cylindrical projections (Robinson, Equal Earth, Mollweide)
latitude maps to a single y value, so alignment is exact; on conic projections
it is exact only along the reference meridian.
