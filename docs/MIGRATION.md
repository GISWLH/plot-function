# Migration from notebook helpers

The original plotting logic still uses xarray to draw on Cartopy GeoAxes with
Matplotlib and mplotutils. Its implementation now lives in
`plot_function/legacy.py`; `utils/plot.py` re-exports the public helpers, so
`from utils import plot` remains valid. The new `plot_map` API wraps this approach
with NetCDF loading, coordinate normalization, styling, and export.

## What changed

| Previous issue | Corrected behavior |
| :--- | :--- |
| Coastline keyword names passed as positional arguments | `resolution`, `alpha`, and other kwargs reach Cartopy correctly. |
| Hatching overwrote the map artist returned by `one_map` | Returns `(mappable, legend_handle)`; the mappable remains usable for colorbars. |
| An inverted all-zero mask produced no hatch | Inversion occurs before checking whether the mask is empty. |
| Hatching modified global `rcParams` | A local style context and artist properties keep styling scoped to the plot. |
| Invalid hatch-value errors omitted actual values | Errors report the observed range. |
| Every contour field received a cyclic column | Only regularly spaced global longitude grids are closed. Regional fields stay regional. |
| A region's `extents` was applied only when `interval` was set | Viewport and tick spacing work independently. Gridlines also work without explicit intervals. |
| Warming-panel legends called undefined helpers | Standard Matplotlib patch handles provide working legends. |
| `colorbar_kwargs` was ignored in warming panels | User overrides are applied. |
| A colorbar always claimed Celsius | Existing field units are used. |
| Boundary paths depended on the working directory | Legacy boundaries resolve relative to the checkout, or accept `shapefile=` in `add_china` / `add_dashline`. |
| Profile mean was labeled “Median” and assumed axis order | The profile reduces by the named latitude dimension and labels the curve “Mean”. |

## Recommended new code

```python
from plot_function import plot_map

result = plot_map(
    "data/ERA5temp_1978_monthly.nc", "t2m",
    reduce="time", offset=-273.15, units="°C",
    cmap="RdYlBu_r", output="temperature.png",
)
```

## Existing advanced layouts

```python
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from plot_function import open_field
from utils import plot

data = open_field("data/ERA5temp_1978_monthly.nc", "t2m", reduce="time")
fig, ax = plt.subplots(subplot_kw={"projection": ccrs.Robinson()})
artist, legend = plot.one_map(data, ax, hatch_data=data > 273.15)
fig.colorbar(artist, ax=ax, orientation="horizontal")
```

The legacy helpers do not perform the new API's coordinate validation or unit
conversion automatically. Use `open_field` when you want that preparation while
keeping a custom layout. Legacy regional tick intervals are intended for
rectangular map projections; use the new API's gridlines for other projections.

The historical notebook and its data remain unchanged. Its China-temperature
example still needs the external `data/tp/tmp_2022.nc` file. The generic NetCDF API
does not depend on the notebook, shapefiles, rioxarray, or Salem. The Python wheel
contains code, not the large repository datasets or legacy boundary files.
