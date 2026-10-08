"""Load a single geographic field without guessing its scientific meaning."""

from collections.abc import Mapping, Sequence
from os import PathLike

import numpy as np
import xarray as xr

Source = str | PathLike | xr.Dataset | xr.DataArray
_STATISTICS = {"mean", "median", "min", "max", "sum", "std"}


def _coordinate(da: xr.DataArray, kind: str, explicit: str | None) -> str:
    if explicit is not None:
        if explicit not in da.coords:
            raise ValueError(f"Coordinate {explicit!r} was not found in {list(da.coords)}.")
        return explicit
    aliases = {"latitude": {"lat", "latitude"}, "longitude": {"lon", "longitude"}}
    units = {
        "latitude": {"degrees_north", "degree_north", "degrees_n", "degree_n"},
        "longitude": {"degrees_east", "degree_east", "degrees_e", "degree_e"},
    }
    candidates = [
        name
        for name, coord in da.coords.items()
        if str(name).lower() in aliases[kind]
        or coord.attrs.get("standard_name", "").lower() == kind
        or coord.attrs.get("units", "").lower() in units[kind]
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"Cannot uniquely identify {kind}; candidates: {candidates}. "
            f"Pass {kind}='coordinate_name'. Projected x/y coordinates in metres "
            "are not geographic longitude/latitude."
        )
    return candidates[0]


def _select_variable(source: xr.Dataset | xr.DataArray, variable: str | None) -> xr.DataArray:
    if isinstance(source, xr.DataArray):
        if variable is not None and variable != source.name:
            raise ValueError(f"DataArray is named {source.name!r}, not {variable!r}.")
        return source
    if variable is None:
        # Ignore coordinate bounds / scalar CRS metadata when choosing the only map field.
        candidates = [name for name, value in source.data_vars.items() if value.ndim >= 2]
        bounds = {c.attrs.get("bounds") for c in source.coords.values()}
        candidates = [name for name in candidates if name not in bounds]
        if len(candidates) != 1:
            raise ValueError(
                f"Choose variable= explicitly. Available fields: {candidates or list(source.data_vars)}."
            )
        variable = candidates[0]
    if variable not in source.data_vars:
        raise ValueError(f"Unknown variable {variable!r}. Available: {list(source.data_vars)}.")
    return source[variable]


def _prepare(
    source, variable, sel, isel, reduce, statistic, latitude, longitude, scale, offset, units
) -> xr.DataArray:
    da = _select_variable(source, variable)
    attrs = dict(da.attrs)
    lat = _coordinate(da, "latitude", latitude)
    lon = _coordinate(da, "longitude", longitude)
    for name in (lat, lon):
        if da[name].ndim != 1:
            raise ValueError(
                "Only rectilinear grids with one-dimensional latitude/longitude are supported; "
                f"{name!r} has {da[name].ndim} dimensions. Regrid curvilinear data first."
            )
    lat_dim, lon_dim = da[lat].dims[0], da[lon].dims[0]
    if lat_dim == lon_dim:
        raise ValueError("Latitude and longitude must index two different dimensions.")
    if sel and isel and set(sel) & set(isel):
        raise ValueError("Do not select the same dimension with both sel and isel.")
    if sel:
        da = da.sel(dict(sel))
    if isel:
        da = da.isel(dict(isel))
    reduce_dims = [reduce] if isinstance(reduce, str) else list(reduce or [])
    if statistic not in _STATISTICS:
        raise ValueError(f"Unknown statistic {statistic!r}; choose from {sorted(_STATISTICS)}.")
    if set(reduce_dims) & {lat_dim, lon_dim}:
        raise ValueError("Reduction must preserve latitude and longitude.")
    if reduce_dims:
        missing = set(reduce_dims) - set(da.dims)
        if missing:
            raise ValueError(f"Reduction dimensions not found: {sorted(missing)}.")
        options = {"dim": reduce_dims, "skipna": True, "keep_attrs": True}
        if statistic == "sum":
            options["min_count"] = 1
        da = getattr(da, statistic)(**options)
    singleton = [d for d in da.dims if d not in (lat_dim, lon_dim) and da.sizes[d] == 1]
    if singleton:
        da = da.squeeze(singleton, drop=True)
    extra = [d for d in da.dims if d not in (lat_dim, lon_dim)]
    if extra:
        raise ValueError(
            f"Select or reduce extra dimensions {dict((d, da.sizes[d]) for d in extra)}. "
            "For example: isel={'time': 0}, sel={'level': 850}, or reduce='time'."
        )
    if lat_dim not in da.dims or lon_dim not in da.dims:
        raise ValueError("Selection must preserve a two-dimensional latitude/longitude grid.")
    da = da.transpose(lat_dim, lon_dim)
    if not np.issubdtype(da.dtype, np.number) or np.issubdtype(da.dtype, np.complexfloating):
        raise ValueError("The plotted variable must contain real numeric values.")
    lat_values = np.asarray(da[lat].values, dtype=float)
    lon_values = np.asarray(da[lon].values, dtype=float)
    for name, values in (("latitude", lat_values), ("longitude", lon_values)):
        if len(values) < 2 or not np.isfinite(values).all():
            raise ValueError(f"{name.capitalize()} needs at least two finite coordinates.")
    if np.any(np.abs(lat_values) > 90):
        raise ValueError("Latitude must be in degrees within [-90, 90].")
    if np.unique(lat_values).size != lat_values.size:
        raise ValueError("Latitude coordinates must be unique.")
    if np.ptp(lon_values) > 360 + 1e-6:
        raise ValueError("Longitude spans more than 360 degrees; check its units.")
    normalized_lon = (lon_values + 180) % 360 - 180
    # Remove only a duplicated global seam, never arbitrary duplicate grid columns.
    _, unique = np.unique(np.round(normalized_lon, 8), return_index=True)
    if len(unique) != len(lon_values):
        seam = len(unique) == len(lon_values) - 1 and np.isclose(
            abs(lon_values[-1] - lon_values[0]), 360, rtol=0, atol=1e-6
        )
        if not seam:
            raise ValueError("Longitude coordinates contain duplicate grid columns.")
        da = da.isel({lon_dim: slice(None, -1)})
        normalized_lon = normalized_lon[:-1]
    if len(normalized_lon) < 2:
        raise ValueError("At least two distinct longitude coordinates are required.")
    if np.any(np.diff(np.sort(normalized_lon)) > 180):
        raise ValueError(
            "The regional grid crosses the antimeridian or contains a longitude gap "
            "larger than 180 degrees. Split or regrid it before plotting."
        )
    # Construct canonical coordinates without rename conflicts (e.g. latitude(y)).
    result = xr.DataArray(
        da.values,
        dims=("lat", "lon"),
        coords={"lat": lat_values, "lon": normalized_lon},
        name=da.name,
        attrs=attrs,
    ).sortby(["lat", "lon"])
    if not np.isfinite(scale) or not np.isfinite(offset):
        raise ValueError("scale and offset must be finite numbers.")
    with xr.set_options(keep_attrs=True):
        result = result * scale + offset
    if units is not None:
        result.attrs["units"] = units
    result.lat.attrs = {"standard_name": "latitude", "units": "degrees_north"}
    result.lon.attrs = {"standard_name": "longitude", "units": "degrees_east"}
    result = result.where(np.isfinite(result))
    if not bool(result.notnull().any()):
        raise ValueError("The selected field contains no finite data to plot.")
    return result


def open_field(
    source: Source,
    variable: str | None = None,
    *,
    sel: Mapping | None = None,
    isel: Mapping | None = None,
    reduce: str | Sequence[str] | None = None,
    statistic: str = "mean",
    latitude: str | None = None,
    longitude: str | None = None,
    scale: float = 1,
    offset: float = 0,
    units: str | None = None,
) -> xr.DataArray:
    """Read one 2D field from NetCDF, Dataset, or DataArray.

    Detect geographic coordinates by common names or CF metadata, sort latitude,
    and wrap longitude to [-180, 180). Extra dimensions require ``sel``, ``isel``,
    or an explicit ``reduce``; only nonspatial singleton dimensions are squeezed.
    Conversion is explicit: ``offset=-273.15, units='°C'`` for Kelvin input.
    The returned array is loaded into memory and independent of the source file.
    Input datasets and arrays are never modified or closed by this function.
    """
    arguments = (variable, sel, isel, reduce, statistic, latitude, longitude, scale, offset, units)
    if isinstance(source, (xr.DataArray, xr.Dataset)):
        return _prepare(source, *arguments)
    with xr.open_dataset(source) as dataset:
        return _prepare(dataset, *arguments)
