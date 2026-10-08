"""NetCDF-first plotting, built on the original plot-function map utilities."""

from .data import open_field
from .maps import MapResult, plot_map

__all__ = ["MapResult", "open_field", "plot_map"]
__version__ = "0.2.0"
