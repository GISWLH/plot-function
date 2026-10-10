"""NetCDF-first plotting, built on the original plot-function map utilities."""

from . import journal
from .data import open_field
from .maps import MapResult, plot_map
from .options import Distribution, MapFeatures, Profile, Significance
from .statistics import histogram, significance_mask, spatial_profile

__all__ = [
    "journal",
    "MapResult",
    "open_field",
    "plot_map",
    "Profile",
    "Distribution",
    "Significance",
    "MapFeatures",
    "spatial_profile",
    "histogram",
    "significance_mask",
]
__version__ = "0.5.0"
