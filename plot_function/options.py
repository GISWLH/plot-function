"""Declarative, reusable options for statistical and cartographic layers."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Profile:
    position: str = "right"
    style: str = "band"
    statistic: str = "mean"
    spread: str | None = "std"
    weights: Any = None
    color: str = "#267f83"
    width: float = 0.20
    pad: float = 0.045
    label: str | None = None
    reference: float | None = None


@dataclass(frozen=True)
class Distribution:
    style: str = "bars"
    bins: Any = 18
    weights: Any = None
    density: bool = True
    bounds: tuple = (0.045, 0.075, 0.27, 0.25)
    color: str = "#267f83"
    label: str | None = None
    show_mean: bool = True


@dataclass(frozen=True)
class Significance:
    data: Any
    variable: str | None = None
    kind: str = "pvalue"
    style: str = "stipple"
    alpha: float = 0.05
    correction: str = "none"
    stride: int = 3
    color: str = "#243e44"
    size: float = 3.0
    hatch: str = "...."
    label: str | None = None
    legend: bool = True
    sel: Any = None
    isel: Any = None
    latitude: str | None = None
    longitude: str | None = None


@dataclass(frozen=True)
class MapFeatures:
    resolution: str = "110m"
    rivers: bool = False
    lakes: bool = False
    borders: bool = False
    land: bool = False
    ocean: bool = False
    river_color: str = "#518eab"
    lake_color: str = "#b6d9e5"
    border_color: str = "#667579"
    land_color: str = "#f0eee8"
    ocean_color: str = "#e5f0f5"
    linewidth: float = 0.5
