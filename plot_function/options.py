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
    color: str = "#2b5d8c"
    width: float = 0.16
    pad: float = 0.03
    label: str | None = None
    reference: float | None = None


@dataclass(frozen=True)
class Distribution:
    style: str = "bars"
    bins: Any = 18
    weights: Any = None
    density: bool = True
    bounds: tuple = (0.06, 0.1, 0.2, 0.22)
    color: str = "#3b6f8f"
    label: str | None = None
    show_mean: bool = True
    background: str | None = "white"  # translucent backdrop over busy maps; None = clear
    background_alpha: float = 0.8


@dataclass(frozen=True)
class Significance:
    data: Any
    variable: str | None = None
    kind: str = "pvalue"
    style: str = "stipple"
    alpha: float = 0.05
    correction: str = "none"
    stride: int = 3
    color: str = "#1a1a1a"
    size: float = 1.2
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
