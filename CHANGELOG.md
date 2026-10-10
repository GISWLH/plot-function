# Changelog

## 0.4.0 — Unreleased

- New `plot_function.journal` module: Nature/Science-style typography
  (`journal_style`, Arial/Helvetica fallback chain, 7–9 pt, editable PDF/SVG text),
  column-width `figsize`, discrete colormaps with extend triangles
  (`discrete_cmap`), slim titled colourbars (`add_colorbar`), clean degree ticks and
  dashed graticules (`geo_ticks`), cream land and borders (`add_land`),
  stippling/hatching helpers with legend handles, `add_inset_bars` (stacked
  low-agreement hatching), `add_inset_histogram` (stacked groups, total outline,
  log bins, cumulative curve on an offset twin axis), projection-aware marginal
  profiles (`add_latitude_profile`, `add_longitude_profile`), panel labels, size
  legends and 600-dpi `save_figure`.
- Marginal profiles are now aligned with the map latitudes on every projection
  (Robinson, Equal Earth, …), not only Plate Carrée, and follow `set_extent`.
- Restyled `plot_map`: white publication theme, thinner lines, slim colourbars,
  degree tick labels on regional maps, Nature-style panel letters, 300-dpi export.
- Restyled distribution insets (white translucent backdrop, open spines, mean
  marker) and profile panels (labels below the axis, dashed latitude guides).
- New gallery (`examples/journal_figures.py`, `docs/gallery/`) and redesigned
  minimalist line-drawing logo and banner (`docs/brand/*.svg`).
- Bilingual README, `CITATION.cff`, richer package metadata.

## 0.3.0 — Unreleased

- Add right/top profiles with line, spread-band, and bar styles.
- Add bars, steps, frequency polygons, and ECDF inset distributions.
- Add explicit spatial weighting, summary scopes, and inspectable xarray results.
- Add aligned p-value/mask overlays with stippling, hatching, contour boundaries,
  and optional Benjamini–Hochberg FDR correction.
- Add configurable Natural Earth features, panel labels, and CLI controls.
- Add three white-background research compositions and multilingual guidance.

## 0.2.0

- Add an installable `plot_function` package and `plot-function` CLI.
- Add NetCDF variable selection, geographic coordinate recognition, explicit
  dimension reduction and conversion, and longitude normalization.
- Add configurable global and regional maps with editable results and export.
- Preserve original notebook imports and correct legacy plotting regressions.
- Add offline tests, CI, a synthetic quick start, and four reproducible ERA5 demos.
- Replace the README with complete English, Chinese, and Japanese guides and
  local branding/gallery assets.
