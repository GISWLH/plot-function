# Contributing

Thank you for helping make geographic data easier to plot. Bug reports, focused
fixes, examples, and Chinese / English / Japanese documentation improvements are
welcome.

## Set up

Use Python 3.10 or newer and a virtual environment:

```bash
python -m pip install -e '.[dev]'
pytest
ruff check plot_function utils tests examples
python examples/quickstart.py
python -m build
```

The tests render with Agg and do not require network downloads or the large
bundled datasets. CI runs the same workflow on Python 3.10 and 3.12.

## Changes and bug reports

- Include a small reproducible example, your Python/package versions, and the
  relevant variable names, dimensions, units, and coordinate metadata.
- Prefer a tiny synthetic NetCDF fixture over adding a large dataset. Do not
  include credentials or private data in an issue or pull request.
- Preserve the xarray / Cartopy / Matplotlib approach and compatibility imports.
  Keep transformations explicit; do not silently choose levels, weights, or units.
- Add a regression test for changed behavior. Keep rendering tests independent
  of internet access and avoid image-pixel assertions that depend on fonts.
- Keep the three READMEs aligned when changing public behavior. Detailed API and
  migration references live in `docs/`.
- Gallery changes should be reproducible with `python examples/gallery.py` and
  should state the input data, transformations, units, and display sampling.

Use `ruff format plot_function utils tests examples` for Python formatting. Submit
a focused pull request explaining the behavior change and the checks you ran.
