"""Command-line access to NetCDF inspection and plotting."""

import argparse
import json
import sys


def _mapping(value):
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError("Use a JSON object, for example '{\"time\": 0}'.") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("Selection must be a JSON object.")
    return parsed


def parser():
    root = argparse.ArgumentParser(description="Plot geographic NetCDF data with one command.")
    commands = root.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser(
        "inspect", help="List variables, dimensions, coordinates, and units"
    )
    inspect.add_argument("file")
    plot = commands.add_parser("plot", help="Save a map without opening a GUI")
    plot.add_argument("file")
    plot.add_argument("-v", "--variable")
    plot.add_argument("-o", "--output", required=True)
    plot.add_argument("--isel", type=_mapping, help="Index selection, e.g. '{\"time\": 0}'")
    plot.add_argument("--sel", type=_mapping, help="Label selection, e.g. '{\"level\": 850}'")
    plot.add_argument("--reduce", nargs="+", help="Dimensions to aggregate, e.g. time")
    plot.add_argument(
        "--statistic", choices=["mean", "median", "min", "max", "sum", "std"], default="mean"
    )
    plot.add_argument("--latitude", help="Latitude coordinate name")
    plot.add_argument("--longitude", help="Longitude coordinate name")
    plot.add_argument("--scale", type=float, default=1)
    plot.add_argument("--offset", type=float, default=0)
    plot.add_argument("--units", help="Output unit label after explicit scale/offset conversion")
    plot.add_argument(
        "--projection",
        choices=["robinson", "platecarree", "equalearth", "mollweide"],
        default="robinson",
    )
    plot.add_argument("--extent", type=float, nargs=4, metavar=("WEST", "EAST", "SOUTH", "NORTH"))
    plot.add_argument("--cmap", default="viridis")
    plot.add_argument("--vmin", type=float)
    plot.add_argument("--vmax", type=float)
    plot.add_argument("--title")
    plot.add_argument("--subtitle")
    plot.add_argument("--theme", choices=["light", "dark"], default="light")
    plot.add_argument(
        "--no-coastlines", action="store_true", help="Render without map-data downloads"
    )
    plot.add_argument("--dpi", type=int, default=180)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "inspect":
            import xarray as xr

            with xr.open_dataset(args.file) as ds:
                print(ds)
                for name, field in ds.data_vars.items():
                    print(f"{name}: {field.dims}, units={field.attrs.get('units', 'unspecified')}")
            return 0
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from .maps import plot_map

        options = vars(args).copy()
        options.pop("command")
        source = options.pop("file")
        options["coastlines"] = not options.pop("no_coastlines")
        result = plot_map(source, **options)
        plt.close(result.figure)
        print(f"Saved {args.output}")
        return 0
    except (ValueError, OSError, KeyError, IndexError, TypeError) as exc:
        print(f"plot-function: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
