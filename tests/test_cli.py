from plot_function.cli import main


def test_inspect(ncfile, capsys):
    assert main(["inspect", str(ncfile)]) == 0
    assert "units=K" in capsys.readouterr().out


def test_cli_plot(ncfile, tmp_path):
    out = tmp_path / "cli.png"
    assert (
        main(
            [
                "plot",
                str(ncfile),
                "--isel",
                '{"time": 0}',
                "--no-coastlines",
                "--offset",
                "-273.15",
                "--units",
                "°C",
                "-o",
                str(out),
            ]
        )
        == 0
    )
    assert out.stat().st_size > 5000


def test_cli_error_is_actionable(ncfile, capsys):
    assert main(["plot", str(ncfile), "-o", "unused.png"]) == 2
    assert "extra dimensions" in capsys.readouterr().err


def test_cli_statistical_composition(dataset, tmp_path):
    import xarray as xr

    dataset["p"] = xr.full_like(dataset.temperature, 0.001)
    path, output = tmp_path / "pvalues.nc", tmp_path / "composition.png"
    dataset.to_netcdf(path)
    assert (
        main(
            [
                "plot",
                str(path),
                "-v",
                "temperature",
                "--isel",
                '{"time": 0}',
                "--profile",
                "right",
                "--distribution",
                "ecdf",
                "--coslat",
                "--pvalues",
                str(path),
                "--pvariable",
                "p",
                "--p-isel",
                '{"time": 0}',
                "--fdr",
                "--no-coastlines",
                "-o",
                str(output),
            ]
        )
        == 0
    )
    assert output.stat().st_size > 5000
