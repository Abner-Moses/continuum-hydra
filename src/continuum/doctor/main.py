from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import typer

# Import registers built-in checks via decorators.
from continuum.doctor.checks import environment as _environment_checks  # noqa: F401
from continuum.doctor.formatters.human import render_report_human
from continuum.doctor.formatters.json import report_to_dict, write_report_json
from continuum.doctor.runner import DoctorRunner


def _resolve_hydra_version() -> str:
    for dist_name in ("continuum-intelligence", "continuum"):
        try:
            return version(dist_name)
        except PackageNotFoundError:
            continue
    return "0.0.0"


def doctor_command(
    json_output: bool = typer.Option(False, "--json", help="Also print JSON report to stdout."),
    export: Path | None = typer.Option(None, "--export", help="Directory to write JSON report."),
    no_write: bool = typer.Option(False, "--no-write", help="Do not write JSON report to disk."),
    verbose: bool = typer.Option(False, "--verbose", help="Reserved for future verbose output."),
) -> None:
    try:
        runner = DoctorRunner(hydra_version=_resolve_hydra_version())
        report = runner.run(context={"verbose": verbose})

        render_report_human(report)

        if json_output:
            typer.echo(json.dumps(report_to_dict(report), indent=2, ensure_ascii=False))

        if not no_write:
            output_dir = export if export is not None else Path(".hydra/reports")
            write_report_json(report, output_dir)

        raise typer.Exit(code=DoctorRunner.exit_code(report))
    except typer.Exit:
        raise
    except Exception:
        raise typer.Exit(code=4)


__all__ = ["doctor_command"]
