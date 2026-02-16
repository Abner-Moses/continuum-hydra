from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import typer

# Import registers built-in checks via decorators.
from continuum.doctor.checks import cuda as _cuda_checks  # noqa: F401
from continuum.doctor.checks import environment as _environment_checks  # noqa: F401
from continuum.doctor.checks import gpu as _gpu_checks  # noqa: F401
from continuum.doctor.checks import gpu_props as _gpu_props_checks  # noqa: F401
from continuum.doctor.checks import nccl as _nccl_checks  # noqa: F401
from continuum.doctor.checks import pytorch as _pytorch_checks  # noqa: F401
from continuum.doctor.checks import system as _system_checks  # noqa: F401
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


def _parse_csv_values(value: str | None) -> set[str]:
    if not value:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def doctor_command(
    json_output: bool = typer.Option(False, "--json", help="Also print JSON report to stdout."),
    export: Path | None = typer.Option(None, "--export", help="Directory to write JSON report."),
    no_write: bool = typer.Option(False, "--no-write", help="Do not write JSON report to disk."),
    only: str | None = typer.Option(None, "--only", help="Comma-separated check ids or categories to include."),
    exclude: str | None = typer.Option(None, "--exclude", help="Comma-separated check ids to exclude."),
    list_checks: bool = typer.Option(False, "--list-checks", help="List available checks and exit."),
    deterministic: bool = typer.Option(False, "--deterministic", help="Stable timestamps/durations for CI diffs."),
    verbose: bool = typer.Option(False, "--verbose", help="Reserved for future verbose output."),
) -> None:
    try:
        runner = DoctorRunner(hydra_version=_resolve_hydra_version())
        selected_checks = DoctorRunner.filter_checks(
            runner.checks,
            only=_parse_csv_values(only),
            exclude=_parse_csv_values(exclude),
        )

        if list_checks:
            for check in selected_checks:
                typer.echo(f"{check.id}\t{check.category}\t{check.title}")
            raise typer.Exit(code=0)

        report = DoctorRunner(hydra_version=_resolve_hydra_version(), checks=selected_checks).run(
            context={"verbose": verbose, "deterministic": deterministic}
        )

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
