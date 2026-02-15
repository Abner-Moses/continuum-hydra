from __future__ import annotations

from rich.console import Console
from rich.table import Table

from continuum.doctor.models import Report

_STATUS_STYLE = {
    "PASS": "green",
    "WARN": "yellow",
    "FAIL": "red",
    "SKIP": "dim",
    "ERROR": "magenta",
}


def render_report_human(report: Report, console: Console | None = None) -> None:
    active_console = console or Console()

    table = Table(title="Continuum Doctor Report")
    table.add_column("Status", no_wrap=True)
    table.add_column("Check", overflow="fold")
    table.add_column("Message", overflow="fold")

    for check in report.checks:
        status_value = check.status.value
        style = _STATUS_STYLE.get(status_value, "")
        check_name = f"{check.id} - {check.title}"
        table.add_row(f"[{style}]{status_value}[/{style}]" if style else status_value, check_name, check.message)

    active_console.print(table)

    fail_count = report.summary.get("FAIL", 0)
    warn_count = report.summary.get("WARN", 0)
    error_count = report.summary.get("ERROR", 0)
    active_console.print(
        f"Summary: FAIL={fail_count} WARN={warn_count} ERROR={error_count}",
    )


__all__ = ["render_report_human"]
