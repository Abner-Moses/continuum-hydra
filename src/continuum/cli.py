from __future__ import annotations

import typer

from continuum.doctor.main import doctor_command

app = typer.Typer(
    help="Continuum CLI — Performance-first ML infrastructure toolkit.",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """Root CLI group for Continuum subcommands."""


app.command(name="doctor")(doctor_command)

__all__ = ["app"]
