from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from continuum.accelerate.models import AccelerationActionResult, ExecutionContext, normalize_profile, parse_csv_set
from continuum.accelerate.plan_builder import build_plan
from continuum.accelerate.plugins.loader import run_shell_hooks
from continuum.accelerate.reporting import build_report, render_summary, write_state_report
from continuum.accelerate.ui.interactive import select_actions_interactively

app = typer.Typer(
    help="Hydra Accelerate: safe performance optimization planner/executor",
    invoke_without_command=True,
)


def _render_plan(plan_dict: dict, console: Console) -> None:
    table = Table(title=f"Hydra Accelerate Plan ({plan_dict['profile']})")
    table.add_column("Recommended", no_wrap=True)
    table.add_column("Supported", no_wrap=True)
    table.add_column("ID")
    table.add_column("Category", no_wrap=True)
    table.add_column("Risk", no_wrap=True)
    table.add_column("Root", no_wrap=True)

    for item in plan_dict.get("recommendations", []):
        table.add_row(
            "yes" if item.get("recommended") else "no",
            "yes" if item.get("supported") else "no",
            item.get("action_id", ""),
            item.get("category", ""),
            item.get("risk", ""),
            "yes" if item.get("requires_root") else "no",
        )

    console.print(table)


def _build_dry_run_results(plan_dict: dict) -> list[AccelerationActionResult]:
    results: list[AccelerationActionResult] = []
    for item in plan_dict.get("recommendations", []):
        results.append(
            AccelerationActionResult(
                action_id=item["action_id"],
                title=item["title"],
                supported=bool(item["supported"]),
                applied=False,
                skipped_reason="Dry run - not applied",
                requires_root=bool(item["requires_root"]),
                risk=item["risk"],
                before={},
                after={},
                commands=list(item.get("commands", [])),
                errors=[],
            )
        )
    return results


def _auto_selection(plan_dict: dict, expert_mode: bool) -> set[str]:
    selected: set[str] = set()
    for item in plan_dict.get("recommendations", []):
        if not item.get("recommended") or not item.get("supported"):
            continue
        if item.get("risk", "").lower() == "high" and not expert_mode:
            continue
        selected.add(item["action_id"])
    return selected


def _is_supported_os(ctx: ExecutionContext) -> bool:
    return ctx.is_linux or ctx.is_windows or ctx.is_macos


@app.callback()
def accelerate(
    dry_run: bool = typer.Option(True, "--dry-run/--apply", help="Plan only (default) or apply actions."),
    interactive: bool = typer.Option(False, "--interactive", help="Interactively choose actions."),
    profile: str = typer.Option("balanced", "--profile", help="minimal|balanced|max|expert"),
    only: str | None = typer.Option(None, "--only", help="Comma-separated categories to include."),
    exclude: str | None = typer.Option(None, "--exclude", help="Comma-separated categories to exclude."),
    json_output: bool = typer.Option(False, "--json", help="Print JSON report."),
    out: Path | None = typer.Option(None, "--out", help="Write report JSON to this path."),
) -> None:
    console = Console()

    try:
        normalized_profile = normalize_profile(profile)
        expert_mode = normalized_profile == "expert"

        plan, internal_data, ctx, hooks = build_plan(
            profile=normalized_profile,
            only=parse_csv_set(only),
            exclude=parse_csv_set(exclude),
            expert_mode=expert_mode,
            cwd=Path.cwd(),
        )

        if not _is_supported_os(ctx):
            console.print("Skipped: not supported on this OS.")
            raise typer.Exit(code=0)

        plan_dict = plan.to_dict()
        _render_plan(plan_dict, console)

        selected_ids: set[str]
        hook_warnings: list[str] = []

        if dry_run:
            selected_ids = _auto_selection(plan_dict, expert_mode)
            results = _build_dry_run_results(plan_dict)
            report = build_report(
                plan=plan,
                action_results=results,
                ctx=ctx,
                selected_action_ids=selected_ids,
                dry_run=True,
                hook_warnings=[],
            )
            write_state_report(report, out=out, cwd=Path.cwd())
            render_summary(report, console)
            if json_output:
                typer.echo(json.dumps(report, indent=2, ensure_ascii=False))
            raise typer.Exit(code=0)

        if interactive:
            selected_ids = select_actions_interactively(plan.recommendations, console=console)
            if not typer.confirm("Apply selected actions?", default=False):
                console.print("Apply cancelled by user.")
                raise typer.Exit(code=0)
        else:
            selected_ids = _auto_selection(plan_dict, expert_mode)

        plan_payload = plan.to_dict()
        ctx_payload = ctx.to_dict()

        hook_warnings.extend(run_shell_hooks(hooks.pre_apply_shell, ctx_payload, plan_payload, selected_ids))
        for callback in hooks.pre_apply_py:
            try:
                callback(ctx_payload, plan_payload, selected_ids)
            except Exception as exc:  # noqa: BLE001
                hook_warnings.append(f"Python pre hook failed: {type(exc).__name__}: {exc}")

        results: list[AccelerationActionResult] = []
        for item in internal_data:
            action = item["action"]
            supported = bool(item["supported"])

            if action.id not in selected_ids:
                results.append(
                    AccelerationActionResult(
                        action_id=action.id,
                        title=action.title,
                        supported=supported,
                        applied=False,
                        skipped_reason="Not selected",
                        requires_root=action.requires_root,
                        risk=action.risk,
                        before=item.get("before", {}),
                        after=item.get("after_preview", {}),
                        commands=list(item.get("commands", [])),
                        errors=[],
                    )
                )
                continue

            if not supported:
                results.append(
                    AccelerationActionResult(
                        action_id=action.id,
                        title=action.title,
                        supported=False,
                        applied=False,
                        skipped_reason="Unsupported on this environment",
                        requires_root=action.requires_root,
                        risk=action.risk,
                        before=item.get("before", {}),
                        after=item.get("before", {}),
                        commands=list(item.get("commands", [])),
                        errors=[],
                    )
                )
                continue

            try:
                results.append(action.apply(ctx))
            except Exception as exc:  # noqa: BLE001
                results.append(
                    AccelerationActionResult(
                        action_id=action.id,
                        title=action.title,
                        supported=True,
                        applied=False,
                        skipped_reason="Action apply raised an exception",
                        requires_root=action.requires_root,
                        risk=action.risk,
                        before=item.get("before", {}),
                        after=item.get("before", {}),
                        commands=list(item.get("commands", [])),
                        errors=[f"{type(exc).__name__}: {exc}"],
                    )
                )

        hook_warnings.extend(run_shell_hooks(hooks.post_apply_shell, ctx_payload, plan_payload, selected_ids))
        for callback in hooks.post_apply_py:
            try:
                callback(ctx_payload, plan_payload, selected_ids)
            except Exception as exc:  # noqa: BLE001
                hook_warnings.append(f"Python post hook failed: {type(exc).__name__}: {exc}")

        report = build_report(
            plan=plan,
            action_results=results,
            ctx=ctx,
            selected_action_ids=selected_ids,
            dry_run=False,
            hook_warnings=hook_warnings,
        )
        write_state_report(report, out=out, cwd=Path.cwd())
        render_summary(report, console)

        if json_output:
            typer.echo(json.dumps(report, indent=2, ensure_ascii=False))

        raise typer.Exit(code=0)
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001
        console.print(f"Accelerate failed: {type(exc).__name__}: {exc}")
        raise typer.Exit(code=4)


__all__ = ["app"]
