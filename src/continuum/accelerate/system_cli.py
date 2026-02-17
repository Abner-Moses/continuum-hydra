from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import typer

from continuum.accelerate.system_formatter import render_status
from continuum.accelerate.system_state import load_state, save_state, utc_now
from continuum.accelerate.system_tuner import apply_acceleration, capture_previous_state, detect_context, restore_acceleration


def _compute_counts(changes: list[dict[str, Any]]) -> dict[str, int]:
    applied = sum(1 for change in changes if change.get("result") == "applied")
    skipped = sum(1 for change in changes if change.get("result") in {"skipped", "not-applied"})
    planned = sum(1 for change in changes if change.get("result") == "planned")
    return {
        "applied_count": applied,
        "skipped_count": skipped,
        "planned_count": planned,
    }


def _active_status(*, active_requested: bool, effective_active: bool, applied_count: int, skipped_count: int) -> str:
    if not active_requested:
        return "False"
    if effective_active and applied_count > 0 and skipped_count == 0:
        return "True"
    # Requested acceleration but either nothing applied or partial skip occurred.
    # This is a degraded/partial mode by design.
    return "Partial"


def _build_payload(
    *,
    platform_name: str,
    timestamp: str,
    mode: str,
    active_requested: bool,
    effective_active: bool,
    changes: list[dict[str, Any]],
    failures: list[str],
    previous_state: dict[str, Any],
    applied_actions: list[str],
    message: str | None = None,
) -> dict[str, Any]:
    counts = _compute_counts(changes)
    payload: dict[str, Any] = {
        "active": effective_active,
        "active_requested": active_requested,
        "effective_active": effective_active,
        "active_status": _active_status(
            active_requested=active_requested,
            effective_active=effective_active,
            applied_count=counts["applied_count"],
            skipped_count=counts["skipped_count"],
        ),
        "platform": platform_name,
        "timestamp": timestamp,
        "mode": mode,
        "changes_applied": changes,
        "failures": failures,
        "applied_actions": sorted(applied_actions),
        "previous_state": previous_state,
        **counts,
    }
    if message is not None:
        payload["message"] = message
    return payload


def _allow_risky() -> bool:
    value = os.environ.get("CONTINUUM_ACCELERATE_RISKY", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def accelerate_command(
    on: bool = typer.Option(False, "--on", help="Enable acceleration mode."),
    off: bool = typer.Option(False, "--off", help="Restore previous system settings."),
    status: bool = typer.Option(False, "--status", help="Show current acceleration state."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be changed."),
    verbose: bool = typer.Option(False, "--verbose", help="Print detailed state/command information."),
    cpu_only: bool = typer.Option(False, "--cpu-only", help="Apply CPU optimizations only."),
    gpu_only: bool = typer.Option(False, "--gpu-only", help="Apply GPU optimizations only."),
) -> None:
    try:
        if cpu_only and gpu_only:
            typer.echo("Usage error: cannot combine --cpu-only and --gpu-only", err=True)
            raise typer.Exit(code=2)

        selected = sum(1 for flag in (on, off, status) if flag)
        if selected > 1:
            typer.echo("Usage error: use only one of --on, --off, --status", err=True)
            raise typer.Exit(code=2)

        if selected == 0:
            status = True

        ctx = detect_context()

        if status:
            current = load_state(Path.cwd())
            if current is None:
                payload = _build_payload(
                    platform_name=ctx["platform"],
                    timestamp=utc_now(),
                    mode="status",
                    active_requested=False,
                    effective_active=False,
                    changes=[],
                    failures=[],
                    previous_state={},
                    applied_actions=[],
                )
                render_status(payload, verbose=verbose)
            else:
                render_status(current, verbose=verbose)
            raise typer.Exit(code=0)

        if on:
            previous_state_full = capture_previous_state(ctx, cpu_only=cpu_only, gpu_only=gpu_only)
            changes, failures, applied_actions, previous_state_applied = apply_acceleration(
                ctx,
                previous_state=previous_state_full,
                dry_run=dry_run,
                cpu_only=cpu_only,
                gpu_only=gpu_only,
                allow_risky=_allow_risky(),
            )

            effective_active = (not dry_run) and len(applied_actions) > 0
            payload = _build_payload(
                platform_name=ctx["platform"],
                timestamp=utc_now(),
                mode="dry-run" if dry_run else "on",
                active_requested=True,
                effective_active=effective_active,
                changes=changes,
                failures=failures,
                previous_state=previous_state_applied,
                applied_actions=applied_actions,
            )

            if not dry_run:
                save_state(payload, Path.cwd())

            if verbose:
                typer.echo(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), err=True)
            render_status(payload, verbose=verbose)
            raise typer.Exit(code=0)

        existing = load_state(Path.cwd())
        if existing is None:
            payload = _build_payload(
                platform_name=ctx["platform"],
                timestamp=utc_now(),
                mode="off",
                active_requested=False,
                effective_active=False,
                changes=[],
                failures=[],
                previous_state={},
                applied_actions=[],
                message="No active acceleration state found.",
            )
            render_status(payload, verbose=verbose)
            raise typer.Exit(code=0)

        previous_state = existing.get("previous_state", {})
        applied_actions = list(existing.get("applied_actions", []))
        changes, failures = restore_acceleration(
            ctx,
            previous_state=previous_state,
            applied_actions=applied_actions,
            dry_run=dry_run,
        )

        payload = _build_payload(
            platform_name=ctx["platform"],
            timestamp=utc_now(),
            mode="dry-run" if dry_run else "off",
            active_requested=False,
            effective_active=False,
            changes=changes,
            failures=failures,
            previous_state=previous_state,
            applied_actions=[],
        )

        if not dry_run:
            save_state(payload, Path.cwd())

        if verbose:
            typer.echo(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), err=True)
        render_status(payload, verbose=verbose)
        raise typer.Exit(code=0)
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"Accelerate critical failure: {type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(code=4)


__all__ = ["accelerate_command"]
