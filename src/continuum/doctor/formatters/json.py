from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from continuum.doctor.models import Report


def report_to_dict(report: Report) -> dict[str, Any]:
    return report.to_dict()


def write_report_json(report: Report, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"doctor_{timestamp}.json"

    payload = report_to_dict(report)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path


__all__ = ["report_to_dict", "write_report_json"]
