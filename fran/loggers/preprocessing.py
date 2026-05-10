from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import traceback

from fran.utils.jsonl import append_jsonl_rows


@dataclass(slots=True)
class PreprocessingLogger:
    output_folder: Path

    @property
    def jsonl_path(self):
        return self.output_folder / "log.jsonl"

    def _stamp_rows(self, rows):
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return [{"timestamp": timestamp, **row} for row in rows]

    def append_rows(self, rows):
        rows = self._stamp_rows(rows)
        append_jsonl_rows(self.jsonl_path, rows)
        return rows

    def exception(
        self,
        exc,
        *,
        error_type=None,
        case_id="",
        image="",
        lm="",
    ):
        return self.append_rows(
            [
                {
                    "case_id": case_id,
                    "status": "ERROR",
                    "image": image,
                    "lm": lm,
                    "error_type": type(exc).__name__
                    if error_type is None
                    else error_type,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            ]
        )
