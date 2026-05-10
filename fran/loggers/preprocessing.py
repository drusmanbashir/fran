from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import traceback

import pandas as pd

from fran.utils.jsonl import append_jsonl_rows


@dataclass(slots=True)
class PreprocessingLogger:
    output_folder: Path
    columns: list[str]

    @property
    def jsonl_path(self):
        return self.output_folder / "log.jsonl"

    @property
    def csv_path(self):
        return self.output_folder / "preprocessing_log.csv"

    def _stamp_rows(self, rows):
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return [{"timestamp": timestamp, **row} for row in rows]

    def append_rows(self, rows):
        rows = self._stamp_rows(rows)
        append_jsonl_rows(self.jsonl_path, rows)
        df = pd.DataFrame(rows).reindex(columns=self.columns)
        write_header = not self.csv_path.exists()
        df.to_csv(self.csv_path, mode="a", header=write_header, index=False)
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
