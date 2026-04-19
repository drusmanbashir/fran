from pathlib import Path

from fran.managers.project import COMMON_PATHS


def profiler_folder(project_title):
    folder = Path(COMMON_PATHS["cold_storage_folder"]) / "logs" / "profiler" / project_title
    folder.mkdir(parents=True, exist_ok=True)
    return folder
