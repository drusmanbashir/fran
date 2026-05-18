import pickle
from pathlib import Path

from lightning.pytorch.callbacks import Callback
from utilz.cprint import cprint


class CrashDumpCallback(Callback):
    def __init__(self, project, run_name, dump_subdir="crash_dumps"):
        self.project = project
        self.run_name = run_name
        self.dump_subdir = dump_subdir

    def on_exception(self, trainer, pl_module, exception):
        dump_root = Path(self.project.log_folder) / self.dump_subdir
        dump_root.mkdir(parents=True, exist_ok=True)
        dump_path = dump_root / f"{self.run_name}.trainer.pkl"
        with dump_path.open("wb") as f:
            pickle.dump(trainer, f)
        cprint(f"CrashDumpCallback wrote {dump_path}", color="yellow")
