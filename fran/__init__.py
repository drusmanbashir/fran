import contextlib
import importlib
import io
import os
import warnings
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings(
    "ignore", message="The cuda.cudart module is deprecated.*"
)
__path__ = [str(Path(__file__).resolve().parent)]

with contextlib.redirect_stdout(io.StringIO()):
    try:
        importlib.import_module("nnunet")
    except ModuleNotFoundError:
        pass
