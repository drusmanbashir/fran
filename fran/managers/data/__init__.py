"""Optional exports for training-time data managers.

Kept guarded to avoid forcing heavy dependencies in light import contexts.
"""

try:
    from .training import (  # noqa: F401
        DataManagerBaseline,
        DataManagerLBD,
        DataManagerPatch,
        DataManagerSource,
        DataManagerWhole,
        DataManagerWID,
    )
except Exception:
    pass
