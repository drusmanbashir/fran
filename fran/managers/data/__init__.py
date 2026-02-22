"""Optional exports for training-time data managers.

Kept guarded to avoid forcing heavy dependencies in light import contexts.
"""

try:
    from .training import (
        DataManagerBaseline,
        DataManagerLBD,
        DataManagerPatch,
        DataManagerSource,
        DataManagerWID,
        DataManagerWhole,
    )
except Exception:
    pass
