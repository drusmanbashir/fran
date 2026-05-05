"""Optional exports for training-time data managers.

Kept guarded to avoid forcing heavy dependencies in light import contexts.
"""

try:
    from .dualssd import DataManagerDualSSD, dual_ssd_manager_class
    from .training import (
        DataManagerBaseline,
        DataManagerRBD,
        DataManagerLBD,
        DataManagerPatch,
        DataManagerSource,
        DataManagerWhole,
        DataManagerWID,
    )
except Exception:
    pass
