"""Optional exports for training-time data managers.

Kept guarded to avoid forcing heavy dependencies in light import contexts.
"""

try:
    from .dualssd import (
        DataManagerDualSSD,
        DataManagerDualSSDBTfms,
        dual_ssd_manager_class,
    )
    from .main import (
        DataManagerDual,
        DataManagerRBD,
        DataManagerLBD,
        DataManagerMulti,
        DataManagerPatch,
        DataManagerSource,
        DataManagerWhole,
    )
    from .batch_tfms import (
        DataManagerDualBTfms,
        DataManagerLBDBTfms,
        DataManagerMultiBTfms,
        DataManagerPatchBTfms,
        DataManagerRBDBTfms,
        DataManagerSourceBTfms,
        DataManagerWholeBTfms,
    )
except Exception:
    pass
