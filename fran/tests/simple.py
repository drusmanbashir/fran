
import torch._dynamo

torch._dynamo.config.suppress_errors = True


try:
    hpc_settings_fn = os.environ["HPC_SETTINGS"]  # noqa: F821
except Exception:
    pass

import torch  # noqa: E402

if __name__ == "__main__":
    print("Imported main modules. This works")
