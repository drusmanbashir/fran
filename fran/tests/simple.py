import ipdb

tr = ipdb.set_trace


import torch._dynamo

torch._dynamo.config.suppress_errors = True


try:
    hpc_settings_fn = os.environ["HPC_SETTINGS"]
except:
    pass

import torch

if __name__ == "__main__":
    print("Imported main modules. This works")
