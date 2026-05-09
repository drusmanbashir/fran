import traceback

import torch

p = "/data/home/mpx588/datasets/cache/nodes/spc_080_080_150_ric03e8a587_ex050/test/fa5554b49ac25a5f0615962c9cff28b5.pt"
try:
    torch.load(p, map_location="cpu", weights_only=False)
    print("LOAD_OK")
except Exception as e:
    print("LOAD_FAIL:", repr(e))
    traceback.print_exc()
