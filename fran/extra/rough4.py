"""
Run:  python repro_keyerror_config.py

What it does:
- Mimics your BaseInferer.__init__ access pattern:
    self.plan = fix_ast(self.params["config"]["plan_train"], ["spacing"])
- But we deliberately pass params WITHOUT "config" so it raises:
    KeyError: 'config'

Try also in IPython to see a "Cell In[...], line N" frame:
    ipython
    In [1]: %run repro_keyerror_config.py
"""

from __future__ import annotations


def fix_ast(x, keys):
    # dummy pass-through: just return what you got
    return x


class BaseInferer:
    def __init__(
        self,
        run_name: str,
        ckpt=None,
        state_dict=None,
        params: dict | None = None,
        bs: int | None = None,
        patch_overlap=None,
        mode: str | None = None,
        devices=None,
        safe_mode: bool = False,
        save_channels=None,
        save: bool = False,
        k_largest: int | None = None,
    ):
        # Emulate your code path:
        if params is None:
            # NOTE: No "config" key -> this will trigger KeyError
            self.params = {"something_else": 123}
        else:
            self.params = params

        # ↓↓↓ This line intentionally raises KeyError: 'config'
        self.plan = fix_ast(self.params["config"]["plan_train"], ["spacing"])

        # if the line above didn’t error, you would continue…
        self.check_plan_compatibility()

    def check_plan_compatibility(self):
        # stub: pretend to validate spacing etc.
        pass


def main():
    # Minimal call that will raise KeyError: 'config'
    T = BaseInferer(run_name="demo", params=None)  # params=None ⇒ missing "config"


if __name__ == "__main__":
    main()

