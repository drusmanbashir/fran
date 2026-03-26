from fran.managers import wandb as wandb_mod


class _Run:
    def __init__(self, run_id=None, name=None):
        self.id = run_id
        self.name = name


class _Api:
    def runs(self, _path):
        return [_Run("KITS-TEAL", None), _Run("legacy-id", "KITS-STEP")]


def test_new_run_id_skips_existing_project_suffixes(monkeypatch):
    words = iter(["teal", "step", "moss"])
    monkeypatch.setattr(wandb_mod.wandb, "Api", lambda: _Api())
    monkeypatch.setattr(wandb_mod, "random_real_word", lambda *_: next(words))
    monkeypatch.delenv("WANDB_MODE", raising=False)

    assert wandb_mod._new_run_id("entity", "kits", width=5) == "KITS-MOSS"
