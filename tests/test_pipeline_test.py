import argparse

import fran.pipelines.test as pipeline


def test_apply_dataset_overrides():
    conf = {
        "plan_train": {"datasources": "old_ds"},
        "plan_valid": {"datasources": "old_ds"},
        "plan_test": {"datasources": "old_ds"},
        "dataset_params": {"fold": 3, "cache_rate": 0.5},
    }

    updated = pipeline.apply_dataset_overrides(conf, dataset="drli_short", fold=0, cache_rate=0.0)

    assert updated["plan_train"]["datasources"] == "drli_short"
    assert updated["plan_valid"]["datasources"] == "drli_short"
    assert updated["plan_test"]["datasources"] == "drli_short"
    assert updated["dataset_params"]["fold"] == 0
    assert updated["dataset_params"]["cache_rate"] == 0.0


def test_run_pipeline_preprocess_and_train_on_gpu1(monkeypatch):
    calls = {"resample": 0, "lbd": 0, "fit": 0}

    class FakeDB:
        def exists(self):
            return False

    class FakeProject:
        def __init__(self, title):
            self.project_title = title
            self.db = FakeDB()
            self.global_properties = {"datasources": [{"ds": "drli_short"}], "labels_all": [0, 1]}
            self.has_folds = True
            self.create_called = False
            self.create_mnemonic = None
            self.create_datasources = None

        def create(self, mnemonic, datasources):
            self.create_called = True
            self.create_mnemonic = mnemonic
            self.create_datasources = datasources

        @property
        def datasources(self):
            return ["drli_short"]

        def maybe_store_projectwide_properties(self, overwrite=False, multiprocess=False):
            _ = overwrite, multiprocess
            self.global_properties["mean_dataset_clipped"] = 0.0

    class FakeConfigMaker:
        def __init__(self, project):
            self.configs = {
                "plan_train": {"mode": "lbd", "datasources": "lits,drli,litq,litqsmall"},
                "plan_valid": {"mode": "lbd", "datasources": "lits,drli,litq,litqsmall"},
                "plan_test": {"mode": "lbd", "datasources": "lits,drli,litq,litqsmall"},
                "dataset_params": {"fold": 2, "cache_rate": 0.2},
            }

        def setup(self, _plan):
            return None

    class FakePreprocessingManager:
        def __init__(self, _args, conf=None):
            self.conf = conf

        def resample_dataset(self, overwrite=False, num_processes=1):
            _ = overwrite, num_processes
            calls["resample"] += 1

        def generate_lbd_dataset(self, overwrite=False, num_processes=1):
            _ = overwrite, num_processes
            calls["lbd"] += 1

    class FakeTrainer:
        def __init__(self, project_title, conf, run_name=None):
            _ = project_title, conf, run_name
            self.setup_kwargs = None
            self.N = type("N", (), {"compiled": True})()

        def setup(self, **kwargs):
            self.setup_kwargs = kwargs
            calls["setup_kwargs"] = kwargs

        def fit(self):
            calls["fit"] += 1

    monkeypatch.setattr(pipeline, "Project", FakeProject)
    monkeypatch.setattr(pipeline, "ConfigMaker", FakeConfigMaker)
    monkeypatch.setattr(pipeline, "PreprocessingManager", FakePreprocessingManager)
    monkeypatch.setattr(pipeline, "Trainer", FakeTrainer)
    monkeypatch.setattr(pipeline, "confirm_plan_analyzed", lambda _p, _plan: {"src_fldr_full": False, "final_fldr_full": False})
    monkeypatch.setattr(pipeline, "DS", {"drli_short": "drli_short_spec"})

    args = argparse.Namespace(
        project_title="liver_test",
        mnemonic="liver",
        dataset="drli_short",
        plan=1,
        fold=0,
        gpu_id=1,
        epochs=5,
        batch_size=2,
        cache_rate=0.0,
        num_processes=1,
        overwrite_preprocess=False,
    )
    pipeline.run_pipeline(args)

    assert calls["resample"] == 1
    assert calls["lbd"] == 1
    assert calls["fit"] == 1
    assert calls["setup_kwargs"]["devices"] == [1]
    assert calls["setup_kwargs"]["epochs"] == 5
