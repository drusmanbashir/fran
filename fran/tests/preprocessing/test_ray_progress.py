from pathlib import Path

import pandas as pd

from fran.run.preproc import analyze_resample as analyze_resample_module
from fran.run.preproc.analyze_resample import (
    CaseOutputCounter,
    ExactCaseOutputCounter,
    OutputFolderProgressMonitor,
    PatchCaseApproxCounter,
    PreprocessingManager,
)


class FakeProgress:
    instances = []

    def __init__(self, *args, **kwargs):
        self.total = kwargs["total"]
        self.desc = kwargs["desc"]
        self.unit = kwargs["unit"]
        self.updates = []
        self.closed = False
        self.__class__.instances.append(self)

    def update(self, n):
        self.updates.append(n)

    def close(self):
        self.closed = True


class FakeGenerator:
    def __init__(self, output_folder, total_cases, remaining_cases, overwrite, result=True):
        self.output_folder = output_folder
        self.df = pd.DataFrame([{"case_id": f"all_{i}"} for i in range(total_cases)])
        self.df_pt = pd.DataFrame(
            [{"case_id": f"remaining_{i}"} for i in range(remaining_cases)]
        )
        self.overwrite = overwrite
        self.process_calls = []
        self.result = result

    def process(self, **kwargs):
        self.process_calls.append(kwargs)
        return self.result


def test_exact_case_output_counter_counts_only_new_complete_cases(tmp_path):
    images = tmp_path / "images"
    lms = tmp_path / "lms"
    images.mkdir()
    lms.mkdir()

    (images / "case_old.pt").touch()
    (lms / "case_old.pt").touch()

    counter = ExactCaseOutputCounter(tmp_path, total=3)
    assert counter.completed_cases() == 0

    (images / "case_a.pt").touch()
    assert counter.completed_cases() == 0

    (lms / "case_a.pt").touch()
    assert counter.completed_cases() == 1

    (images / "case_b.pt").touch()
    (lms / "case_b.pt").touch()
    assert counter.completed_cases() == 2


def test_case_output_counter_counts_new_image_cases(tmp_path):
    images = tmp_path / "images"
    lms = tmp_path / "lms"
    images.mkdir()
    lms.mkdir()

    (images / "case_old.pt").touch()
    (lms / "case_old.pt").touch()

    counter = CaseOutputCounter(tmp_path, total=3)
    assert counter.completed_cases() == 0

    (images / "case_a.pt").touch()
    assert counter.completed_cases() == 1

    (images / "case_b.pt").touch()
    assert counter.completed_cases() == 2


def test_patch_case_approx_counter_uses_overall_patch_ratio(tmp_path):
    images = tmp_path / "images"
    images.mkdir()

    for name in ["kits_0001_0.pt", "kits_0001_1.pt", "kits_0002_0.pt", "kits_0002_1.pt"]:
        (images / name).touch()

    counter = PatchCaseApproxCounter(tmp_path, total=4)
    assert counter.completed_cases() == 0

    for name in ["kits_0003_0.pt", "kits_0003_1.pt", "kits_0004_0.pt", "kits_0004_1.pt"]:
        (images / name).touch()

    assert counter.completed_cases() == 2


def test_output_folder_progress_monitor_sync_uses_counter(monkeypatch):
    FakeProgress.instances.clear()
    monkeypatch.setattr(analyze_resample_module, "tqdm", FakeProgress)

    class FakeCounter:
        total = 3

        def __init__(self):
            self.values = iter([0, 1, 3])

        def completed_cases(self):
            return next(self.values)

    monitor = OutputFolderProgressMonitor(FakeCounter())
    assert monitor.sync() == 0
    assert monitor.sync() == 1
    assert monitor.sync() == 3

    progress = FakeProgress.instances[-1]
    assert progress.total == 3
    assert progress.desc == "Analyze/resample"
    assert progress.unit == "case"
    assert progress.updates == [1, 2]


def test_process_with_output_progress_stops_monitor_on_success(monkeypatch):
    events = []

    class FakeMonitor:
        def __init__(self, counter, desc):
            events.append(("init", counter.output_folder, counter.total, desc))

        def start(self):
            events.append("start")
            return self

        def stop(self):
            events.append("stop")

    generator = FakeGenerator(
        output_folder="/tmp/out",
        total_cases=5,
        remaining_cases=2,
        overwrite=False,
        result=False,
    )
    manager = PreprocessingManager.__new__(PreprocessingManager)
    monkeypatch.setattr(
        analyze_resample_module,
        "OutputFolderProgressMonitor",
        FakeMonitor,
    )

    result = manager._process_with_output_progress(
        generator,
        PatchCaseApproxCounter,
        desc="Patches",
        num_processes=1,
    )

    assert result is False
    assert generator.process_calls == [
        {
            "overwrite": False,
            "num_processes": 1,
            "src_dims": analyze_resample_module.DEFAULT_HDF5_SRC_DIMS,
            "cases_per_shard": 5,
            "max_shard_bytes": None,
            "overwrite_hdf5_shards": False,
            "hdf5_compression": "gzip",
            "hdf5_compression_opts": 1,
        }
    ]
    assert events == [("init", Path("/tmp/out"), 2, "Patches"), "start", "stop"]


def test_process_with_output_progress_uses_full_df_total_when_overwriting(monkeypatch):
    events = []

    class FakeMonitor:
        def __init__(self, counter, desc):
            events.append(("init", counter.output_folder, counter.total, desc))

        def start(self):
            events.append("start")
            return self

        def stop(self):
            events.append("stop")

    generator = FakeGenerator(
        output_folder="/tmp/out",
        total_cases=5,
        remaining_cases=2,
        overwrite=False,
    )
    manager = PreprocessingManager.__new__(PreprocessingManager)
    monkeypatch.setattr(
        analyze_resample_module,
        "OutputFolderProgressMonitor",
        FakeMonitor,
    )

    manager._process_with_output_progress(
        generator,
        CaseOutputCounter,
        desc="LBD",
        overwrite=True,
        num_processes=1,
    )

    assert generator.process_calls == [
        {
            "overwrite": True,
            "num_processes": 1,
            "src_dims": analyze_resample_module.DEFAULT_HDF5_SRC_DIMS,
            "cases_per_shard": 5,
            "max_shard_bytes": None,
            "overwrite_hdf5_shards": False,
            "hdf5_compression": "gzip",
            "hdf5_compression_opts": 1,
        }
    ]
    assert events == [("init", Path("/tmp/out"), 5, "LBD"), "start", "stop"]


def test_process_with_output_progress_uses_generator_overwrite_when_omitted(monkeypatch):
    events = []

    class FakeMonitor:
        def __init__(self, counter, desc):
            events.append(("init", counter.output_folder, counter.total, desc))

        def start(self):
            events.append("start")
            return self

        def stop(self):
            events.append("stop")

    generator = FakeGenerator(
        output_folder="/tmp/out",
        total_cases=7,
        remaining_cases=3,
        overwrite=False,
    )
    manager = PreprocessingManager.__new__(PreprocessingManager)
    monkeypatch.setattr(
        analyze_resample_module,
        "OutputFolderProgressMonitor",
        FakeMonitor,
    )

    manager._process_with_output_progress(
        generator,
        CaseOutputCounter,
        desc="Whole",
        num_processes=1,
    )

    assert generator.process_calls == [
        {
            "overwrite": False,
            "num_processes": 1,
            "src_dims": analyze_resample_module.DEFAULT_HDF5_SRC_DIMS,
            "cases_per_shard": 5,
            "max_shard_bytes": None,
            "overwrite_hdf5_shards": False,
            "hdf5_compression": "gzip",
            "hdf5_compression_opts": 1,
        }
    ]
    assert events == [("init", Path("/tmp/out"), 3, "Whole"), "start", "stop"]
