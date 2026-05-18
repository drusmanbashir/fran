from concurrent.futures import Future
from types import SimpleNamespace

from PIL import Image

from fran.callback.wandb.wandb_bk import WandbImageGridCallback


def _trainer(logged):
    return SimpleNamespace(
        current_epoch=0,
        global_step=123,
        logger=SimpleNamespace(experiment=SimpleNamespace(log=logged.append)),
    )


def test_drain_completed_async_grid_render_logs_and_clears_pending(tmp_path, monkeypatch):
    callback = WandbImageGridCallback(classes=2, patch_size=(8, 8, 8), epoch_freq=1)
    callback.local_folder = tmp_path
    logged = []
    trainer = _trainer(logged)
    image_path = tmp_path / "grid.png"
    Image.new("RGB", (8, 8), color=(0, 0, 0)).save(image_path)

    future = Future()
    future._fran_grid_logged = False
    future._fran_grid_epoch = 5
    future.set_result(
        {"epoch": 5, "image_path": str(image_path), "key": "images/grid", "step": 123}
    )
    callback._pending_grid_renders = [future]

    monkeypatch.setattr(
        "fran.callback.wandb.wandb_bk.wandb.Image",
        lambda image: ("wandb-image", image.size),
    )

    callback._drain_completed_grid_renders(trainer)

    assert callback._pending_grid_renders == []
    assert callback._logged_grid_epochs == [5]
    assert len(logged) == 1
    assert list(logged[0]) == ["images/grid"]
    assert not image_path.exists()


def test_drain_keeps_running_future_pending():
    callback = WandbImageGridCallback(classes=2, patch_size=(8, 8, 8), epoch_freq=1)
    future = Future()
    future._fran_grid_logged = False
    future._fran_grid_epoch = 5
    callback._pending_grid_renders = [future]

    callback._drain_completed_grid_renders(_trainer([]))

    assert callback._pending_grid_renders == [future]
    assert callback._logged_grid_epochs == []


def test_submit_skips_when_single_pending_future_exists(monkeypatch):
    callback = WandbImageGridCallback(classes=2, patch_size=(8, 8, 8), epoch_freq=1)
    pending = Future()
    pending._fran_grid_logged = False
    pending._fran_grid_epoch = 5
    callback._pending_grid_renders = [pending]

    submit_called = False

    class FakeExecutor:
        def submit(self, fn, job):
            nonlocal submit_called
            submit_called = True
            done = Future()
            done._fran_grid_logged = False
            done._fran_grid_epoch = int(job["epoch"])
            done.set_result(
                {
                    "epoch": int(job["epoch"]),
                    "image_path": job["image_path"],
                    "key": "images/grid",
                    "step": int(job["step"]),
                }
            )
            return done

    callback._grid_render_executor = FakeExecutor()
    callback._build_async_grid_job = lambda trainer: {
        "epoch": 10,
        "image_path": "/tmp/fake.png",
        "key": "images/grid",
        "step": 999,
    }

    callback._submit_async_grid_render(_trainer([]))

    assert not submit_called
    assert callback._pending_grid_renders == [pending]
