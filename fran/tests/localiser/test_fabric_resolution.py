from localiser.inference import base as localiser_base


def test_setup_fabric_resolves_cpu_and_gpu_device_forms(monkeypatch):
    fabric_calls = []

    class DummyFabric:
        def __init__(self, *, accelerator, devices, precision):
            fabric_calls.append(
                {
                    "accelerator": accelerator,
                    "devices": devices,
                    "precision": precision,
                }
            )
            self.device = "cpu" if accelerator == "cpu" else "cuda:0"

    monkeypatch.setattr(localiser_base, "Fabric", DummyFabric)
    inferer = localiser_base.LocaliserInfererBase.__new__(
        localiser_base.LocaliserInfererBase
    )

    inferer.devices = "cpu"
    inferer.setup_fabric()
    assert fabric_calls[-1] == {
        "accelerator": "cpu",
        "devices": 1,
        "precision": 32,
    }
    assert inferer.fabric_device == "cpu"

    inferer.devices = [1]
    inferer.setup_fabric()
    assert fabric_calls[-1] == {
        "accelerator": "gpu",
        "devices": [1],
        "precision": "bf16-mixed",
    }

    inferer.devices = 0
    inferer.setup_fabric()
    assert fabric_calls[-1] == {
        "accelerator": "gpu",
        "devices": [0],
        "precision": "bf16-mixed",
    }

    inferer.devices = None
    inferer.setup_fabric()
    assert fabric_calls[-1] == {
        "accelerator": "gpu",
        "devices": "auto",
        "precision": "bf16-mixed",
    }
