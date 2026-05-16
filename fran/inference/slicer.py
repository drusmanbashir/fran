from dataclasses import dataclass

from fran.run.inference import infer


@dataclass(frozen=True)
class SlicerInferenceSelection:
    body_part: str
    run_name: str | None = None
    localiser_type: str | None = None
    localiser_run_name: str | None = None
    devices: list[int] | str = (0,)
    patch_overlap: float = 0.2
    safe_mode: bool = True
    save: bool = False
    save_channels: bool = False


def build_inferer(selection: SlicerInferenceSelection):
    spec = infer.resolve_inference_spec(
        mnemonic_or_run=selection.body_part,
        run_name=selection.run_name,
        localiser_type=selection.localiser_type,
        localiser_run_name=selection.localiser_run_name,
    )
    inferer = infer.build_inferer(
        spec=spec,
        gpus=list(selection.devices)
        if selection.devices != "cpu"
        else selection.devices,
        safe_mode=selection.safe_mode,
        patch_overlap=selection.patch_overlap,
        save=selection.save,
        save_channels=selection.save_channels,
    )
    return inferer, spec


class SlicerInfererAdapter:
    def __init__(self, selection: SlicerInferenceSelection):
        self.selection = selection
        self.inferer, self.spec = build_inferer(selection)

    def process_data_sublist(self, images):
        self.inferer.setup()
        return self.inferer.process_data_sublist(images)
