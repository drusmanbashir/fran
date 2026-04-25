import torch

from fran.inference.base import BaseInferer
from fran.inference.cascade import CascadeInferer, WholeImageInferer
from fran.inference.cascade_yolo import CascadeInfererYOLO
from fran.trainers.base import checkpoint_from_model_id


def load_ckpt_payload(run_name):
    ckpt = checkpoint_from_model_id(run_name)
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    return ckpt, payload


def infer_train_mode(run_name):
    _, payload = load_ckpt_payload(run_name)
    return payload["datamodule_hyper_parameters"]["configs"]["plan_train"]["mode"]


def resolve_inferer_cls(run_name, mode=None):
    resolved_mode = mode if mode is not None else infer_train_mode(run_name)
    if resolved_mode == "source":
        return BaseInferer, resolved_mode
    if resolved_mode == "whole":
        return WholeImageInferer, resolved_mode
    if resolved_mode == "kbd":
        return CascadeInfererYOLO, resolved_mode
    if resolved_mode in ["patch", "lbd"]:
        return CascadeInferer, resolved_mode
    raise ValueError(f"Unsupported mode='{resolved_mode}' for run_name='{run_name}'")


def build_inferer(
    run_name,
    mode=None,
    run_w=None,
    localiser_labels=None,
    devices=None,
    project_title=None,
):
    if localiser_labels is None:
        localiser_labels = [1]
    if devices is None:
        devices = [0]

    cls, resolved_mode = resolve_inferer_cls(run_name, mode=mode)
    if cls in [BaseInferer, WholeImageInferer]:
        inferer = cls(
            run_name=run_name,
            project_title=project_title,
            devices=devices,
            save=False,
            save_channels=False,
        )
        return inferer, resolved_mode

    run_w_final = run_name if run_w is None else run_w
    if cls is CascadeInferer:
        inferer = cls(
            run_w=run_w_final,
            run_p=run_name,
            localiser_labels=localiser_labels,
            project_title=project_title,
            devices=devices,
            save=False,
            save_localiser=False,
            save_channels=False,
        )
        return inferer, resolved_mode

    inferer = cls(
        run_w=run_w_final,
        run_p=run_name,
        localiser_labels=localiser_labels,
        project_title=project_title,
        devices=devices,
        save=False,
        save_localiser=False,
        save_channels=False,
    )
    return inferer, resolved_mode


if __name__ == "__main__":
    pass
