# %%
from __future__ import annotations

from abc import ABC, abstractmethod

import monai
from monai.data.utils import create_file_basename
from utilz.string import headline, strip_extension

__all__ = ["FolderLayoutBase", "FolderLayout", "default_name_formatter"]

import itertools as il

import ipdb
import numpy as np
from label_analysis.totalseg import TotalSegmenterLabels
from monai.transforms.compose import Compose
from monai.transforms.io.dictionary import SaveImaged
from monai.transforms.post.dictionary import (AsDiscreted, MeanEnsembled,
                                              VoteEnsembled)
from monai.transforms.utility.dictionary import CastToTyped, ToDeviceD

from fran.data.dataset import FillBBoxPatchesd
from fran.managers import Project
from fran.transforms.imageio import LoadSITKd
from fran.transforms.inferencetransforms import (
    KeepLargestConnectedComponentWithMetad, MakeWritabled, ToCPUd)
from fran.utils.misc import parse_devices

tr = ipdb.set_trace

import gc
from pathlib import Path
from typing import List, Optional, Sequence, Union

import torch
from fastcore.basics import store_attr

# import your existing inferers
from fran.inference.base import (BaseInferer, filter_existing_files,
                                 load_images, load_params)
from fran.inference.cascade import (CascadeInferer, PatchInferer,
                                    WholeImageInferer, apply_bboxes,
                                    img_bbox_collated)

# from monai.transforms.utility.dictionary import AddChanneld, EnsureTyped


# from utilz.itk_sitk import *


class Formatter:
    def __init__(self, keys):
        self.keys = keys
        self.counter=0

    def default_name_formatter(
        self, metadict: dict, saver: monai.transforms.Transform
    ) -> dict:
        """Returns a kwargs dict for :py:meth:`FolderLayout.filename`,
        according to the input metadata and SaveImage transform."""
        subject = (
            metadict.get(
                monai.utils.ImageMetaKey.FILENAME_OR_OBJ,
                getattr(saver, "_data_index", 0),
            )
            if metadict
            else getattr(saver, "_data_index", 0)
        )
        patch_index = (
            metadict.get(monai.utils.ImageMetaKey.PATCH_INDEX, None)
            if metadict
            else None
        )
        subject = subject.replace(".nii","_{}.nii".format(self.keys[self.counter]))
        self.counter+=1
        self.counter = self.counter % len(self.keys)
        return {"subject": f"{subject}", "idx": patch_index}

class _InferenceSession:
    def __init__(self, inferer):
        self.inf = inferer

    def __enter__(self):
        self.inf.setup()
        return self.inf

    def __exit__(self, exc_type, exc, tb):
        for attr in ("model", "pred_dl", "ds"):
            if hasattr(self.inf, attr):
                try:
                    delattr(self.inf, attr)
                except Exception:
                    pass
        torch.cuda.empty_cache()
        gc.collect()


def get_mode_outchannels(run_name: str) -> str:
    params = load_params(run_name)
    plan = params["configs"]["plan_train"]
    out_channels = params["configs"]["model_params"]["out_channels"]
    return plan.get("mode", "source"), out_channels


def _localiser_for(run_w: str, devices, safe_mode, save_channels):
    """Pick Base or WholeImage localiser based on the localiser run's mode."""
    mode, _ = get_mode_outchannels(run_w)
    if mode == "whole":
        return WholeImageInferer(
            run_name=run_w,
            devices=devices,
            safe_mode=safe_mode,
            save_channels=save_channels,
        )
    # default: source
    return BaseInferer(
        run_name=run_w,
        devices=devices,
        safe_mode=safe_mode,
        save_channels=save_channels,
    )


def _decide_inferer_for_run(
    run_name: str,
    devices: Union[str, List[int]] = [0],
    safe_mode: bool = False,
    save_channels: bool = False,
    localiser_run: Optional[str] = None,
    localiser_labels: Optional[Sequence[int]] = None,
):
    """
    Pick Base, WholeImage, or Cascade based on the run's plan/mode.
    """
    params = load_params(run_name)
    plan = params["configs"]["plan_train"]
    mode = plan.get("mode", "source")

    if mode == "source":
        return BaseInferer(
            run_name=run_name,
            devices=devices,
            safe_mode=safe_mode,
            save_channels=save_channels,
        )

    if mode == "whole":
        return WholeImageInferer(
            run_name=run_name,
            devices=devices,
            safe_mode=safe_mode,
            save_channels=save_channels,
        )

    if mode in ("lbd", "patch"):
        hint = plan.get("source_plan_run") or params["configs"].get("source_plan_run")
        run_w = localiser_run or hint
        if run_w is None:
            raise ValueError(
                f"{run_name} requires a whole-image localiser run ('source_plan_run' or --localiser-run)."
            )
        if localiser_labels is None:
            localiser_labels = params["configs"].get("label_localiser", None)
            if isinstance(localiser_labels, set):
                localiser_labels = list(localiser_labels)

        return CascadeInferer(
            run_w=run_w,
            run_p=run_name,
            localiser_labels=list(localiser_labels) if localiser_labels else [],
            devices=devices,
            safe_mode=safe_mode,
            save_channels=save_channels,
            save=True,
        )

    # fallback
    return BaseInferer(
        run_name=run_name,
        devices=devices,
        safe_mode=safe_mode,
        save_channels=save_channels,
    )


class EnsembleInferer:
    """
    Mixed ensemble orchestrator.

    - Groups cascade members: runs their localiser once per chunk to get bboxes;
      then applies each patch inferer on cropped ROIs.
    - Runs base/whole-image members on full images in parallel (sequentially for memory).
    - Fuses per-member outputs (mean logits if any, else majority vote).
    """

    def __init__(
        self,
        project: Project,
        runs: Sequence[str],
        devices: Union[str, List[int]] = [0],
        safe_mode: bool = False,
        localiser_run: Optional[str] = None,
        localiser_labels: Optional[Sequence[int]] = None,
        k_largest: Optional[int] = None,
        save_channels: bool = False,
        save_casc_preds: bool = False,  # save member predictions of ensemble (only cascaded)
        save: bool = True,  # save voted out prediction
        debug: bool = False,
        debug_base: bool = False,
        debug_patch: bool = False,
    ):
        if debug==True:
            debug_base = True
            debug_patch= True
        store_attr(but="localiser_labels")
        self.localiser_labels = list(localiser_labels) if localiser_labels else None

        # Partition runs into cascade vs base-like
        self.cascade_runs: List[Tuple[str, str]] = []  # (run_w, run_p)
        self.patch_runs: List[str] = []
        self.base_runs: List[str] = []
        for r in self.runs:
            mode, out_channels = get_mode_outchannels(r)
            if mode in ("lbd", "patch"):
                # Resolve localiser run (prefer explicit; else checkpoint hint)
                params = load_params(r)
                hint = params["configs"]["plan_train"].get("source_plan_run") or params[
                    "configs"
                ].get("source_plan_run")
                run_w = self.localiser_run or hint
                if not run_w:
                    raise ValueError(
                        f"{r} requires a localiser run (set --localiser-run or add 'source_plan_run' in its config)."
                    )
                self.cascade_runs.append((run_w, r))
                self.patch_runs.append(r)
            else:
                self.base_runs.append(r)

        self.out_channels = out_channels
        # Group cascades by their localiser run to share bbox extraction
        self.cascade_groups: Dict[str, List[str]] = {}  # run_w -> [run_p...]
        for run_w, run_p in self.cascade_runs:
            self.cascade_groups.setdefault(run_w, []).append(run_p)

    def _extract_bboxes(self, run_w: str, data):
        """Run the localiser once to produce per-image bounding boxes."""
        from monai.transforms.compose import Compose

        from fran.inference.base import get_patch_spacing
        from fran.transforms.inferencetransforms import BBoxFromPTd
        from fran.transforms.misc_transforms import SelectLabels

        spacing = get_patch_spacing(run_w)
        # labels: prefer explicit; else borrow from localiser run's configs
        if self.localiser_labels is None:
            p = load_params(run_w)
            lab = p["configs"].get("label_localiser", None)
            labels = list(lab) if isinstance(lab, (list, set, tuple)) else lab
        else:
            labels = self.localiser_labels

        Sel = SelectLabels(keys=["pred"], labels=labels or [])
        B = BBoxFromPTd(keys=["pred"], spacing=spacing, expand_by=10)
        post = Compose([Sel, B])

        # Build appropriate localiser inferer (Base vs WholeImage) and run once
        bboxes = []
        W = _localiser_for(run_w, self.devices, self.safe_mode, self.save_channels)
        with _InferenceSession(W) as loc:
            loc.prepare_data(data)
            loc.create_postprocess_transforms(loc.ds.transform)
            for batch in loc.predict():
                pred = loc.postprocess(batch)
                pred = post(pred)
                bb = pred["bounding_box"]
                if bb is None or (isinstance(bb, (list, tuple)) and len(bb) == 0):
                    raise ValueError("No bounding box found by localiser.")
                bboxes.append(bb)

        # free localiser model asap
        if hasattr(W, "model"):
            del W.model
        torch.cuda.empty_cache()
        gc.collect()
        return bboxes

    def decollate_patches(self, pa, bboxes, keys):
        num_cases = len(bboxes)
        output = []
        for case_idx in range(num_cases):
            img_bbox_preds = {}
            for i, run_name in enumerate(keys):
                pred = pa[run_name][case_idx]["pred"]
                img_bbox_preds[run_name] = pred
            # img_bbox_preds.update(self.ds[case_idx])
            img_bbox_preds["bounding_box"] = bboxes[case_idx]
            output.append(img_bbox_preds)
        # num_cases = len(bboxes)
        # output = []
        # for case_idx in range(num_cases):
        #     run_preds_bbox = {}
        #     for i, run_name in enumerate(keys):
        #         preds_run = pa[run_name]
        #         pred = preds_run[case_idx]["pred"]
        #         run_preds_bbox[run_name] = pred
        #     run_preds_bbox["bounding_box"] = bboxes[case_idx]
        # img_bbox_preds.update(self.ds[case_idx])
        # output.append(run_preds_bbox)

        return output

    def tfms_from_dict(self, keys: str, tfms_dict):
        keys = keys.split(",")
        tfms = []
        for key in keys:
            tfm = tfms_dict[key]
            tfms.append(tfm)
        return tfms

    def _apply_bboxes(self, data: List[dict], bboxes):
        """Crop data in-memory the same way CascadeInferer.apply_bboxes does."""
        data2 = []
        for i, dat in enumerate(data):
            dat2 = dict(dat)
            dat2["image"] = dat2["image"][bboxes[i][1:]]
            dat2["bounding_box"] = bboxes[i]
            data2.append(dat2)
        return data2

    def _run_patch_member(self, run_p: str, cropped_data: List[dict]) -> List[dict]:
        """Run a single PatchInferer over cropped ROIs and return postprocessed batches."""
        P = PatchInferer(
            run_name=run_p,
            devices=self.devices,
            save_channels=self.save_channels,
            # save=self.save_members,
            params=load_params(run_p),
        )
        preds_all_runs = []
        with _InferenceSession(P) as pinf:
            pinf.prepare_data(data=cropped_data, collate_fn=img_bbox_collated)
            pinf.create_postprocess_transforms(pinf.ds.transform)
            for b in pinf.predict():
                preds_all_runs.append(pinf.postprocess_transforms(b))
        return preds_all_runs

    def patch_prediction(self, data, runs_p):
        preds_all_runs = {}
        print("Starting patch data prep and prediction")
        for run in runs_p:
            P = PatchInferer(
                run_name=run,
                devices=self.devices,
                save_channels=self.save_channels,
                params=load_params(run),
                debug=self.debug,
                safe_mode=self.safe_mode,
                save=self.save_casc_preds,
            )
            P.setup()
            P.prepare_data(data=data, collate_fn=img_bbox_collated)
            P.create_and_set_postprocess_transforms()
            preds_all_runs[run] = []
            preds = P.predict()
            for batch in preds:
                output = P.postprocess(batch)
                preds_all_runs[run].append(output)
        return preds_all_runs

    def _merge_member_outputs(self, per_member_batches: List[List[dict]]) -> List[dict]:
        """Mean logits if any member outputs multi-channel; else majority vote."""
        n_cases = len(per_member_batches[0])
        merged = []
        for i in range(n_cases):
            cases = [m[i] for m in per_member_batches]
            preds = [c["pred"] for c in cases]
            has_logits = any(
                p.dtype.is_floating_point and p.ndim >= 5 and p.shape[1] > 1
                for p in preds
            )
            if has_logits:
                T = torch.stack(preds, dim=0)  # [M,B,C,D,H,W]
                out = cases[0].copy()
                out["pred"] = T.mean(dim=0)
            else:
                T = torch.stack([p.long() for p in preds], dim=0)  # [M,B,1,D,H,W]
                out = cases[0].copy()
                out["pred"] = T.mode(dim=0).values.to(torch.uint8)
            merged.append(out)
        return merged

    # ---------- public API -----------------------------------------------------
    def _cascade_runs(self, data):
        preds_all_patches = []
        for run_w, run_ps in self.cascade_groups.items():
            self.create_and_set_postprocess_transforms_casc(run_ps)
            self.bboxes = self._extract_bboxes(run_w, data)
            cropped = apply_bboxes(data, self.bboxes)
            pred_patches = self.patch_prediction(cropped, run_ps)
            pat = self.decollate_patches(pred_patches, self.bboxes, run_ps)
            pat2 = self.postprocess_casc(pat)
            preds_all_patches.extend(pat2) # extend vs append can cause problems
        return preds_all_patches

    def _base_runs(self, data):
        prds_all_base = {}
        for r in self.base_runs:
            prds_all_base[r] = []
            mode ,_= get_mode_outchannels(r)
            member = (
                WholeImageInferer(
                    r,
                    devices=self.devices,
                    safe_mode=self.safe_mode,
                    save_channels=self.save_channels,
                    debug=self.debug_base,
                )
                if mode == "whole"
                else BaseInferer(
                    r,
                    devices=self.devices,
                    safe_mode=self.safe_mode,
                    save_channels=self.save_channels,
                    debug=self.debug_base,
                )
            )
            with _InferenceSession(member) as inf:
                inf.setup()
                inf.prepare_data(data, collate_fn=img_bbox_collated if "bounding_box" in data[0].keys() else None)
                inf.create_and_set_postprocess_transforms()
                for num_batches, batch in enumerate(inf.predict(), 1):
                    batch = inf.postprocess(batch)
                    prds_all_base[r].append(batch)

        preds_all_base = self.decollate_base_predictions(prds_all_base)
        return preds_all_base

    def create_and_set_postprocess_transforms_casc(self, runs_keys):
        self.create_postprocess_transforms_casc(runs_keys=runs_keys)
        self.set_postprocess_tfms_keys_casc()
        self.set_postprocess_transforms_casc()

    def create_postprocess_transforms_casc(self, runs_keys):
        F = Formatter(runs_keys)
        self.postprocess_transforms_dict_casc = {
            # "U": ToDeviceD(keys=keys, device=self.device),
            "Int": CastToTyped(keys=runs_keys, dtype=np.uint8),
            "W": MakeWritabled(keys=runs_keys),
            "F": FillBBoxPatchesd(keys=runs_keys),
            "S": SaveImaged(
                keys=runs_keys,
                output_dir=self.output_folder,
                output_postfix="",
                separate_folder=False,
                output_dtype=np.uint8,
                output_name_formatter=F.default_name_formatter,
            ),
        }

    def set_postprocess_tfms_keys_casc(self):
        if self.safe_mode== False:
            self.postprocess_tfms_keys_casc = "A,Int,W,F"
        else:
            self.postprocess_tfms_keys_casc = "W,F"
        if self.save_casc_preds == True:
            self.postprocess_tfms_keys_casc += ",S"

    def set_postprocess_transforms_casc(self):
        self.postprocess_transforms_casc = self.tfms_from_dict(
            self.postprocess_tfms_keys_casc, self.postprocess_transforms_dict_casc
        )
        self.postprocess_compose_casc=Compose(self.postprocess_transforms_casc)

    def postprocess_casc(self, preds):
        if self.debug == False:
            output = self.postprocess_compose_casc(preds)
        else:
            output = self.postprocess_iterate_casc(preds[0])
        return output

    def postprocess_iterate_casc(self, batch):
        for tfm in self.postprocess_transforms_casc:
            headline(tfm)
            tr()
            batch = tfm(batch)
        return batch

    def run(
        self, images: List[Union[str, Path]], chunksize: int = 4, overwrite=False
    ) -> List[dict]:
        if not isinstance(images, list):
            images = [images]
        if overwrite == False and (
            isinstance(images[0], str) or isinstance(images[0], Path)
        ):
            images = filter_existing_files(images, self.output_folder)
        chunksize = max(1, chunksize)
        all_outputs: List[dict] = []
        self.create_and_set_postprocess_transforms()
        for start in range(0, len(images), chunksize):

            chunk = images[start : start + chunksize]
            data = load_images(chunk)
            # 1) Prepare data once for base-like members (they each handle their own transforms)
            # 2) Handle cascade groups: run localiser ONCE per run_w, then fan out to each run_p
            preds_all_patches = (
                self._cascade_runs(data) if len(self.cascade_runs) > 0 else []
            )
            data = load_images(chunk)
            preds_all_base = self._base_runs(data) if len(self.base_runs) > 0 else []
            # 3) Run base/whole members directly on full images
            preds_all = self.combined_patch_base_preds(
                preds_all_patches, preds_all_base
            )

            preds_final = self.postprocess(preds_all)
            all_outputs.append(preds_final)

            torch.cuda.empty_cache()
            gc.collect()

        return all_outputs

    def combined_patch_base_preds(self, preds_all_patches, preds_all_base):
        preds_all = []
        if len(preds_all_patches) > 0 and len(preds_all_base) > 0:
            for pred_patch, pred_base in zip(preds_all_patches, preds_all_base):
                preds_ = pred_patch | pred_base
                preds_all.append(preds_)
        elif len(preds_all_patches) > 0:
            preds_all = preds_all_patches
        elif len(preds_all_base) > 0:
            preds_all = preds_all_base
        return preds_all

    def decollate_base_predictions(self, preds):
        preds_out = []
        num_batches = len(preds[self.base_runs[0]])
        for i in range(num_batches):
            prds_decolled = {}
            for r in self.base_runs:
                prds = preds[r][i]["pred"]
                prds_decolled[r] = prds
            preds_out.append(prds_decolled)
        return preds_out

    def create_postprocess_transforms(self):
        dev0 = self.devices if isinstance(self.devices, int) else self.devices[0]
        device = parse_devices(dev0)

        # Pre/post utils
        W = MakeWritabled(keys=self.runs)

        # Move member outputs only (not "pred" yet)
        GPU_members = ToDeviceD(keys=self.runs, device=device)
        CPU_members = ToCPUd(keys=self.runs)

        # Fuse members -> "pred"
        MR = VoteEnsembled(keys=self.runs, output_key="pred", num_classes=self.out_channels)

        # Everything from here acts on "pred"
        CPU_pred = ToCPUd(keys=["pred"])                      # ensure CPU before saving
        Int      = CastToTyped(keys=["pred"], dtype=np.uint8) # force uint8 labels
        K        = KeepLargestConnectedComponentWithMetad(
                      keys=["pred"], independent=False, num_components=self.k_largest
                  ) if self.k_largest else None
        S        = SaveImaged(
                      keys=["pred"],
                      output_dir=self.output_folder,
                      output_postfix="",
                      separate_folder=False,
                      output_dtype=np.uint8,
                  ) if self.save else None

        # Stash for key-driven assembly
        self.postprocess_transforms_dict = {
            "W": W,
            "GPU_members": GPU_members,
            "MR": MR,
            "CPU_members": CPU_members,
            "CPU_pred": CPU_pred,
            "Int": Int,
            "K": K,
            "S": S,
        }
    def set_postprocess_tfms_keys(self):
        # Safe mode: keep everything on CPU; otherwise hop members to GPU for voting
        if self.safe_mode is False:
            keys = "W,GPU_members,MR,CPU_members,CPU_pred,Int"
        else:
            keys = "W,MR,Int"

        if self.k_largest:
            keys += ",K"
        if self.save:
            keys += ",S"

        self.postprocess_tfms_keys = keys


    def set_postprocess_transforms(self):
        def _tfms_from_dict(keys: str, tfms_dict: dict):
            out = []
            for k in keys.split(","):
                t = tfms_dict[k]
                if t is not None:
                    out.append(t)
            return out

        self.postprocess_transforms = _tfms_from_dict(
            self.postprocess_tfms_keys, self.postprocess_transforms_dict
        )
        self.postprocess_compose = Compose(self.postprocess_transforms)

    def postprocess_iterate(self, batch):
        if isinstance(batch, list):
            batch = batch[0]
        bbox = batch.get("bounding_box")
        if bbox and isinstance(bbox[0], list):
            bbox = bbox[0]
        batch["bounding_box"] = bbox
        for tfm in self.postprocess_transforms:
            headline(tfm)
            tr()
            batch = tfm(batch)
        return batch

    def postprocess(self, preds):
        if self.debug == False:
            output = self.postprocess_compose(preds)
        else:
            output = self.postprocess_iterate(preds)
        return output
    #
    # def postprocess(self, preds):
    #     device = self.devices if isinstance(self.devices, int) else self.devices[0]
    #     device = parse_devices(device)
    #     CPU = ToCPUd(keys=self.runs)
    #     GPU = ToDeviceD(keys=self.runs, device=device)
    #     W = MakeWritabled(keys=self.runs)
    #
    #     MR = VoteEnsembled(
    #         output_key="pred", keys=self.runs, num_classes=self.out_channels
    #     )
    #
    #     S = SaveImaged(
    #         keys=["pred"],
    #         output_dir=self.output_folder,
    #         output_postfix="",
    #         separate_folder=False,
    #         output_dtype=np.uint8,
    #     )
    #     K = KeepLargestConnectedComponentWithMetad(
    #         keys=["pred"], independent=False, num_components=self.k_largest
    #     )  # label=1 is the organ
    #
    #     tfms = [W, GPU, MR, CPU]
    #     if self.k_largest:
    #         tfms.append(K)
    #     if self.save == True:
    #         tfms.append(S)
    #
    #     self.pp_transforms = {
    #         "MakeWritable": W,
    #         "ToDevice": GPU,
    #         "VoteEnsemble": MR,
    #         "ToCPU": CPU,
    #     }
    #     if self.k_largest:
    #         self.pp_transforms["KeepLargest"] = K
    #     if self.save == True:
    #         self.pp_transforms["SaveImage"] = S
    #
    #     if self.debug == False:
    #         output = Compose(tfms)(preds)
    #     else:
    #         batch = preds[0]
    #         for key, tfm in self.pp_transforms.items():
    #             headline(key)
    #             batch= tfm(batch)
    #     return output
    #
    def create_and_set_postprocess_transforms(self ):
        self.create_postprocess_transforms()
        self.set_postprocess_tfms_keys()
        self.set_postprocess_transforms()

    @property
    def output_folder(self):
        fldr_name = "_".join(self.runs)
        try:
            fldr = self.project.predictions_folder / fldr_name
        except Exception as e:
            print(
                "Project has not been estimated yet. You have to run the inferer first."
            )
            print(e)
        return fldr

    # dici =    preds[1]
    #
    # dici=MR(dici)
    # # dici['pred'].unique()
    #
    # dici=S(dici)


# %%
if __name__ == "__main__":
# SECTION:-------------------- SETUP-------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR> <CR> <CR> <CR> <CR> <CR> <CR> <CR> <CR>
    #
    run_w = "LITS-1439"  # this run has localiser_labels not full TSL.

    run_lidc2 = ["LITS-902"]
    run_nodes = ["LITS-1290", "LITS-1230", "LITS-1288"]
    run_nodes1 = ["LITS-1326","LITS-1327", "LITS-1328"]
    run_nodes2 = ["LITS-1405","LITS-1416", "LITS-1417"]

    run_lidc2 = ["LITS-842"]
    run_lidc2 = ["LITS-913"]
    run_lidc2 = ["LITS-911"]
    run_litsmc = ["LITS-933"]
    run_litsmc2 = ["LITS-1018"]
    run_litsmc2 = ["LITS-1217", "LITS-1297"]
    run_ts = ["LITS-827"]
    run_totalseg = ["LITS-1246"]

    litq_test_fldr = "/s/xnat_shadow/litq/test/images_ub/"
    fns = "/s/datasets_bkp/drli_short/images/"
    img_fldr = Path("/s/xnat_shadow/lidc2/images/")
    img_fn2 = "/s/xnat_shadow/crc/wxh/images/crc_CRC198_20170718_CAP1p51.nii.gz"
    img_fn3 = "/s/xnat_shadow/crc/srn/images/crc_CRC002_20190415_CAP1p5.nii.gz"

    # fldr_crc = Path("/s/xnat_shadow/crc/images_train_rad/images/")
    fldr_crc = Path("/s/xnat_shadow/crc/images")
    # srn_fldr = "/s/xnat_shadow/crc/srn/cases_with_findings/images/"
    litq_fldr = "/s/xnat_shadow/litq/test/images_ub/"
    litq_imgs = list(Path(litq_fldr).glob("*"))
    t6_fldr = Path("/s/datasets_bkp/Task06Lung/images")
    imgs_t6 = list(t6_fldr.glob("*"))
    react_fldr = Path("/s/insync/react/sitk/images")
    imgs_react = list(react_fldr.glob("*"))
    imgs_crc = list(fldr_crc.glob("*"))
    nodesthick_fldr = Path("/s/xnat_shadow/nodesthick/images")
    nodes_fldr = Path("/s/xnat_shadow/nodes/images_pending/thin_slice/images")
    nodes_fldr_test =Path("/s/xnat_shadow/nodes/images_test")
    nodes_test = list(nodes_fldr_test.glob("*"))
    nodes = list(nodes_fldr.glob("*"))
    capestart_fldr = Path("/s/insync/datasets/capestart/nodes_2025/images")
    capestart = list(capestart_fldr.glob("*"))

    img_fns = [imgs_t6][:20]
    localiser_labels = [45, 46, 47, 48, 49]
    localiser_labels_litsmc = [1]
    TSL = TotalSegmenterLabels()
    proj_nodes = Project("nodes")

# %%
# SECTION:-------------------- NODES -------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR> <CR> <CR> <CR> <CR> <CR> <CR> <CR> <CR>
    localiser_labels = set(TSL.label_localiser)
    runs = run_nodes2
    safe_mode = True
    devices = [1]
    overwrite = True
    overwrite = False
    save_channels = False
    save_localiser = True
    save_casc_preds = True
    chunksize = 2
    localiser_run = run_w
    debug_ = True
    debug_ = False
    debug_base=True
    debug_base=False
    # En = CascadeInferer(

    #     run_w,
    #     run_nodes,
    #     save_channels=save_channels,
    #     devices=devices,
    #     overwrite=overwrite,
    #     localiser_labels=localiser_labels,
    #     save_localiser=save_localiser,
    #     safe_mode=safe_mode,
    #     k_largest=None,
    # )

# %%
    E = EnsembleInferer(
        project=proj_nodes,
        runs=runs,
        devices=devices,
        localiser_run=localiser_run,
        localiser_labels=localiser_labels,
        safe_mode=safe_mode,
        save_channels=save_channels,
        save_casc_preds=save_casc_preds,
        debug=debug_,
        debug_base=debug_base,
    )

# %%
    # nodes = nodes[:3]
    imgs = nodes_test
    imgs = nodes
    # node_fn = "/s/insync/datasets/capestart/nodes_nov2025/images/nodes_43_20220805_CAP1p5SoftTissue.nii.gz"
    preds = E.run(imgs, chunksize=chunksize, overwrite=overwrite)
# %%
    # preds = En.run(img_fns, chunksize=2)

    pap = preds[0]
    pap[0].keys()
# %%
    # batch['image'].shape

    # S
    params["configs"].keys()
    params["configs"]["dataset_params"]["plan_train"]
    # S
# %%
# SECTION:-------------------- LITSMC-------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR> <CR> <CR>

    run = run_litsmc2
    debug_ = False
    localiser_labels_litsmc = [3]
    devices = [1]
    overwrite = True
    safe_mode = True
    save_localiser = True
    save_channels = False
    project = Project(project_title="litsmc")
    if project.project_title == "litsmc":
        k_largest = 1
    else:
        k_largest = None
    En = EnsembleInferer(
        project=project,
        localiser_run=run_w,
        runs=run,
        save_channels=save_channels,
        devices=devices,
        localiser_labels=localiser_labels_litsmc,
        safe_mode=safe_mode,
        k_largest=k_largest,
        debug=debug_,
    )

# %%
    fns_litq = list(Path(litq_fldr).glob("*"))

    preds = En.run(fns_litq[:2], chunksize=chunksize, overwrite=overwrite)
    # preds = En.run(imgs_crc[:30], chunksize=4)
# %%

# %%
# SECTION:-------------------- TS run()-------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR> <CR>

    images = nodes

    all_outputs: List[dict] = []

    start = 0
    chunk = images[start : start + chunksize]
    data = load_images(chunk)
    # 1) Prepare data once for base-like members (they each handle their own transforms)
    # 2) Handle cascade groups: run localiser ONCE per run_w, then fan out to each run_p
    preds_all_patches = En._cascade_runs(data) if len(En.cascade_runs) > 0 else []

    preds_all_patches[0].keys()
    preds_all_patches[0]["LITS-1217"].dtype
    preds_all_patches[0]["LITS-1217"].max()
# %%
    preds_all_base = En._base_runs(data) if len(En.base_runs) > 0 else []
    # 3) Run base/whole members directly on full images
    preds_all = En.combined_patch_base_preds(preds_all_patches, preds_all_base)

    preds_final = En.postprocess(preds_all)
    all_outputs.append(preds_final)

    torch.cuda.empty_cache()
    gc.collect()

# %%

# SECTION:-------------------- TS-------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR> <CR> <CR> <CR> <CR> <CR> <CR>
    images = nodes[:2]
    if not isinstance(images, list):
        images = [images]
    chunksize = max(1, chunksize)

    all_outputs: List[dict] = []

    # for start in range(0, len(images), chunksize):

    start = 0
    chunk = images[start : start + chunksize]

    # 1) Prepare data once for base-like members (they each handle their own transforms)
    per_member_batches: List[List[dict]] = []

# %%
# %%
#SECTION:-------------------- patch--------------------------------------------------------------------------------------

    E.debug=False
    run_w = 'LITS-1088'
    run_ps = E.cascade_groups[run_w]
    preds_all_patches = []
# %%

    data = load_images(images[1])
    E.create_and_set_postprocess_transforms_casc(run_ps)
    E.bboxes = E._extract_bboxes(run_w, data)
    cropped = apply_bboxes(data, E.bboxes)
    pred_patches = E.patch_prediction(cropped, run_ps)
    pp = E.decollate_patches(pred_patches, E.bboxes, run_ps)
    pp = E.postprocess_casc(pp)
    preds_all_patches.extend(pp)
    # 2) Handle cascade groups: run localiser ONCE per run_w, then fan out to each run_p
# %%
    meta_org = data[0]['image'].meta.copy()
    meta2 = data

    meta2 = data[0]['image'].meta.copy()
# %%
#SECTION:--------------------  BASE RUN--------------------------------------------------------------------------------------

        chunk = nodes[1]
        data = load_images(chunk)
        prds_all_base = {}
        for r in E.base_runs:
            r = E.base_runs[0]
            prds_all_base[r] = []
            mode ,channels = get_mode_outchannels(r)
            member = (
                WholeImageInferer(
                    r,
                    devices=E.devices,
                    safe_mode=E.safe_mode,
                    save_channels=E.save_channels,
                )
                if mode == "whole"
                else BaseInferer(
                    r,
                    devices=E.devices,
                    safe_mode=E.safe_mode,
                    save_channels=E.save_channels,
                )
            )
# %%
        inf = member
        inf.debug=True
        inf.setup()
        inf.prepare_data(data, collate_fn=None)
        inf.create_and_set_postprocess_transforms()
        batch = next(inf.predict())
# %%
        # for num_batches, batch in enumerate(inf.predict(), 1):
        batch = inf.postprocess(batch)
        prds_all_base[r].append(batch)

        preds_all_base = E.decollate_base_predictions(prds_all_base)
# %%
# SECTION:-------------------- patch pred-------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR> <CR> <CR> <CR> <CR> <CR>
    # 3) Run base/whole members directly on full images
    prds_all_base = {}
    for r in E.base_runs:
        prds_all_base[r] = []
        mode = get_mode_outchannels(r)
        member = (
            WholeImageInferer(
                r,
                devices=E.devices,
                safe_mode=E.safe_mode,
                save_channels=E.save_channels,
            )
            if mode == "whole"
            else BaseInferer(
                r,
                devices=E.devices,
                safe_mode=E.safe_mode,
                save_channels=E.save_channels,
            )
        )
        with _InferenceSession(member) as inf:
            # data = inf.load_images(chunk)
            inf.prepare_data(data, tfms_keys=inf.preprocess_tfms_keys)
            inf.create_postprocess_transforms(inf.ds.transform)
            for num_batches, batch in enumerate(inf.predict(), 1):
                # for batch in inf.predict():
                batch = inf.postprocess(batch)
                if E.save:
                    inf.save_pred(batch)
                # batches.append(b)
                prds_all_base[r].append(batch)

# %%
    preds_all_base = []
    for i in range(num_batches):
        prds_decolled = {}
        for r in E.base_runs:
            prds = prds_all_base[r][i]["pred"]
            prds_decolled[r] = prds
        preds_all_base.append(prds_decolled)

# %%
    preds_all = []
    for i in range(num_batches):
        preds_ = preds_all_base[i] | preds_all_patches[i]
        preds_all.append(preds_)
# %%

    U = ToCPUd(keys=E.runs)
    MR = MeanEnsembled(output_key="pred", keys=E.runs)
    C = Compose([U, MR])
    ind = 0
    batch = preds_all[ind]
    out1 = U(preds_all[0])
    out2 = MR(out1)

    output = C(preds_all)
    # [[x['pred'].shape for x in b] for b in per_member_batches  ]
# %%
    outs = En.postprocess(preds_all)
# %%

# %%
    # 4) Fuse
    if per_member_batches:
        merged = E._merge_member_outputs(per_member_batches)
        all_outputs.extend(merged)

    torch.cuda.empty_cache()
    gc.collect()

# %%
    run_p = En.patch_runs[0]
    P = PatchInferer(
        run_name=run_p,
        devices=En.devices,
        save_channels=En.save_channels,
        # save=En.save_members,
        params=load_params(run_p),
    )

# %%
    batch.keys()
    batch["image"].shape
    batch["pred"].shape

    batch.keys()
    batch["bounding_box"]
    batch["LITS-1290"].shape
    batch["LITS-1290"].dtype
    batch["LITS-1290"]

# %%
    fn = strip_extension(subject)
# %%



    chunk = nodes[1]
    data = load_images(chunk)
    # 1) Prepare data once for base-like members (they each handle their own transforms)
    # 2) Handle cascade groups: run localiser ONCE per run_w, then fan out to each run_p
    preds_all_patches = (
        E._cascade_runs(data) if len(E.cascade_runs) > 0 else []
    )
    preds_all_base = E._base_runs(data) if len(E.base_runs) > 0 else []
    # 3) Run base/whole members directly on full images
    preds_all = E.combined_patch_base_preds(
        preds_all_patches, preds_all_base
    )

    preds_final = E.postprocess(preds_all)
# %%


    preds_all_patches[0]['LITS-1290']
    preds_all_patches[0]['LITS-1288'].max()
# %%
    preds_all_base[0]['LITS-1230']

# %%

    data = load_images(chunk)
# %%
# %%
    # pred_patches['LITS-1290'][0]['pred'].shape
# %%
    batch.keys()
    batch['image'].meta
    batch['pred'].meta
    batch['LITS-1290'].shape
    batch['LITS-1288'].shape
    batch['LITS-1230'].shape
    batch['LITS-1230'].dtype
    batch['LITS-1230'].max()
    batch['LITS-1290'].max()
    batch['LITS-1288'].max()
    batch['pred'].shape
    batch['pred'].dtype
    batch['pred'].max()
    batch['pred'].device
# .max()%%
    preds_all_base[0]['LITS-1230'].max()
    preds_all_base[0]['LITS-1230'].dtype

    preds_all_base[0]['LITS-1230']
    torch.save(preds_all_base[0]["LITS-1230"],"pred_base.pt")
# %%
    batch['image'].meta
    batch['pred'].meta
    batch['pred'].shape
    batch.keys()

# %%
    data['pred'].shape
    data['pred'].shape
    pred_patches.keys()
    pp[0].keys()
# %%
