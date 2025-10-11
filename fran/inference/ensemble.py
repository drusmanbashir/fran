# %%
from __future__ import annotations

import itertools as il

import ipdb
from label_analysis.totalseg import TotalSegmenterLabels
from monai.transforms.compose import Compose
from monai.transforms.io.dictionary import SaveImaged
from monai.transforms.post.dictionary import MeanEnsembled, VoteEnsembled
from monai.transforms.utility.dictionary import ToDeviceD

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
                                    WholeImageInferer, img_bbox_collated)

# from monai.transforms.utility.dictionary import AddChanneld, EnsureTyped


# from utilz.itk_sitk import *


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
    out_channels = params['configs']['model_params']["out_channels"]
    return plan.get("mode", "source"), out_channels


def _localiser_for(run_w: str, devices, safe_mode, save_channels):
    """Pick Base or WholeImage localiser based on the localiser run's mode."""
    mode,_ = get_mode_outchannels(run_w)
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
        save_members: bool = True,  # save member predictions of ensemble
        save: bool = True,  # save voted out prediction
        debug: bool = False,
    ):
        store_attr(but="localiser_labels")
        self.localiser_labels = list(localiser_labels) if localiser_labels else None

        # Partition runs into cascade vs base-like
        self.cascade_runs: List[Tuple[str, str]] = []  # (run_w, run_p)
        self.patch_runs: List[str] = []
        self.base_runs: List[str] = []

        for r in self.runs:
            mode,out_channels = get_mode_outchannels(r)
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

        self.out_channels=out_channels
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
            pinf.prepare_data(
                data=cropped_data, tfms_keys="ESN", collate_fn=self._img_bbox_collated
            )
            pinf.create_postprocess_transforms(pinf.ds.transform)
            for b in pinf.predict():
                preds_all_runs.append(pinf.postprocess_transforms(b))
        return preds_all_runs

    def patch_prediction(self, data, runs_p):
        F = FillBBoxPatchesd()
        preds_all_runs = {}
        print("Starting patch data prep and prediction")
        for run in runs_p:
            P = PatchInferer(
                run_name=run,
                devices=self.devices,
                save_channels=self.save_channels,
                params=load_params(run),
                debug=self.debug,
            )
            P.setup()
            P.prepare_data(data=data, collate_fn=img_bbox_collated)
            P.create_postprocess_transforms(P.ds.transform)
            preds_all_runs[run] = []
            preds = P.predict()
            for batch in preds:
                output = P.postprocess(batch)
                output = F(output)
                if self.save_members == True:
                    S = SaveImaged(
                        keys=["pred"],
                        output_dir=P.output_folder,
                        output_postfix="",
                        separate_folder=False,
                    )
                    S(output)
                preds_all_runs[run].append(output)

        return preds_all_runs

    @staticmethod
    def _img_bbox_collated(batch):
        """Keep parity with cascade.img_bbox_collated."""
        imgs, bboxes = [], []
        for item in batch:
            imgs.append(item["image"])
            bboxes.append(item["bounding_box"])
        return {"image": torch.stack(imgs, 0), "bounding_box": bboxes}

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
            # load raw images once (BaseInferer handles type resolution)
            # tmp_loader = BaseInferer(run_name=run_ps[0], devices=self.devices, safe_mode=self.safe_mode, save_channels=self.save_channels)
            # 2a) localiser → bbox (once)
            self.bboxes = self._extract_bboxes(run_w, data)
            # 2b) apply crop, then run each patch member on the same cropped ROIs
            cropped = self._apply_bboxes(data, self.bboxes)
            pred_patches = self.patch_prediction(cropped, run_ps)
            pp = self.decollate_patches(pred_patches, self.bboxes, run_ps)
            preds_all_patches.extend(pp)

        return preds_all_patches

    def _base_runs(self, data):
        prds_all_base = {}
        for r in self.base_runs:
            prds_all_base[r] = []
            mode = get_mode_outchannels(r)
            member = (
                WholeImageInferer(
                    r,
                    devices=self.devices,
                    safe_mode=self.safe_mode,
                    save_channels=self.save_channels,
                )
                if mode == "whole"
                else BaseInferer(
                    r,
                    devices=self.devices,
                    safe_mode=self.safe_mode,
                    save_channels=self.save_channels,
                )
            )
            with _InferenceSession(member) as inf:
                inf.prepare_data(data, tfms_keys=inf.preprocess_tfms_keys)
                inf.create_postprocess_transforms(inf.ds.transform)
                for num_batches, batch in enumerate(inf.predict(), 1):
                    # for batch in inf.predict():
                    batch = inf.postprocess(batch)
                    # if self.save:
                    #     inf.save_pred(batch)
                    # batches.append(b)
                    prds_all_base[r].append(batch)

        preds_all_base = self.decollate_base_predictions(prds_all_base)
        return preds_all_base

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

        for start in range(0, len(images), chunksize):

            chunk = images[start : start + chunksize]
            data = load_images(chunk)
            # 1) Prepare data once for base-like members (they each handle their own transforms)
            # 2) Handle cascade groups: run localiser ONCE per run_w, then fan out to each run_p
            preds_all_patches = self._cascade_runs(data) if len(self.cascade_runs) > 0 else []
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

    def postprocess(self, preds):
        device = self.devices if isinstance(self.devices, int) else self.devices[0]
        device = parse_devices(device)
        CPU = ToCPUd(keys=self.runs)
        GPU = ToDeviceD(keys=self.runs, device=device)
        W = MakeWritabled(keys=self.runs)

        MR = VoteEnsembled(
            output_key="pred", keys=self.runs, num_classes=self.out_channels
        )  

        S = SaveImaged(
            keys=["pred"],
            output_dir=self.output_folder,
            output_postfix="",
            separate_folder=False,
        )
        K = KeepLargestConnectedComponentWithMetad(
            keys=["pred"], independent=False, num_components=self.k_largest
        )  # label=1 is the organ
        
        tfms = [W, GPU, MR, CPU]
        if self.k_largest:
            tfms.append(K)
        if self.save == True:
            tfms.append(S)
            
        self.pp_transforms = {
            "MakeWritable": W,
            "ToDevice": GPU, 
            "VoteEnsemble": MR,
            "ToCPU": CPU
        }
        if self.k_largest:
            self.pp_transforms["KeepLargest"] = K
        if self.save == True:
            self.pp_transforms["SaveImage"] = S

        if self.debug == False:
            output = Compose(tfms)(preds)
        else:
            output = self.cascade_postprocess_iterate(preds[0])
        return output
    
    def cascade_postprocess_iterate(self, batch):
        from utilz.string import headline
        
        for name, tfm in self.pp_transforms.items():
            headline(f"{name}: {tfm}")
            tr()
            batch = tfm(batch)
        return batch

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
# SECTION:-------------------- SETUP-------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR> <CR> <CR> <CR>
    #
    # p = argparse.ArgumentParser()
    # p.add_argument("--runs", type=str, required=True, help="Comma-separated run_names")
    # p.add_argument("--images", type=str, nargs="+", required=True)
    # p.add_argument("--devices", type=str, default="0")
    # p.add_argument("--localiser-run", type=str, default=None)
    # p.add_argument("--localiser-labels", type=int, nargs="*", default=None)
    # p.add_argument("--chunksize", type=int, default=8)
    # p.add_argument("--safe-mode", action="store_true")
    # p.add_argument("--save-channels", action="store_true")
    # args = p.parse_args()
    #
    # runs = [r.strip() for r in args.runs.split(",") if r.strip()]
    # devices = [int(d) for d in args.devices.split(",")]
    # ens = EnsembleInferer(
    #     runs=runs,
    #     devices=devices,
    #     safe_mode=args.safe_mode,
    #     save_channels=args.save_channels,
    #     chunksize=args.chunksize,
    #     localiser_run=args.localiser_run,
    #     localiser_labels=args.localiser_labels,
    # )
    # ens.run(args.images)
    # ... run your application ...
    pass

    run_w2 = "LIT-145"
    run_w = "LITS-1088"  # this run has localiser_labels not full TSL.

    run_lidc2 = ["LITS-902"]
    run_nodes = ["LITS-1290", "LITS-1230", "LITS-1288"]
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
    nodes = list(nodes_fldr.glob("*"))
    capestart_fldr = Path("/s/insync/datasets/capestart/nodes_2025/images")
    capestart = list(capestart_fldr.glob("*"))

    img_fns = [imgs_t6][:20]
    localiser_labels = [45, 46, 47, 48, 49]
    localiser_labels_litsmc = [1]
    TSL = TotalSegmenterLabels()
    proj_nodes = Project("nodes")

# SECTION:-------------------- NODES -------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR> <CR> <CR> <CR>
    localiser_labels = set(TSL.label_localiser)
    runs = run_nodes
    safe_mode = False
    devices = [0]
    overwrite = False
    overwrite = True
    save_channels = False
    save_localiser = True
    save_members = True
    chunksize = 2
    localiser_run = run_w
    debug_= True
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
        save_members=save_members,
        debug=debug_
    )

# %%
    preds = E.run(nodes[:3], chunksize=chunksize, overwrite=overwrite)
    # preds = En.run(img_fns, chunksize=2)

# %%
    # batch['image'].shape

    # S
    params["configs"].keys()
    params["configs"]["dataset_params"]["plan_train"]
    # S
# %%
# SECTION:-------------------- LITSMC-------------------------------------------------------------------------------------- <CR>

    run = run_litsmc2
    debug_=True
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
        debug=debug_
    )

# %%
    fns_litq = list(Path(litq_fldr).glob("*"))

    preds = En.run(fns_litq, chunksize=chunksize, overwrite=overwrite)
    # preds = En.run(imgs_crc[:30], chunksize=4)
# %%

# %%
#SECTION:-------------------- TS run()--------------------------------------------------------------------------------------

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
    preds_all = En.combined_patch_base_preds(
        preds_all_patches, preds_all_base
    )

    preds_final = En.postprocess(preds_all)
    all_outputs.append(preds_final)

    torch.cuda.empty_cache()
    gc.collect()


# %%

# SECTION:-------------------- TS-------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR> <CR>
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

    # 2) Handle cascade groups: run localiser ONCE per run_w, then fan out to each run_p
# %%
    preds_all_patches = []
    for run_w, run_ps in E.cascade_groups.items():
        # load raw images once (BaseInferer handles type resolution)
        # tmp_loader = BaseInferer(run_name=run_ps[0], devices=E.devices, safe_mode=E.safe_mode, save_channels=E.save_channels)
        data = E.load_images(chunk)
        # 2a) localiser → bbox (once)
        E.bboxes = E._extract_bboxes(run_w, chunk)
        # 2b) apply crop, then run each patch member on the same cropped ROIs
        cropped = E._apply_bboxes(data, E.bboxes)
        pred_patches = E.patch_prediction(cropped, run_ps)
        pp = E.decollate_patches(pred_patches, E.bboxes, run_ps)
        preds_all_patches.extend(pp)

# %%
# SECTION:-------------------- patch pred-------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR>
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

    r = En.runs[0]
        for r in En.runs:
            mode = get_mode_outchannels(r)
            if mode in ("lbd", "patch"):
                # Resolve localiser run (prefer explicit; else checkpoint hint)
                params = load_params(r)
                hint = params["configs"]["plan_train"].get("source_plan_run") or params[
                    "configs"
                ].get("source_plan_run")
                run_w = En.localiser_run or hint
                if not run_w:
                    raise ValueError(
                        f"{r} requires a localiser run (set --localiser-run or add 'source_plan_run' in its config)."
                    )
                En.cascade_runs.append((run_w, r))
                En.patch_runs.append(r)
            else:
                En.base_runs.append(r)
# %%
