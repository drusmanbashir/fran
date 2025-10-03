# %%
from fran.managers import Project
import itertools as il

from __future__ import annotations

import ipdb
from fastcore.all import listify
from label_analysis.totalseg import TotalSegmenterLabels
from monai.transforms.compose import Compose
from monai.transforms.io.dictionary import SaveImaged
from monai.transforms.post.dictionary import (AsDiscreteD, MeanEnsembled,
                                              VoteEnsembled)

from fran.data.dataset import FillBBoxPatchesd
from fran.transforms.imageio import LoadSITKd
from fran.transforms.inferencetransforms import (
    KeepLargestConnectedComponentWithMetad, RenameDictKeys, ToCPUd)

tr = ipdb.set_trace

import gc
from pathlib import Path
from typing import List, Optional, Sequence, Union

import torch
from fastcore.basics import store_attr

# import your existing inferers
from fran.inference.base import BaseInferer, load_params
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


def _mode_of(run_name: str) -> str:
    params = load_params(run_name)
    plan = params["configs"]["plan_train"]
    return plan.get("mode", "source")


def _localiser_for(run_w: str, devices, safe_mode, save_channels):
    """Pick Base or WholeImage localiser based on the localiser run's mode."""
    mode = _mode_of(run_w)
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
    ):
        store_attr(but="localiser_labels")
        self.localiser_labels = list(localiser_labels) if localiser_labels else None

        # Partition runs into cascade vs base-like
        self.cascade_runs: List[Tuple[str, str]] = []  # (run_w, run_p)
        self.patch_runs: List[str] = []
        self.base_runs: List[str] = []

        for r in self.runs:
            mode = _mode_of(r)
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
            loc.prepare_data(data, tfms="ESN")
            loc.create_postprocess_transforms(loc.ds.transform)
            for batch in loc.predict():
                pred = loc.postprocess_transforms(batch)
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
                data=cropped_data, tfms="ESN", collate_fn=self._img_bbox_collated
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
            )
            P.setup()
            P.prepare_data(data=data, tfms="ESN", collate_fn=img_bbox_collated)
            P.create_postprocess_transforms(P.ds.transform)
            preds_all_runs[run] = []
            preds = P.predict()
            for batch in preds:
                output = P.postprocess_compose(batch)
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

    def parse_input(self, imgs_inp):
        """
        input types:
            folder of img_fns
            nifti img_fns
            itk imgs (slicer)
        returns list of img_fns if folder. Otherwise just the imgs
        """

        if not isinstance(imgs_inp, list):
            imgs_inp = [imgs_inp]
        imgs_out = []
        for dat in imgs_inp:
            if any([isinstance(dat, str), isinstance(dat, Path)]):
                self.input_type = "files"
                dat = Path(dat)
                if dat.is_dir():
                    dat = list(dat.glob("*"))
                else:
                    dat = [dat]
            else:
                self.input_type = "itk"
                if isinstance(dat, sitk.Image):
                    pass
                    # do nothing
                    # dat = ConvertSimpleItkImageToItkImage(dat, itk.F)
                elif isinstance(dat, itk.Image):
                    dat = itm(dat)
                else:
                    tr()
                dat = [dat]
            imgs_out.extend(dat)
        imgs_out = [{"image": img} for img in imgs_out]
        return imgs_out

    def load_images(self, data):
        """
        data can be filenames or images. InferenceDatasetNii will resolve data type and add LoadImaged if it is a filename
        """

        Loader = LoadSITKd(["image"])
        data = self.parse_input(data)
        data = [Loader(d) for d in data]
        return data

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
            mode = _mode_of(r)
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
                inf.prepare_data(data, tfms=inf.tfms)
                inf.create_postprocess_transforms(inf.ds.transform)
                for num_batches, batch in enumerate(inf.predict(), 1):
                    # for batch in inf.predict():
                    batch = inf.postprocess_transforms(batch)
                    if self.save:
                        inf.save_pred(batch)
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
            images = self.filter_existing_preds(images)
        chunksize = max(1, chunksize)

        all_outputs: List[dict] = []

        for start in range(0, len(images), chunksize):

            chunk = images[start : start + chunksize]
            data = self.load_images(chunk)
            # 1) Prepare data once for base-like members (they each handle their own transforms)
            # 2) Handle cascade groups: run localiser ONCE per run_w, then fan out to each run_p
            preds_all_patches = self._cascade_runs(data)
            preds_all_base = self._base_runs(data)
            # 3) Run base/whole members directly on full images
            preds_all = self.combined_patch_base_preds(
                preds_all_patches, preds_all_base
            )
            preds_final = self.cascade_postprocess(preds_all)
            all_outputs.append(preds_final)

            torch.cuda.empty_cache()
            gc.collect()

        return all_outputs

    def filter_existing_preds(self, imgs):

        print(
            "Filtering existing predictions\nNumber of images provided: {}".format(
                len(imgs)
            )
        )
        out_fns = [self.output_folder / img.name for img in imgs]
        to_do = [not fn.exists() for fn in out_fns]
        imgs = list(il.compress(imgs, to_do))
        print("Number of images remaining to be predicted: {}".format(len(imgs)))
        return imgs

    def combined_patch_base_preds(self, preds_all_patches, preds_all_base):
        preds_all = []
        for pred_patch, pred_base in zip(preds_all_patches, preds_all_base):
            preds_ = pred_patch | pred_base
            preds_all.append(preds_)
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

    def cascade_postprocess(self, preds):
        CPU = ToCPUd(keys=self.runs)

        MR = VoteEnsembled(
            output_key="pred", keys=self.runs, num_classes=2
        )  # HACK: These num_classes etc shold be fixed for multiclass runs
        # MR = VoteEnsembled(output_key="pred", keys=self.runs)
        # MR = MeanEnsembled(output_key="pred", keys=self.runs)

        S = SaveImaged(
            keys=["pred"],
            output_dir=self.output_folder,
            output_postfix="",
            separate_folder=False,
        )
        K = KeepLargestConnectedComponentWithMetad(
            keys=["pred"], independent=False, num_components=self.k_largest
        )  # label=1 is the organ
        # tfms = [MR, A, D, K, F]
        tfms = [CPU, MR]
        if self.k_largest:
            tfms.append(K)
        if self.save == True:
            tfms.append(S)
        C = Compose(tfms)

        tr()
        output = C(preds)
        return output

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
# SECTION:-------------------- SETUP-------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR>
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
    run_litsmc2 = ["LITS-1217"]
    run_ts = ["LITS-827"]
    run_totalseg = ["LITS-1246"]

    img_fna = "/s/xnat_shadow/litq/test/images_ub/"
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
    proj_nodes = Project("nodes"     )

# %%
# SECTION:-------------------- NODES -------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR>
    localiser_labels = set(TSL.label_localiser)
    runs = run_nodes
    safe_mode = False
    devices = [0]
    overwrite = False
    save_channels = False
    save_localiser = True
    save_members = True
    chunksize = 2
    localiser_run = run_w
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
    )

# %%
    preds = E.run(nodes, chunksize=chunksize)
    # preds = En.run(img_fns, chunksize=2)

# %%
    # batch['image'].shape

    # S
    params["configs"].keys()
    params["configs"]["dataset_params"]["plan_train"]
    # S
# %%
# SECTION:-------------------- TS-------------------------------------------------------------------------------------- <CR> <CR>
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
# SECTION:-------------------- patch pred-------------------------------------------------------------------------------------- <CR>
    # 3) Run base/whole members directly on full images
    prds_all_base = {}
    for r in E.base_runs:
        prds_all_base[r] = []
        mode = _mode_of(r)
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
            inf.prepare_data(data, tfms=inf.tfms)
            inf.create_postprocess_transforms(inf.ds.transform)
            for num_batches, batch in enumerate(inf.predict(), 1):
                # for batch in inf.predict():
                batch = inf.postprocess_transforms(batch)
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
    outs = En.cascade_postprocess(preds_all)
# %%

# %%
    # 4) Fuse
    if per_member_batches:
        merged = E._merge_member_outputs(per_member_batches)
        all_outputs.extend(merged)

    torch.cuda.empty_cache()
    gc.collect()


# %%
