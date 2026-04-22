import os
from pathlib import Path

from fran.inference.cascade import CascadeInferer
from fran.transforms.spatialtransforms import CropByYolo
from localiser.inference.base import LocaliserInfererPT, load_yolo_specs
from ultralytics import YOLO
from utilz.fileio import load_yaml


class CascadeInfererYOLO(CascadeInferer):
    def __init__(
        self,
        run_w,
        run_p,
        localiser_labels: list[str],
        project_title=None,
        devices=[0],
        safe_mode=False,
        patch_overlap=0.2,
        profile=None,
        save_channels=False,
        save=True,
        save_localiser=True,
        k_largest=None,
        debug=False,
        yolo_folder=None,
        localiser_regions=None,
        yolo_classes=None,
        yolo_batch_size=64,
        yolo_overwrite=True,
    ):
        super().__init__(
            run_w=run_w,
            run_p=run_p,
            localiser_labels=localiser_labels,
            project_title=project_title,
            devices=devices,
            safe_mode=safe_mode,
            patch_overlap=patch_overlap,
            profile=profile,
            save_channels=save_channels,
            save=save,
            save_localiser=save_localiser,
            k_largest=k_largest,
            debug=debug,
        )
        self.yolo_folder = Path(yolo_folder) if yolo_folder is not None else None
        self.localiser_regions = localiser_regions
        self.yolo_classes = yolo_classes
        self.yolo_batch_size = yolo_batch_size
        self.yolo_overwrite = yolo_overwrite
        self.yolo_specs = None
        self.Y = None
        self.cropper_yolo = CropByYolo(
            keys=["image"],
            lm_key="image",
            bbox_key="bbox",
            margin=20,
            sanitize=False,
        )
        self._setup_yolo_inferer()
        self.W = self.Y

    def _resolve_yolo_folder(self):
        if self.yolo_folder is not None:
            return self.yolo_folder
        conf_fldr = os.environ["FRAN_CONF"]
        best_runs = load_yaml(Path(conf_fldr) / "best_runs.yaml")
        return Path(best_runs["localiser"][0])

    def _resolve_yolo_classes(self):
        if self.yolo_classes is not None:
            return sorted(set(int(v) for v in self.yolo_classes))

        names = self.yolo_specs["data"]["names"]
        if isinstance(names, dict):
            class_to_ind = {str(v): int(k) for k, v in names.items()}
        else:
            class_to_ind = {str(name): idx for idx, name in enumerate(names)}

        if self.localiser_regions is None:
            return sorted(class_to_ind.values())

        regions = str(self.localiser_regions).replace(" ", "")
        regions_list = [r for r in regions.split(",") if r]
        classes = []
        for class_name, class_idx in class_to_ind.items():
            if any(region in class_name for region in regions_list):
                classes.append(class_idx)
        classes = sorted(set(classes))
        if len(classes) == 0:
            raise ValueError(
                f"No localiser classes matched localiser_regions='{self.localiser_regions}'."
            )
        return classes

    def _setup_yolo_inferer(self):
        yolo_folder = self._resolve_yolo_folder()
        self.yolo_specs = load_yolo_specs(yolo_folder)
        model = YOLO(self.yolo_specs["ckpt"])
        classes = self._resolve_yolo_classes()
        imsize = self.yolo_specs["specs"]["imgsz"]
        out_folder = self.output_folder / "localisers_yolo"
        self.Y = LocaliserInfererPT(
            model,
            classes,
            imsize=imsize,
            window="a",
            projection_dim=(1, 2),
            out_folder=out_folder,
            batch_size=self.yolo_batch_size,
        )

    def process_data_sublist(self, imgs_sublist):
        self.create_and_set_postprocess_transforms()
        data = self.load_images(imgs_sublist)
        self.bboxes = self.extract_fg_bboxes(data)
        data = self.apply_bboxes(data, self.bboxes)
        pred_patches = self.patch_prediction(data)
        pred_patches = self.decollate_patches(pred_patches, self.bboxes)
        output = self.postprocess(pred_patches)
        self.cuda_clear()
        return output

    def infer_yolo(self, data):
        image_fns = [Path(dat["image"].meta["filename_or_obj"]) for dat in data]
        outputs = self.Y.run(image_fns, overwrite=self.yolo_overwrite)
        by_name = {
            Path(out["projection_meta"][0]["filename_or_obj"]).name: out["bboxes_final"]
            for out in outputs
        }
        bboxes = []
        for dat in data:
            fn = Path(dat["image"].meta["filename_or_obj"]).name
            bboxes.append(by_name[fn])
        return bboxes

    def extract_fg_bboxes(self, data):
        return self.infer_yolo(data)

    def apply_bboxes(self, data, bboxes):
        data2 = []
        for dat, bbox in zip(data, bboxes):
            cropped = self.cropper_yolo({"image": dat["image"], "bbox": bbox})
            dat["image"] = cropped["image"]
            dat["bounding_box"] = bbox
            data2.append(dat)
        return data2

    def patch_prediction(self, data):
        if hasattr(self.W, "model") and self.W is not self.Y:
            del self.W.model
        return super().patch_prediction(data)


if __name__ == '__main__':
    
    

