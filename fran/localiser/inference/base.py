from pathlib import Path

from fran.transforms.spatialtransforms import Project2D
import SimpleITK as sitk
import ipdb
import torch
from fastcore.all import listify
from fran.localiser.helpers import (
    draw_tensor_boxes,
    jpg_to_tensor,
    make_multiwindow_inference_tensor,
    make_singlewindow_inference_tensor,
)
from fran.localiser.yolo_ct_augment import apply_window_tensor
from monai.data.dataloader import DataLoader
from monai.data.dataset import Dataset
from monai.transforms.compose import Compose
from monai.transforms.croppad.dictionary import SpatialPadd
from monai.transforms.io.dictionary import LoadImaged
from monai.transforms.spatial.dictionary import Orientationd
from monai.transforms.spatial.dictionary import Resized
from monai.transforms.transform import MapTransform
from monai.transforms.utility.dictionary import EnsureChannelFirstd
from torch.nn.functional import interpolate
from ultralytics import YOLO
from utilz.dictopts import DictToAttr
from utilz.fileio import maybe_makedirs
from utilz.stringz import headline, strip_extension

tr = ipdb.set_trace


def list_to_chunks(items, chunksize):
    return [items[i : i + chunksize] for i in range(0, len(items), chunksize)]


def load_images_nifti(images):
    loader = LoadImaged(keys=["image"], image_only=True)
    return [loader({"image": image}) for image in listify(images)]


class WindowOned(MapTransform):
    def __init__(self, keys, window="a"):
        super().__init__(keys)
        self.window = window

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            image = d[key].float()
            image = apply_window_tensor(image, self.window)
            d[key] = image
        return d


class PermuteFlip2Dd(MapTransform):
    def __init__(self, keys, permute=(0, 2, 1), flip_direction="vertical"):
        super().__init__(keys)
        self.permute = permute
        self.flip_dims = {
            "vertical": (1,),
            "height": (1,),
            "h": (1,),
            "y": (1,),
            "updown": (1,),
            "horizontal": (2,),
            "width": (2,),
            "w": (2,),
            "x": (2,),
            "leftright": (2,),
        }[flip_direction.lower()]

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            image = d[key].permute(*self.permute)
            d[key] = torch.flip(image, dims=self.flip_dims).contiguous()
        return d


class Resize2Dd(MapTransform):
    def __init__(self, keys, spatial_size=(256, 256), mode="bilinear"):
        super().__init__(keys)
        self.spatial_size = spatial_size
        self.mode = mode

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.resize_image(d[key])
        return d

    def resize_image(self, image):
        image = image.float().unsqueeze(0)
        kwargs = {}
        if self.mode in {"linear", "bilinear", "bicubic", "trilinear"}:
            kwargs["align_corners"] = False
        return interpolate(
            image,
            size=self.spatial_size,
            mode=self.mode,
            **kwargs,
        ).squeeze(0)


class NormaliseZeroToOned(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.normalise_image(d[key])
        return d

    def normalise_image(self, image):
        image = image.float()
        image = image - image.min()
        denom = image.max()
        if denom > 0:
            image = image / denom
        return image


class RepeatChannelsd(MapTransform):
    def __init__(self, keys, channels=3):
        super().__init__(keys)
        self.channels = channels

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.repeat_image(d[key]).contiguous()
        return d

    def repeat_image(self, image):
        return image.repeat(self.channels, 1, 1)


def collate_projections(batch):
    images = []
    projection_meta = []
    for case_index, item in enumerate(batch):
        for projection_index, key in enumerate(("image1", "image2")):
            images.append(item[key])
            meta = dict(item[key].meta)
            meta.update(letterbox_meta(item[key]))
            meta["case_index"] = case_index
            meta["projection_index"] = projection_index
            meta["projection_key"] = key
            projection_meta.append(meta)
    return {
        "image": torch.cat(images, dim=0),
        "projection_meta": projection_meta,
    }


def letterbox_meta(image):
    spatial_pad = [
        op for op in image.applied_operations if op["class"] == "SpatialPad"
    ][-1]
    resize = [op for op in image.applied_operations if op["class"] == "Resize"][-1]
    return {
        "letterbox_padded": spatial_pad["extra_info"]["padded"],
        "letterbox_orig_size": resize["orig_size"],
        "letterbox_resized_size": spatial_pad["orig_size"],
    }


def yolo_bbox_to_slices(bbox, spatial_shape, padding=None, padded_shape=None, resized_shape=None):
    spatial_shape = torch.as_tensor(spatial_shape, dtype=torch.float64)
    bbox = torch.as_tensor(bbox, dtype=torch.float64)
    centers = bbox[[1, 0]]
    sizes = bbox[[3, 2]]
    n_dims = 2

    if padding is not None:
        padding = torch.as_tensor(padding, dtype=torch.float64)[-n_dims:]
        pad_before = padding[:, 0]
        pad_total = padding.sum(dim=1)
        if padded_shape is None:
            padded_shape = torch.as_tensor(resized_shape, dtype=torch.float64) + pad_total
        else:
            padded_shape = torch.as_tensor(padded_shape, dtype=torch.float64)
        resized_shape = padded_shape - pad_total
        centers = (centers * padded_shape - pad_before) / resized_shape
        sizes = sizes * padded_shape / resized_shape

    eps = 1e-6
    starts = torch.floor((centers - sizes / 2) * spatial_shape + eps).long()
    stops = torch.ceil((centers + sizes / 2) * spatial_shape - eps).long()
    shape = spatial_shape.long()
    starts = torch.minimum(torch.maximum(starts, torch.zeros_like(starts)), shape - 1)
    stops = torch.minimum(torch.maximum(stops, starts + 1), shape)
    return tuple(slice(int(start), int(stop)) for start, stop in zip(starts, stops))


def yolo_bbox_to_3d_slices(
    bbox,
    spatial_shape_3d,
    projection_key,
    padding=None,
    padded_shape=None,
    resized_shape=None,
):
    x, y, z = spatial_shape_3d
    projection_key = projection_key.replace("image", "p")
    if projection_key in {"p1", "lat"}:
        z_slice, y_slice = yolo_bbox_to_slices(
            bbox,
            spatial_shape=(z, y),
            padding=padding,
            padded_shape=padded_shape,
            resized_shape=resized_shape,
        )
        return (slice(None), y_slice, _flip_slice(z_slice, z))

    z_slice, x_slice = yolo_bbox_to_slices(
        bbox,
        spatial_shape=(z, x),
        padding=padding,
        padded_shape=padded_shape,
        resized_shape=resized_shape,
    )
    return (x_slice, slice(None), _flip_slice(z_slice, z))


def _flip_slice(slice_, size):
    return slice(size - slice_.stop, size - slice_.start)


class LocaliserInferer():
    def __init__(
        self,
        model,
        imsize=256,
        window="a",
        projection_dim=(1, 2),
        batch_size=None,
        device=0,
        debug=False,
        out_folder=None,
        letterbox=True,
        keys_preproc="E,O,W,P1,P2,PF,R,Rep,E2,N",
    ):
        self.model = model
        self.imsize = imsize
        self.window = window
        self.projection_dim = projection_dim
        self.batch_size = batch_size
        self.device = device
        self.debug = debug
        self.out_folder = Path(out_folder) if out_folder is not None else None
        self.letterbox = letterbox
        self.keys_preproc = keys_preproc
        self.projections_dims = {"lat": 1, "ap": 2, "ax": 3}
        self.image_key = "image"
        self.image2d_keys = ["image1","image2"]
        self.setup_model()
        self.create_and_set_preprocess_transforms()

    def setup_model(self):
        if self.device is not None:
            self.model.to(self.device)
        self.model.eval()

    def create_and_set_preprocess_transforms(self):
        self.create_preprocess_transforms()
        self.set_preprocess_transforms()

    def create_preprocess_transforms(self):
        self.preprocess_transforms_dict = {
            "O": Orientationd(keys=[self.image_key], axcodes="RAS", labels=None),
            "E": EnsureChannelFirstd(keys=[self.image_key], channel_dim="no_channel"),
            "W": WindowOned(
                keys=[self.image_key],
                window=self.window,
            ),
            "PF": PermuteFlip2Dd(keys=self.image2d_keys),

            "E2": EnsureChannelFirstd(keys=self.image2d_keys, channel_dim="no_channel"),
            "R": self.create_resize_transform(),
            "N": NormaliseZeroToOned(keys=self.image2d_keys),
            "Rep": RepeatChannelsd(keys=self.image2d_keys, channels=3),
            "P1":  Project2D(
            keys=[ self.image_key],
            operations=["mean"],
            dim=self.projections_dims["lat"],
            suffix="lat",
            output_keys=[ "image1"],
        ),

        "P2" : Project2D(
            keys=[ self.image_key],
            operations=[ "mean"],
            dim=self.projections_dims["ap"],
            suffix="ap",
            output_keys=[ "image2"],
        )
            
        }

    def create_resize_transform(self):
        if self.letterbox:
            return Compose(
                [
                    Resized(
                        keys=self.image2d_keys,
                        spatial_size=self.imsize,
                        size_mode="longest",
                        mode="bilinear",
                    ),
                    SpatialPadd(
                        keys=self.image2d_keys,
                        spatial_size=(self.imsize, self.imsize),
                        method="symmetric",
                    ),
                ]
            )
        return Resize2Dd(
            keys=self.image2d_keys,
            spatial_size=(self.imsize, self.imsize),
        )

    def set_preprocess_transforms(self):
        self.preprocess_transforms = self.tfms_from_dict(
            self.keys_preproc, self.preprocess_transforms_dict
        )
        self.preprocess_compose = Compose(self.preprocess_transforms)

    def preprocess_iterate(self, data):
        for tfm in self.preprocess_transforms:
            headline(tfm)
            tr()
            data = tfm(data)
        return data

    def tfms_from_dict(self, keys, transforms_dict):
        keys = keys.replace(" ", "").split(",")
        return [transforms_dict[key] for key in keys]

    def load_images(self, images):
        return load_images_nifti(images)

    def prepare_data(self, data):
        transform = self.preprocess_iterate if self.debug else self.preprocess_compose
        self.ds = Dataset(data=data, transform=transform)
        batch_size = self.batch_size or len(data)
        self.pred_dl = DataLoader(
            self.ds,
            batch_size=batch_size,
            num_workers=0,
            collate_fn=collate_projections,
        )

    def run(self, images, chunksize=64):
        if self.out_folder is not None:
            maybe_makedirs(self.out_folder)
        outputs = []
        for image_chunk in list_to_chunks(listify(images), chunksize):
            outputs.extend(self.process_data_sublist(image_chunk))
        return outputs

    def process_data_sublist(self, images):
        data = self.load_images(images)
        self.prepare_data(data)
        return list(self.predict())

    def predict(self):
        with torch.inference_mode():
            for batch in self.pred_dl:
                image = batch["image"].float()
                if self.device is not None:
                    image = image.to(self.device)
                pred = self.model(image, verbose=False)
                if self.out_folder is not None:
                    self.save_prediction_images(image, pred, batch["projection_meta"])
                yield {
                    "pred": pred,
                    "image": image,
                    "batch": batch,
                }

    def save_prediction_images(self, images, preds, projection_meta):
        for index, (image, pred, meta) in enumerate(zip(images, preds, projection_meta)):
            src = strip_extension(Path(meta["filename_or_obj"]).name)
            projection = meta["projection_key"].replace("image", "p")
            filename = self.out_folder / f"{src}_{index:03d}_{projection}.jpg"
            draw_tensor_boxes(image[None], pred, filename=filename, show=False)
# %%
#SECTION:--------------------  setup--------------------------------------------------------------------------------------
if __name__ == '__main__':
    model = YOLO(
        "/s/fran_storage/yolo_output/totalseg_localiser/train32/weights/best.pt"
    )

# %%
    def show(dici):
        def _imi(im):
            print(im.shape)
            print(f"Max: {im.max()}")
            print(f"Min: {im.min()}")

        print(dici.keys())
        img = dici['image']
        _imi(img)

# %%
    data_folder =Path("/media/UB/datasets/kits23/") 
    imgs_fldr = data_folder / "images"
    out_fldr = data_folder / "loc2d"
    nii_path = Path("/media/UB/datasets/kits23/images/kits23_00007.nii.gz")
    assert nii_path.exists(), f"{nii_path} does not exist"
    M = LocaliserInferer(model, imsize=256, window="a", projection_dim=(1, 2), out_folder=out_fldr)
    imgs = list(imgs_fldr.glob("*"))
    out = M.run(imgs)
    out[0].keys()
    data = M.load_images([nii_path])
    dici = data[0]
    dici = M.preprocess_transforms_dict["E"](dici)
    dici = M.preprocess_transforms_dict["O"](dici)
    dici = M.preprocess_transforms_dict["W"](dici)
    show(dici)
    dici = M.preprocess_transforms_dict["P1"](dici)
    dici = M.preprocess_transforms_dict["P2"](dici)
    dici = M.preprocess_transforms_dict["PF"](dici)
    M.image2d_keys=[ "image1","image2"]
    dici['image1'].shape
    R= Resize2Dd(keys=M.image2d_keys, spatial_size=(M.imsize, M.imsize))
    dici = R(dici)
    
    dici['image1'].shape
# %%
# %%
    Rep  = RepeatChannelsd(keys=M.image2d_keys, channels=3)
    dici = Rep(dici)
    E2 = EnsureChannelFirstd(keys=M.image2d_keys, channel_dim="no_channel")
    dici = E2(dici)
    dici['image2'].max()
    dici['image2'].shape
    N = NormaliseZeroToOned(keys=M.image2d_keys)
    dici = N(dici)
    dici['image1'].max()
    dici['image1'].shape


    im1 = dici['image1']
    im2 = dici['image2']
    ressi = model(im1)
    ress2 = model(im2)
    rr = ressi[0]
    rr2= ress2[0]
    draw_tensor_boxes(im1,rr)
    draw_tensor_boxes(im2,rr2)
# %%
    from matplotlib import pyplot as plt
    plt.imshow(dici['image1'][0])
    plt.show()
# %%
    rr = out[0]

    rr.keys()
    yol = rr['pred']
    yol[0].boxes
    draw_tensor_boxes(rx4,yol[0])
# %%
    

# %%
    img = sitk.ReadImage(nii_path)

    arr = sitk.GetArrayFromImage(img)
    imsize=256
    raw_x = torch.from_numpy(arr).float().unsqueeze(0)
    rx_win = apply_window_tensor(raw_x, "a")
    rx2 = torch.mean(rx_win, 1)
    rx2_perm = rx2.permute(0, 2, 1)
    rx4 = rx2.unsqueeze(0)
    rx4 = rx4.permute(0, 1, 3, 2)
    rx4 = interpolate(rx4, (imsize, imsize))
    min_ = rx4.min()
    max_ = rx4.max()
    rx4 = (rx4 - min_) / (max_ - min_)

    # min_t = rx2.min()
    # max_t = rx2.max()
    # rx3 = (rx2-min_t)/(max_t-min_t)
    # rx4 = rx3.unsqueeze(0)
    rx4 = rx4.repeat(1, 3, 1, 1)
    print(rx4.shape)
    plt.imshow(rx4[0,0])
# %%
    res2 = model(rx4)
    rr2 = res2[0]
# %%

    from fran.localiser.helpers import draw_tensor_boxes

    draw_tensor_boxes(rx4, rr2)

# %%

# %%
# SECTION:-------------------- jpg--------------------------------------------------------------------------------------

    import torch

# %%
    fn = "/s/xnat_shadow/totalseg2d/jpg/valid/images/totalseg_s0029_a2.jpg"
    img = jpg_to_tensor(fn)
    img_torch = img.permute(2, 0, 1).float()
    img_torch = img_torch / img_torch.max()
    img_torch = img_torch.unsqueeze(0)
    im2 = interpolate(img_torch, (imsize, imsize))

    im3 = torch.flip(im2, dims=[2, 3])
    res = model(im3)
    rr = res[0]
    # draw_tensor_boxes(im3, rr)

# %%

# %%

# SECTION:-------------------- 3window--------------------------------------------------------------------------------------
    x = make_multiwindow_inference_tensor(raw_x)

    res = model(x)

# %%

# %%
# SECTION:-------------------- one window--------------------------------------------------------------------------------------
    x2 = make_singlewindow_inference_tensor(raw_x, "b")

# %%
    res = model(x2)

# %%
    rr = res[0]
    imtnsr = x2
# %%
