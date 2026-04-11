import torch
from monai.transforms import Compose, Transform
from monai.transforms.utility.dictionary import MapTransform


def tfms_from_dict(keys, transforms_dict):
    keys = keys.replace(" ", "").split(",")
    tfms = []
    for key in keys:
        tfms.append(transforms_dict[key])
    return Compose(tfms)


class NormaliseZeroToOne(MapTransform):
    def __call__(self, data):
        for key in self.keys:
            image = data[key]
            image = image - image.min()
            image = image / image.max()
            data[key] = image
        return data


class WindowTensor3Channeld(Transform):
    def __init__(self, image_key):
        self.windows = {
            "b": [-450.0, 1050.0],
            "c": [-1350.0, 150.0],
            "a": [-150.0, 250.0],
        }
        self.image_key = image_key

    def __call__(self, data):
        image = data[self.image_key]
        outs = []
        for L, U in self.windows.values():
            img = torch.clamp(image, L, U)
            img = (img - L) / (U - L)
            outs.append(img)

        data[self.image_key] = torch.cat(outs, dim=0)
        return data

