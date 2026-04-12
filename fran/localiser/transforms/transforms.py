from monai.transforms.transform import Randomizable, RandomizableTransform
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

class CombineProjections(RandomizableTransform):

        def __init__(self, proj1_key = "image1", proj2_key = "image2",windows=["a","b","c"], projection_key: str|None='1a,2b,2c',allow_missing_keys=False):#     projection_keys = 
            self.proj1_key = proj1_key
            self.proj2_key = proj2_key
            for win in windows:
                assert win in projection_key, f"Window {win} not in projection keys {projection_key}"
            self.windows=windows
            self.projection_key = projection_key
        def proj_builder(self):
               aa = self.R.randint(1, 3)  # 1 or 2
               bb = self.windows[self.R.randint(0, len(self.windows))]
               return f"{aa}{bb}"



        def randomize(self):
            projection_key = ""
            for _ in range(3):
                projection_key += self.proj_builder() + ","



        def __call__(self, data):
            if self.projection_key is None:
                projection = self.randomize()
            else: projection = self.projection_key

            image1 = data[self.proj1_key]
            image2 = data[self.proj2_key]
            image_out = self.combine_projections(image1,image2,projection)
            data["image"] = image_out
            return data

        def combine_projections(self,proj1,proj2,projection_key):
            projection_key_list = projection_key.split(",")
            rgb = []
            assert proj1.ndim==3 and proj2.ndim==3, f"Projections must be 3D tensors with shape (C,H,W), got {proj1.shape} and {proj2.shape}"
            assert proj1.shape[0] == len(self.windows), f"Number of channels in projections must match number of windows, got {proj1.shape[0]} channels and {len(self.windows)} windows"
            for kss in projection_key_list:
                img_suff = kss[0]
                win_suff = kss[1]
                ind = WINDOW_TO_IND_MAP[win_suff]
                if img_suff == "1":
                    img = proj1
                else:
                    img = proj2
                rgb.append(img[ind])
            rgb = torch.stack(rgb, dim=0)
            return rgb

class WindowTensor3ChanneldRand( RandomizableTransform):
    '''
    randomizes  and jitter the window
    '''
    def __init__(self, image_key, prob=0.5, jitter=100):
        self.jitter = jitter
        self.windows = {
            "b": [-450.0, 1050.0],
            "c": [-1350.0, 150.0],
            "a": [-150.0, 250.0],
        }
        self.image_key = image_key
        RandomizableTransform.__init__(self, prob)

    def randomize(self):
        do_transform = self.R.rand() < self.prob
        if do_transform:
            jitter = self.R.uniform(-self.jitter, self.jitter)
            new_windows = {}
            for key, (L, U) in self.windows.items():
                new_windows[key] = (L + jitter, U + jitter)
            return new_windows
        else:
            return self.windows


    def __call__(self, data):
        image = data[self.image_key]
        outs = []
        window_values = self.randomize()
        for L, U in window_values.values():
            img = torch.clamp(image, L, U)
            img = (img - L) / (U - L)
            outs.append(img)

        if img.ndim== 4:
            stack_dim = 0
        else: raise ValueError(f"Unsupported image dimension: {img.ndim}")
        data[self.image_key] = torch.concat(outs, dim=stack_dim)   # assumes a free channel dimension
        return data

if __name__ == "__main__":
# %%
    img = torch.tensor([[-1000.0, -500.0, 0.0, 500.0, 1000.0]])
    dici = {"image": img}
    Win = WindowTensor3ChanneldRand(image_key="image", prob=1.0, jitter=100)
    dici = Win(dici)
