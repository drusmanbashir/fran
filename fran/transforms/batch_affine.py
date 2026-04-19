import torch
import torch.nn.functional as F
from monai.transforms.transform import MapTransform, RandomizableTransform


class BatchRandAffined3D(MapTransform, RandomizableTransform):
    def __init__(
        self,
        keys=("image", "lm"),
        mode=("bilinear", "nearest"),
        prob=1.0,
        rotate_range=(0.0, 0.0, 0.0),
        scale_range=(0.0, 0.0, 0.0),
        allow_missing_keys=False,
    ):
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob=1.0)
        self.mode = tuple(mode)
        self.item_prob = float(prob)
        self.rotate_range = tuple(float(v) for v in rotate_range)
        self.scale_range = tuple(float(v) for v in scale_range)

    def randomize(self, data):
        image = data[self.keys[0]]
        batch_size = image.shape[0]
        self._active = self.R.rand(batch_size) < self.item_prob
        self._angles = (self.R.rand(batch_size, 3) * 2.0 - 1.0) * self.rotate_range
        self._scales = 1.0 + (self.R.rand(batch_size, 3) * 2.0 - 1.0) * self.scale_range

    def _theta(self, device, dtype):
        active = torch.as_tensor(self._active, device=device)
        angles = torch.as_tensor(self._angles, device=device, dtype=torch.float32)
        scales = torch.as_tensor(self._scales, device=device, dtype=torch.float32)
        batch_size = angles.shape[0]

        cx, cy, cz = torch.cos(angles[:, 0]), torch.cos(angles[:, 1]), torch.cos(angles[:, 2])
        sx, sy, sz = torch.sin(angles[:, 0]), torch.sin(angles[:, 1]), torch.sin(angles[:, 2])

        theta = torch.zeros(batch_size, 3, 4, device=device, dtype=torch.float32)
        theta[:, 0, 0] = cy * cz
        theta[:, 0, 1] = sx * sy * cz - cx * sz
        theta[:, 0, 2] = cx * sy * cz + sx * sz
        theta[:, 1, 0] = cy * sz
        theta[:, 1, 1] = sx * sy * sz + cx * cz
        theta[:, 1, 2] = cx * sy * sz - sx * cz
        theta[:, 2, 0] = -sy
        theta[:, 2, 1] = sx * cy
        theta[:, 2, 2] = cx * cy
        theta[:, :, :3] = theta[:, :, :3] * scales[:, None, :]

        identity = torch.eye(3, 4, device=device, dtype=torch.float32).expand(batch_size, -1, -1)
        theta = torch.where(active[:, None, None], theta, identity)
        return theta.to(dtype=dtype)

    def __call__(self, data):
        d = dict(data)
        self.randomize(d)
        image = d[self.keys[0]]
        theta = self._theta(image.device, image.dtype)
        grid = F.affine_grid(theta, image.shape, align_corners=False)

        for key, mode in zip(self.keys, self.mode):
            src = d[key]
            src_sample = src.float() if mode == "nearest" else src
            dst = F.grid_sample(
                src_sample,
                grid,
                mode=mode,
                padding_mode="border",
                align_corners=False,
            ).to(dtype=src.dtype)
            dst.meta = src.meta
            d[key] = dst
        return d
