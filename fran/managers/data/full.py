
from torch.utils.data import DataLoader


from monai.data import Dataset
from monai.transforms import Compose, EnsureChannelFirstd

# fran / project utils (already used elsewhere in your codebase)

from monai.transforms import Compose, EnsureChannelFirstd

from fran.twop5d.datamanagers import DataManager

class DataManagerFullScan(DataManager):
    def set_collate_fn(self):
        self.collate_fn = None  # default

    def setup(self, stage=None):
        self.create_transforms()
        self.set_transforms("L,E,N")  # no crops, no patches, no rand tfms
        self.create_dataset()
        self.dl = DataLoader(
            self.ds,
            batch_size=1,
            num_workers=2,
            collate_fn=self.collate_fn,
            persistent_workers=True,
            pin_memory=True,
        )

    def create_dataset(self):
        if not hasattr(self, "data") or len(self.data) == 0:
            self.prepare_data()
        self.ds = Dataset(data=self.data, transform=self.transforms)

    def create_transforms(self):
        Dev = ToDeviceD(keys=["image", "lm"], device=self.device)
        E = EnsureChannelFirstd(keys=["image", "lm"], channel_dim="no_channel")
        N = NormaliseClipd(
            keys=["image"],
            clip_range=self.dataset_params["intensity_clip_range"],
            mean=self.dataset_params["mean_fg"],
            std=self.dataset_params["std_fg"],
        )
        L = LoadImaged(keys=["image", "lm"], image_only=True, ensure_channel_first=False, simple_keys=True)
        L.register(TorchReader())
        self.transforms_dict = {"L": L, "E": E, "N": N, "Dev": Dev}
        self.transforms = Compose([L, E, N, Dev])
