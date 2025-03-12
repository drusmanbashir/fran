# %%
import torch
from fran.managers import Project

from fran.trainers.trainer import Trainer
from fran.utils.config_parsers import ConfigMaker
from utilz.imageviewers import ImageMaskViewer

#
# class TrainerBaseline(Trainer):
#     def init_dm(self):
#         cache_rate = self.config["dataset_params"]["cache_rate"]
#         ds_type = self.config["dataset_params"]["ds_type"]
#         # DMClass = self.resolve_datamanager(self.config["plan"]["mode"])
#         D = DataManagerBaseline(
#             self.project,
#             dataset_params=self.config["dataset_params"],
#             config=self.config,
#             transform_factors=self.config["transform_factors"],
#             affine3d=self.config["affine3d"],
#             batch_size=self.config["dataset_params"]["batch_size"],
#             cache_rate=cache_rate,
#             ds_type=ds_type,
#             training_aug=False
#         )
#         return D




# %%
if __name__ == "__main__":
# SECTION:-------------------- SETUP-------------------------------------------------------------------------------------- <CR>
    import warnings



    warnings.filterwarnings("ignore", "TypedStorage is deprecated.*")

    torch.set_float32_matmul_precision("medium")


    project_title = "nodes"
    proj = Project(project_title=project_title)

    configuration_filename = (
        "/s/fran_storage/projects/lits32/experiment_config_wholeimage.xlsx"
    )
    configuration_filename = "/s/fran_storage/projects/litsmc/experiment_config.xlsx"
    configuration_filename = None

    conf = ConfigMaker(proj, raytune=False).config

    # conf['dataset_params']['ds_type']=None
    # conf['model_params']['lr']=1e-3

    # run_name = "LITS-1007"
# %%
    # device_id = 0
    device_id = 0
    bs = 2# 5 is good if LBD with 2 samples per case
    run_name = None
    run_name='LITS-1072'
    compiled = False
    profiler = False
    # NOTE: if Neptune = False, should store checkpoint locally
    batch_finder = False
    neptune = True
    tags = []
    description = f"Settingh up a baseline with small dttaset"
# %%

    Tm = Trainer(proj, conf, run_name)
# %%
    Tm.setup(
        compiled=compiled,
        batch_size=bs,
        devices=[device_id],
        epochs=600 if profiler == False else 1,
        batchsize_finder=batch_finder,
        profiler=profiler,
        neptune=neptune,
        tags=tags,
        description=description,
    )

# %%
    Tm.fit()
# %%
    Tm.D.train_cases
    len(Tm.D.data_train)
    # Tm.D.batch_size=8
    Tm.N.compiled = compiled


    Tm.D.setup()
    D = Tm.D
    ds = Tm.D.valid_ds
    # ds2 = Tm.D.train_ds
    # print(ds[0]['image'].shape)
    # print(ds2[0]['image'].shape)

# %%
    dl = Tm.D.train_dataloader()

    iteri = iter(dl)
    bb = next(iteri)
    # len(ds)

    images = bb['image']
    lms = bb['lm']
    ind = 1
    ImageMaskViewer([images[ind][0],lms[ind][0]],'ii')
# %%

    bb['lm'].shape
# %%
    # model(inputs)
# %%
#SECTION:-------------------- TROUBLESHOOTING--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# %%
#     pred = torch.load("pred.pt")
#     target = torch.load("target.pt")
# # %%
#
#     Tm.trainer.model.to('cpu')
#     pred = [a.cpu() for a in pred]
#     loss = Tm.trainer.model.loss_fnc(pred.cpu(), target.cpu())
#     loss_dict = Tm.trainer.loss_fnc.loss_dict
#     Tm.trainer.maybe_store_preds(pred)
#     # preds = [pred.tensor() if hasattr(pred, 'tensor') else pred for pred in preds]
#     torch.save(preds, 'new_pred.pt')
#     torch.save(targ.tensor(),'new_target.pt')
#
#     tt = torch.tensor(targ.clone().detach())
#     torch.save(tt,"new_target.pt")
#

# %%
    Tm.D.setup()
    D = Tm.D
    ds = Tm.D.valid_ds
# %%


    dl = Tm.D.train_dataloader()
    iteri = iter(dl)
    batch = next(iteri)
    batch['image'].meta['filename_or_obj']

# %%
    dl2 = Tm.D.val_dataloader()
    iteri2 = iter(dl)
# %%
    while iteri:
        bb = next(iteri)
        # pred = Tm.trainer.model(bb['image'].cuda())
        print(bb['lm'].unique())


    dicis=[]
# %%
    bb['image'].shape
    ImageMaskViewer([bb['image'][1][0], bb['lm'][1][0]])
# %%
    for i, id in enumerate(ds):
        
        lm = id['lm']
        vals = lm.unique()
        print(vals)
        # print(vals)
        if vals.max()>8:
            tr()
            # print("Rat")
            dici = {'lm':lm.meta['filename_or_obj'], 'vals':vals}
            dicis.append(dici) 
#
            # vals = [  0.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,   8., 118.]
# %%
    dici = ds[7]
    dici = ds.data[7]
    dici = ds.transform(dici)

# %%
    L = LoadImaged(
        keys=["image", "lm"],
        image_only=True,
        ensure_channel_first=False,
        simple_keys=True,
    )
    L.register(TorchReader())

    E = EnsureChannelFirstd(keys=["image", "lm"], channel_dim="no_channel")
    Rtr = RandCropByPosNegLabeld(
        keys=["image", "lm"],
        label_key="lm",
        image_key="image",
        fg_indices_key="lm_fg_indices",
        bg_indices_key="lm_bg_indices",
        image_threshold=-2600,
        spatial_size=D.src_dims,
        pos=3,
        neg=1,
        num_samples=D.plan["samples_per_file"],
        lazy=True,
        allow_smaller=False,
    )
    Ld = LoadTorchDict(keys=["indices"], select_keys=["lm_fg_indices", "lm_bg_indices"])

    Rva = RandCropByPosNegLabeld(
        keys=["image", "lm"],
        label_key="lm",
        image_key="image",
        image_threshold=-2600,
        fg_indices_key="lm_fg_indices",
        bg_indices_key="lm_bg_indices",
        spatial_size=D.dataset_params["patch_size"],
        pos=3,
        neg=1,
        num_samples=D.plan["samples_per_file"],
        lazy=False,
        allow_smaller=True,
    )
    Re = ResizeWithPadOrCropd(
        keys=["image", "lm"],
        spatial_size=D.dataset_params["patch_size"],
        lazy=False,
    )

    Ind = MetaToDict(keys=["lm"], meta_keys=["lm_fg_indices", "lm_bg_indices"])
# %%
    D.prepare_data()
    D.setup(None)
# %%
    dici = D.data_train[0]
    D.valid_ds.data[0]

# %%
    dici = D.valid_ds.data[7]
    dici = L(dici)
    dici = Ind(dici)
    dici = Ld(dici)
    dici = D.transforms_dict["E"](dici)
    dici = D.transforms_dict["Rva"](dici)
    dici = Re(dici[1])

# %%
    ImageMaskViewer([dici[0]["image"][0], dici[0]["lm"][0]])

# %%
    Ld = LoadTorchDict(keys=["indices"], select_keys=["lm_fg_indices", "lm_bg_indices"])
    dici = Ld(dici)
# %%

# %%

    fn = "/r/datasets/preprocessed/litsmc/lbd/spc_080_080_150/images/lits_115.pt"
    fn2 = "/r/datasets/preprocessed/litsmc/lbd/spc_080_080_150/lms/lits_115.pt"
    tt = torch.load(fn)
    tt2 = torch.load(fn2)
    ImageMaskViewer([tt, tt2])
# %%

    Re = ResizeWithPadOrCropd(
        keys=["image", "lm"],
        spatial_size=D.dataset_params["patch_size"],
        lazy=False,
    )
# %%
    dici = Re(dici)
# %%
    dici = ds[1]
    dici = ds.data[0]
    keys_tr = "L,E,Ind,Rtr,F1,F2,A,Re,N,I"
    keys_val = "L,E,Ind,Rva,Re,N"
    keys_tr = keys_tr.split(",")
    keys_val = keys_val.split(",")

# %%
    dici = ds.data[5].copy()
    for k in keys_val[:3]:
        tfm = D.transforms_dict[k]
        dici = tfm(dici)
# %%

    ind = 0
    dici = ds.data[ind]
    ImageMaskViewer([dici["image"][0], dici["lm"][0]])
    ImageMaskViewer([dici[ind]["image"][0], dici[ind]["lm"][0]])
# %%
    tfm2 = D.transforms_dict[keys_tr[5]]

# %%
    for didi in dici:
        dd = tfm2(didi)
# %%
    idx = 0
    ds.set_bboxes_labels(idx)
    if ds.enforce_ratios == True:
        ds.mandatory_label = ds.randomize_label()
        ds.maybe_randomize_idx()

    filename, bbox = ds.get_filename_bbox()
    img, lm = ds.load_tensors(filename)
    dici = {"image": img, "lm": lm, "bbox": bbox}
    dici = ds.transform(dici)

    E = EnsureChannelFirstd(keys=["image", "lm"], channel_dim="no_channel")
    dici = E(dici)
# %%
    # img = ds.create_metatensor(img_fn)
    # label = ds.create_metatensor(label_fn)
    dici = ds.data[3]
    dici = ds[3]
    dici[0]["image"]
    dat = ds.data[5]
    dici = ds.transform(dat)
    type(dici)
    dici = ds[4]
    dici.keys()
    dat
    dici = {"image": img_fn, "lm": label_fn}
    im = torch.load(img_fn)

    im.shape

# %%

# %%
    b = next(iteri2)

    b["image"].shape
    m = Tm.N.model
    N = Tm.N

# %%
    for x in range(len(ds)):
        casei = ds[x]
        for a in range(len(casei)):
            print(casei[a]["image"].shape)
# %%
    for i, b in enumerate(dl):
        print("----------------------------")
        print(b["image"].shape)
        print(b["label"].shape)
# %%
    # b2 = next(iter(dl2))
    batch = b
    inputs, target, bbox = batch["image"], batch["lm"], batch["bbox"]

    [pp(a["filename"]) for a in bbox]
# %%
    preds = N.model(inputs.cuda())
    pred = preds[0]
    pred = pred.detach().cpu()
    pp(pred.shape)
# %%
    n = 1
    img = inputs[n, 0]
    mask = target[n, 0]
# %%
    ImageMaskViewer([img.permute(2, 1, 0), mask.permute(2, 1, 0)])
# %%
    fn = "/s/fran_storage/datasets/preprocessed/fixed_spacings/litsmall/spc_080_080_150/images/lits_4.pt"
    fn = "/s/fran_storage/datasets/preprocessed/fixed_spacings/litstp/spc_080_080_150/images/lits_4.pt"
    fn2 = "/home/ub/datasets/preprocessed/lits32/patches/spc_080_080_150/dim_192_192_128/masks/lits_4_1.pt"
    img = torch.load(fn)
    mask = torch.load(fn2)
    pp(img.shape)
# %%

    ImageMaskViewer([img, mask])
# %%
# %%

    Tm.trainer.callback_metrics
# %%
    ckpt = Path(
        "/s/fran_storage/checkpoints/litsmc/Untitled/LITS-709/checkpoints/epoch=81-step=1886.ckpt"
    )
    kk = torch.load(self.ckpt)
    kk["datamodule_hyper_parameters"].keys()
    kk.keys()
    kk["datamodule_hyper_parameters"]
# %%
