from fran.utils.common import *
P = Project(project_title="lits"); proj_defaults= P
configs_excel = ConfigMaker(proj_defaults.configuration_filename,raytune=False).config

train_list, valid_list, test_list = get_fold_case_ids(
        fold=configs_excel['metadata']["fold"],
        json_fname=proj_defaults.validation_folds_filename,
    )
fldr =Path("/home/ub/datasets/preprocessed/lits/patches/spc_080_080_150/dim_192_192_128") 


bboxes_fname = fldr/ ("bboxes_info")
dd = load_dict(bboxes_fname)



# %%
train_ds = ImageMaskBBoxDataset(
        proj_defaults,
        train_list,
        bboxes_fname,
        [0,0,1]
    )
# %%
a,b,c = train_ds[0]
# %%
 
# Create geometries and projector.
# %%
import matplotlib.pyplot as plt
f = np.load('/home/ub/code/fran/fran/extra/SLphan.npy')
plt.imshow(im)
# %%
fname = "/s/datasets_bkp/litq/nifti/7/case7_02092019.nii.gz"
import astra
a = a.permute(2,0,1)
d,w,h = a.shape
vol_geom = astra.create_vol_geom(  w,h,d )
pp(vol_geom)
rec_id = astra.data3d.create('-vol', vol_geom,data=a.numpy())
# %%
det_count = 32
angles = np.linspace(0, 2*np.pi, 360, False)
proj_geom = astra.create_proj_geom('parallel3d', 1.0, 1.0, 32, 64, angles)
# Create projector
projector_id = astra.data3d.create('-proj3d', proj_geom)
# %%
cfg = astra.astra_dict('SIRT3D_CUDA')
cfg['ReconstructionDataId'] = rec_id
cfg['ProjectionDataId'] = projector_id
alg_id = astra.algorithm.create(cfg)
astra.algorithm.run(alg_id, 100)
rec = astra.data3d.get(rec_id)
# %%
from fran.utils.imageviewers import ImageMaskViewer
img2 = rec.swapaxes(0,1)
img2.shape
ImageMaskViewer([rec,a],data_types=['img','img'])
#
