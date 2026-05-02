# %%
import itertools as il
import os
from pathlib import Path

from fran.data.dataregistry import DS
from fran.managers import Project
from fran.utils.common import *
from label_analysis.totalseg import TotalSegmenterLabels
from monai.transforms.post.dictionary import Activationsd, AsDiscreted
from utilz.fileio import load_yaml
from utilz.helpers import pp

# SECTION:-------------------- SETUP--------------------------------------------------------------------------------------

conf_fldr = os.environ["FRAN_CONF"]

def load_best_runs_yaml(conf_fldr):
    best_runs = load_yaml(Path(conf_fldr) / "best_runs.yaml")
    localiser_runs = best_runs["runs"]["localiser"]
    best_runs["runs"]["localiser"] = {
        key: os.path.expandvars(value) for key, value in localiser_runs.items()
    }
    return best_runs


best_runs = load_best_runs_yaml(conf_fldr)
best_runs = best_runs["runs"]
run_w = best_runs["run_w"]
runs_2d = best_runs["localiser"]
totalseg_runs_all = best_runs["totalseg"]
totalseg_run_big = totalseg_runs_all["full"][0]
totalseg_proj = Project(project_title="totalseg")

# SECTION:-------------------- FILES and FOLDERS--------------------------------------------------------------------------

crc_fldr = Path("/s/xnat_shadow/crc/images")
crc_imgs = list(crc_fldr.glob("*"))

curvas_fldr = DS["curvaspdac"].folder / "images"
curvas_imgs = list(curvas_fldr.glob("*"))

lidc_fldr = DS["lidc"].folder / "images"
lidc_imgs = list(lidc_fldr.glob("*"))

nodes_pending_fldr = Path("/s/xnat_shadow/nodes/images_pending/thin_slice/images")
nodes_pending_imgs = list(nodes_pending_fldr.glob("*"))
nodes_training_fldr = Path("/s/xnat_shadow/nodes/images")
nodes_training_imgs = list(nodes_training_fldr.glob("*"))

liver_fldrs = (
    Path(DS["litq"].folder),
    Path(DS["drli"].folder),
    Path(DS["lits"].folder),
    Path(DS["litqsmall"].folder),
)
liver_imgs = [list((fld / "images").glob("*")) for fld in liver_fldrs]
liver_imgs = list(il.chain.from_iterable(liver_imgs))

bosniak_fldr = Path("/s/datasets_bkp/bosniak/bosniak/kits/nifti")
bosniak_imgs = list(bosniak_fldr.glob("*"))

colonmsd_fldr = DS["colonmsd10"].folder / "images"
colonmsd_imgs = list(colonmsd_fldr.glob("*"))

litq_test_fldr = Path("/s/xnat_shadow/litq/test/images_ub/")
litq_test_imgs = list(litq_test_fldr.glob("*"))
drli_short_fldr = Path("/s/datasets_bkp/drli_short/images/")
drli_fldr = Path("/s/datasets_bkp/drli/images/")
lidc2_shadow_fldr = Path("/s/xnat_shadow/lidc2/images/")
crc_wxh_img_fn = "/s/xnat_shadow/crc/wxh/images/crc_CRC198_20170718_CAP1p51.nii.gz"
crc_srn_img_fn = "/s/xnat_shadow/crc/srn/images/crc_CRC002_20190415_CAP1p5.nii.gz"

t6_fldr = Path("/s/datasets_bkp/Task06Lung/images")
t6_imgs = list(t6_fldr.glob("*"))
react_fldr = Path("/s/insync/react/sitk/images")
react_imgs = list(react_fldr.glob("*"))
nodesthick_fldr = Path("/s/xnat_shadow/nodesthick/images")
nodesthick_imgs = list(nodesthick_fldr.glob("*"))
bones_fldr = Path("/s/xnat_shadow/bones/images")
bones_imgs = list(bones_fldr.glob("*"))
capestart_fldr = Path("/s/insync/datasets/capestart/nodes_2025/images")
capestart_imgs = list(capestart_fldr.glob("*"))

misc_fldr = Path("/s/xnat_shadow/misc/images")
misc_imgs = list(misc_fldr.glob("*"))
t6_img_fns = t6_imgs[:20]
localiser_labels = [45, 46, 47, 48, 49]
localiser_labels_litsmc = [1]
TSL = TotalSegmenterLabels()
lidc2_fldr = DS.lidc2.folder / "images"
lidc2_imgs = list(lidc2_fldr.glob("*"))

kits_fldr = DS.kits23.folder / "images"
kits_imgs = list(kits_fldr.glob("*"))

# SECTION:-------------------- BACKWARDS-COMPATIBLE ALIASES---------------------------------------------------------------

runs_tot_all = totalseg_runs_all
run_tot_big = totalseg_run_big
proj = totalseg_proj

fldr_crc = crc_fldr
imgs_crc = crc_imgs
fldr_curvas = curvas_fldr
imgs_curvas = curvas_imgs
fldr_lidc = lidc_fldr
imgs_lidc = lidc_imgs
fldr_nodes = nodes_pending_fldr
fldr_nodes2 = nodes_training_fldr
img_nodes = nodes_pending_imgs
img_nodes2 = nodes_training_imgs
fldr_litsmc = liver_fldrs
imgs_litsmc = liver_imgs
fldr_bosniak = bosniak_fldr
imgs_bosniak = bosniak_imgs
fldr_colonmsd = colonmsd_fldr
imgs_colonmsd = colonmsd_imgs
img_fna = str(litq_test_fldr)
fns = str(drli_short_fldr)
img_fldr = lidc2_shadow_fldr
img_fn2 = crc_wxh_img_fn
img_fn3 = crc_srn_img_fn
litq_fldr = str(litq_test_fldr)
litq_imgs = litq_test_imgs
imgs_t6 = t6_imgs
imgs_react = react_imgs
nodes_fldr = nodes_pending_fldr
nodes_fldr_training = nodes_training_fldr
nodes_imgs = nodes_pending_imgs
nodes_imgs_training = nodes_training_imgs
capestart = capestart_imgs
fldr_misc = misc_fldr
imgs_misc = misc_imgs
img_fns = t6_img_fns
imgs_lidc2 = lidc2_imgs

# %%
# nodes_pending_imgs = ["/s/xnat_shadow/nodes/images_pending/nodes_24_20200813_ChestAbdoC1p5SoftTissue.nii.gz"]
