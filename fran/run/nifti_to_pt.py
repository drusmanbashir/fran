# %%
import pandas as pd
from label_analysis.totalseg import TotalSegmenterLabels
from fran.preprocessing.fixed_spacing import ResampleDatasetniftiToTorch
from fran.preprocessing.imported import LabelBoundedDataGeneratorImported
from fran.utils.config_parsers import ConfigMaker


if __name__ == '__main__':
    from fran.utils.common import *

    from fran.managers import Project
    project = Project("litsmc")

# %%
#SECTION:-------------------- SETUP--------------------------------------------------------------------------------------


    from fran.utils.common import *
    from fran.managers import Project

    P = Project(project_title="litsmc")
    spacing = [0.8, 0.8, 1.5]
    P.maybe_store_projectwide_properties()

    conf = ConfigMaker(P, raytune=False, configuration_filename=None).config
    plan = conf["plan_valid"]
    nifti_fldr = "/s/xnat_shadow/crc/sampling/nifti"
    fixed_fldr ="/s/xnat_shadow/crc/sampling/tensors/fixed_spacing/"
    lbd_fldr ="/s/xnat_shadow/crc/sampling/tensors/lbd"
    fn_results = "/s/fran_storage/predictions/litsmc/LITS-933_fixed_mc/results/summary_LITS-933.xlsx"
    df_res = pd.read_excel(fn_results)
    overwrite=False
# %%
#SECTION:-------------------- ResampleDatasetniftiToTorch--------------------------------------------------------------------------------------
    Rs = ResampleDatasetniftiToTorch(project, spacing=[0.8, 0.8, 1.5], data_folder=nifti_fldr,output_folder=fixed_fldr)
    Rs.setup(overwrite=overwrite)
    Rs.process()
# %%
    if not "labels_all" in P.global_properties.keys():
        P.set_lm_groups(plan["lm_groups"])
        P.maybe_store_projectwide_properties(overwrite=False)
    lm_group = P.global_properties["lm_group1"]
    TSL = TotalSegmenterLabels()
    imported_folder = "/s/fran_storage/predictions/totalseg/LITS-1088"
    imported_labelsets = lm_group["imported_labelsets"]
    imported_labelsets = [TSL.get_labels("liver",localiser=True)]
    remapping = TSL.create_remapping(
        imported_labelsets,
        [
            1,
        ]
        * len(imported_labelsets),
        localiser=True,
    )
    merge_imported_labels = True
# %%

    L = LabelBoundedDataGeneratorImported(
        project=P,
        data_folder=fixed_fldr,
        output_folder=lbd_fldr,
        plan=plan,
        imported_folder=imported_folder,
        merge_imported_labels=merge_imported_labels,
        remapping=remapping,
        folder_suffix=plan['plan_name'],
    )
    L.setup(overwrite=overwrite)
    L.process()
# %%
