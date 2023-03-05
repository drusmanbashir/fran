
import cc3d
from tqdm.std import tqdm
from fran.callback.neptune import NeptuneManager
from fran.inference.helpers import BoundingBoxes_to_lists
from fran.managers.trainer import create_model_from_conf, load_checkpoint
from fran.inference.inference_base import WholeImagePredictor
import pandas as pd

import SimpleITK as sitk

from fran.utils.helpers import get_train_valid_test_lists_from_json
# %%

if __name__ == "__main__":



    common_paths_filename=os.environ['FRAN_COMMON_PATHS']
    P = Project(project_title="lits"); proj_defaults= P.proj_summary
    train_list,valid_list,test_list = get_train_valid_test_lists_from_json(project_title=proj_defaults.project_title,fold=0, json_fname=proj_defaults.validation_folds_filename,image_folder=proj_defaults.raw_data_folder/"images",ext=".nii.gz")
    mask_files=[ proj_defaults.raw_data_folder/"masks"/test_file.name.replace(".npy",".nii.gz") for test_file in test_list]
    img_files =  [proj_defaults.raw_data_folder/"images"/mask_file.name for mask_file in mask_files]
    run_name  ="kits_637_096"
    Nep = NeptuneManager(proj_defaults)
    Nep.load_run(run_name,param_names = 'default',nep_mode='read-only')
    Nep.run_name = run_name
    config_dict = Nep.config_dict
    model_params = config_dict['model_params']
    dataset_params=  config_dict['dataset_params']

    model = create_model_from_conf(Nep.model_params,out_channels=2)
    load_checkpoint(Nep.checkpoints_folder,model)
# %%


    # P = LearnerManagerFromNeptune(proj_defaults,run_name="KITS-1672")
    # model = create_model_from_conf(Nep.model_params,out_channels_from_dict_or_cell(dest_labels))

    p = WholeImagePredictor(proj_defaults=proj_defaults, patch_size=dataset_params['patch_size'])
    p.load_model(model, model_id=run_name)
    # %%
    results = []
    #note: horseshoe kidney  Path('/s/datasets/raw_database/raw_data/kits21/images/kits21_00005.nii.gz')
# %%
    N= len(img_files)-27
    BB = BoundingBoxes_to_lists()
    for n in tqdm(range(N)):
        p.load_case(img_filename=img_files[n],mask_filename=mask_files[n])
        p.make_prediction(save=True)
        p.create_binary_mask(threshold=0.2)
        pred_filename = p.save_binary_prediction()
        mask_stats = cc3d.statistics(p.mask)
        voxel_counts = mask_stats['voxel_counts']
        centroids = mask_stats['centroids']
        bounding_boxes = mask_stats['bounding_boxes']
        bounding_boxes = BB.encodes(bounding_boxes)
        mask_filename = [case_ for case_ in mask_files if p.case_id in str(case_)][0]
        p.mask_sitk = sitk.ReadImage(str(mask_filename))
        score = p.score_prediction()
        print("Score: {}".format(score))
        results.append({
            'case_id': p.case_id,
            'img_filename': img_files[n],
            'mask_filename': mask_filename,
            'prediction_filename': pred_filename,
            'score0': score[0],
            'score1': score[1],
            'voxel_counts': list(voxel_counts),
            'centroids':list(centroids),
            'bounding_boxes':bounding_boxes
        })
    df_fn =p.output_image_folder / "results.xlsx"
    df = pd.DataFrame.from_dict(results)
    df.to_excel( df_fn,index=False)

# %%
