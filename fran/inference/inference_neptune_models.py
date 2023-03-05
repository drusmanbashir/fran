# %
import os
# from fran.inference.transforms import *
from fran.inference.scoring import compute_dice_fran
from fran.utils.common import *
from fran.inference.inference_raytune_models import ModelFromTuneTrial
from fran.transforms.spatialtransforms import *
# def load_model(file, model, opt, with_opt=True, device=None, strict=True):
#     "Load `model` from `file` along with `opt` (if available, and if `with_opt`)"
#     distrib_barrier()
#     if isinstance(device, int): device = torch.device('cuda', device)
#     elif device is None: device = 'cpu'
#     state = torch.load(file, map_location=device)
#     hasopt = set(state)=={'model', 'opt'}
#     model_state = state['model'] if hasopt else state
#     get_model(model).load_state_dict(model_state, strict=strict)
#     if hasopt and with_opt:
#         try: opt.load_state_dict(state['opt'])
#         except:
#             if with_opt: warn("Could not load the optimizer state.")
#     elif with_opt: warn("Saved filed doesn't contain an optimizer state.")
#

from fran.managers.trainer import *
from fran.managers.tune import *
from fran.inference.inference_base import *
from fran.utils.imageviewers import *

# %%
if __name__ == "__main__":


# %%
    common_paths_filename=os.environ['FRAN_COMMON_PATHS']
    P = Project(project_title="lits"); proj_defaults= P.proj_summary

    configs_excel = ConfigMaker(proj_defaults.configuration_filename,raytune=False).config
# %%
    train_list, valid_list, test_list = get_fold_case_ids(
            fold=configs_excel['metadata']["fold"],
            json_fname=proj_defaults.validation_folds_filename,
        )

    

    mask_files = list((proj_defaults.raw_data_folder/("masks")).glob("*nii*"))
    img_files= list((proj_defaults.raw_data_folder/("images")).glob("*nii*"))
    masks_valid = [filename for filename in mask_files if  get_case_id_from_filename(proj_defaults.project_title, filename) in valid_list]
    imgs_valid =  [proj_defaults.raw_data_folder/"images"/mask_file.name for mask_file in masks_valid]
    imgs_test =  [filename for filename in img_files if  get_case_id_from_filename(proj_defaults.project_title, filename) in test_list]

    masks_train = [filename for filename in mask_files if  get_case_id_from_filename(proj_defaults.project_title, filename) in train_list]
    imgs_train =  [proj_defaults.raw_data_folder/"images"/mask_file.name for mask_file in masks_train]
    imgs_test =  [filename for filename in img_files if  get_case_id_from_filename(proj_defaults.project_title, filename) in test_list]
# %%


    #note: horseshoe kidney  Path('/s/datasets/raw_database/raw_data/kits21/images/kits21_00005.nii.gz')

    run_name_w= "LITS-206" # best trial
    run_name_p  ="LITS-133"

    E = EndToEndPredictor(proj_defaults,run_name_w,run_name_p,use_neptune=True,device=1)

# %%
    n= 16
    img_fn = imgs_train[n]
    mask_fn = img_fn.str_replace("images","masks")
    E.predict(img_fn=img_fn , save_localiser=True)
# %%

    n_classes= 3
    pred_fn = E.output_image_folder/(img_fn.name+".gz")
    pred_sitk,mask_sitk = map(sitk.ReadImage,[pred_fn,mask_fn])
    score = compute_dice_fran(mask_sitk,pred_sitk,n_classes=n_classes)
    print(score)
# %%
    bb1 = E.predictor_w.get_bbox_from_pred()
# %%
    patch_size = [64,160,160]
    patch_overlap = [int(x/2) for x in patch_size]
    p1= PatchPredictor(proj_defaults=proj_defaults, patch_size=patch_size,patch_overlap=patch_overlap, 
                                        stride = [1,1,1], batch_size=4)
    p1.load_model(model_p,model_id=run_name)
# %%

    p1.load_case(img_filename=img_fn,mask_filename=mask_fn,bboxes=bb1)

    p1.make_prediction()
    p1.scores
# %%

# %%
    ImageMaskViewer([E.predictor_w.img_np_orgres.transpose(2,1,0),E.predictor_w.pred_orgres.transpose(2,1,0)])
    ImageMaskViewer([p.img_np_orgres.transpose(2,1,0),p.pred_k_largest.transpose(2,1,0)])
# %%

# %%

    run_name_w = "kits_479_706"

    P = ModelFromTuneTrial(proj_defaults, trial_name=run_name_w,out_channels= 2)
    model_w = P.model
    patch_size_w= make_patch_size(P.params_dict['dataset_params']['patch_dim0'], P.params_dict['dataset_params']['patch_dim1'])

    p = WholeImagePredictor(proj_defaults=proj_defaults, patch_size=patch_size_w,device=device)
    p.load_model(model_w, model_id=run_name_w)
# %%

    imgs = sitk.ReadImage(imgs_valid[n])
    im = E.predictor_w.img_np_orgres
    ImageMaskViewer([im[bb1[0]].transpose(2,1,0),im[bb1[0]].transpose(0,1,2)])
# %%
    N= len(img_files)
    BB = BoundingBoxes_to_lists()
    results = []
# %%
# %% [markdown]
## Getting crude localizing bounding boxes for all validation cases
# %%
    
    for n in tqdm.tqdm(range(1,N)):
        p.load_case(img_filename=img_files[n])
        p.make_prediction(save=True)
        p.save_prediction()
        p.score_prediction(mask_files[n])
        bb = p.get_bbox_from_pred()
        bounding_boxes = BB.encodes(bb)
        score = p.scores

        print("Score: {}".format(p.scores))
        results.append({
            'case_id': p.case_id,
            'img_filename': img_files[n],
            'mask_filename': mask_files[n],
            'prediction_filename': p.pred_fn,
            'score0': score[0,0],
            'score1': score[0,1],
            'bounding_boxes':bounding_boxes
    })
# %%
    df = pd.DataFrame(results)
    df.to_csv(p.output_image_folder/('results.csv'))
    print("Saved results to : {}".format(p.output_image_folder/('results.csv')))
# %%
    B = BBoxesToPatchSize(p.patch_size,p.sz_dest,0.1)
    BB = BBoxesToLists()


# %%

    n=10
    df_fn = '/s/datasets/predictions/kits21/segmentations_kits_134_716_220522_1352/results.xlsx'
    df = pd.read_excel(df_fn)
# %% df=  df.sort_values(by=["score0"],ascending=True) df.columns
    ser = df.iloc[n]
    img_fn =df.iloc[n]['img_filename'] 
    pred_fn =df.iloc[n]['prediction_filename'] 
    voxel_counts= ser['voxel_counts']
    centroids = ser['centroids']
    bb2 = ast.literal_eval(ser['bounding_boxes'])
    print(ser)
    mask_fn= df.iloc[n]['mask_filename']
    bboxes= BB.decodes(bb2)
# %%
    img = sitk_filename_to_numpy(img_fn)
    mask = sitk_filename_to_numpy(mask_fn)
    pred = sitk_filename_to_numpy(pred_fn)
# %%
    img_t , mask_t , pred_t= img.transpose(2,1,0), mask.transpose(2,1,0), pred.transpose(2,1,0)
    ImageMaskViewer([pred_t,mask_t],['mask','mask'],intensity_slider_range_percentile=[0,100])
# %%
    ind=1
    ImageMaskViewer([img[bboxes[ind]].transpose(2,0,1),mask[bboxes[ind]].transpose(2,0,1)])
# %%
    ImageMaskViewer([img[bb1[ind]].transpose(2,0,1),mask[bb1[ind]].transpose(2,0,1)])
# %%
    output_image_subfolder = "tumour_segmentation"

######################################################################################
# %% [markdown]
## Doing patch-based prediction %%

   # df_fn =p.output_image_folder / "results.xlsx"
    # df = pd.DataFrame.from_dict(results)
    # df.to_excel( df_fn,index=False)
    #
# %%
    configs_excel = load_config_from_workbook(proj_defaults.configuration_filename, raytune=False)
    run_name = "KITS-1869"
    # run_name = None
    La = Trainer.fromNeptuneRun(proj_defaults, config_dict=None, run_name=run_name, update_nep_run_from_config=False)
    La.dataset_params['fake_tumours']=False
    La.create_transforms()
    learn = La.create_learner()
    model = learn.model
# %%

# %%
# %%
    inx = 1
    b = p1.bboxes_transformed
    im = p1.img_transformed
    ImageMaskViewer([im[b[inx]],im[b[inx]]])
# %%
    inx = 1
    im_org = p1.img_np_orgres
    b_org =     p1.bboxes
    ImageMaskViewer([im_org[b_org[inx]],im_org[b_org[inx]]])

# %%

# %%
    x =    p1.img_np_orgres,     p1.pred_orgres,p1.pred_final, p1.gt_orgres
    x_t=[]
    for xx in x: 
        xx=xx.transpose(2,1,0)
        x_t.append(xx)
# %%
    ImageMaskViewer([x_t[0],x_t[2]],data_types=['img','mask'])
    ImageMaskViewer([x_t[0],x_t[3]],data_types=['img','mask'])
    ImageMaskViewer([x_t[2],x_t[3]],data_types=['mask','mask'])
# %%
    dl1 = p1.dls[0][0]
    dl1
    iteri = iter(dl1)
    aa = next(iteri)
    im = aa['image']['data']

    agg = p1.dls[0][1]
    ot = agg.get_output_tensor()
    ot = F.softmax(ot,dim=0)
# %%
    ind =0
    ImageMaskViewer([im[0,0],ot[1]],['img','img'])
    ImageMaskViewer([ot[ind],ot[ind+1]],['img','img'])
    
# %%
    ims = []
    for fn in [img_fn, pred_fn]:
        img = sitk.ReadImage(str(fn))
        arr = sitk.GetArrayFromImage(img)
        ims.append(arr)
# %%
# bb = bboxes[1][2],bboxes[1][1],bboxes[1][0]
    i =1
    ImageMaskViewer([ims[0][bboxes[i]].transpose(2,1,0), ims[1][bboxes[i]].transpose(2,1,0)])
# %%
    ImageMaskViewer([ims[0].transpose(2,1,0), ims[1].transpose(2,1,0)])
# %%
    ims = []
    img_fn, pred_fn ='/s/datasets/raw_data/kits21/images/kits21_00008.nii.gz', '/s/datasets/predictions/kits21/segmentations_KITS-1672_150622_1759/kits21_00008.nii.gz' 
    for fn in [img_fn, pred_fn]:
        img = sitk.ReadImage(str(fn))
        arr = sitk.GetArrayFromImage(img)
        ims.append(arr)
        

# %%

