# %%
from fran.managers.tune import *
from fran.inference.cascade import *




# %%
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
class BoundingBoxes_to_lists(Transform):

    def encodes(self,bounding_boxes):
        slices_as_num=[]
        for bb in bounding_boxes[1:]:
            cc = [[a.start,a.stop] for a in bb]
            slices_as_num.append(cc)
        return slices_as_num
    def decodes(self,bboxes):
        bboxes_out=[]
        for bb in bboxes:
            slices=[]
            for b in bb:
                slc = slice(b[0],b[1])
                slices.append(slc)
            bboxes_out.append(slices)
        return bboxes_out
# %%
if __name__ == "__main__":
    # %%
    results = []
    #note: horseshoe kidney  Path('/s/datasets/raw_database/raw_data/kits21/images/kits21_00005.nii.gz')
# %%
#     N= len(img_files)-29
    BB = BoundingBoxes_to_lists()
#     for n in tqdm.tqdm(range(N)):
#         p.load_case(img_filename=img_files[n])
#         p.make_prediction(save=True)
#         p.create_binary_mask(threshold=0.2)
#         pred_filename = p.save_binary_prediction()
#         mask_stats = cc3d.statistics(p.mask)
#         voxel_counts = mask_stats['voxel_counts']
#         centroids = mask_stats['centroids']
#         bounding_boxes = mask_stats['bounding_boxes']
#         bounding_boxes = BB.encodes(bounding_boxes)
#         mask_filename = [case_ for case_ in mask_files if p.case_id in str(case_)][0]
#         p.mask_sitk = sitk.ReadImage(str(mask_filename))
#         score = p.score_prediction()
#         print("Score: {}".format(score))
#         results.append({
#             'case_id': p.case_id,
#             'img_filename': img_files[n],
#             'mask_filename': mask_filename,
#             'prediction_filename': pred_filename,
#             'score0': score[0],
#             'score1': score[1],
#             'voxel_counts': list(voxel_counts),
#             'centroids':list(centroids),
#             'bounding_boxes':bounding_boxes
#         })
#     df_fn =p.output_image_folder / "results.xlsx"
#     df = pd.DataFrame.from_dict(results)
#     df.to_excel( df_fn,index=False)

# %%

# %%
    df = pd.read_excel(df_fn)
# %%
    df=  df.sort_values(by=["score0"],ascending=True)
    df.columns
# %%
    i = 0
    ser = df.iloc[i]
    img_fn =df.iloc[i]['img_filename'] 
    pred_fn =df.iloc[i]['prediction_filename'] 
    voxel_counts= ser['voxel_counts']
    centroids = ser['centroids']
    bb = ast.literal_eval(ser['bounding_boxes'])
    bboxes= BB.decodes(bb)
# %%
    ims = []
    for fn in [img_fn, pred_fn]:
        img2 = sitk.ReadImage(str(fn))
        arr = sitk.GetArrayFromImage(img2)
        ims.append(arr)
# %%
# bb = bboxes[1][2],bboxes[1][1],bboxes[1][0]
    i =1
    ImageMaskViewer([ims[0][bboxes[i]].transpose(2,1,0), ims[1][bboxes[i]].transpose(2,1,0)])
# %%
    ImageMaskViewer([ims[0].transpose(2,1,0), ims[1].transpose(2,1,0)])
# %%

