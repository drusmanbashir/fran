# %%
import shutil

import torch
import time
import SimpleITK as sitk
import re
from pathlib import Path

from label_analysis.helpers import get_labels
from fran.managers.data import find_matching_fn, pbar
from fran.utils.fileio import maybe_makedirs
from fran.utils.imageviewers import ImageMaskViewer
# %%
if __name__ == "__main__":
    fldr = Path("/s/xnat_shadow/crc/lms/")
    img_fldr = Path("/s/xnat_shadow/crc/images/")
    lm_fns = list(fldr.glob("*"))

    out_fldr_img = Path("/s/crc_upload/images")
    out_fldr_lm = Path("/s/crc_upload/lms")
    maybe_makedirs([out_fldr_lm,out_fldr_img])
# %%
    im_fn = Path("/r/datasets/preprocessed/nodes/lbd/spc_080_080_150/lms/nodesthick_13_20230322_NCAPC_thick_label-Segment_1-label.pt")
    im_fn2 = Path("/r/datasets/preprocessed/nodes/lbd/spc_080_080_150/lms/nodesthick_10_20200713_Body1p0CE_thick.pt")
    im = torch.load(im_fn)
    lm = torch.load(im_fn2)


    ImageMaskViewer([im,lm])
    lm_fn = lm_fns[0]
    im_fns = list(img_fldr.glob("*"))
    im_fn= im_fns[0]
# %%
    for lm_fn in pbar(lm_fns):

        lm = sitk.ReadImage(str(lm_fn))
        labs = get_labels(lm)
        print(labs)

# %%
        new_filename = re.sub(r'_\d{8}_', '_', im_fn.name)
        out_lm_fname = out_fldr_lm / new_filename
        out_img_fname = out_fldr_img / new_filename
        shutil.copy(im_fn, out_img_fname)
        if not ".nii.gz" in lm_fn.name:
            lm = sitk.ReadImage(str(lm_fn))
            sitk.WriteImage(lm, out_lm_fname)
        else:
            shutil.copy(lm_fn, out_lm_fname)
# %%
        lm = sitk.ReadImage(str(lm_fn))
        labels = get_labels(lm)
        if not labels == [1]:
            lm= to_binary(lm)

# %%

    im_fn = Path("/s/fran_storage/datasets/preprocessed/fixed_spacing/nodes/spc_080_080_150/images/nodes_66_410921_ChestAbdomenPelviswithIVC1p00Br40S3.pt")
    lm_fn = Path("/s/fran_storage/datasets/preprocessed/fixed_spacing/nodes/spc_080_080_150/lms/nodes_66_410921_ChestAbdomenPelviswithIVC1p00Br40S3.pt")

    im = torch.load(im_fn)
    lm = torch.load(lm_fn)
    ImageMaskViewer([im,lm], 'im')


    lm = sitk.ReadImage(str(lm_fn))
    get_labels(lm)
    lm = relabel(lm,{1:2})
    sitk.WriteImage(lm,lm_fn)
    img_fn = Path("/r/datasets/preprocessed/litsmc/lbd/spc_080_080_150_liver_only/images/drli_001ub.pt")


    img = torch.load(img_fn)
    lm = torch.load(lm_fn)
    ImageMaskViewer([img,lm])

# %%


    register_writer("pt", TorchWriter)
    dim= 1
    L = LoadSITKd(keys=['image','lm'])
    N = NormalizeIntensityd(['image'])
    E = EnsureChannelFirstd(['image', 'lm'])
    P1 = Project2D(keys = ['lm', 'image'], operations = ['sum','mean'],dim=1, output_keys=['lm1','image1'])
    P2 = Project2D(keys = ['lm', 'image'], operations = ['sum','mean'],dim=2, output_keys=['lm2','image2'])
    P3 = Project2D(keys = ['lm', 'image'], operations = ['sum','mean'],dim=3, output_keys=['lm3','image3'])
    BB1 = BoundingRectd(keys = ['lm1'])
    BB2 = BoundingRectd(keys = ['lm2'])
    BB3 = BoundingRectd(keys = ['lm3'])
    M = MetaToDict(keys = ['image'], meta_keys = ['filename_or_obj'])
    B1= BoundingBoxYOLOd(['lm1_bbox'],2,key_template_tensor='lm1')
    B2= BoundingBoxYOLOd(['lm2_bbox'],2, key_template_tensor='lm2')
    B3= BoundingBoxYOLOd(['lm3_bbox'],2, key_template_tensor='lm3')
    D1 = DictToMeta(keys=['image1'], meta_keys=['lm1_bbox'])
    D2 = DictToMeta(keys=['image2'], meta_keys=['lm2_bbox'])
    D3 = DictToMeta(keys=['image3'], meta_keys=['lm3_bbox'])


    tfms = Compose([L,N,E,P1,P2,P3,BB1,BB2,BB3, B1,B2,B3,D1,D2,D3])

# %%
    fldr_imgs = Path("/s/xnat_shadow/lidc2/images/")
    fldr_lms = Path("/s/fran_storage/predictions/totalseg/LITS-827/")
    lms = list(fldr_lms.glob("*"))
    imgs = list(fldr_imgs.glob("*"))[:20]

# %%
    data_dicts= []
    for img in imgs:
        dici = {'image':img, 'lm':  find_matching_fn(img,lms,tag='all')}
        data_dicts.append(dici)



# %%

    ds = Dataset(data=data_dicts, transform=tfms)

    dl = DataLoader(ds, batch_size=2, num_workers=0, collate_fn = as_is_collated)
    ii = iter(dl)
    aa = next(ii)
# %%
    S1 = SaveImage(output_ext='pt',  output_dir='tmp', output_postfix=str(1), output_dtype='float32', writer=TorchWriter)
    S2 = SaveImage(output_ext='pt',  output_dir='tmp', output_postfix=str(2), output_dtype='float32', writer=TorchWriter)
    S3 = SaveImage(output_ext='pt',  output_dir='tmp', output_postfix=str(3), output_dtype='float32', writer=TorchWriter)
# %%
    images1 = aa['image1']
    images2 = aa['image2']
    images3 = aa['image3']


# %%
    for img in images1:
        S1(img)

    for img in images2:
        S1(img)

    for img in images3:
        S1(img)
# %%


    dici =data_dicts[0]
    dici = L(dici)
    # dici = M(dici)


    dici = E(dici)
    dici = N(dici)
    dici = P1(dici)
    dici = P2(dici)
    dici = P3(dici)
# %%
    dici = BB1(dici)
    dici = BB2(dici)
    dici = BB3(dici)
# %%

    dici = B1(dici)
    dici = B2(dici)
    dici = B3(dici)
    dici = D1(dici)
    dici = D2(dici)
    dici = D3(dici)
# %%

    d2 = dici['lm_bbox'].copy()
    dici = B(dici)


# %%
# %%

    lmv = dici['lm'][0]
    lmv = torch.permute(lmv,(1,0))
    imv = dici['image'][0]
    imv = torch.permute(imv,(1,0))
# %%
    box_converter = ConvertBoxMode(src_mode="xxyy", dst_mode="ccwh")
    box_converter(dici['lm_bbox'])

# %%

# convert boxes with format [xmin, ymin, xmax, ymax] to [xcenter, ycenter, xsize, ysize].
    bc2 = ConvertBoxMode(src_mode= "xxyy", dst_mode= "xywh")
    bb_im = bc2(dici['lm_bbox'])
    bb_im = bb_im.flatten()
    box_converter(bb2)

    dici = B(dici)

    lm = dici['lm']
    lm2 = lm.sum(0)
    lmv = torch.permute(lm2,(1,0))
    lmv.unsqueeze_(0)
    lm3 = lm2.unsqueeze(0)



# %%
    fig,ax = plt.subplots()
    ax.imshow(lmv)
    ax.imshow(imv)

    rect = patches.Rectangle((bb_im[0],bb_im[1]), bb_im[2],bb_im[3],linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect)

# %%
# %%
    lm = sitk.ReadImage(str(lm_fn))



    fns = list(fldr.glob("*"))
    fns = [fn for fn in fns if is_sitk_file(fn)]
    fil = sitk.LabelShapeStatisticsImageFilter()
    dicis = []
    for fn in fns:
        lm = sitk.ReadImage(str(fn))
        n_nodes = get_labels(lm)
        n_nodes = len(n_nodes)
        lmb = to_binary(lm)
        fil.Execute(lmb)
        bbox = fil.GetBoundingBox(1)
        bbs = [a+b for a,b in zip(bbox[:3],bbox[3:])]
        tot_size = lm.GetSize()
        spacing = lm.GetSpacing()
        # L = LabelMapGeometry(lm)
        sz = print(lm.GetSize())
        dici = {'fn': fn , 'node':n_nodes, 'spacing':spacing,'bbox':bbox}
        dicis.append(dici)
    df = pd.DataFrame(dicis)
    df.to_csv(fldr.parent/("info.csv"))
# %%
# %%
#SECTION:-------------------- FIXING--------------------------------------------------------------------------------------

    fn = '/s/fran_storage/checkpoints/litsmc/litsmc/LITS-999/checkpoints/epoch=106-val_loss=0.78.ckpt'
    ckp = torch.load(fn)
    print(ckp.keys())
    conf['plan'] =ckp['datamodule_hyper_parameters']['plan']
    ckp['datamodule_hyper_parameters']['plan']
    ckp['datamodule_hyper_parameters'].pop('plan')
    ckp['datamodule_hyper_parameters']['config'] = conf
    torch.save(ckp,fn)
# %%

# %%
    src_fn  =Path('/s/fran_storage/datasets/preprocessed/fixed_spacing/nodes/spc_080_080_150/lms/nodes_70_20210804_ChestAbdomenPelviswithIVC1p00Hr40S3.pt')

    lm = torch.load(src_fn)
    mask_fnames = Path('/r/datasets/preprocessed/nodes/lbd/spc_080_080_150/lms')
    fn2 = find_matching_fn(fn,fldr1)
    fn = "/s/fran_storage/datasets/preprocessed/fixed_spacing/litsmc/spc_080_080_150/indices/drli_017.pt"

    lm = torch.load(fn)
# %%
    fldr = Path("/s/xnat_shadow/litq/images")
    fns = fldr.glob("*")
    for fn in fns: compress_img(fn)

# %%
    fn = "/s/xnat_shadow/crc/lms/crc_CRC133_20130102_ABDOMEN.nii.gz"
    dici = {'lm':fn}
    L = LoadSITKd(keys=['lm'])
    dici = L(dici)
    lm = dici['lm']

# %%

    fn = "/s/xnat_shadow/crc/lms/crc_CRC133_20130102_ABDOMEN.nii.gz"
    dici = {'lm':fn}
    L = LoadSITKd(keys=['lm'])
    dici = L(dici)
    lm = dici['lm']
# %%

# %%
    def mini(lm,remapping):
        lm_sitk = sitk.GetImageFromArray(lm)
        lm_sitk = relabel(lm_sitk, remapping)
        lm_np = sitk.GetArrayFromImage(lm_sitk)
        lm_pt = torch.tensor(lm_np)
        lm_out= MetaTensor(lm_pt)
        lm_out.copy_meta_from(lm) 
        return lm_out

    def mini2( lm, remapping):
            lm_sitk = sitk.GetImageFromArray(lm)
            lm_sitk = relabel(lm_sitk, remapping)
            lm_np = sitk.GetArrayFromImage(lm_sitk)
            lm_pt = torch.tensor(lm_np)
            lm_out= MetaTensor(lm_pt)
            lm_out.copy_meta_from(lm) 
            return lm_out

# %%
    sta = time.time()
    for i in range(30):
        x = mini2(lm,remapping)

    en =time.time()
    take = en-sta
    print(take)
# %%

    sta = time.time()
    for i in range(30):
        x = mini(lm,remapping)

    en =time.time()
    take = en-sta
    print(take)
# %%
    bbox_fn = Path("/r/datasets/preprocessed/litsmc/patches/spc_080_080_150_plan5/dim_320_320_192/bboxes_info.pkl")
    bboxes = load_dict(bbox_fn)
    bboxes_out = []
# %%
for bbox in bboxes:

    fn = bbox['filename']  
    fn = fn.str_replace("plan3","plan4")
    bbox['filename'] = fn
    bboxes_out.append(bbox)
    save_pickle(bbox_fn,bboxes_out)
# %%
    imgs_fldr = Path("/s/fran_storage/predictions/litsmc/LITS-933")
    img = [fn for fn in imgs_fldr.glob("*") if "CRC164" in fn.name][0]
    lm = sitk.ReadImage(str(img))

    L = LabelMapGeometry(lm, ignore_labels=[2,3])



    view_sitk(lm,L.lm_cc)

    fil.Execute(lm)
    bbox = fil.GetBoundingBox(1)
    bbs = [a+b for a,b in zip(bbox[:3],bbox[3:])]
    print(bbs)




    out_fldr = Path("/s/xnat_shadow/crc/sbh/lms")

    imgs = list(imgs_fldr.glob("*"))
    cids2 = [info_from_filename(fn.name)['case_id'] for fn in imgs]
    moves = [cid not in cids for cid in cids2]
    barts  =list(il.compress(imgs,moves))

# %%
    for fn in barts:
        fn_out = out_fldr/fn.name
        shutil.move(fn,fn_out)
# %%
    # fn = "/home/ub/Dropbox/AIscreening/data/metadata_published.xlsx"
    ind = next(fk)

    ind = 0

    fn = "/s/datasets_bkp/totalseg/meta.xlsx"
    df = pd.read_excel(fn,sheet_name="labels")
    df.location_localiser



    row  = df.iloc[ind]

# %%
    localisers_done = []
    labels_short=[]
    fk = fk_generator(0)
    for row in df.iterrows():
        lab_loc = row[1]['location_localiser']
        if not lab_loc in localisers_done:
            ind = next(fk)
            localisers_done.append(lab_loc)
        else:
            ind = localisers_done.index(lab_loc)

        labels_short.append(ind)

# %%

    df['labels_short']= labels_short
    df.to_excel("/s/datasets_bkp/totalseg/meta2.xlsx", index=False)
# %%
    fldr  = Path("/s/xnat_shadow/crc/lms")

    fns = list(fldr.glob("*"))
# %%
    fns_1=[]
    for fn in pbar(fns):
        lm = sitk.ReadImage(str(fn))
        lbs = get_labels(lm)
        print(lbs)
        if 1 in lbs:
            fns_1.append(fn)
# %%
    df = pd.DataFrame(fns_1)

    df.to_csv()


    df = pd.read_csv("gt_fns_with_label1.csv")
# %%
    import ast
    for row in pbar(df.iterrows()):
        fn = row[1].iloc[1]
        lm = sitk.ReadImage(str(fn))
        # remapping = ast.literal_eval(row[1].iloc[2])
        print(fn)
        print(get_labels(lm))
        # lm = relabel(lm,remapping)


# %%


# %%
    fns_fin = []
    for cid in cids:
        fn = [fn for fn in fns if cid in fn.name]
        if len(fn)>1:
            tr()
        else:
            fns_fin.append(  fn[0])

# %%

    bkp_fldr = Path("/s/xnat_shadow/crc/lms_staging/")
    fns_fin = list(bkp_fldr.glob("*"))
    for fn in fns_fin:
        print(fn)
        lm = sitk.ReadImage(str(fn))
        print(get_labels(lm))
        lm = relabel(lm,{1:2,2:3})
        print(get_labels(lm))
        # lm = relabel(lm,{1:0})
        sitk.WriteImage(lm,str(fn))
        print("------------"*10)
# %%

    fn_df = "/s/fran_storage/predictions/litsmc/LITS-933_fixed_mc/results/summary_LITS-933.xlsx"
    df2 = pd.read_excel(fn_df)


    fldr_lms = "/s/fran_storage/predictions/nodes/LITS-702/"
    fldr_out = Path("/s/xnat_shadow/nodes/capestart/lms")
    preds = list(Path(fldr_lms).glob("*"))
    preds = [fn for fn in preds if not "_1.nii" in fn.name]



# %%
    fn2 = "/s/xnat_shadow/crc/srn/lms_slicer_fixed/crc_CRC284_20160527_CAP1p5Soft.nii.gz-Segment_2-label_2.nrrd"
    fn = "/s/xnat_shadow/crc/lms_staging/crc_CRC198_20170718_CAP1p51.nii.gz"
    lm = sitk.ReadImage(fn)
    get_labels(lm)
# %%
    dici ={}
    for lab in range(1,7):
        dici.update({lab:3})
        
# %%
    lm = relabel(lm,dici)
    sitk.WriteImage(lm,fn)
# %%

    lm.GetSize()
    lm.GetOrigin()
# %%
    fn2 = "/s/xnat_shadow/crc/srn/lms_slicer_fixed/crc_CRC284_20160527_CAP1p5Soft.nii.gz-Segment_2-label_2.nrrd"
    lm2 = sitk.ReadImage(fn2)

    get_labels(lm)

    lm2.GetSize()
    lm2.GetOrigin()

# %%
    for i, row in df.iterrows():

        fn_org = Path(row.fn_org)
        fn_out = Path(row.fn_out)
        fn_out_name = Path(fn_out).name
        fn = find_matching_fn(fn_org.name,preds)
        fn_out_full = fldr_out/fn_out_name
        shutil.copy(fn,fn_out_full)

# %%

    flr = Path("/s/xnat_shadow/crc/srn/lms")
    flr_wxh =Path("/s/xnat_shadow/crc/wxh/lms_manual_final") 
    lm_wxh = list(flr.glob("*"))
    lms_done = list(flr.glob("*"))
    lms_done = [fn.name for fn in lms_done]
    imgs_all = Path("/s/xnat_shadow/crc/srn/images/")
    imgs_ex= Path("/s/xnat_shadow/crc/srn/excluded/images/")
    imgs_all = list(imgs_all.glob("*"))
# %%
    for img in imgs_all:
        if info_from_filename(img.name,True)["case_id"] in excluded:
            img_neo = img.str_replace("images","excluded/images")
            shutil.move(img,img_neo)
# %%

    imgs_processed =Path("/s/xnat_shadow/crc/srn/cases_with_findings/images/")
    imgs_processed = list(imgs_processed.glob("*"))
    imgs_processed = [fn.name for fn in imgs_processed]
    df = pd.read_excel("/s/xnat_shadow/crc/srn/srn_summary_latest.xlsx", sheet_name="A")
    excluded = df.loc[df['labels']=="exclude", "case_id"].to_list()
# %%

    for img in imgs_all:
        if img.name  in imgs_processed:
            os.remove(img)
# %%
    imgs_new = Path("/s/xnat_shadow/crc/srn/images_done/")
    cids = list(set([info_from_filename(fn.name,True)["case_id"] for fn in lm_wxh]))
    imgs= list(imgs_all.glob("*"))

    done = [im for im in lms_done if info_from_filename(im.name,True)['case_id'] in cids]
    in_wxh = [fn for fn in done if fn.name in lms_done] 
    imgs_fldr =Path("/s/xnat_shadow/crc/images_more/images")
    masks_fldr = Path("/s/xnat_shadow/crc/images_more/segs")
# %%
    imgs = list(imgs_fldr.glob("*"))
    masks = list(masks_fldr.glob("*"))
    for fn in masks:
        find_matching_fn(fn,imgs)

# %%
    for fn in in_wxh:
        os.remove(fn)
# %%
    for img in done: 
        img_neo = img.str_replace("images","images_done")
        shutil.move(img,img_neo)
# %%
    fldr = Path("/s/fran_storage/checkpoints/totalseg/totalseg/LITS-836")

    fn = "/r/datasets/preprocessed/nodes/lbd/spc_080_080_150/images/nodesthick_110_20190508_CAP1p5_thick.pt"
    fn2 = "/r/datasets/preprocessed/nodes/lbd/spc_080_080_150/lms/nodesthick_110_20190508_CAP1p5_thick.pt"
    im = torch.load(fn)
    im2= torch.load(fn2)
    im = torch.permute(im,[2,1,0])
    im2 = torch.permute(im2,[2,1,0])
    ImageMaskViewer([im,im2],dtypes=['image','mask'])


    def add_dataset_params_key(fn,key,val):
            ckp =torch.load(fn)
            if not key in ckp['datamodule_hyper_parameters']['dataset_params'].keys():
                print ("Key '{}' not present. Adding".format(key))
                ckp['datamodule_hyper_parameters']['dataset_params'][key] = val
                torch.save(ckp,fn)
# %%
    fldr = Path("/r/datasets/preprocessed/litsmc/lbd/spc_080_080_150/")
    fls = list(fldr.rglob("*.pt"))
# %%
    for fn in fls:
            pt = torch.load(fn)
            print(pt.shape)
# %%
            
    ckpt = "/s/fran_storage/checkpoints/lits32/Untitled/LIT-145/checkpoints/epoch=198-step=2189.ckpt"

    sd = torch.load(ckpt)
    sd['datamodule_hyper_parameters']['dataset_params']['spacing'] =    sd['datamodule_hyper_parameters']['dataset_params']['spacings'] 
    torch.save(sd,ckpt)
# %%

    file  = QFile ("/home/ub/code/qt/regexbrowser/assets/UB_all_CT_abdo_short.txt");
    fn = "/s/fran_storage/datasets/preprocessed/fixed_spacing/litsmc/spc_080_080_150/images/drli_057.pt"
    tt = torch.load(fn)
    fn2 = "/s/fran_storage/datasets/preprocessed/fixed_spacing/lidc2/spc_080_080_150/lms/lidc2_0001b.pt"
    fn = "/r/datasets/preprocessed/litsmc/lbd/spc_080_080_150/images/drli_020.pt"
    fn2 = "/r/datasets/preprocessed/litsmc/lbd/spc_080_080_150/lms/drli_020.pt"
# %%

    tb_dir ="/r/datasets/preprocessed/litsmc/lbd/spc_080_080_150/thumbnails"
    wr=SummaryWriter(log_dir = tb_dir)
# %%
    fldr = Path("/r/datasets/preprocessed/litsmc/lbd/spc_080_080_150/")
    lms_fldr = fldr/("lms")
    lms = list(lms_fldr.glob("*"))
    imgs =list((fldr/("images")).glob("*"))
    S = ScaleIntensity()
    for img_fn in pbar(imgs[:50]):
        spatial_size = [int(sz/4) for sz in img.shape]
        Re = Resize(spatial_size=spatial_size)
        lm_fn = find_matching_fn(img_fn,lms)
        img = torch.load(img_fn)
        lm = torch.load(lm_fn)
        img[lm==0]=0
        img2 = img.unsqueeze(0)
        lm = lm.unsqueeze(0)
        img3 = Re(img2,mode='trilinear')
        img3= S(img3)
        lm = Re(lm)
        img_tag = str(img_fn)
        lm_tag = str(lm_fn)
        img2tensorboard.add_animated_gif(writer=wr,image_tensor=img3,tag=img_tag,scale_factor=500)
        # img2tensorboard.add_animated_gif(writer=wr,image_tensor=lm,tag=lm_tag,scale_factor=1)

# %%
    ImageMaskViewer([img2[0],lm[0]])
# %%


    N = NormalizeIntensity()
    tt = torch.load(fn2)
    tt= tt.unsqueeze(0)
    fn = "/r/datasets/preprocessed/litsmc/lbd/spc_080_080_150/lms/litq_15_20190809.pt"


    tt= torch.load(fn)

    tt.meta

    # %load_ext tensorboard
    # %tensorboard --logdir=$tb_dir

    ImageMaskViewer([tt,tt])
    fn2 = "/home/ub/datasets/preprocessed/tmp/lbd/spc_080_080_150/images/lidc2_0001.pt"
    t2 = torch.load(fn2)
    ImageMaskViewer([t2[0],tt[0]])
    fn = "/s/fran_storage/projects/nodes/raw_dataset_properties.pkl"
    dici = load_dict(fn)
    dici[10]
    h5fn = "/s/xnat_shadow/nodes/fg_voxels.h5"
    h5f_file = h5py.File(h5fn, 'r')
    h5f_file.keys()
        for fn in h5f_file[cid]:

                    cs = h5f_file[cid]
    fl = h5py.File(fn, "r")
    aa= fl['litqsmall_00008']
    # file  = QFile ("/home/ub/code/qt/regexbrowser/assets/sample.csv");
# %%
    aa = 10
    print(aa)
    file.open(QFile.OpenModeFlag.ReadOnly | QFile.OpenModeFlag.Text);
    aa =file.readLine()
    aa = str(aa)
    print(aa)
    file.close()

# %%
    import matplotlib.pyplot as plt
    ind = 1

    pred = torch.load("pred_prefix.pt")
    plt.imshow(im[ind,0])
    plt.imshow(pred[ind,0])
    im= torch.load("image.pt")
    im = im.cpu()
    pred = torch.load("pred.pt")
    pred =pred[0].float()
    pred = pred.cpu()
    ind = 2
    pred = F.softmax(pred,dim=1)
    ImageMaskViewer([im[ind,0],pred[ind,1]])
# %%
# %%

    bb  = aa.split("\";\"")
    print(bb)
    print(len(bb))
# %%

    file.readData(100)
# %%
    fn = "/s/xnat_shadow/crc/srn/cases_with_findings/preds_fixed/crc_CRC138_20180812_Abdomen3p0I30f3.nii.gz-label.nrrd"
    img_fn = sitk.ReadImage(fn)
    view_sitk(img_fn,img_fn)
# %%

