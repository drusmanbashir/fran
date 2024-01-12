# %%
from monai.transforms.utility.dictionary import SqueezeDimd
from prompt_toolkit.shortcuts import input_dialog
from fran.inference.cascade import *
from monai.data import ImageDataset

class InferenceDatasetNii(Dataset):
    def __init__(self, imgs,dataset_params):

        self.dataset_params= dataset_params
        self.imgs= self.parse_input(imgs)
        self.create_transforms()



    def __len__(self) -> int:
        return len(self.imgs)


    def parse_input(self,imgs_inp):
        '''
        input types:
            folder of img_fns
            nifti img_fns 
            itk imgs (slicer)
        returns list of img_fns if folder. Otherwise just the imgs
        '''

        if not isinstance(imgs_inp,list): imgs_inp=[imgs_inp]
        imgs_out = []
        for dat in imgs_inp:
            if any([isinstance(dat,str),isinstance(dat,Path)]):
                self.input_type= 'files'
                dat = Path(dat)
                if dat.is_dir():
                    dat = list(dat.glob("*"))
                else:
                    dat=[dat]
            else:
                self.input_type= 'itk'
                if isinstance(dat,sitk.Image):
                    dat= ConvertSimpleItkImageToItkImage(dat, itk.F)
                # if isinstance(dat,itk.Image):
                dat=itm(dat) 
            imgs_out.extend(dat)
        imgs_out = [{'image':img} for img in imgs_out]
        return imgs_out

    def create_transforms(self):
        if self.input_type=='files':
            L=LoadImaged(keys=['image'],image_only=True,ensure_channel_first=False,simple_keys=True)
            tfms =[L]
        else:
            tfms = []

        E = EnsureChannelFirstd(keys=["image"], channel_dim="no_channel")
        S = Spacingd(keys=["image"], pixdim=self.dataset_params['spacings'])
        N = NormaliseClipd(
            keys=["image"],
            clip_range=self.dataset_params["intensity_clip_range"],
            mean=self.dataset_params["mean_fg"],
            std=self.dataset_params["std_fg"],
        )
        tfms += [E,S,N]
        self.transform=Compose(tfms)


    def __getitem__(self, index):
        dici = self.imgs[index]
        dici=self.transform(dici)
        return dici


class SimpleInferer(GetAttr, DictToAttr):
    def __init__(self, project,run_name,bs=8,patch_overlap=.25,mode='gaussian', devices=[1],debug=True):
        '''
        data is a dataset from Ensemble in this base class
        '''

        store_attr('project,run_name,devices,debug')
        self.ckpt = checkpoint_from_model_id(run_name)
        dic1=torch.load(self.ckpt)
        dic2={}
        relevant_keys=['datamodule_hyper_parameters']
        for key in relevant_keys:
            dic2[key]=dic1[key]
            self.assimilate_dict(dic2[key])
    
        self.inferer = SlidingWindowInferer(
            roi_size=self.dataset_params['patch_size'],
            sw_batch_size=bs,
            overlap=patch_overlap,
            mode=mode,
            progress=True,
        )
        self.prepare_model()
        # self.prepare_data(data)

    def run(self,imgs,chunksize=12):
        '''
        chunksize is necessary in large lists to manage system ram
        '''
        imgs  = list_to_chunks(imgs,chunksize)
        for imgs_sublist in imgs:
            self.prepare_data(imgs_sublist)
            self.create_postprocess_transforms()
            preds= self.predict()
            # preds = self.decollate(preds)
            output= self.postprocess(preds)
            # if self.save==True: self.save_pred(output)
        return output



    def prepare_data(self,imgs):
        '''
        imgs: list
        '''

        self.ds = InferenceDatasetNii(imgs,self.dataset_params)
        self.pred_dl = DataLoader(
                self.ds, num_workers=0, batch_size=1, collate_fn = None
            )


    def save_pred(self,preds):
        S = SaveImaged(keys = ['pred'],output_dir=self.output_folder,output_postfix='',separate_folder=False)
        for pp in preds:
            S(pp)

    def create_postprocess_transforms(self):

        Sq = SqueezeDimd(keys = ['pred'], dim=0)
        I = Invertd(keys=['pred'],transform=self.ds.transform,orig_keys=['image'])
        D = AsDiscreted(keys=['pred'],argmax=True,threshold=0.5)
        C = ToCPUd(keys=['image','pred'])
        tfms = [Sq,I,D,C]
        if self.debug==True:
            Sa = SaveImaged(keys = ['pred'],output_dir=self.output_folder,output_postfix='',separate_folder=False)
            tfms.insert(1,Sa)
        C = Compose(tfms)
        self.postprocess_transforms=C

    def prepare_model(self):
        model = UNetTrainer.load_from_checkpoint(
            self.ckpt, project=self.project, dataset_params=self.dataset_params, strict=False
        )

        fabric = Fabric(precision="16-mixed",devices=self.devices)
        self.model=fabric.setup(model)

    def predict(self):
        outputs = []
        for i ,batch in enumerate(self.pred_dl):
                with torch.no_grad():
                    img_input=batch['image']
                    img_input = img_input.cuda()
                    output_tensor = self.inferer(inputs=img_input, network=self.model)
                    output_tensor = output_tensor[0]
                    batch['pred']=output_tensor
                    batch['pred'].meta = batch['image'].meta
                    outputs.append(batch)
        return outputs


    def postprocess(self, preds):
        out_final=[]
        for batch in preds:
            tmp=self.postprocess_transforms(batch) 
            out_final.append(tmp)
        return out_final

    @property
    def output_folder(self):
        run_name = listify(self.run_name)
        fldr='_'.join(run_name)
        fldr = self.project.predictions_folder/fldr
        return fldr


# %%

if __name__ == "__main__":
    # ... run your application ...
    proj= Project(project_title="nodes")



    run_ps=['LITS-702']
    run_name = run_ps[0]

# %%
    img_fn = "/s/xnat_shadow/nodes/imgs_no_mask/nodes_4_20201024_CAP1p5mm_thick.nii.gz"

    img_fns = [img_fn]
    input_data = [{'image':im_fn} for im_fn in img_fns]
    debug = True


# %%
    P=SimpleInferer(proj, run_ps[0], debug=debug)

    preds= P.run(img_fns)
# %%
    imgs = img_fns
    P.prepare_data(imgs)
    P.create_postprocess_transforms()
    preds= P.predict()
# %%
    a = P.ds[0]
    im = a['image']
    im = im[0]
    ImageMaskViewer([im,im])
# %%
    # preds = P.decollate(preds)
    # output= P.postprocess(preds)
# %%

    out_final=[]
    # for batch in preds:

    batch= preds[0]

    C = ToCPUd(keys=['image','pred'])
    Sq = SqueezeDimd(keys = ['pred'], dim=0)
    batch = C(batch)
    batch = Sq(batch)
    batch['pred'].shape

    I = Invertd(keys=['pred'],transform=P.ds.transform,orig_keys=['image'])
    tmp = I(batch)
    tmp=P.postprocess_transforms(batch) 
    out_final.append(tmp)
# %%
    data = P.ds[0]

    P.ds.transform
    P.ds.transform.inverse(data)
# %%
    I = Invertd(keys=['pred'],transform=P.ds.transform,orig_keys=['image'])
    I = Invertd(keys=['image'],transform=P.ds.transform,orig_keys=['image'])
    pp = preds[0].copy()
    print(pp.keys())
    pp['pred']=pp['pred'][0:1,0]

    pp['pred'].shape
    pp['pred'].meta
    a =  I(pp)

    dici = {'image': img_fn}
    L=LoadImaged(keys=['image'],image_only=True,ensure_channel_first=True,simple_keys=True)

    S = Spacingd(keys=["image"], pixdim=P.ds.dataset_params['spacings'])
    tfms = ([L,S])
    Co = Compose(tfms)

    dd = L(dici)
    dda = S(dd)

# %%
    dd = Co(dici)
# %%
    Co.inverse(dd)
# %%
