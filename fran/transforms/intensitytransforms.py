# %%
from functools import wraps
from fastai.basics import *
from scipy.ndimage.filters import gaussian_filter
from fastcore.transform import Pipeline, Transform
from fran.transforms.basetransforms import *

class _IntensityAugmentation():

    def __init__(self, aug_func):
        self.aug_func = aug_func
    def __call__(self,x,factor_range):
            factor = np.random.uniform(*factor_range)
            return self.aug_func(x[0],factor),x[1]


#reversible transforms
class IntensityNorm(Transform):
    def __init__(self,zero_center=True):
        '''
        params: zero_center = True, returns (x-mean)/std, False, returns x in range [0,1]
        '''
        self.zero_center=zero_center
    def encodes(self,img):
        if self.zero_center==True:
            self.mean= img.mean()
            self.std = img.std()
            img = (img-self.mean)/self.std
            return img
        else:
            self.min = img.min()
            self.range = img.max()-self.min
            if self.min<0:
                img= img -self.min
            img= img/(self.range+1e-5)
            return img
    def decodes(self,img):
        if self.zero_center==True:
            img = self.std*(img+self.mean)
        else:
            img = img*self.range+self.min
        return img
            

# %%
def zero_center(func):
    @wraps(func)
    def _inner(img,*args,**kwargs):
          mean= img.mean()
          std = img.std()
          img = (img-mean)/std
          img = func(img,*args,**kwargs)
          return img
    return _inner

def zero_to_one(func):
    @wraps(func)
    def _inner(img,*args,**kwargs):
        min = img.min()
        range = img.max()-min
        if min<0:
            img= img -min
        img= img/(range+1e-5)
        img = func(img,*args,**kwargs)
        return img
    return _inner

class ClipCenter(ItemTransform):
    def __init__(self,clip_range,mean,std):
        store_attr()
    def encodes(self,x):
        img,anything = x
        clip_func = torch.clip if isinstance(img,Tensor) else np.clip # inference uses numpy
        img = clip_func(img,self.clip_range[0],self.clip_range[1])
        img = standardize(img,self.mean,self.std)
        return img,anything
    def decodes (self,x):return x   # clipping cannot be reversed



# %%


@_IntensityAugmentation
@zero_to_one
def log(img,factor): return np.log(img+1e-5)*factor

@_IntensityAugmentation
@zero_to_one
def power(img,factor): return img**factor

@_IntensityAugmentation
def noise(img,scale):
     scale=np.abs(scale) # has to be non_negative
     noise = torch.normal(mean=0,std= scale,size= img.shape)
     return img+noise

@_IntensityAugmentation
def brightness(img,factor): return img*factor

@_IntensityAugmentation
def invert(img,factor): return -img + factor

@_IntensityAugmentation
def shift(img,factor): return img+factor

@_IntensityAugmentation
def contrast(img,factor): 
     clip_min ,clip_max = img.min(),img.max()
     img = img*factor
     img = torch.clip(img,clip_min,clip_max)
     return img

@_IntensityAugmentation
def gaussian_blur (img,sigma): return  gaussian_filter(img, sigma =sigma)

def clip_image(img,clip_range):
        img_clipped = np.copy(img)
        img_clipped[img>=clip_range[1]]=clip_range[1]
        img_clipped[img<clip_range[0]]=clip_range[0]
        return img_clipped


def standardize(img,mn,std):
        img= (img-mn)/std
        return img




# %%
if __name__ == "__main__":
    import os
    
    if 'get_ipython' in globals():
    
        from IPython import get_ipython
        ipython = get_ipython()
        ipython.run_line_magic('load_ext', 'autoreload')
        ipython.run_line_magic('autoreload', '2')
    from fran.data.dataset import *
    
    from fran.utils.helpers import *
    from fran.utils.fileio import *
    from fran.utils.imageviewers import *
    from fran.transforms.spatialtransforms import *
    P = Project(project_title="lits"); proj_defaults= P.proj_summary
    P = Project(project_title="lits"); proj_defaults= P.proj_summary
    bboxes_19 = proj_defaults_kits19.bboxes_info_filename
    bboxes_21= proj_defaults.bboxes_voxels_info_filename
    train_list,valid_list = get_train_valid_test_lists_from_json(project_title=proj_defaults.project_title,fold=0, json_fname="experiments/kits21/metadata/validation_folds.pkl")
    train_ds = ImageMaskBBoxDataset(proj_defaults,train_list,[bboxes_21],[0,1,2])
# %%
    mask_fn = Path('/home/ub/datasets/preprocessed/kits21/masks/kits21_00088.npy')
    img_fn = Path('/home/ub/datasets/preprocessed/kits21/images/kits21_00088.npy')
    bb = load_dict(bboxes_21)
    bboxes = [b for b in bb if b['filename'] == mask_fn][0]
    patch_size = [64, 256, 256]
# %%
    x ,y,_= train_ds[0]
# # %%
    xx,yy = invert([x,y],factor_range=[0.6,1.2])
    xx,yy = contrast([x,y],factor_range=[0,1])
    plt.hist(x.flatten())
    plt.hist(xx.flatten())
# %%
    b = partial(brightness,factor_range=[1.2,1.3])
    c = partial(brightness,factor_range=[1.2,1.3])
    xx,yy = b([x,y])


# %%
#
#     A = AffineTrainingTransform3D(0.99,pi/8)
#     a,b = A.encodes([xx,yy])
#     C = CropBatch(patch_size)
#     aa,bb= C.encodes([a,b])
#     # ImageMaskViewer([a[n,0],b[n,0]])
#
# # %%
#     xxx,yyy = P2([xx,yy])
# # %%
#     n=0
#     ImageMaskViewer([xxx[n,0],yyy[n,0]])
#     ImageMaskViewer([x,y])
# # %%
#     xl,_ = power_transform([x,y],scale=1)
#     ImageMaskViewer([x,xl],cmap_mask="Greys_r")
#     xx, yy = T.encodes([x,y])
#
#     ImageMaskViewer([x,xx],cmap_mask="Greys_r")

# %%
