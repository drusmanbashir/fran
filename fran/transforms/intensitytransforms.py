# %%
from functools import wraps
from typing import Mapping, Hashable

from monai.data.meta_obj import get_track_meta
from fastcore.basics import *
from monai.config.type_definitions import DtypeLike, NdarrayOrTensor
from monai.transforms import RandomizableTransform , MapTransform
from monai.transforms.intensity.array import RandGaussianNoise
from monai.utils.type_conversion import convert_to_tensor
from scipy.ndimage.filters import gaussian_filter
from fastcore.transform import Transform
from fran.transforms.base import *


class RandRandGaussianNoised(RandomizableTransform, MapTransform):
    def __init__(
        self,
        keys,
        std_limits,
        prob: float = 1,
        do_transform: bool = True,
        dtype: DtypeLike = np.float32,
    ):
        MapTransform.__init__(self, keys, False)
        RandomizableTransform.__init__(self, prob)
        store_attr("std_limits,dtype")

    def randomize(self):
        super().randomize(None)
        rand_std = self.R.uniform(low=self.std_limits[0], high=self.std_limits[1])
        self.rand_gaussian_noise = RandGaussianNoise(
            mean=0, std=rand_std, prob=1.0, dtype=self.dtype
        )

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        self.randomize()
        if not self._do_transform:
            for key in self.key_iterator(d):
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            return d

        # all the keys share the same random noise
        first_key: Hashable = self.first_key(d)
        if first_key == ():
            for key in self.key_iterator(d):
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            return d

        self.rand_gaussian_noise.randomize(d[first_key])
        for key in self.key_iterator(d):
            d[key] = self.rand_gaussian_noise(img=d[key], randomize=False)
        return d


class MakeBinary(MonaiDictTransform):
    def func(self,x):
        x[x>0]=1
        return x



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

class ClipCenter(KeepBBoxTransform):
    def __init__(self,clip_range,mean,std):
        store_attr()
    def func(self,x):
        img,mask= x
        clip_func = torch.clip if isinstance(img,Tensor) else np.clip # inference uses numpy
        img = clip_func(img,self.clip_range[0],self.clip_range[1])
        img = standardize(img,self.mean,self.std)
        return img,mask

class ClipCenterI(ItemTransform):
    def __init__(self,clip_range,mean,std):
        store_attr()
    def encodes(self,x):
        img,mask= x
        clip_func = torch.clip if isinstance(img,Tensor) else np.clip # inference uses numpy
        img = clip_func(img,self.clip_range[0],self.clip_range[1])
        img = standardize(img,self.mean,self.std)
        return img,mask





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
    
    from fran.data.dataset import *
    
    from fran.utils.common import *
    from fran.utils.helpers import *
    from fran.utils.fileio import *
    from fran.utils.imageviewers import *
    from fran.transforms.spatialtransforms import *
    from matplotlib import pyplot as plt
    P = Project(project_title="litsmc"); proj_defaults= P
# %%



    import torchvision
    model = torchvision.models.detection.ssd300_vgg16(pretrained = True)
    model.eval()
# %%
    fn = Path("/s/fran_storage/datasets/preprocessed/fixed_spacing/litsmc/spc_080_080_150/images/drli_024.pt")
    im = torch.load(fn)
    
    im1 = im.mean(0)
    im1 = im.mean(2)
    im2 =  im.mean(1)
    plt.imshow(im2)


# %%
    x = torch.tensor([1, 2, 3])
    x.repeat(4, 2)
    x.repeat(4, 2, 1).size()
# %%
    im1 = im1.repeat(3,1,1)

    confidence_threshold = 0.8
    pred = model([im1])
    bbox,scores,labels = pred[0]['boxes'],pred[0]['scores'], pred[0]['labels']
    indices = torch.nonzero(scores > confidence_threshold).squeeze(1)
 
    filtered_bbox = bbox[indices]
    filtered_scores = scores[indices]
    filtered_labels = labels[indices]
    
# %%
    import cv2


    def draw_boxes_and_labels(image, bbox, labels ):
        img_copy = image.copy()
     
        for i in range(len(bbox)):
            x, y, w, h = bbox[i].astype('int')
            cv2.rectangle(img_copy, (x, y), (w, h), (0, 0, 255), 5)
     
            class_index = labels[i].numpy().astype('int')
            # class_detected = class_names[class_index - 1]
     
            class_index = str(class_index)
            cv2.putText(img_copy, class_index, (x, y + 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)
     
        return img_copy


# %%

    plt.imshow(img)
    im1 = im1.detach().cpu()
    img = np.array(im1)
    img = np.transpose(img,(1,2,0))
    cv2_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bbox = bbox.detach().cpu().numpy()
    result_img = draw_boxes_and_labels(cv2_image, bbox, labels )
    cv2.imshow('image',result_img)
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
