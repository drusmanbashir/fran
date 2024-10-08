# %%
import torch

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RangeSlider
import numpy as np
from torch.functional import Tensor

import SimpleITK as sitk

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RangeSlider, CheckButtons
import ipywidgets 
plt.ion()
import ipdb
tr = ipdb.set_trace
# %%

import matplotlib as mpl
from matplotlib import pyplot as plt
def discrete_cmap():
    cmap = plt.cm.jet  # define the colormap
# extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
# force the first color entry to be grey
    cmaplist[0] = (.0, .0, .0, 1.0)
# create the new map
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'Custom cmap', cmaplist, cmap.N)

    bounds = np.linspace(0, 255, 256)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm
def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)
import numpy as np
x = np.random.rand(20)  # define the data
y = np.random.rand(20)  # define the data
tag = np.random.randint(0, 20, 20)

def fix_labels(x):
            if x.GetPixelID()==22:
                x = sitk.Cast(x,sitk.sitkUInt8)
            return x
            

def view_sitk(img,mask,dtypes='im',**kwargs):
    img = fix_labels(img)
    mask = fix_labels(mask)
    img,mask=list(map(sitk.GetArrayFromImage,[img,mask]))
    ImageMaskViewer([img,mask],dtypes,**kwargs)


def view_3d_np(x):
    ImageMaskViewer([np.expand_dims(x[0],0), np.expand_dims(x[1],0)])
def view_5d_torch(x,n=0):
    ImageMaskViewer([x[0][n,0,:].cpu().detach().numpy(),x[1][n,0,:].cpu().detach().numpy()])

def view(a,b,n=0, cmap_img='Greys_r', cmap_mask='RdPu_r'):
    '''
    4d or 5d input possible, tensor or numpy array
    '''
    
    if  len(a.shape)>4: # i.e., has batch etc
            a,b = [x[n] for x in [a,b]]
    arrs = []
    for arr in a,b:
        if isinstance(arr,Tensor):
            arr = arr.cpu().detach().numpy()
            arrs.append(arr)
        else:
            arrs.append(arr)
    ImageMaskViewer(arrs,
                     # nn.Sigmoid()(b[n, 0, :]).cpu().detach().numpy()],
                    cmap_img=cmap_img,
                    cmap_mask=cmap_mask)



def get_window_level_numpy_array(
    image_list,
    intensity_slider_range_percentile=[2, 98],
    dtypes='im'
):
    # to the original images. If they are deleted outside the view would become
    # invalid, so we use a copy wich guarentees that the gui is consistent.
    if isinstance(image_list[0] ,np.ndarray):  # if images are already np_array..
        npa_list = image_list
    elif isinstance(image_list[0], torch.Tensor):
        image_list = [a.detach().cpu() for a in image_list]
        npa_list = [image_list[0].numpy(), image_list[1].numpy()]

    else:
        npa_list = list(map(sitk.GetArrayFromImage, image_list))

    wl_range = []
    wl_init = []
    # We need to iterate over the images because they can be a mix of
    # grayscale and color images. If they are color we set the wl_rangenotebooks/nbs/gui_building.sync.py
    # to [0,255] and the wl_init is equal, ignoring the window_level_list
    # entry.
    for npa, data_type in zip(npa_list, dtypes):
        if data_type == "i":
            min_max = np.percentile(npa.flatten(),
                                    intensity_slider_range_percentile)
        else:
            min_max = [npa.min(), npa.max()]
        wl_range.append((min_max[0], min_max[1]))
        wl_init.append(wl_range[-1])
    return (npa_list, wl_range, wl_init)


# %%

# %%
# %%

class ImageMaskViewer(object):
    '''
    expects numpy  depth x width x height
    '''
    
    def __init__(self,
                 image_list,
                 dtypes="im",
                 figure_size=(10, 8),
                 intensity_slider_range_percentile=[2, 98],
                 cmap_img= 'Greys_r',
                 cmap_mask = None,
                 apply_transpose=True) ->None:
        self.cmap_img ,self.cmap_mask, self.dtypes= cmap_img, cmap_mask, dtypes

        self.npa_list, self.wl_range, self.wl_init = get_window_level_numpy_array(
            image_list, intensity_slider_range_percentile, self.dtypes)

        # if apply_transpose==True :
        #     self.npa_list= [a.transpose(2,1,0) for a in self.npa_list]
        self.fig, self.axises = plt.subplots(1, 2, figsize=figure_size)
        self.axamp = plt.axes([0.1, 0.05, 0.8, 0.03])
        self.axamp_wl = plt.axes([0.1, 0.0, 0.8, 0.03])
        self.slider = Slider(ax=self.axamp,
                             label="image",
                             valmin=0,
                             valmax=self.npa_list[0].shape[0] - 1,
                             valstep=1)
        self.slider_wl = RangeSlider(ax=self.axamp_wl,
                                     label="Window level",
                                     valmin=self.wl_range[0][0],
                                     valmax=self.wl_range[0][1])
        self.ax_imgs = self.create_images()
        self.slider.on_changed(self.update_fig_fast)
        self.slider_wl.on_changed(lambda vals: self.ax_imgs[0].set_clim(*vals))
        plt.tight_layout()

    def create_images(self):
        ax_imgs=[]
        for ax, img, data_type in zip(self.axises, self.npa_list,
                                      self.dtypes):
            img_slice = img[0,:,:]
            if data_type == 'm':
                if not self.cmap_mask:
                    N = img.max()
                    # cmap = discrete_cmap(N,'cubehelix')
                    # ax_imgs.append(ax.imshow(img_slice,cmap=cmap))
                    cmap_mask ='nipy_spectral'
                    ax_imgs.append(ax.imshow(img_slice,cmap=cmap_mask,vmin=img.min(),vmax=img.max() ))
                else:
                    ax_imgs.append(ax.imshow(img_slice,cmap=self.cmap_mask,vmin=img.min(),vmax=img.max() ))

            else:
                ax_imgs.append(ax.imshow(img_slice,
                          cmap=self.cmap_img,
                          vmin=self.slider_wl.val[0],
                          vmax=self.slider_wl.val[1]))
        return ax_imgs

    def update_fig_fast(self,val):
        for i, img in enumerate(self.npa_list):
            img_slice = img[val,:,:]
            self.ax_imgs[i].set_array(img_slice)

class ImageMaskViewer(object):
    '''
    expects numpy  depth x width x height
    '''
    
    def __init__(self,
                 image_list,
                 dtypes="im",
                 figure_size=(10, 8),
                 intensity_slider_range_percentile=[2, 98],
                 cmap_img= 'Greys_r',
                 cmap_mask = None,
                 apply_transpose=True) ->None:
        self.cmap_img ,self.cmap_mask, self.dtypes= cmap_img, cmap_mask, dtypes

        self.npa_list, self.wl_range, self.wl_init = get_window_level_numpy_array(
            image_list, intensity_slider_range_percentile, self.dtypes)

        # if apply_transpose==True :
        #     self.npa_list= [a.transpose(2,1,0) for a in self.npa_list]
        self.fig, self.axises = plt.subplots(1, 2, figsize=figure_size)
        self.axamp = plt.axes([0.1, 0.05, 0.8, 0.03])
        self.axamp_wl = plt.axes([0.1, 0.0, 0.8, 0.03])
        self.slider = Slider(ax=self.axamp,
                             label="image",
                             valmin=0,
                             valmax=self.npa_list[0].shape[0] - 1,
                             valstep=1)
        self.slider_wl = RangeSlider(ax=self.axamp_wl,
                                     label="Window level",
                                     valmin=self.wl_range[0][0],
                                     valmax=self.wl_range[0][1])
        self.ax_imgs = self.create_images()
        self.slider.on_changed(self.update_fig_fast)
        self.slider_wl.on_changed(lambda vals: self.ax_imgs[0].set_clim(*vals))
        plt.tight_layout()

    def create_images(self):
        ax_imgs=[]
        for ax, img, data_type in zip(self.axises, self.npa_list,
                                      self.dtypes):
            img_slice = img[0,:,:]
            if data_type == 'm':
                if not self.cmap_mask:
                    N = img.max()
                    # cmap = discrete_cmap(N,'cubehelix')
                    # ax_imgs.append(ax.imshow(img_slice,cmap=cmap))
                    cmap_mask ='nipy_spectral'
                    ax_imgs.append(ax.imshow(img_slice,cmap=cmap_mask,vmin=img.min(),vmax=img.max() ))
                else:
                    ax_imgs.append(ax.imshow(img_slice,cmap=self.cmap_mask,vmin=img.min(),vmax=img.max() ))

            else:
                ax_imgs.append(ax.imshow(img_slice,
                          cmap=self.cmap_img,
                          vmin=self.slider_wl.val[0],
                          vmax=self.slider_wl.val[1]))
        return ax_imgs

    def update_fig_fast(self,val):
        for i, img in enumerate(self.npa_list):
            img_slice = img[val,:,:]
            self.ax_imgs[i].set_array(img_slice)


class ImageMaskViewer:
    '''
    Expects images as a list of numpy arrays, each with dimensions [depth, width, height].
    '''
    
    def __init__(self,
                 image_list,
                 dtypes="im",
                 figure_size=(10, 8),
                 intensity_slider_range_percentile=[2, 98],
                 cmap_img='Greys_r',
                 cmap_mask=None,
                 apply_transpose=True) -> None:
                     
        self.cmap_img = cmap_img
        self.cmap_mask = cmap_mask
        self.dtypes = dtypes

        # Process images and obtain window level range and initial values
        self.npa_list, self.wl_range, self.wl_init = get_window_level_numpy_array(
            image_list, intensity_slider_range_percentile, self.dtypes)

        # Optionally transpose if needed
        if apply_transpose:
            self.npa_list = [img.transpose(2, 1, 0) for img in self.npa_list]

        # Create figure and axes
        self.fig, self.axes = plt.subplots(1, len(self.npa_list), figsize=figure_size)

        # Ensure self.axes is iterable
        if len(self.npa_list) == 1:
            self.axes = [self.axes]

        # Create slice slider
        self.slice_slider_ax = plt.axes([0.1, 0.01, 0.8, 0.03])
        self.slice_slider = Slider(self.slice_slider_ax, "Slice", 0, self.npa_list[0].shape[0] - 1, valinit=0, valstep=1)
        self.slice_slider.on_changed(self.update_fig_fast)

        # Create specific window level sliders
        self.wl_sliders = []
        for i, img in enumerate(self.npa_list):
            ax_wl = plt.axes([0.1, 0.15 + i * 0.05, 0.8, 0.03])
            slider_wl = RangeSlider(ax_wl, f"Win lev {i}",
                                    valmin=self.wl_range[i][0],
                                    valmax=self.wl_range[i][1],
                                    valinit=(self.wl_init[i][0], self.wl_init[i][1]))
            slider_wl.on_changed(lambda vals, index=i: self.update_windowing(index, vals))
            self.wl_sliders.append(slider_wl)

        plt.tight_layout()
        self.ax_imgs = self.create_images()

    def create_images(self):
        # Initializes and draw images on each corresponding axis
        ax_imgs = []
        for ax, img, data_type, wl in zip(self.axes, self.npa_list, self.dtypes, self.wl_init):
            img_slice = img[0, :, :]
            if data_type == 'm':
                cmap = self.cmap_mask if self.cmap_mask else 'nipy_spectral'
                ax_img = ax.imshow(img_slice, cmap=cmap, vmin=img.min(), vmax=img.max())
            else:
                ax_img = ax.imshow(img_slice, cmap=self.cmap_img, vmin=wl[0], vmax=wl[1])
            ax_imgs.append(ax_img)
        return ax_imgs

    def update_fig_fast(self, val):
        # Updates image slices based on slice slider value
        val_index = int(val)
        for i, img_slice in enumerate(self.npa_list):
            new_slice = img_slice[val_index, :, :]
            self.ax_imgs[i].set_array(new_slice)
        self.fig.canvas.draw_idle()

    def update_windowing(self, index, wl_vals):
        # Update colormap limits based on the window level slider(s)
        vmin, vmax = wl_vals
        self.ax_imgs[index].set_clim(vmin, vmax)
        self.fig.canvas.draw_idle()


# %%
class ImageMaskViewer_J():
    '''
    viewer for jupyter
    expects images of type ndarray strictly
    '''
    
    def __init__(self, f):
        self.arg1 = 110
        self.arg2 = 15
        self.f = f

    def __call__(self,*args,**kwargs):

        self.image_list, self.data_types= self.f(*args,**kwargs)
        self.image_list, self.wl_range, self.wl_init = get_window_level_numpy_array(
            self.image_list,[2,98],  self.data_types)

        horizontal =True
        figure_size=(10, 8)
        # Create a figure.
        col_num, row_num = (len(self.image_list), 1) if horizontal else (1, len(self.image_list))
        self.fig, self.axes = plt.subplots(row_num, col_num, figsize=figure_size)
        if len(self.image_list) == 1:
            self.axes = [self.axes]
        self.fig, self.axises = plt.subplots(1, 2, figsize=figure_size)

        self.cmap_img = 'Greys_r'
        self.cmap_mask='Greys_r'
        self.create_ui()
        self.update_fig(0)



    def create_ui(self):
        self.axamp = plt.axes([0.1, 0.05, 0.8, 0.03])
        self.axamp_wl = plt.axes([0.1, 0.0, 0.8, 0.03])
        self.slider =ipywidgets.IntSlider(ax=self.axamp,
                             label="image",
                             valmin=0,
                             valmax=self.image_list[0].shape[1] - 1,
                             valstep=1)
        self.slider_wl = ipywidgets.IntRangeSlider(ax=self.axamp_wl,
                                     label="Window level",
                                     valmin=self.wl_range[0][0],
                                     valmax=self.wl_range[0][1])
        self.slider.observe(self.update_fig)
        self.slider_wl.on_changed(self.change_fig_windowing)
        plt.tight_layout()

    def update_fig(self, val):
        for ax, img, data_type in zip(self.axises, self.image_list,
                                      self.data_types):
            img_slice = img[:,val,:].transpose(1,2,0)
            if data_type == 'mask':
                ax.imshow(img_slice,cmap=self.cmap_mask)

            else:
                ax.imshow(img_slice,
                          cmap=self.cmap_img,
                          vmin=self.slider_wl.val[0],
                          vmax=self.slider_wl.val[1])

    def change_fig_windowing(self, wl_range):
        tr()
        self.axises[0].imshow(self.image_list[0][self.slider.val, :],
                              cmap=self.cmap_img,
                              vmin=wl_range[0],
                              vmax=wl_range[1])

class ImageMaskViewer:
    '''
    Expects a list of image stacks (numpy arrays), each of shape [depth, width, height].
    '''
    
    def __init__(self, image_list, 
                 dtypes="im",
                 figure_size=(10, 8), 
                 intensity_slider_range_percentile=[2, 98],
                 cmap_img='Greys_r',
                 cmap_mask=None,
                 apply_transpose=True,
                 sync=True) -> None:
        
        self.cmap_img = cmap_img
        self.cmap_mask = cmap_mask
        self.dtypes = dtypes
        self.sync = sync
        self.image_list, self.wl_range, self.wl_init = get_window_level_numpy_array(
            image_list, intensity_slider_range_percentile, dtypes)

        if apply_transpose:
            self.image_list = [img.transpose(2, 1, 0) for img in self.image_list]

        # Create figure and axes
        self.fig, self.axes = plt.subplots(1, len(self.image_list), figsize=figure_size)
        if len(self.image_list) == 1:
            self.axes = [self.axes]

        self.ax_imgs = self.create_images()
        
        if sync:
            # Single slider for synchronized view, extent determined by max depth
            max_slices = max(img.shape[0] for img in self.image_list)
            ax_slider = plt.axes([0.2, 0.01, 0.6, 0.03])
            self.sync_slider = Slider(ax_slider, 'Slice', 0, max_slices - 1, valinit=0, valstep=1)
            self.sync_slider.on_changed(self.update_fig_fast)
        else:
            # Independent sliders and window level sliders
            self.sliders = []
            self.wl_sliders = []
            for i, (ax, img, wl) in enumerate(zip(self.axes, self.image_list, self.wl_init)):
                ax_slider = plt.axes([ax.get_position().x0, ax.get_position().y0 - 0.05, ax.get_position().width, 0.02])
                slider = Slider(ax_slider, '', 0, img.shape[0] - 1, valinit=0, valstep=1)
                slider.on_changed(self.update_individual_fig(img, slider, i))
                self.sliders.append(slider)
                
                ax_wl_slider = plt.axes([ax.get_position().x0, ax.get_position().y0 - 0.1, ax.get_position().width, 0.02])
                wl_slider = RangeSlider(ax_wl_slider, f'WL {i}', valmin=self.wl_range[i][0], valmax=self.wl_range[i][1], valinit=wl)
                wl_slider.on_changed(self.update_windowing(i))
                self.wl_sliders.append(wl_slider)
        
        self.fig.tight_layout()

    def create_images(self):
        # Initialization and display
        ax_imgs = []
        for ax, img, data_type, wl in zip(self.axes, self.image_list, self.dtypes, self.wl_init):
            img_slice = img[0, :, :]
            cmap = self.cmap_mask if data_type == 'm' and self.cmap_mask else self.cmap_img
            ax_img = ax.imshow(img_slice, cmap=cmap, vmin=wl[0], vmax=wl[1])
            ax_imgs.append(ax_img)
        return ax_imgs

    def update_individual_fig(self, img, slider, index):
        # Updates the slice based on the individual slider
        def _update(val):
            val_idx = int(round(slider.val))
            new_slice = img[val_idx, :, :]
            self.ax_imgs[index].set_array(new_slice)
            self.fig.canvas.draw_idle()
        return _update

    def update_fig_fast(self, val):
        # Updates all the images to the slice of the single slider
        val_idx = int(round(val))
        for img, ax_img in zip(self.image_list, self.ax_imgs):
            slice_idx = min(val_idx, img.shape[0] - 1)
            new_slice = img[slice_idx, :, :]
            ax_img.set_array(new_slice)
        self.fig.canvas.draw_idle()

    def update_windowing(self, index):
        # Update window level based on the window level slider
        def _update(vals):
            self.ax_imgs[index].set_clim(vals)
            self.fig.canvas.draw_idle()
        return _update


# %%


# %%
if __name__ =="__main__":

    i = np.load('toydata/image.npy')
    m = np.load('toydata/preds.npy')[0]
    I = ImageMaskViewer([i,m])
# %%
    I = ImageMaskViewer([i2,m2],intensity_slider_range_percentile=[0,100])
# %%
    image_list, data_types= [i2,m2],["image","mask"]
    image_list, wl_range, wl_init = get_window_level_numpy_array(
        image_list,[2,98],  data_types)

# %%
    horizontal =True
    figure_size=(10, 8)
    # Create a figure.
    col_num, row_num = (len(image_list), 1) if horizontal else (1, len(image_list))

    # viewer expects 4d input
    viewer(np.expand_dims(i,0),m)
# %%
