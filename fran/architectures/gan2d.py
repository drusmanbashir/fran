# %%
from fastai.vision.all import *
from fastai.vision.gan import *
from fastai.basics import *
from fran.data.dataset import *
from fran.architectures.gan import *

from fran.utils.fileio import *

from torch.utils.data import DataLoader as DLT
from fran.architectures.gan import *

class GANLoss2D(GANModule):
    "Wrapper around `crit_loss_func` and `gen_loss_func`"
    def __init__(self,
        gen_loss_func:callable, # Generator loss function
        crit_loss_func:callable, # Critic loss function
        gan_model:GANModule # The GAN model
    ):
        super().__init__()
        store_attr('gen_loss_func,crit_loss_func,gan_model')

    def generator(self,
        output, # Generator outputs
        target # Real images
    ):
        "Evaluate the `output` with the critic then uses `self.gen_loss_func` to evaluate how well the critic was fooled by `output`"
        output = torch.cat([target[:,:-1,:],output],1)
        fake_pred = self.gan_model.critic(output)

        self.gen_loss = self.gen_loss_func(fake_pred, output, target)
        return self.gen_loss

    def critic(self,
        real_pred, # Critic predictions for real images
        input # Input noise vector to pass into generator
    ):
        "Create some `fake_pred` with the generator from `input` and compare them to `real_pred` in `self.crit_loss_func`."
        fake = self.gan_model.generator(input).requires_grad_(False)
        mask = input[:,:4,:]
        fake = torch.cat([mask,fake],1)
        fake_pred = self.gan_model.critic(fake)
        self.crit_loss = self.crit_loss_func(real_pred, fake_pred)
        return self.crit_loss

@delegates()
class GANLearner2D(Learner):
    "A `Learner` suitable for GANs."
    def __init__(self,
        dls:DataLoaders, # DataLoaders object for GAN data
        generator:nn.Module, # Generator model
        critic:nn.Module, # Critic model
        gen_loss_func:callable, # Generator loss function
        crit_loss_func:callable, # Critic loss function
        switcher:Callback|None=None, # Callback for switching between generator and critic training, defaults to `FixedGANSwitcher`
        gen_first:bool=False, # Whether we start with generator training
        switch_eval:bool=True, # Whether the model should be set to eval mode when calculating loss
        show_img:bool=True, # Whether to show example generated images during training
        clip:None|float=None, # How much to clip the weights
        cbs:Callback|None|list=None, # Additional callbacks
        metrics=None,
        **kwargs
    ):
        gan = GANModule(generator, critic)
        loss_func = GANLoss(gen_loss_func, crit_loss_func, gan)
        if switcher is None: switcher = FixedGANSwitcher()
        trainer = GANTrainer(clip=clip, switch_eval=switch_eval, gen_first=gen_first, show_img=show_img)
        cbs = L(cbs) + L(trainer, switcher)
        metrics = L(metrics) + L(*LossMetrics('gen_loss,crit_loss'))
        super().__init__(dls, gan, loss_func=loss_func, cbs=cbs, metrics=metrics, **kwargs)

    @classmethod
    def from_learners(cls,
        gen_learn:Learner, # A `Learner` object that contains the generator
        crit_learn:Learner, # A `Learner` object that contains the critic
        switcher:Callback|None=None, # Callback for switching between generator and critic training, defaults to `FixedGANSwitcher`
        weights_gen:None|list|tuple=None, # Weights for the generator and critic loss function
        **kwargs
    ):
        "Create a GAN from `learn_gen` and `learn_crit`."
        losses = gan_loss_from_func(gen_learn.loss_func, crit_learn.loss_func, weights_gen=weights_gen)
        return cls(gen_learn.dls, gen_learn.model, crit_learn.model, *losses, switcher=switcher, **kwargs)

    @classmethod
    def wgan(cls,
        dls:DataLoaders, # DataLoaders object for GAN data
        generator:nn.Module, # Generator model
        critic:nn.Module, # Critic model
        switcher:Callback|None=None, # Callback for switching between generator and critic training, defaults to `FixedGANSwitcher(n_crit=5, n_gen=1)`
        clip:None|float=0.01, # How much to clip the weights
        switch_eval:bool=False, # Whether the model should be set to eval mode when calculating loss
        **kwargs
    ):
        "Create a [WGAN](https://arxiv.org/abs/1701.07875) from `dls`, `generator` and `critic`."
        if switcher is None: switcher = FixedGANSwitcher(n_crit=5, n_gen=1)
        return cls(dls, generator, critic, _tk_mean, _tk_diff, switcher=switcher, clip=clip, switch_eval=switch_eval, **kwargs)

GANLearner.from_learners = delegates(to=GANLearner.__init__)(GANLearner.from_learners)
GANLearner.wgan = delegates(to=GANLearner.__init__)(GANLearner.wgan)

class GeneratorLearner(nn.Module):
    def __init__(self,netG):
        super().__init__()
        self.netG= netG
    def forward(self,input):
        mask= input[:,:-1,:]
        fakeImage = self.netG.forward(mask)
        return torch.cat([mask,fakeImage],1)
#
# @delegates()
# class GANLearner2D(Learner):
#     "A `Learner` suitable for GANs."
#     def __init__(self,
#         dls:DataLoaders, # DataLoaders object for GAN data
#         generator:nn.Module, # Generator model
#         critic:nn.Module, # Critic model
#         gen_loss_func:callable, # Generator loss function
#         crit_loss_func:callable, # Critic loss function
#         switcher:Callback|None=None, # Callback for switching between generator and critic training, defaults to `FixedGANSwitcher`
#         gen_first:bool=False, # Whether we start with generator training
#         switch_eval:bool=True, # Whether the model should be set to eval mode when calculating loss
#         show_img:bool=True, # Whether to show example generated images during training
#         clip:None|float=None, # How much to clip the weights
#         cbs:Callback|None|list=None, # Additional callbacks
#         # metrics:None|list|callable=None, # Metrics
#         metrics:None|list=None, # Metrics
#         **kwargs
#     ):
#         gan = GANModule(generator, critic)
#         loss_func = GANLoss2D(gen_loss_func, crit_loss_func, gan)
#         if switcher is None: switcher = FixedGANSwitcher()
#         trainer = GANTrainer(clip=clip, switch_eval=switch_eval, gen_first=gen_first, show_img=show_img)
#         cbs = L(cbs) + L(trainer, switcher)
#         metrics = L(metrics) + L(*LossMetrics('gen_loss,crit_loss'))
#         super().__init__(dls, gan, loss_func=loss_func, cbs=cbs, metrics=metrics, **kwargs)
#
#     @classmethod
#     def from_learners(cls,
#         gen_learn:Learner, # A `Learner` object that contains the generator
#         crit_learn:Learner, # A `Learner` object that contains the critic
#         switcher:Callback|None=None, # Callback for switching between generator and critic training, defaults to `FixedGANSwitcher`
#         weights_gen:None|list|tuple=None, # Weights for the generator and critic loss function
#         **kwargs
#     ):
#         "Create a GAN from `learn_gen` and `learn_crit`."
#         losses = gan_loss_from_func(gen_learn.loss_func, crit_learn.loss_func, weights_gen=weights_gen)
#         return cls(gen_learn.dls, gen_learn.model, crit_learn.model, *losses, switcher=switcher, **kwargs)
#
#     @classmethod
#     def wgan(cls,
#         dls:DataLoaders, # DataLoaders object for GAN data
#         generator:nn.Module, # Generator model
#         critic:nn.Module, # Critic model
#         switcher:Callback|None=None, # Callback for switching between generator and critic training, defaults to `FixedGANSwitcher(n_crit=5, n_gen=1)`
#         clip:None|float=0.01, # How much to clip the weights
#         switch_eval:bool=False, # Whether the model should be set to eval mode when calculating loss
#         **kwargs
#     ):
#         "Create a [WGAN](https://arxiv.org/abs/1701.07875) from `dls`, `generator` and `critic`."
#         if switcher is None: switcher = FixedGANSwitcher(n_crit=5, n_gen=1)
#         return cls(dls, generator, critic, _tk_mean, _tk_diff, switcher=switcher, clip=clip, switch_eval=switch_eval, **kwargs)
#

# %%
if __name__ == "__main__":
    common_paths_filename=os.environ['FRAN_COMMON_PATHS']
    P = Project(project_title="lits"); proj_defaults= P.proj_summary
    images_folder = proj_defaults.stage1_folder/("128_128_128")
    bboxes_fn =images_folder/"bboxes_info"  
    bboxes = load_dict(bboxes_fn)
    fold = 0

    json_fname=proj_defaults.validation_folds_filename

    imgs =list((proj_defaults.raw_data_folder/("images")).glob("*"))
    masks =list((proj_defaults.raw_data_folder/("masks")).glob("*"))
    img_fn = imgs[0]
    mask_fn = masks[0]
    train_ids,val_ids,_ =  get_fold_case_ids(project_title=proj_defaults.project_title,fold=0, json_fname=proj_defaults.validation_folds_filename)
    train_ds = ImageMaskBBoxDataset(proj_defaults,train_ids, bboxes_fn,,[0,1,2])
    valid_ds = ImageMaskBBoxDataset(proj_defaults,val_ids, bboxes_fn,,[0,1,2])


    images_folder = proj_defaults.stage2_folder/("64_160_160")
    bboxes_fn =images_folder/"bboxes_info"  
    bb = load_dict(bboxes_fn)
    bb[0]
    train_ds = ImageMaskBBoxDatasetWrapper(train_ids, bboxes_fn,)
    valid_ds = ImageMaskBBoxDatasetWrapper(val_ids, bboxes_fn,)
    train_ds[0]
# %%
    sampler = tio.data.UniformSampler([1,128,128])
    qs = [tio.Queue(
        subjects_dataset = ds,
        max_length= 256,
        samples_per_volume = 10,
        sampler=sampler,
        num_workers=0
    ) for ds in [train_ds, valid_ds]]
# %%
    class DictToTensors(ItemTransform):
        def encodes(self,x):
            img, mask= x['A']['data'], x['B']['data']

            return img,mask
    
        

    class OneHot2DMask(ItemTransform):
        def encodes(self,x):
            n_classes=4
            img,mask = x
            mask = one_hot(mask,n_classes, axis = 1 ,fnc=torch.cat)
            return img,mask

    class Pix2PixFormat(ItemTransform):
        def encodes(self,x):
            img,mask =x
            return {'A':img, 'B':mask}

    class BToA (ItemTransform):
        def encodes(self,x):
            return x[1],x[0]


    @Transform
    def squeeze_dim2(x):  
        
        return x.squeeze(2)

    @Transform
    def to_float(x):  
        if not x.dtype==torch.float32: 
            x = x.to(torch.float32)
        return x

    after_batch_t = Pipeline([DictToTensors,
                                 OneHot2DMask,
                                 squeeze_dim2,
                                 to_float,
                                 cGANBatchMaker,
                                 # Pix2PixFormat
                             ])

# %%

    dls = DataLoaders(*[DataLoaderForTIO(qq,batch_size=256,num_workers=24, after_batch = after_batch_t) for qq in qs] ) # 0 num_workers required by Queue

# %% [markdown]
## FASTAI workflow
# %%

# %%
#      y =   model.netG(x[1])
# %%


    # iteri  = iter(dls)
    # a = next(iteri)
    # dls = DataLoaders([dls,_])
# %%

# %%
    import argparse

    parser = argparse.ArgumentParser()

    cuda = get_available_device()
    parser.add_argument('-config', help="configuration file *.yml", type=str, required=False, default='config.yml')
    parser.add_argument('-args', help="learning rate", type=bool, required=False, default=False)
# training parameters
    parser.add_argument('-epochs', help="num of epochs for train", type=int, required=False, default=100)
    parser.add_argument('-lr', help="learning rate", type=float, required=False, default=0.00005)
    parser.add_argument('-batch_size', help="batch size", type=int, required=False, default=64)

    parser.add_argument('-dataroot', type = str, default="/s/tmp", help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
    parser.add_argument('-name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
    parser.add_argument('--gpu_ids', type=str, default=str(cuda), help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--checkpoints_dir', type=str, default='/s/checkpoints/pix2pix', help='models are saved here')
    # model parameters
    parser.add_argument('--model', type=str, default='pix2pix', help='chooses which model to use. [cycle_gan | pix2pix | test | colorization]')
    parser.add_argument('--input_nc', type=int, default=4, help='# of input image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--output_nc', type=int, default=1, help='# of output image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
    parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
    parser.add_argument('--netD', type=str, default='n_layers', help='specify discriminator architecture [basic | n_layers | pixel]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
    parser.add_argument('--netG', type=str, default='unet_128', help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')
    parser.add_argument('--n_layers_D', type=int, default=2, help='only used if netD==n_layers')
    parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization [instance | batch | none]')
    parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
    parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
    parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
    parser.add_argument('--isTrain', action='store_false')
    # dataset parameters
    parser.add_argument('--dataset_mode', type=str, default='unaligned', help='chooses how datasets are loaded. [unaligned | aligned | single | colorization]')
    parser.add_argument('--direction', type=str, default='BtoA', help='AtoB or BtoA')
    parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
    parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--load_size', type=int, default=286, help='scale images to this size')
    parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
    parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
    parser.add_argument('--preprocess', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
    parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
    parser.add_argument('--display_winsize', type=int, default=256, help='display window size for both visdom and html')
    # additional parameters
    parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
    parser.add_argument('--load_iter', type=int, default='0', help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')
    parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
    parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')
    # wandb parameters
    parser.add_argument('--use_wandb', action='store_true', help='if specified, then init wandb logging')
    parser.add_argument('--wandb_project_name', type=str, default='CycleGAN-and-pix2pix', help='specify wandb project name')

    parser.add_argument('--display_freq', type=int, default=400, help='frequency of showing training results on screen')
    parser.add_argument('--display_ncols', type=int, default=4, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
    parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
    parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
    parser.add_argument('--display_env', type=str, default='main', help='visdom display environment name (default is "main")')
    parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
    parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
    parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
    parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
    # network saving and loading parameters
    parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
    parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
    parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
    parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
    parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
    parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
    # training parameters
    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs with the initial learning rate')
    parser.add_argument('--n_epochs_decay', type=int, default=100, help='number of epochs to linearly decay learning rate to zero')
    parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
    parser.add_argument('--gan_mode', type=str, default='lsgan', help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
    parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
    parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
    parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
# %%
    #further training args from pix2pix_model.py line 33:

    parser.set_defaults(pool_size=0, gan_mode='vanilla')
    parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
# %%
    args = parser.parse_known_args()[0]
    import time
    from torchinfo import summary
    from fran.extra.pix2pix.util.visualizer import Visualizer


    model =  Pix2PixModel(args)
    

# %%



# %%
# %%
    model.isTrain = True
    model.setup(args)
    model.netG.to(cuda)
    model.netD.to(cuda)
    dls = dls.to(cuda)
    # summary(model.netG,input_size = [1,128,128])

    learn = GANLearner2D(dls=dls,generator=GeneratorLearner(model.netG),critic=model.netD, gen_loss_func=gen_loss, crit_loss_func=crit_loss,gen_first=False,cbs=[SaveModelCallback])
    learn.to(cuda)
# %%
    learn.fit(50)

# %%
    visualizer = Visualizer(args)   # create a visualizer that display/save images and plots
    total_iters=0
    dataset_size = len(dls.train)

    fn = "/s/checkopints/pix2pix/experiment_name/168_net_G.pth"
    # model.netD(inp.to('cuda'))
# %%
    for epoch in range(args.epoch_count, args.n_epochs + args.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        for i, data in enumerate(dls.train):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % args.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += args.batch_size
            epoch_iter += args.batch_size
            model.set_input_fastai(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % args.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % args.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % args.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / args.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if args.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % args.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if args.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % args.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, args.n_epochs + args.n_epochs_decay, time.time() - epoch_start_time))
# %%
    total_iters = 0                # the total number of training iterations
# %

# %%
    a = dls.valid.one_batch()
    model.set_input_fastai(a)
    fake_B = model.netG(model.real_A)  # G(A)
    B = fake_B.detach().cpu()[:,0,:]
    A = model.real_A.detach().cpu().squeeze(1)[:,1,:]

    ImageMaskViewer([B,A])

# %%
    a,b,c = train_ds[200]
    ImageMaskViewer([a,b])
# %%
    for x in tqdm.tqdm(range(len(train_ds))):
        a,b,c =train_ds[x]
        d,e = Faker.encodes([a,b,c])

# %%
    ImageMaskViewer([d,e])
# %%
    train_list_w,valid_list_w,_ = get_train_valid_test_lists_from_json(project_title=proj_defaults.project_title,fold=fold,image_folder =images_folder/"images", json_fname=json_fname)
    train_ds = ImageMaskDatasetWithFakeTumours(train_list_w,bboxes_fn)
    len(train_ds)
    x,y = train_ds[12]
    ImageMaskViewer([x,y])
# %%
    train_ds = ImageMaskBBoxDataset(proj_defaults,train_list_w,bboxes_fn,[0,1,2])
    len(train_ds)
    x,y,z = train_ds[10]
# %%

#
#     bboxes_fname = proj_default_folders.preprocessing_output_folder / "bboxes_info.pkl"
#     bboxes = load_pickle(bboxes_fname)
#     case_id = get_case_id_from_filenameimg_fn)
#     bbox = [x for x in bboxes if x['case_id']==case_id][0]
#     stats_tumour= bbox['stats_tumour']
#     stats_other = bbox['stats_kidneys']
#     size_tumour = stats_tumour['sizes'][1]
#     size_mask = stats_other['sizes'][0]
# 

# %%
    shape = [64,192,192]
    parent_folder = images_folder
# %%
# %%
    stage1_subfolder = foldername_from_shape(images_folder,shape)
    print("Folder {} exists : {}".format(stage1_subfolder, stage1_subfolder.exists())) 
# %%
         
    dest_labels={"kidney":1,"tumour":1,"cyst":1}
    crop_center= ["kidney"]
# %%
    ######################################################################################
    # %% [markdown]
    ## Nifty Dataset
    # %%
    
    bboxes_nii_fn= proj_defaults.stage1_folder/("cropped/images_nii/bboxes_info")
    bboxes_nii = load_dict(bboxes_nii_fn)
    train_list,valid_list,_ = get_train_valid_test_lists_from_json(project_title=proj_defaults.project_title,fold=0, json_fname=proj_defaults.validation_folds_filename,image_folder =proj_defaults.stage1_folder/("cropped/images_nii/images"),ext=".nii.gz")

    train_ds = ImageMaskBBoxDataset_Nifty(train_list,[bboxes_nii_fn])
    valid_ds = ImageMaskBBoxDataset_Nifty(valid_list,[bboxes_nii_fn])
    a,b,c = train_ds[0]

    bb = c['bbox_stats'][0]['bounding_boxes'][1]
# %%
    ######################################################################################
    # %% [markdown]
    ## Torch Dataset
    # %%
    bboxes_pt_fn= proj_defaults.stage1_folder/("cropped/images_pt/bboxes_info")
    train_list,valid_list,_ = get_train_valid_test_lists_from_json(project_title=proj_defaults.project_title,fold=0, json_fname=proj_defaults.validation_folds_filename,image_folder =proj_defaults.stage1_folder/("cropped/images_pt/images"),ext=".pt")

    train_ds = ImageMaskBBoxDataset(proj_defaults,train_list,[bboxes_pt_fn],[0,1,2])
    valid_ds = ImageMaskBBoxDataset(proj_defaults,valid_list,[bboxes_pt_fn],[0,1,2])
    a,b,c = train_ds[0]

    bb = c['bbox_stats'][0]['bounding_boxes'][1]
   # %%
    ImageMaskViewer([a[bb],b[bb]])
# %%
    train_dl = TfmdDL(train_ds, shuffle=True,bs=bs,num_workers=bs*4,after_item=[
                                Contextual2DSampler(),
                                TrainingAugmentations(augs=[rotate90,flip_vertical],p=[0.2,0.2]),
                                # TrainingAugmentations(augs = [brightness,gaussian_noise,gaussian_blur, power_transform],p=[0.1,.1,.1,.1]),
                               MaskLabelRemap(src_dest_labels),
                                ToTensorImageMask,
                                Unsqueeze,
                                
                                
                          ],

                                create_batch=collate_fn_contextual_sampler,
                                after_batch=[  
                                    AffineTrainingTransform3D(),
                                    CropImgMask(patch_size=patch_size),
                                    PadDeficitImgMask(patch_size,5),
                          ]
                          )

# %%
    a,b,c = train_ds[0]
    aa,bb = Contextual2DSampler().encodes([a,b,c])


# %%
    ImageMaskViewer([aa[:,0,:],bb[:,0,:]])

# %%
    valid_dl= TfmdDL(valid_tl, shuffle=True,bs=bs,num_workers=bs*3,after_item=[
                                CenterCropOrPad(patch_size=patch_size, crop_center = crop_center,expand_by=0, random_sample=0.3),
                               MaskLabelRemap(src_dest_labels),
                                ToTensorImageMask,
                                Unsqueeze,
                                
                                
                          ],
                                after_batch=[  
                                    PadDeficitImgMask(patch_size,5),
                          ]
                          )

# %%
    im = torch.load('input.pt').to('cpu')
    plt.imshow(im[0,-1,:])
    ImageMaskViewer([im[0,0,:], im[0,-1,:]])
# %%
    dls = DataLoaders(train_dl,valid_dl)     

    a,b = train_dl.one_batch()
# %%
# ImageMaskViewer([x[0].numpy(),x[1].numpy()])
