



# %%
import argparse
import  time
from datetime import timedelta
from typing import OrderedDict
import torch.optim as optim
from torch.autograd import Variable
from fran.architectures.unet3d.buildingblocks import *
from utilz.imageviewers import ImageMaskViewer
from fastcore.foundation import L

from fran.transforms.batchtransforms import *
from fran.architectures.unet3d.model import Generator, Discriminator_ub
from fran.data.dataset import *
from fran.evaluation.losses import *
from fran.utils.config_parsers import *
from utilz.helpers import *
from fran.callback.nept import *
from fran.callback.tune import *
from fran.transforms.misc_transforms import *


from fran.data.dataset import ImageMaskBBoxDataset
def gen_loss(fake_pred, output, target,loss_fnc = nn.BCEWithLogitsLoss(), L1=nn.L1Loss()):
    # loss = torch.sum(torch.log(1-fake_pred))
    real_labels = torch.ones(fake_pred.shape,device=fake_pred.device)
    loss =loss_fnc(fake_pred,real_labels)
    if L1 :
            lambda_L1=100
            loss_G_L1 = L1(output,target) * lambda_L1
            loss = loss+loss_G_L1
    return loss

def crit_loss(real_pred,fake_pred,loss_fnc=nn.BCEWithLogitsLoss()):
    real_labels = torch.ones(real_pred.shape,device=real_pred.device)
    fake_labels = torch.zeros(fake_pred.shape,device=fake_pred.device)
    real_loss = loss_fnc(real_pred,real_labels)
    fake_loss = loss_fnc(fake_pred,fake_labels)
    return (real_loss + fake_loss)*0.5

class GANLoss_ub(gan.GANModule):
    "Wrapper around `crit_loss_func` and `gen_loss_func`"
    def __init__(self,
        gen_loss_func:callable, # Generator loss function
        crit_loss_func:callable, # Critic loss function
        gan_model:gan.GANModule # The GAN model
    ):
        super().__init__()
        store_attr('gen_loss_func,crit_loss_func,gan_model')

    def generator(self,
        output, # Generator outputs
        target # Real images
    ):
        "Evaluate the `output` with the critic then uses `self.gen_loss_func` to evaluate how well the critic was fooled by `output`"
        fake_pred = self.gan_model.critic(output)
        self.gen_loss = self.gen_loss_func(fake_pred, output, target)
        return self.gen_loss

    def critic(self,
        real_pred, # Critic predictions for real images
        input # Input noise vector to pass into generator
    ):
        "Create some `fake_pred` with the generator from `input` and compare them to `real_pred` in `self.crit_loss_func`."
        fake = self.gan_model.generator
        for f in fake:
          f.requires_grad_(False)
        fake_pred = self.gan_model.critic(fake)
        self.crit_loss = self.crit_loss_func(real_pred, fake_pred)
        return self.crit_loss

@delegates()
class GANLearner_ub(Learner):
    "A `Learner` suitable for GANs."
    def __init__(self,
        dls:DataLoaders, # DataLoaders object for GAN data
        generator:nn.Module, # Generator model
        critic:nn.Module, # Critic model
        gen_loss_func:callable, # Generator loss function
        crit_loss_func:callable, # Critic loss function
        switcher:Callback=None, # Callback for switching between generator and critic training, defaults to `FixedGANSwitcher`
        gen_first:bool=False, # Whether we start with generator training
        switch_eval:bool=True, # Whether the model should be set to eval mode when calculating loss
        show_img:bool=True, # Whether to show example generated images during training
        clip:float=None, # How much to clip the weights
        cbs=None, # Additional callbacks
        metrics=None, # Metrics
        **kwargs
    ):
        gan = GANModule(generator, critic)
        loss_func = GANLoss_ub(gen_loss_func, crit_loss_func, gan)
        if switcher is None: switcher = FixedGANSwitcher()
        trainer = GANTrainer(clip=clip, switch_eval=switch_eval, gen_first=gen_first, show_img=show_img)
        cbs = L(cbs) + L(trainer, switcher)
        metrics = L(metrics) + L(*LossMetrics('gen_loss,crit_loss'))
        super().__init__(dls, gan, loss_func=loss_func, cbs=cbs, metrics=metrics, **kwargs)

    def _do_one_batch(self):
        self.pred = self.model(*self.xb)
        self('after_pred')
        if len(self.yb):
            self.loss_grad = self.loss_func(self.pred, *self.yb)
            self.loss = self.loss_grad.clone()
        self('after_loss')
        if not self.training or not len(self.yb): return
        self._with_events(self._backward, 'backward', CancelBackwardException)
        self._with_events(self._step, 'step', CancelStepException)
        self.opt.zero_grad()

    @classmethod
    def wgan(cls,
        dls:DataLoaders, # DataLoaders object for GAN data
        generator:nn.Module, # Generator model
        critic:nn.Module, # Critic model
        switcher=None, # Callback for switching between generator and critic training, defaults to `FixedGANSwitcher(n_crit=5, n_gen=1)`
        clip=0.01, # How much to clip the weights
        switch_eval:bool=False, # Whether the model should be set to eval mode when calculating loss
        **kwargs
    ):
        "Create a [WGAN](https://arxiv.org/abs/1701.07875) from `dls`, `generator` and `critic`."
        if switcher is None: switcher = FixedGANSwitcher(n_crit=5, n_gen=1)
        return cls(dls, generator, critic, _tk_mean, _tk_diff, switcher=switcher, clip=clip, switch_eval=switch_eval, **kwargs)

class ImageMaskBBoxDatasetSinglePatchVols(ImageMaskBBoxDataset):
    def __init__(self,case_ids, bbox_fn, ensure=['tumour']):
            bboxes = load_dict(bbox_fn)
            images_folder = bboxes[0]['filename'].parent
            all_files = list(images_folder.glob("*.pt"))
            single_patch_vols =[] 
            for case_id in case_ids:
                files_this_case = [fns for fns in all_files if case_id in str(fns)]
                if len(files_this_case)==1: single_patch_vols.append(case_id)
            super().__init__(case_ids=single_patch_vols,bbox_fn=bbox_fn,ensure=ensure)

class cGANBatchMaker(ItemTransform):
        def __init__(self,mean=0.,std=.5,n_labels=4):
            store_attr()
        def encodes(self, x):
            img, mask = x

            # rand_ch = torch.normal(mean=self.mean,std=self.std,size = img.shape,device=img.device)
            # return torch.cat([mask,rand_ch],1), torch.cat([mask,img],1)
            return [torch.cat([mask,img],1),]*2

class cGANMaskOneHot(ItemTransform):
    def __init__(self,n_classes ) -> None:
        self.n_classes = n_classes

    def encodes(self, x):
        output=[]
        for tnsr in x:
            mask , img= tnsr[:,0:1,:], tnsr[:,1:,:]
            mask = one_hot(mask,self.n_classes, axis = 1 ,fnc=torch.cat)
            tnsr = torch.cat([mask,img],axis=1)
            output.append(tnsr)
        return output





# %%


if __name__ == "__main__":


    bs,max_workers=8,16
    patch_size = target_size=[64,160,160]

    import os; print(os.getcwd())
# %%
    after_item_intensity=    {'brightness': [[0.7, 1.3], 0.1],
     'shift': [[-0.2, 0.2], 0.1],
     'noise': [[0, 0.1], 0.1],
     'brightness': [[0.7, 1.5], 0.01],
     'contrast': [[0.7, 1.3], 0.1]}
    after_item_spatial = {'flip_random':0.5}
    intensity_augs,spatial_augs = create_augmentations(after_item_intensity,after_item_spatial)

    probabilities_intensity,probabilities_spatial = 0.1,0.5
    after_item_intensity = TrainingAugmentations(augs=intensity_augs, p=probabilities_intensity)
    after_item_spatial = TrainingAugmentations(augs=spatial_augs, p=probabilities_spatial)
    affine_vals ={'rotate_max': 0.392699081698724,
     'translate': [0.1, 0.1, 0.1],
     # 'scale_ranges': [0.75, 1.25],
     'shear': 1,
     'p': 0.4}
    after_batch_affine =            AffineTrainingTransform3D(**affine_vals), 
    dest_labels = {"kidney":1,"tumour":2,"cyst":3}

# %%
    after_item_train = Pipeline([
            DropBBoxFromDataset(),
           MaskLabelRemap(remapping_train),
            PermuteImageMask,
            # StrideRandom(patch_size=patch_size, stride_max=[2, 2, 2], pad_value=-0.49),
            # CropExtra(patch_size=patch_size), 
                                    # after_item_intensity, 
                                    # after_item_spatial, 
                                    Unsqueeze
        ])
    after_item_valid = Pipeline([
            DropBBoxFromDataset(),
           MaskLabelRemap(remapping_train), Unsqueeze],
                                         )
    after_batch_train = Pipeline([
            # AffineTrainingTransform3D(**affine_vals),
            CropImgMask(patch_size),
            PadDeficitImgMask(patch_size, 5),
            # ResizeBatch(target_size=target_size),
            cGANBatchMaker(),
            cGANMaskOneHot(n_classes=4)
        ])
    after_batch_valid = Pipeline([PadDeficitImgMask(patch_size, 5),
                    ResizeBatch(target_size=target_size),
                     cGANBatchMaker(),
            cGANMaskOneHot(n_classes=4)
                                 ])

# %%
    common_vars_filename=os.environ['FRAN_COMMON_PATHS']
    P = Project(project_title="lits"); proj_defaults= P
    
    dim0,dim1=64,160
    dataset_folder = proj_defaults.stage2_folder / "{0}_{1}_{1}".format(dim0, dim1)
    images_folder = dataset_folder / "volumes"
    bboxes_fname = dataset_folder / "bboxes_info"
    train_list,valid_list ,_= get_fold_case_ids(fold=0, json_fname=proj_defaults.validation_folds_filename)
    train_ds= ImageMaskBBoxDatasetSinglePatchVols(train_list,bboxes_fname,ensure='tumour')
    valid_ds = ImageMaskBBoxDatasetSinglePatchVols(valid_list,bboxes_fname,ensure='tumour')
# %%
    train_dl = TfmdDL(train_ds,
                      shuffle=True,
                      bs=bs,
                      num_workers=np.minimum(max_workers, bs * 2),
                      after_item=after_item_train,
                      after_batch=after_batch_train,
                      )
    valid_dl = TfmdDL(valid_ds,
                      shuffle=False,
                      bs=bs,
                      num_workers=np.minimum(max_workers, bs * 4),
                      after_item=after_item_valid,
                      after_batch=after_batch_valid,
                      )

# %%
    class OneHotMask(Callback):
        '''
        Assumes mask is the first index in the tensor
        '''
        def __init__(self, n_classes=4,mask_first=True):
            store_attr()
        
        def before_batch(self):
            self.learn.xb = self._process(*self.learn.xb)
            self.learn.yb = self._process(*self.learn.yb)


        def _process(self,*tnsr_list):
            output=[]
            for tnsr in tnsr_list:
                if self.mask_first==True:
                    mask , img= tnsr[:,0:1,:], tnsr[:,1:,:]
                    mask = one_hot(mask,self.n_classes, axis = 1 ,fnc=torch.cat)
                    mask = mask.to(img.dtype)
                    tnsr = torch.cat([mask,img],axis=1)
                    output.append(tnsr)
                else: print("Not imptlemented")
            return output
# %%
    dls = DataLoaders(train_dl,valid_dl,device='cuda')

    fake, real = dls.one_batch()



    D =Discriminator_ub(in_channels=5, num_levels=4,f_maps=32,patch_size=target_size)

    G = Generator(in_channels= 5,num_levels=3,f_maps=16, layer_order='sbl')
    device = 'cuda'
# %%
    # ImageMaskViewer([fake[0,0].detach().cpu(), real[0,-1].detach().cpu()])
    #
    # ImageMaskViewer([mask[0,0],mask[0,1]])
# %%
    learn = GANLearner(dls=dls,generator=G,critic=D, gen_loss_func=gen_loss, crit_loss_func=crit_loss,gen_first=False)

    learn.to(device)
    learn = learn.to_fp16()
    # learn.fit(200, lr=2e-4)
    outputs=[]
    n_classes=4
    for tnsr in x:
            mask , img= tnsr[:,0:1,:], tnsr[:,1:,:]
            mask = one_hot(mask,n_classes, axis = 1 ,fnc=torch.cat)
            tnsr = torch.cat([mask,img],axis=1)
            outputs.append(tnsr)

# %%
    y = G(fake)
    fake, real= outputs
    ImageMaskViewer([y[0,-1].detach().cpu(),y[0,1].detach().cpu()])
    ImageMaskViewer([real[0,0],fake[0,2]])
# %%
    learn.to(device)
    learn.dls.to(device)
    learn.fit(50, 2e-2)

# %%

    fake ,real= learn.dls.valid.one_batch()

    n_classes = 4
    dd(img, mask, n_classes, output, x)
    pred = G(noise)
    ImageMaskViewer([pred[0,1].detach().cpu(), real[0,1].detach().cpu()])
# %%
    criterion_GAN = torch.nn.MSELoss()
    criterion_voxelwise = diceloss()

    generator = GeneratorUNet()
    discriminator = Discriminator()


    # Loss weight of L1 voxel-wise loss between translated image and real image
    lambda_voxel = 100
    
    # Calculate output of image discriminator (PatchGAN)
    img_height=img_width = patch_size[1]
    img_depth = patch_size[0]
    patch = (1, img_depth// 2 ** 4, img_width // 2 ** 4, img_height// 2 ** 4)

# %%
    batch = dls.one_batch()
    real_A = Variable(batch["A"])

    real_B = Variable(batch["B"])
# %%

    generator = generator.cuda()
    discriminator = discriminator.cuda()
    criterion_GAN.cuda()
    criterion_voxelwise.cuda()
    glr,dlr,b1,b2=0.0002,0.0002,0.5,0.999

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=glr, betas=(b1, b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=dlr, betas=(b1, b2))

# %%

    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--dataset_name", type=str, default="leftkidney_3d", help="name of the dataset")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--glr", type=float, default=0.0002, help="adam: generator learning rate")
    parser.add_argument("--dlr", type=float, default=0.0002, help="adam: discriminator learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_height", type=int, default=128, help="size of image height")
    parser.add_argument("--img_width", type=int, default=128, help="size of image width")
    parser.add_argument("--img_depth", type=int, default=128, help="size of image depth")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--disc_update", type=int, default=5, help="only update discriminator every n iter")
    parser.add_argument("--d_threshold", type=int, default=.8, help="discriminator threshold")
    parser.add_argument("--threshold", type=int, default=-1, help="threshold during sampling, -1: No thresholding")
    parser.add_argument(
        "--sample_interval", type=int, default=1, help="interval between sampling of images from generators"
    )
    parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between model checkpoints")
    opt = parser.parse_known_args()[0]

    lambda_voxel = 100
# %%




    dls = dls.to('cuda')
# %%
    prev_time = time.time()
    discriminator_update = 'False'
    for epoch in range(opt.epoch, opt.n_epochs):
        for i, batch in enumerate(dls.train):

            # Model inputs
            real_A = Variable(batch["A"])
            real_B = Variable(batch["B"])

            # Adversarial ground truths
            valid = torch.ones((real_A.size(0), *patch),device='cuda',requires_grad=False)
            fake= torch.zeros((real_A.size(0), *patch),device='cuda',requires_grad=False)


            # ---------------------
            #  Train Discriminator, only update every disc_update batches
            # ---------------------
            # Real loss
            fake_B = generator(real_A)
            pred_real = discriminator(real_B, real_A)
            loss_real = criterion_GAN(pred_real, valid)

            # Fake loss
            pred_fake = discriminator(fake_B.detach(), real_A)
            loss_fake = criterion_GAN(pred_fake, fake)
            # Total loss
            loss_D = 0.5 * (loss_real + loss_fake)

            d_real_acu = torch.ge(pred_real.squeeze(), 0.5).float()
            d_fake_acu = torch.le(pred_fake.squeeze(), 0.5).float()
            d_total_acu = torch.mean(torch.cat((d_real_acu, d_fake_acu), 0))

            if d_total_acu <= opt.d_threshold:
                optimizer_D.zero_grad()
                loss_D.backward()
                optimizer_D.step()
                discriminator_update = 'True'

            # ------------------
            #  Train Generators
            # ------------------
            optimizer_D.zero_grad()
            optimizer_G.zero_grad()

            # GAN loss
            fake_B = generator(real_A)
            pred_fake = discriminator(fake_B, real_A)
            loss_GAN = criterion_GAN(pred_fake, valid)
            # Voxel-wise loss
            loss_voxel = criterion_voxelwise(fake_B, real_B)

            # Total loss
            loss_G = loss_GAN + lambda_voxel * loss_voxel

            loss_G.backward()
            optimizer_G.step()

            batches_done = epoch * len(dls.train) + i

            # --------------
            #  Log Progress
            # --------------

            # Determine approximate time left
            batches_left = opt.n_epochs * len(dls.train) - batches_done
            time_left = timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f, D accuracy: %f, D update: %s] [G loss: %f, voxel: %f, adv: %f] ETA: %s"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dls),
                    loss_D.item(),
                    d_total_acu,
                    discriminator_update,
                    loss_G.item(),
                    loss_voxel.item(),
                    loss_GAN.item(),
                    time_left,
                )
            )
            # If at sample interval save image
            discriminator_update = 'False'

        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(generator.state_dict(), "saved_models/%s/generator_%d.pth" % (opt.dataset_name, epoch))
            torch.save(discriminator.state_dict(), "saved_models/%s/discriminator_%d.pth" % (opt.dataset_name, epoch))



# %%
    c= ConvLayer(1,16,ndim=3)
    order = 'cil'
    C = create_conv(1,16,3,1,order,1)
    S = nn.Sequential(OrderedDict(C))
    x = torch.rand(1,1,128,128,128)
    y = S(x)
# %%
# %%

    ImageMaskViewer([fake_B[0,0].detach().cpu(), real_B[0,0].detach().cpu()])

# %%
    num_epochs=5
    criterion=nn.BCELoss()
    a,b = dls.one_batch()
    fixed_noise = a.clone()
    G_losses,D_losses = [],[]
    lr = 0.0002
    real_label=1.
    fake_label = 0.
    beta1 = 0.5
    optimizerD = optim.Adam(D.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(G.parameters(), lr=lr, betas=(beta1, 0.999))
    iters=0
    img_list=[]

    iteri = iter(train_dl)
    a,b = next(iteri)

    ImageMaskViewer([fake[0,1].cpu().detach(),b [0,1].cpu().detach()])
    b = G(a)
# %%
    for epoch in range(num_epochs):
        for i, data in enumerate(train_dl):   

            real_gt= torch.ones((bs),*D.output_shape,device=device,requires_grad=True)
            fake_gt = torch.zeros((bs),*D.output_shape,device=device,requires_grad=True)

            D.zero_grad()
            noise, real = data

            # b_size = real.size(0)
            real_pred = D(real)
            real_loss = criterion(real_pred,real_gt)
            real_loss.backward()
            D_x = real_pred.mean().item()

            fake = G(noise)

            fake_pred = D(fake.detach())
            fake_loss = criterion(fake_pred,fake_gt)
            fake_loss.backward()
            
            D_G_z1=fake_loss.mean().item()
            loss = real_loss=fake_loss
            optimizerD.step()

            G.zero_grad()
            gen_pred = D(fake)
            gen_loss = criterion(gen_pred,real_gt)
            gen_loss.backward()
            D_G_z2 = fake_loss.mean().item()
            optimizerG.step()
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dls.train),
                         loss.item(), fake_loss.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(gen_loss.item())
            D_losses.append(loss.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dls.train)-1)):
                with torch.no_grad():
                    fake = G(fixed_noise).detach().cpu()
                img_list.append(fake)

            iters += 1
# %%
    aa = img_list[2]
# %%

from fran.architectures.unet3d.buildingblocks import *
class ResNetBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, order='cge', num_groups=8, **kwargs):
        super().__init__()

        self.conv1 = SingleConv(in_channels, out_channels, kernel_size=kernel_size, order=order, num_groups=num_groups,padding='same')
        n_order = order
        for c in 'rel':
            n_order = n_order.replace(c, '')
        # remove non-linearity from the 2nd convolution since it's going to be applied after adding the residual
        self.conv2 = SingleConv(out_channels, out_channels, kernel_size=kernel_size, order=n_order, num_groups=num_groups,padding='same')

        # create non-linearity separately
        if 'l' in order:
            self.non_linearity = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif 'e' in order:
            self.non_linearity = nn.ELU(inplace=True)
        else:
            self.non_linearity = nn.ReLU(inplace=True)

    def forward(self, x):
        # apply first convolution and save the output as a residual
        out = self.conv1(x)
        residual = out

        # residual block
        out = self.conv2(out)

        out += residual
        out = self.non_linearity(out)

        return out

# %%
def create_conv_transpose_block(in_channels,out_channels,kernel_size=3,stride=1,padding=0):
        co=nn.ConvTranspose3d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding)
        no= nn.InstanceNorm3d(in_channels)
        nl= nn.LeakyReLU()
        return nn.Sequential(co,no,nl)


# %%
    # a = learn.generator(dls.valid.one_batch()[0].cuda()).detach().cpu()
    # ImageMaskViewer([a[:,-1,:],a[:,1,:]])
# %%
   # img  = torch.load ("gen_false.pt").to('cpu')
   #
   # plt.imshow(img[0,:])
#


#     x1  = torch.rand(1,1,64,128,128)
#     x2  = torch.rand(1,1,64,160,160)
#     f_maps = 64
#     num_levels = 4 
#     if isinstance(f_maps, int):
#                 f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)
#     f_maps= [1]+f_maps
# # %%
#     x = torch.clone(x1)
# # %%
#     encoders= [SingleConv(f_maps[i],f_maps[i+1],kernel_size=4,stride=2, order='cil') for i in range(len(f_maps)-1)]
#     encoders_features = []
# # %%
#     for en in encoders: 
#         x = en(x)
#         encoders_features.insert(0, x)
#
# # %%
#     encoders_features = encoders_features[1:]
#     [a.shape for a in encoders_features]
# # %%
#     blks = [ResNetBlock(f_maps[-1],f_maps[-1],kernel_size=4,order='cbr') for i in range(4)]
#     bottleneck = nn.Sequential(*blks)
#     x2= bottleneck(x1)
#     x3 = decoders[0](x2)
#     
# # %%
#     i=0
#     reversed_f_maps = list(reversed(f_maps))
#     decoders = []
#     for i in range(len(reversed_f_maps)-1):
#         decoders.append(create_conv_transpose_block(reversed_f_maps[i],reversed_f_maps[i+1],kernel_size=4,stride=2,padding=1))
# # %%
#
# # %%
#     decoders = []
#     for i in range(len(reversed_f_maps) - 1):
#         in_feature_num = reversed_f_maps[i]
#
#         out_feature_num = reversed_f_maps[i + 1]
#
#         _upsample = True
#         if i == 0:
#             # upsampling can be skipped only for the 1st decoder, afterwards it should always be present
#             _upsample = upsample
#
#         decoder = Decoder(in_feature_num, out_feature_num,
#                           basic_module=basic_module,
#                           conv_layer_order=layer_order,
#                           conv_kernel_size=conv_kernel_size,
#                           num_groups=num_groups,
#                           padding=conv_padding,
#                           upsample=_upsample)
#                        
#         decoders.append(decoder)
#     return nn.ModuleList(decoders)
# # %%
# # ImageMaskViewer([fake[0][0,0], fake[1][0,0]],intensity_slider_range_percentile=[0,])
#     ImageMaskViewer([fake[0][0,0], fake[1][0,0]])
#
# # %%
#     ImageMaskViewer([fixed_noise[0, 0,:].cpu().detach(),img_list[2][0,1,:]]  ,data_types=['mask','img'])
#     ImageMaskViewer([fixed_noise[0,0,:].cpu().detach(),fixed_noise[0,1,:].cpu().detach()])
