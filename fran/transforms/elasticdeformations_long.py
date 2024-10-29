# %%
from torch.autograd.variable import Variable
from fran.transforms.spatialtransforms import *
from fran.utils.helpers import *
from fran.utils.imageviewers import *


# %%

sys.path+=['/home/ub/Dropbox/code/nurbspy/nurbspy', '/home/ub/Dropbox/code/nurbspy/','/home/ub/Dropbox/code']

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.ion()
    patch_size = [128,128,128]

    P = Project(project_title="lits"); proj_defaults= P
    folder = proj_defaults.stage2_folder/"64_160_160"
    # fake_tumours = list((project_title=proj_defaults.project_title,folder/"tumour_only").glob("*.pt"))
    json_fname=proj_defaults.validation_folds_filename
    fold = 0
    train_list,valid_list,_ = get_fold_case_ids(project_title=proj_defaults.project_title,fold=fold,json_fname=json_fname)

    print(len(train_ds))
# %%
    ind=23
    a, b, c = train_ds[ind]
    ImageMaskViewer([a,b])
    img = a.clone()
# %%
        # for a ,b in zip(tmr_center, tmr_final_size):
        #     print()
        #     tumour_slcs.append(slice(int(a-np.floor(b/2)),int(a+np.ceil(b/2))))
        #
        # mask_fullsize= tmr_fullsize= torch.zeros(img_shape)
        # tmr_fullsize[tumour_slcs]=tmr_final
        # tmr_backup = tmr_fullsize.clone()
        # mask_fullsize[tumour_slcs]=mask_final
        # inds_tmr = torch.where(mask_fullsize==self.label_index)
        #
# %%
######################################################################################
# %% [markdown]
## Spline surfact 
# %%
    
    p=3
    q=3
    h = m+p+1
    U_clamped= torch.cat([torch.zeros(p),torch.linspace(0,1,h+3-2*p), torch.ones(p)])
    k = n+q+1
    V_clamped= torch.cat([torch.zeros(q),torch.linspace(0,1,k+3-2*q), torch.ones(q)])
    u= torch.linspace(0,1,40)
    uv = torch.meshgrid(u,u)
    # U_clamped= torch.cat([torch.zeros(p),torch.rand(h+3-2*p), torch.ones(p)])
    # V_clamped= torch.cat([torch.zeros(q),torch.rand(k+3-2*q), torch.ones(q)])

    # V= torch.linspace(0,1,len(V_clamped))
    # U= torch.linspace(0,1,len(U_clamped))
    U = torch.tensor([0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1,1,1 ])
    V = torch.tensor([0, 0, 0, 0, 0.33, 0.66, 1, 1, 1, 1 ,1,1])
# %%
    
    UV= torch.meshgrid(U,V)
    knots = torch.stack(UV,0)
    knot_int = torch.zeros(uv.shape)
    xs = uv[20:23,20:23]
    a,b = 3,4
    int_start =knots[:,a,b]
    int_stop = knots[:,a+1,b+1]
# %%
    z = torch.rand(m+1,n+1)
    z = torch.tensor([0.,0.3,0.4,0.]).repeat(4,1)
    print(z.shape)
    Cp_bspl= torch.stack([*cp,z],0)
    m=Cp_bspl.shape[-2]
    n=Cp_bspl.shape[-1]
    curve = torch.zeros(len(u),len(v),n_dim)
    for i in range(m):
        for j in range(n):
            N_m= N_i_p_vec(u,U,i,p)
            # N_m= N_i_scipy(u,U,i,p)
            N_n= N_i_p_vec(v,V,j,q)
            # N_n= N_i_scipy(v,V,j,q)
            P_ij = Cp_bspl[:,i,j]
            # cur = N_m*N_n
            # c2 = torch.einsum('ij,k',cur,c)
            c_tmp =torch.einsum('ij,ij,k->ijk',N_m,N_n,P_ij) 
            curve+=c_tmp
# %%
    plt.show()
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(*curve.permute(2,0,1))

    ax.scatter3D(*Cp_bspl,color='r')
# %%
    patch_size= [64,160,160]
    folder_p = proj_defaults.stage2_folder / ("{0}_{1}_{1}".format(patch_size[0],patch_size[1]))

    ind = 10
    a, b, c = train_ds_p[ind]
    # ImageMaskViewer([a,b])
# %%
    bbox_all = c['bbox_stats']
    tumour_bb = [a for a in bbox_all if a['tissue_type']=='tumour'][0]['bounding_boxes'][1]
    print(tumour_bb)
# %%
    lins=[]
    for sh in a.shape:
        lins.append(torch.linspace(-1,1,sh))
# %%
    x ,y, z= torch.meshgrid(*lins)
    base_mesh = torch.stack([z,y,x],3)
    base_mesh.unsqueeze_(0)
    aa= a.unsqueeze(0).unsqueeze(0)
    a2 = F.grid_sample(aa,base_mesh)
    ImageMaskViewer([aa[0,0],a2[0,0]])

# %%
    X = np.arange(-10, 10, 1)
    Y = np.arange(-10, 10, 1)
    U, V = np.meshgrid(X, Y)

    fig, ax = plt.subplots()
    qa = ax.quiver(X, Y, U, V)
    plt.show()
    [print(a.shape) for a in [X,Y,U,V]]
# %%
    plt.contour(U,V,U)
    plt.show()
# %%
    N_mn_basis = torch.einsum('ij,ij->ij',N_m,N_n)
    plt.show()
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(u,v,N_mn_basis)
# %%
    i=0

    y = torch.zeros(u.shape)
    # for i in range(m):
    y+= N_i_p_vec(u,U,i,p=p)
    inds = torch.where((u>=U[i]) & (u<U[i+1]))  
    print(y)
    plt.plot(u,y)
    y[inds].sum()
# %%
    x_dim=10        
    U = torch.tensor([0,.25,.5,.75,1])
    UV = torch.meshgrid([U,U])
    UV = torch.stack(UV,2)
    u = torch.linspace(0,1,x_dim)
    u,v = torch.meshgrid([u,u])
    uv = torch.stack([u,v],2)
    u
# %%
    num_points=4 # includes ends
    gap = x_dim/num_points # must be int
    inds_1d = [0,]+[int(gap*mult-1) for mult in range(1,num_points+1)]


# %%
# %%
    inds_2d = np.zeros((len(inds_1d),len(inds_1d)))
    for i in range(len(inds_1d)):
        for j in range(len(inds_1d)):
            inds_2d[i,j]=inds_1d[i],inds_1d[j]
# %%
    ind=6
    uv[inds_2d[ind][0],inds_2d[ind][1]]
# %%
    # ImageMaskViewer([x,y])
# %%

    _,_,test_list = get_train_valid_test_lists_from_json(project_title=proj_defaults.project_title,fold=0, json_fname=proj_defaults.validation_folds_filename,image_folder=proj_defaults.raw_data_folder/"images",ext=".nii.gz")
    mask_files=[ proj_defaults.raw_data_folder/"lms"/test_file.name.replace(".npy",".nii.gz") for test_file in test_list]
    img_files =  [proj_defaults.raw_data_folder/"images"/mask_file.name for mask_file in mask_files]
# %%
# %%
# %% [markdown]
## Applying RBF using contour / edge of mask and mask centroid
# %%

    # folder = proj_defaults.stage1_folder/"patches_48_128_128"
    # train_ds = ImageMaskBBoxDataset(proj_defaults,train_list,bbox_fn= folder/"bboxes_info",,[0,1,2])
    ind=23
    a, b, c = train_ds[ind]
# %%
    img = a.clone()
# %%
    bb = b.clone()
    bb[bb==1]=0
    bb = bb.unsqueeze(0).unsqueeze(0).float()
# %%
    # ImageMaskViewer([a,bb])
    depth = 1
    channels = 3
    sobel = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
    sobel_kernel = torch.tensor(sobel, dtype=torch.float32).unsqueeze(0).expand(depth, 1, channels, 3, 3)
# %%
    con= F.conv3d(bb, sobel_kernel, stride=1, padding=1)
    con=con.abs()[0,0]
    indices = torch.nonzero(con)
    quarts = int(len(indices)/4)
    cp_inds =np.array( [0,]+[random.randint(0,len(indices)) for x in range(14)]+[-1])
    i =indices[cp_inds,:]

# %%
# %%
    n=1
    plt.imshow(bb[0,0][22,:])
    this = i[n,0].item()
    plt.imshow(con[this,:])
    plt.plot(i[1][2],i[1][1],marker='o')

# %%
    ImageMaskViewer([a,con])
    ImageMaskViewer([a,b])
        
# %%
    bb = c['bbox_stats']
    tumour_info=[c for c in bb if c['tissue_type']=='tumour'][0]
    cent = tumour_info['centroids'][1]
    bbox = tumour_info['bounding_boxes'][1]
# %%
# %%
    def distance(p1,p2):
        d = (p1-p2)**2
        d = torch.sqrt(d.sum())
        return d
    def weight_curve(d,alpha):
        return 1-d**alpha
    def weight_fold(d,alpha):
        return alpha/(d+alpha)

# %%
######################################################################################
# %% [markdown]
## Elastic deforms
# %%
######################################################################################
# %% [markdown]
## Splines

    from torch.distributions.multivariate_normal  import MultivariateNormal as MN
# %%
    cov = torch.eye(3)
    mn = torch.zeros(3)
    x = torch.rand(3)
    torch.exp((x-mean).transpose()*torch.inv(conv))
# %%
    
# %%
    M = MN(loc=mn,covariance_matrix=cov)

# %%
######################################################################################
# %% [markdown]
## 2D gaussian mesh
# %%
    mn2= torch.zeros(2,requires_grad=True)
    cov2=torch.eye(2)
    cov2=torch.tensor([3.5,5.2,5.2,10.5]).reshape(2,2)
# %%
# %%
# %% [markdown]
##  2D version
# %%
    
# %%
    bboxes = c['bbox_stats']
    bbox = [b for b in bboxes if b['tissue_type']=='tumour'][0]
    mask_centre = bbox['centroids'][1]
    a2 = a[22,:120,60:180]
    a2 = a[22,:]
    a2.unsqueeze_(0).unsqueeze_(0)
    # plt.imshow(a2[0,0])
    x_dim = a2.shape[-1]
    aa = torch.linspace(-1,1,int(x_dim))
    x,y = torch.meshgrid(aa,aa)
    base_mesh = torch.stack([y,x],2)
    base_mesh.unsqueeze_(0)
    plt.imshow(x,cmap='Greys')
    plt.imshow(a2[0,0],cmap='Greys_r')
# %%
    def mvn(mn2,cov2=cov2,base_mesh=base_mesh):
        dist = torch.distributions.multivariate_normal.MultivariateNormal(mn2, cov2)
        logprobs = dist.log_prob(base_mesh)
        return logprobs
# %%


    aa = torch.linspace(-1,1,int(10))
    x,y = torch.meshgrid(aa,aa)
    minimesh= torch.stack([y,x],2)
    mm = Variable(minimesh,requires_grad=True)
# %%
    plt.imshow(minimesh[:,:,1],cmap='Greys')
    plt.show()
# %%
    symm= mvn(mn2,cov2=torch.tensor([[.2,0],[0,.3]]))
    torch.exp(symm).max()
# %%
    fig, ax = plt.subplots()
    # ax.quiver(*minimesh.permute(2,0,1).numpy(),*grads.permute(2,0,1).numpy())
    ax.quiver(*minimesh.permute(2,0,1).numpy(),*minimesh.permute(2,0,1).numpy())
# %%
    vals = mvn(mn2)
    grads = torch.autograd.functional.jacobian(mvn, mn2)
    grads = grads/grads.max()

    base_mesh_sqrd = base_mesh*base_mesh**(2)
    fig, ax = plt.subplots()
    plt.quiver(base_mesh[0,:,:,0],base_mesh[0,:,:,1],base_mesh_sqrd[0,:,:,0],base_mesh_sqrd[0,:,:,1])
    plt.quiver(base_mesh[0,10:25,25:35,0],base_mesh[0,10:25,25:35,1],base_mesh[0,10:25,25:35,0],base_mesh[0,10:25,25:35,1])
# %%
    base_mesh[0,10:25,25:35,0],base_mesh[0,10:25,25:35,1]
# %%
    plt.imshow(a2[0,0])
# %%
    y_1,y_2=10,15
    y_11,y_22= y_1-2,y_2+3
    x_1,x_2 = 18,25
    x_11,x_22 = x_1-2,x_2+2
# %%
    aa_1,aa_2 = aa[y_1], aa[y_2]
    bb = torch.linspace(-1,aa_1,y_11)
    bbb = torch.linspace(aa_1,aa_2,y_22-y_11+1)[1:]
    bbbb = torch.linspace(aa_2,1,int(x_dim)-y_22+1)[1:]
    bbbb_y = torch.cat([bb,bbb,bbbb])
# %%

# %%
# %%
    xx= expand_lesion(18,25,aa, expand_factor=-0.2)
    yy= expand_lesion(10,15,aa,expand_factor=-0.2)

    plt.plot(yy)
    plt.plot(xx)
# %%

    y,x = torch.meshgrid(yy,xx)
    mesh_wonky=torch.stack([x,y],2)
    mesh_wonky.unsqueeze_(0)
    outp2 = F.grid_sample(a2,mesh_wonky)
    fig, ax = plt.subplots()
    plt.imshow(outp2[0,0])
# %%
    aa = torch.linspace(-1,1,int(x_dim))
    y,x = torch.meshgrid(aa,aa)
    base_mesh = torch.stack([x,y],2)
    base_mesh.unsqueeze_(0)

# %%
# %%

    fig, ax = plt.subplots()
    plt.quiver(base_mesh[0,:,:,0],base_mesh[0,:,:,1],base_mesh[0,:,:,0],base_mesh[0,:,:,1])
    fig, ax = plt.subplots()
    plt.quiver(base_mesh[0,:,:,0],base_mesh[0,:,:,1],mesh_wonky[0,:,:,0],mesh_wonky[0,:,:,1])
# %%


# %%
    fig, ax = plt.subplots()
    plt.imshow(a2[0,0])
# %%
# %%
######################################################################################
# %% [markdown]
## 3D gaussian mesh
# %%
    import torch.nn.functional as F
    aa = torch.linspace(0,0.5,10)
    bb = torch.linspace(0.51,1.0,20)
    cc = torch.cat([aa,bb])
    cc.unsqueeze_(0).unsqueeze_(0)
    plt.plot(cc[0,0])
    dd = F.interpolate(cc,mode='linear', scale_factor=1.0)
    plt.plot(dd[0,0])
    plt.show()
    fig,ax =plt.figure()
    plt.plot(dd[0,0])
# %%

    sz = 128
    aa = torch.linspace(-1,1,20)
    base_mesh3d = torch.meshgrid(aa,aa,aa,indexing ='xy')
    base_mesh3d = torch.stack(base_mesh3d)
    base_mesh3d.unsqueeze_(0)
    # base_mesh3d=base_mesh3d.permute(0,2,3,4,1)
    base_mesh3d=base_mesh3d.permute(0,4, 2,3,1)
# %%
    def f_3d(mn2):
        Sigma_k = torch.rand(3,3)
        Sigma_k = torch.mm(Sigma_k, Sigma_k.t())
        # Sigma_k=torch.zeros(3,3)
        Sigma_k.add_(torch.eye(3,3))*10
        dist = torch.distributions.multivariate_normal.MultivariateNormal(mn2, Sigma_k)
        logprobs = dist.log_prob(base_mesh3d)
        return logprobs
# %%
    mn3 = torch.zeros(3)
    mn3 = torch.rand(3)*0.05
    ff = f_3d(mn3)

    grads3d = torch.autograd.functional.jacobian(f_3d, mn3)
    # grads3d = grads3d/grads3d.abs().max()
    base_mesh3d_big=F.interpolate(base_mesh3d.permute(0,4,1,2,3),size=(128,128,128),mode='trilinear').permute(0,2,3,4,1)
# %%
    n=0
    im = torch.tensor(nii_sitk_to_np(str(img_files[n])))
    img = im.clone().permute(2,1,0)
    img = img.float()
    ImageMaskViewer([img,img])
# %%
    grads3d_big = F.interpolate(grads3d.permute(0,4,1,2,3),size=(128,128,128),mode='trilinear')
    grads3d_big = grads3d_big.permute(0,2,3,4,1)
    # outp2 = F.grid_sample(img.unsqueeze(0).unsqueeze(0),base_mesh3d_big)
    outp2 = F.grid_sample(img.unsqueeze(0).unsqueeze(0),grads3d_big)
# %%
    base_mesh3d_mod = base_mesh3d_big*2*base_mesh3d_big**2
    outp2 = F.grid_sample(img.unsqueeze(0).unsqueeze(0),base_mesh3d_mod)

    ImageMaskViewer([outp2[0,0],img])
# %%
    ImageMaskViewer([outp2[0,0],outp2[0,0]])
    ImageMaskViewer([img,img])
# %%
# %%
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(base_mesh3d[0,:,:,0], base_mesh[0,:,:,1], grads3d[0,:,:,0],grads3d[0,:,:,1])
# %%
    import matplotlib.pyplot as plt
    import numpy as np
    x,y = np.meshgrid(np.arange(-2, 2, .2), np.arange(-2, 2, .25))
    z = x*np.exp(-x**2 - y**2)
    v, u = np.gradient(z, .2, .2)
    fig, ax = plt.subplots()
    q = ax.quiver(x,y,u,v)

# %%
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(x,y,z)
# %%

    x.shape
# %%
    surf = ax.quiver(minimesh[:,:,0], minimesh[:,:,1], z,
                           linewidth=0, antialiased=False)
# %%
    ff_def = base_mesh.clone()
    plt.quiver(base_mesh[0,:,:,0],base_mesh[0,:,:,1],ff_def[0,:,:,0],ff_def[0,:,:,1])
    plt.quiver(base_mesh[0,:,:,0],base_mesh[0,:,:,1],grads[0,:,:,0],grads[0,:,:,1])
# %%
    knots_x = [0,0,0,0.25,0.5,0.75,1,1,1]
    knots_y = [0,0,0,0,0.33,0.66,1,1,1,1]
    deg_x = 2
    deg_y=3


# %%
    n_dims=2
    num_points=4 # includes ends
    gap = x_dim/num_points # must be int
    inds_1d = [0,]+[int(gap*mult-1) for mult in range(1,num_points+1)]
    inds_2d = []
    for i in inds_1d:
        for j in inds_1d:
            inds_2d.append([i,j])
# %%
    print(inds_2d)
    i= 6
    ind =  inds_2d[i]
    print(ind)

# %%
    print(base_mesh[0,ind[0],ind[1],:])
    print(ff_def[0,ind[0],ind[1],:])
# %%
    cp_inds =[]
    for points in range(num_points):
        cp_inds.append([int(gap*points),int(gap*(points+1))])
    ind = 0
    base_mesh[0,cp_inds[ind][0],cp_inds[ind][1],:]
    print(cp_inds)
# %%


    tot = num_points**n_dims
    locs = torch.randint(0,x_dim,[p,2])
    locs
######################################################################################
# %% [markdown]
## Random elastic deformation
# %%

    def get_gaussian_kernel(kernel_size=3, sigma=2, channels=3):
        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        mean = (kernel_size - 1)/2.
        variance = sigma**2.

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        gaussian_kernel = (1./(2.*math.pi*variance)) *\
                          torch.exp(
                              -torch.sum((xy_grid - mean)**2., dim=-1) /\
                              (2*variance)
                          )
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
        return gaussian_kernel
# %%
    kern = get_gaussian_kernel(channels=1,sigma=18)
    kern= kern.unsqueeze(0).unsqueeze(0)
    conv = torch.nn.Conv2d(1,1,kernel_size=3,padding='same',bias=None)
    conv.weight.data=kern


# %%
    scale=1
    x_delta = torch.rand(1,1,128,128)*2-1
    y_delta = torch.rand(1,1,128,128)*2-1
    x_delta2 = conv(x_delta*scale)
    y_delta2 = conv(y_delta*scale)
    elast = torch.stack([x_delta2.squeeze(0),y_delta2.squeeze(0)],dim=-1)
    elast.shape
# %%
    outp2 = F.grid_sample(a,base_mesh+elast).detach().cpu()
    plt.imshow(outp2[0,0])
# %%
    ff_rand = torch.rand(base_mesh.shape)*2-1
# %%
# %%
    import numpy as np
    import matplotlib.pyplot as plt
    from nurbspy.nurbs_curve import NurbsCurve

    # Define the array of control points
    P = np.zeros((2,5))
    P[:, 0] = [0.20, 0.50]
    P[:, 1] = [0.40, 0.70]
    P[:, 2] = [0.80, 0.60]
    P[:, 3] = [0.80, 0.40]
    P[:, 4] = [0.40, 0.20]

    # Create and plot the Bezier curve
    bezierCurve = NurbsCurve(control_points=P)
    bezierCurve.plot()
    plt.show()
# %%
    import numpy as np
    import nurbspy as nrb
    import matplotlib.pyplot as plt

# Define the array of control points
    n_dim, n, m = 3, 4, 3
    P = np.zeros((n_dim, n, m))

# First row
    P[:, 0, 0] = [0.00, 0.00, 0.00]
    P[:, 1, 0] = [0.25, 0.00, 0.25]
    P[:, 2, 0] = [0.75, 0.00, 0.25]
    P[:, 3, 0] = [1.0, 0.00, 0.00]

# Second row
    P[:, 0, 1] = [0.00, 0.25, 0.25]
    P[:, 1, 1] = [0.25, 0.25, 0.75]
    P[:, 2, 1] = [0.75, 0.25, 0.75]
    P[:, 3, 1] = [1.0, 0.25, 0.25]

# Third row
    P[:, 0, 2] = [0.00, 0.75, 0.00]
    P[:, 1, 2] = [0.25, 0.75, 0.25]
    P[:, 2, 2] = [0.75, 0.75, 0.25]
    P[:, 3, 2] = [1.0, 0.75, 0.00]

# %%

# %%
    PP = np.random.rand(1,4,3)
# %%
# Create and plot the Bezier surface
# %%
    XX= 128
    N=128
    u = np.linspace(0,1,N)
    uv = np.meshgrid(u,u,u)
    uv2 = [a.reshape(-1) for a in uv]
    from datetime import datetime
    start = datetime.now()
    bezierSurface = nrb.NurbsSurface(control_points=P)
    w = torch.tensor(bezierSurface.get_value(*uv2))
    w = w.reshape(3,128,128).permute(1,2,0).to(torch.float32)
# %%
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    uu = torch.stack([torch.tensor(aa) for aa in uv])
    ww = w.permute(2,0,1)
    ax.plot_surface(*ww)
# %%
    ax.plot_surface(*uu,ww[2,:,:])
# %%


# %%
    bezierSurface.plot(control_points=True, isocurves_u=6, isocurves_v=6)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    bezierSurface.plot_control_points(fig,ax)
    ww = torch.stack(a)
    ww = ww-ww.max()/2
    ww=ww.unsqueeze(0)
# %%
   
# %%
    ww[2,:,:]= ww[2,:,:]*-1 # inverting
    ww=  ww+ww.min()

# %%
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(*ww)

# %%
    aa = a.unsqueeze(0).unsqueeze(0)

    grid = get_affine_grid(aa.shape,scale_ranges=[1.,1.])
    g = grid+ww
    aaa= F.grid_sample(aa,grid)
    aa2= F.grid_sample(aa,g)
    aa3= F.grid_sample(aa,ww)
    ImageMaskViewer([aa[0,0],b])
    ImageMaskViewer([aaa[0,0],aa[0,0]])
    ImageMaskViewer([aaa[0,0],aa3[0,0]])
# %%
# %%

    ww_multiplier = ww[2,:,:]
    ww_multiplier=ww_multiplier-ww_multiplier.min()
    ww_multiplier=ww_multiplier/(ww_multiplier.max())
    ww2= torch.einsum('ij,hijkl->hijkl',ww_multiplier,base_mesh3d_big)

# %%
    outp2 = F.grid_sample(img.unsqueeze(0).unsqueeze(0),ww2)

    ImageMaskViewer([outp2[0,0],img])
    # ax3 = plt.figure().add_subplot(projection='2d')
    # ax3.quiver(*base_mesh3d.squeeze(0).permute(3,0,1,2),*base_mesh3d.squeeze(0).permute(3,0,1,2),length=0.1,normalize=True)

# %%
    import matplotlib.pyplot as plt
    from matplotlib import cm
    import numpy as np
    X = np.arange(-5, 5, 0.25)
    Y = np.arange(-5, 5, 0.25)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X**2 + Y**2)
    Z = np.sin(R)

# Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
# %%

    alpha = 0.8
    dists  = torch.zeros(y.shape)
# %%
    p_rand = base_mesh[0,20,30,:]
    v = torch.tensor([0.12,0.73])
    v_normed = v/v.norm()
    line = p_rand-v
    for i in range(base_mesh.shape[1]):
        for j in range(base_mesh.shape[2]):
              p_ij = base_mesh[0,i,j,:]
              dists[i,j] =               ((p_ij-p_rand)-(torch.dot(p_ij-p_rand,v_normed))*v_normed).norm()

# %%
    weights = weight_curve(dists,alpha)
    weights2 = weight_fold(dists,alpha)

# %%
    for i in range(base_mesh.shape[1]):
# %%
        for j in range(base_mesh.shape[2]):
            ff_def[0,i,j,:] = base_mesh[0,i,j,:]+v*weights[i,j]
# %%
# %%
    n=20
    grads2= base_mesh.clone()
    out2 = F.grid_sample(a2,grads.unsqueeze(0))
    plt.imshow(outp[0,0])
    plt.imshow(out2[0,0])
# %%
    ff_def = base_mesh.clone()
    for i in range(base_mesh.shape[1]):
        for j in range(base_mesh.shape[2]):
            ff_def[0,i,j,:] = base_mesh[0,i,j,:]+v*weights2[i,j]
# %%
    outp2 = F.grid_sample(a,ff_def)
    plt.imshow(outp2[0,0])
# %%
    plt.figure()
# %%
    img,mask = a.clone(), b.clone()
    img.unsqueeze_(0).unsqueeze_(0)

    mask.unsqueeze_(0).unsqueeze_(0)

# %%
    plt.quiver(base_mesh[0,:,:,0],base_mesh[0,:,:,1],ff_def[0,:,:,0],ff_def[0,:,:,1])
    plt.figure()
    plt.quiver(base_mesh[0,:,:,0],base_mesh[0,:,:,1],grads2[0,:,:,0]*5,grads2[0,:,:,1]*5)
# %%

    sz = [56,56,56]
    img = F.interpolate(img, size=sz, mode='trilinear')
    mask = F.interpolate(mask, size=sz, mode='nearest')
    ImageMaskViewer([img[0,0],mask[0,0]])

# %%
######################################################################################
# %% [markdown]
## Cubic b-spline
# %%

    def compute_basis_function (l: int, vox_idx, tile_dim):
        u = vox_idx / tile_dim

        if l==0:
            B = (1.0/6.0) * (- 1.0 * u*u*u + 3.0 * u*u - 3.0 * u + 1.0)
        elif l==1: 
            B = (1.0/6.0) * (+ 3.0 * u*u*u - 6.0 * u*u+ 4.0)
        elif l==2:
            B = (1.0/6.0) * (- 3.0 * u*u*u + 3.0 * u*u + 3.0 * u + 1.0)
        elif l==3:
            B = (1.0/6.0) * (+ 1.0 * u*u*u)
        else:
            B = 0.0;
        return B;
# %%

    tile_dims=[6,6,6]
    LUT_Bspline_x,LUT_Bspline_y,LUT_Bspline_z=[np.zeros((aa*4)) for aa in tile_dims]
    for  j in  range(4):
             for i in range(tile_dims[0]):
                LUT_Bspline_x [ j*tile_dims[0] + i] = compute_basis_function ( j, i, tile_dims[0])
             for i in range(tile_dims[1]):
                LUT_Bspline_y [ j*tile_dims[1] + i] = compute_basis_function ( j, i, tile_dims[1]);
             for i in range(tile_dims[2]):
                LUT_Bspline_z [ j*tile_dims[2] + i] = compute_basis_function ( j, i, tile_dims[2]);

# %%

