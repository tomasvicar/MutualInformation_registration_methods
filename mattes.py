import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from get_overlay import get_overlay

from skimage.io import imread
from skimage.transform import AffineTransform, warp, rotate

from config import Config
from MutualInformation import MutualInformation

config = Config()

fixed = imread('knee1.png')


moving = imread('knee2.png')
transform = AffineTransform(translation=[-10,-10])
moving = warp(moving, transform, mode='wrap', preserve_range=True)
moving = rotate(moving,7,preserve_range=True)
moving = moving.astype(fixed.dtype)


# transform = AffineTransform(translation=[5,5])
# moving = warp(fixed, transform, mode='wrap', preserve_range=True)
# moving = rotate(moving,3)
# moving = moving.astype(fixed.dtype)





fixed = fixed.astype(config.np_dtype)/255 
moving = moving.astype(config.np_dtype)/255 


fixed = fixed[::2,::2]
moving = moving[::2,::2]

# moving = moving + 0.05 * np.random.randn(moving.shape[0],moving.shape[1]).astype(config.np_dtype)




tx = torch.zeros(1).to(config.device)
ty = torch.zeros(1).to(config.device)
angle_rad = torch.zeros(1).to(config.device)

tx.requires_grad = True
ty.requires_grad = True
angle_rad.requires_grad = True

params = [tx,ty,angle_rad]



optimizer = torch.optim.Adam(params,lr=config.init_lr,amsgrad=True)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, config.iterations, gamma=config.gamma, last_epoch=-1)



MI = MutualInformation(num_bins=128, sample=20000, kernel=MutualInformation.KERNEL_BSPLINE).to(config.device)


img = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(moving),0),1).to(config.device)
img_ref = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(fixed),0),1).to(config.device)


losses = []
for it in range(config.iterations[-1]):

    theta = torch.zeros(1, 3, 3).to(config.device)
    
    theta[:,0,0] = torch.cos(angle_rad)
    theta[:,1,1] = torch.cos(angle_rad)
    theta[:,0,1] = torch.sin(angle_rad)
    theta[:,1,0] = -torch.sin(angle_rad)
    
    theta[:,0,2] = tx
    theta[:,1,2] = ty
    
    
    grid = F.affine_grid(theta[:,0:2,:], img.shape, align_corners=config.align_corners)
    output = F.grid_sample(img, grid, padding_mode="zeros", align_corners=config.align_corners, mode=config.interp_mode)
    
    
    # loss = torch.mean((img_ref - output) ** 2)
    
    loss = -MI(img_ref,output)
    
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    
    losses.append(loss.detach().cpu().numpy())
    
    if (it % 20) == 0:
        print(losses[-1])
        plt.plot(losses)
        plt.show()
        
        plt.imshow(get_overlay(fixed, output[0,0,:,:].detach().cpu().numpy()))
        plt.show()
    


plt.imshow(get_overlay(fixed, moving))
plt.show()


plt.imshow(get_overlay(fixed, output[0,0,:,:].detach().cpu().numpy()))
plt.show()







