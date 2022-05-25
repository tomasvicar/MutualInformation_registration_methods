import numpy as np
import torch


class Config:
    
    device = torch.device('cuda:0')
    
    np_dtype = np.float32
    torch_dtype = torch.float32
    
    init_lr = 1e-3
    init_lr_mine = 1e-6
    
    # iterations = [600,700,750]
    # iterations = [400,500,550]
    # iterations = [100,150,175]
    # iterations = [500,750,1000]
    iterations = [1000,1250,1500]
    # iterations = [5000,6000,6500]
    
    interp_mode = 'bilinear'
    
    align_corners = False
    
    gamma = 0.1