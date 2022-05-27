import numpy as np
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
import torch
from skimage.transform import rotate

from Cmif import Cmif




class Wrapper(object):
    def __init__(self, fixed, moving, overlap, device, bins):
        self.fixed = fixed
        self.moving = moving
        self.overlap = overlap
        self.device = device
        self.bins = bins
        
        self.it = -1
        self.MI_max_poss = []
        self.MI_map = []
        
    def __call__(self, **params):
        
        self.it = self.it + 1
        
        rot = params['rot']
        cmif = Cmif(self.overlap, self.device, self.bins)
        
        MI_max_pos, MI_max, MI_map = cmif.get_transormation(fixed,moving,rot_deg=rot)
        
        self.MI_max_poss.append(MI_max_pos)
        self.MI_map.append(MI_map)
        
        return MI_max


class BayesOptimizedRotationCmif():
    
    
    def __init__(self,fixed, moving, init_points, n_iter, rotation_range=[-10,10], overlap=0.8, device=torch.device('cuda:0'), bins=50):
        
        self.fixed = fixed
        self.moving = moving
        self.init_points = init_points
        self.n_iter = n_iter
        self.rotation_range = rotation_range
        self.overlap = overlap
        self.device = device
        self.bins = bins
    
    def optimize(self):
        
        wrapper = Wrapper(self.fixed, self.moving, self.overlap, self.device, self.bins)
        
        pbounds = dict()
        pbounds['rot'] = self.rotation_range
        
        optimizer = BayesianOptimization(f=wrapper,pbounds=pbounds,random_state=42) 
        
        optimizer.maximize(init_points=self.init_points,n_iter=self.n_iter)
        
        
        best_iter = np.argmax([x['target'] for x in optimizer.res])
        
        best_shift = wrapper.MI_max_poss[best_iter]
        
        best_rot = optimizer.max['params']['rot']
        
        return best_shift, best_rot
        






if __name__ == '__main__':
    
    from skimage.io import imread
    from skimage.transform import AffineTransform, warp
    from get_overlay import get_overlay
	
    device = 'cuda:0'
    
    fixed = imread('knee1.png')


    moving = imread('knee2.png')
    transform = AffineTransform(translation=[-10,-10])
    moving = warp(moving, transform, mode='wrap', preserve_range=True)
    moving = rotate(moving,9,preserve_range=True)
    moving = moving.astype(fixed.dtype)


    
    # transform = AffineTransform(translation=[-10,-10])
    # moving = warp(fixed, transform, mode='wrap', preserve_range=True)
    # moving = rotate(moving,7.5,preserve_range=True)
    # moving = moving.astype(fixed.dtype)



    fixed = fixed.astype(np.float32)/255
    moving = moving.astype(np.float32)/255

    fixed = fixed[::2,::2]
    moving = moving[::2,::2]
    
    
    bayesOptimizedRotationCmif = BayesOptimizedRotationCmif(fixed, moving, init_points=5, n_iter=50, rotation_range=[-10,10], overlap=0.8, device=device, bins=128)
    
    best_shift, best_rot = bayesOptimizedRotationCmif.optimize()
    
    print(best_shift, best_rot)
    
    corrected = rotate(moving,best_rot)
    transform = AffineTransform(translation=best_shift)
    corrected = warp(corrected, transform, mode='wrap', preserve_range=True)
    


    plt.imshow(get_overlay(fixed, moving))
    plt.show()
    
    plt.imshow(get_overlay(fixed, corrected))
    plt.show()








