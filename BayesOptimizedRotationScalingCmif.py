import numpy as np
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
import torch
from skimage.transform import rotate, AffineTransform, warp
from bayes_opt.event import Events
from itertools import accumulate
from bayes_opt import SequentialDomainReductionTransformer

from Cmif import Cmif


class BayesPloter():
    
    def update(self, event, instance):
        
        if event == Events.OPTIMIZATION_STEP:
            
            values = [x['target'] for x in instance.res]
            cummax = list(accumulate(values, max))
            plt.plot(cummax)
            plt.plot(values)
            plt.title(str(cummax[-1]))
            plt.show()
            
            



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
        scale = params['scale']
        cmif = Cmif(self.overlap, self.device, self.bins)
        
        fixed = self.fixed
        moving = self.moving
        
        transform = AffineTransform(scale=scale)
        moving = warp(moving, transform, mode='wrap', preserve_range=True)
        
        MI_max_pos, MI_max, MI_map = cmif.get_transormation(fixed,moving,rot_deg=rot)
        
        self.MI_max_poss.append(MI_max_pos)
        self.MI_map.append(MI_map)
        
        return MI_max


class BayesOptimizedRotationScalingCmif():
    
    
    def __init__(self,fixed, moving, init_points, n_iter, rotation_range=[-10,10], scaling_range=[0.95, 1.05], overlap=0.8, device=torch.device('cuda:0'), bins=50, bounds_transformer=SequentialDomainReductionTransformer()):
        
        self.fixed = fixed
        self.moving = moving
        self.init_points = init_points
        self.n_iter = n_iter
        self.rotation_range = rotation_range
        self.scaling_range = scaling_range
        self.overlap = overlap
        self.device = device
        self.bins = bins
        self.optimizer = None
        self.bounds_transformer = bounds_transformer
    
    def optimize(self):
        
        wrapper = Wrapper(self.fixed, self.moving, self.overlap, self.device, self.bins, )
        
        pbounds = dict()
        pbounds['rot'] = self.rotation_range
        pbounds['scale'] = self.scaling_range
        
        
        self.optimizer = BayesianOptimization(f=wrapper,pbounds=pbounds,random_state=42, bounds_transformer=self.bounds_transformer) 

        
        self.optimizer.subscribe(Events.OPTIMIZATION_STEP, BayesPloter())

        
        self.optimizer.maximize(init_points=self.init_points,n_iter=self.n_iter)
        

        
        
        best_iter = np.argmax([x['target'] for x in self.optimizer.res])
        
        best_shift = wrapper.MI_max_poss[best_iter]
        
        best_rot = self.optimizer.max['params']['rot']
        best_scale = self.optimizer.max['params']['scale']
        
        return best_shift, best_rot, best_scale
        






if __name__ == '__main__':
    
    from skimage.io import imread
    from get_overlay import get_overlay
	
    device = 'cuda:0'
    
    fixed = imread('knee1.png')
    
    moving = imread('knee2.png')
    # moving = fixed.copy()
    
    ##  make example faster with smaller image
    fixed = fixed[::2,::2]
    moving = moving[::2,::2]

    
    transform = AffineTransform(translation=[-8,-8])
    moving = warp(moving, transform, mode='wrap', preserve_range=True)
    moving = rotate(moving,5,preserve_range=True)
    transform = AffineTransform(scale=0.93)
    moving = warp(moving, transform, mode='wrap', preserve_range=True)
    moving = moving.astype(fixed.dtype)



    fixed = fixed.astype(np.float32)/255
    moving = moving.astype(np.float32)/255


    
    bayesOptimizedRotationScalingCmif = BayesOptimizedRotationScalingCmif(fixed, moving, init_points=5, n_iter=200, rotation_range=[-10,10], scaling_range=[0.90, 1.1], overlap=0.8, device=device, bins=50, bounds_transformer=SequentialDomainReductionTransformer())


    best_shift, best_rot, best_scale = bayesOptimizedRotationScalingCmif.optimize()
    
    print(best_shift, best_rot, best_scale)
    print(bayesOptimizedRotationScalingCmif.optimizer.max['target'])
    
    transform = AffineTransform(scale=best_scale)
    moving = warp(moving, transform, mode='wrap', preserve_range=True)
    corrected = rotate(moving,best_rot, preserve_range=True)
    transform = AffineTransform(translation=best_shift)
    corrected = warp(corrected, transform, mode='wrap', preserve_range=True)
    


    plt.imshow(get_overlay(fixed, moving))
    plt.show()
    
    plt.imshow(get_overlay(fixed, corrected))
    plt.show()








