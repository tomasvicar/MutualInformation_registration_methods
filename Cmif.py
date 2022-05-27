import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.fft
from skimage.transform import rotate
import time

class Cmif():
    
    eps = 1e-7
    
    def __init__(self, overlap, device, bins=50):
        
        self.overlap = overlap
        self.bins = bins
        self.device = device
        self.MI_map = None
        
    def float_compare(self, A, c):
        return torch.clamp(1-torch.abs(A-c), 0.0)

    def compute_entropy(self, C, N, eps=1e-7):
        p = C/N
        return p*torch.log2(torch.clamp(p, min=eps, max=None))

    
    def get_transormation(self,fixed, moving, rot_deg=0):
        
        
        moving = rotate(moving,rot_deg,preserve_range=True)
        
        # expect input 0-1
        fixed = np.round(fixed * self.bins).astype(np.float32)
        moving = np.round(moving * self.bins).astype(np.float32)
        
        
        Q_A = self.bins
        Q_B = self.bins
        
        A = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(fixed),0),1).to(self.device)
        B = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(moving),0),1).to(self.device)


        M_A = torch.ones_like(A)
        M_B = torch.ones_like(B) 
        

        shape_req = (np.array([max(x,y) for x,y in zip(A.shape,B.shape)]) * (1 +(1.0-self.overlap)))[2:]
        pad_sz_A = (shape_req - np.array(A.shape[2:])) // 2
        pad_sz_A = pad_sz_A.astype(int) 

        A =  F.pad(A, (pad_sz_A[1], pad_sz_A[1], pad_sz_A[0], pad_sz_A[0]), mode='constant', value=Q_A + 1)
        M_A = F.pad(M_A, (pad_sz_A[1], pad_sz_A[1], pad_sz_A[0], pad_sz_A[0]), mode='constant', value=0)


        ### solve odd size difference 
        pad_sz_B = (np.array(A.shape[2:]) - np.array(B.shape[2:])) / 2
        pad_sz_B_1 = pad_sz_B.copy()
        pad_sz_B_2 = pad_sz_B.copy()
        pad_sz_B_1[(pad_sz_B % 2) != 0] = np.floor(pad_sz_B_1[(pad_sz_B % 2) != 0])
        pad_sz_B_2[(pad_sz_B % 2) != 0] = np.ceil(pad_sz_B_2[(pad_sz_B % 2) != 0])
        pad_sz_B_1 = pad_sz_B_1.astype(int)
        pad_sz_B_2 = pad_sz_B_2.astype(int)

        B =  F.pad(B, (pad_sz_B_1[1], pad_sz_B_2[1], pad_sz_B_1[0], pad_sz_B_2[0]), mode='constant', value=Q_B + 1)
        M_B = F.pad(M_B, (pad_sz_B_1[1], pad_sz_B_2[1], pad_sz_B_1[0], pad_sz_B_2[0]), mode='constant', value=0)
         

        A[M_A == 0] = Q_A + 1
        B[M_B == 0] = Q_B + 1
        

        M_A_fft = torch.fft.rfft2(M_A)
        M_B_fft = torch.conj(torch.fft.rfft2(M_B))
        N = torch.fft.fftshift(torch.fft.irfft2(M_A_fft * M_B_fft))


    
        ### batchwise implementation - it is not faster...
        # A_ffts = []
        # for a in range(Q_A):
        #     A_ffts.append(self.float_compare(A, a))
        # A_ffts = torch.fft.rfft2(torch.cat(A_ffts,0))

        # B_ffts= []
        # for b in range(Q_B):
        #     B_ffts.append(self.float_compare(B, b))
        # B_ffts = torch.conj(torch.fft.rfft2(torch.cat(B_ffts,0)))
        

        # C_A_a = torch.round(torch.fft.fftshift(torch.fft.irfft2(A_ffts * M_B_fft)))
        # H_A = -torch.sum(self.compute_entropy(C_A_a,N),dim=0)
        
        # C_B_b = torch.round(torch.fft.fftshift(torch.fft.irfft2(M_A_fft * B_ffts)))
        # H_B = -torch.sum(self.compute_entropy(C_B_b,N),dim=0)
        
        # H_AB = torch.zeros_like(A)
        # for a in range(Q_A):
        #     C_AB_ab = torch.round(torch.fft.fftshift(torch.fft.irfft2(A_ffts[[a],:,:,:] * B_ffts)))
        #     H_AB = H_AB - torch.sum(self.compute_entropy(C_AB_ab,N),dim=0)
        

        
        
        
        #### iterative implementation -  same speed but nicer
        A_ffts = []
        for a in range(Q_A):
            A_ffts.append(torch.fft.rfft2(self.float_compare(A, a)))


        B_ffts= []
        for b in range(Q_B):
            B_ffts.append(torch.conj(torch.fft.rfft2(self.float_compare(B, b))))
        
        H_A = torch.zeros_like(A)
        for a in range(Q_A):
            C_A_a = torch.round(torch.fft.fftshift(torch.fft.irfft2(A_ffts[a] * M_B_fft)))
            H_A = H_A - self.compute_entropy(C_A_a,N)

        H_B = torch.zeros_like(A)
        for b in range(Q_B):
            C_B_b = torch.round(torch.fft.fftshift(torch.fft.irfft2(M_A_fft * B_ffts[b])))
            H_B = H_B - self.compute_entropy(C_B_b,N)

        H_AB = torch.zeros_like(A)
        for a in range(Q_A):
            for b in range(Q_B):
                C_AB_ab = torch.round(torch.fft.fftshift(torch.fft.irfft2(A_ffts[a] * B_ffts[b])))
                H_AB = H_AB - self.compute_entropy(C_AB_ab,N)
                
                
        MI = H_A + H_B - H_AB

        MI = MI.detach().cpu().numpy()[0,0,:,:]
        MI = MI[MI.shape[0]//2-pad_sz_A[0]:MI.shape[0]//2+pad_sz_A[0],MI.shape[1]//2-pad_sz_A[1]:MI.shape[1]//2+pad_sz_A[1]]

        MI_max = np.max(MI)
        
        MI_max_pos = np.unravel_index(np.argmax(MI), MI.shape)
        MI_max_pos = np.array(MI_max_pos) - (np.array(MI.shape) / 2) 
        MI_max_pos = -MI_max_pos
        MI_map = MI

        return MI_max_pos, MI_max, MI_map
        
        
    
if __name__ == '__main__':
    
    from skimage.io import imread
    from skimage.transform import AffineTransform, warp
    
	
    device = 'cuda:0'
    
    fixed = imread('knee1.png')


    moving = imread('knee2.png')
    transform = AffineTransform(translation=[-10,-10])
    moving = warp(moving, transform, mode='wrap', preserve_range=True)
    # moving = rotate(moving,7,preserve_range=True)
    moving = moving.astype(fixed.dtype)


    # transform = AffineTransform(translation=[-20,-20])
    # moving = warp(fixed, transform, mode='wrap', preserve_range=True)
    # # moving = rotate(moving,3)
    # moving = moving.astype(fixed.dtype)



    fixed = fixed.astype(np.float32)/255
    moving = moving.astype(np.float32)/255

    fixed = fixed[::2,::2]
    moving = moving[::2,::2]
    
    cmif = Cmif(overlap=0.8, device=device, bins=50)
    
    
    MI_max_pos, MI_max, MI_map = cmif.get_transormation(fixed,moving)
    
    
    plt.imshow(MI_map)
    plt.show()
    print(MI_max_pos)
    
    
    
    
    
    
    
    
    

    

