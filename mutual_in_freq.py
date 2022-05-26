import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.fft


from skimage.io import imread
from skimage.transform import AffineTransform, warp, rotate

from config import Config
from get_overlay import get_overlay


def float_compare(A, c):
    return torch.clamp(1-torch.abs(A-c), 0.0)

def compute_entropy(C, N, eps=1e-7):
    p = C/N
    return p*torch.log2(torch.clamp(p, min=eps, max=None))


eps=1e-7

config = Config()

fixed = imread('knee1.png')


# moving = imread('knee2.png')
# transform = AffineTransform(translation=[-10,-10])
# moving = warp(moving, transform, mode='wrap', preserve_range=True)
# # moving = rotate(moving,7,preserve_range=True)
# moving = moving.astype(fixed.dtype)





transform = AffineTransform(translation=[-20,-20])
moving = warp(fixed, transform, mode='wrap', preserve_range=True)
# moving = rotate(moving,45,preserve_range=True)
moving = moving.astype(fixed.dtype)



fixed = fixed.astype(np.float32)/255
moving = moving.astype(np.float32)/255


fixed = fixed[::2,::2]
moving = moving[::2,::2]

# fixed = fixed[1:-1,1:-1]
# moving = moving[1:-1,1:-1]
# fixed = fixed[7:-7,7:-7]
# moving = moving[7:-7,7:-7]
fixed = fixed[15:-15,12:-12]
moving = moving[7:-7,6:-6]





A = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(fixed),0),1).to(config.device)
B = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(moving),0),1).to(config.device)


M_A = torch.ones_like(A)
M_B = torch.ones_like(B)


Q_A = 255 // 3
Q_B = 255 // 3


overlap = 0.8


shape_req = (np.array([max(x,y) for x,y in zip(A.shape,B.shape)]) * (1 +(1.0-overlap)))[2:]
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
 


M_A_fft = torch.fft.rfft2(M_A)

A_ffts = []
for a in range(Q_A):
    A_ffts.append(torch.fft.rfft2(float_compare(A, a)))



M_B_fft = torch.conj(torch.fft.rfft2(M_B))

B_ffts= []
for b in range(Q_B):
    B_ffts.append(torch.conj(torch.fft.rfft2(float_compare(B, b))))


N = torch.fft.fftshift(torch.fft.irfft2(M_A_fft * M_B_fft))






H_A = torch.zeros_like(A)
for a in range(Q_A):
    C_A_a = torch.round(torch.fft.fftshift(torch.fft.irfft2(A_ffts[a] * M_B_fft)))
    H_A = H_A - compute_entropy(C_A_a,N)

H_B = torch.zeros_like(A)
for b in range(Q_B):
    C_B_b = torch.round(torch.fft.fftshift(torch.fft.irfft2(M_A_fft * B_ffts[b])))
    H_B = H_B - compute_entropy(C_B_b,N)

H_AB = torch.zeros_like(A)
for a in range(Q_A):
    for b in range(Q_B):
        C_AB_ab = torch.round(torch.fft.fftshift(torch.fft.irfft2(A_ffts[a] * B_ffts[b])))
        H_AB = H_AB - compute_entropy(C_AB_ab,N)
        
        

MI = H_A + H_B - H_AB

MI = MI.detach().cpu().numpy()[0,0,:,:]
MI = MI[MI.shape[0]//2-pad_sz_A[0]:MI.shape[0]//2+pad_sz_A[0],MI.shape[1]//2-pad_sz_A[1]:MI.shape[1]//2+pad_sz_A[1]]


max_pos = np.unravel_index(np.argmax(MI), MI.shape)
pos = np.array(max_pos) - (np.array(MI.shape) / 2) 




plt.imshow(MI)
plt.show()
print(pos)
