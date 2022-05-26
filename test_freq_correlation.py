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


transform = AffineTransform(translation=[50,50])
moving = warp(fixed, transform, mode='wrap', preserve_range=True)
# moving = rotate(moving,3)
moving = moving.astype(fixed.dtype)


fixed = (fixed // 3).astype(np.float32)
moving = (moving // 3).astype(np.float32)


fixed = fixed[::2,::2]
moving = moving[::2,::2]



A = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(moving),0),1).to(config.device)
B = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(fixed),0),1).to(config.device)

overlap = 0.2


partial_overlap_pad_sz = (round(A.shape[-1]*(1.0-overlap)), round(A.shape[-2]*(1.0-overlap)))
A =  F.pad(A, (partial_overlap_pad_sz[0], partial_overlap_pad_sz[0], partial_overlap_pad_sz[1], partial_overlap_pad_sz[1]))

partial_overlap_pad_sz = (round(B.shape[-1]*(1.0-overlap)), round(B.shape[-2]*(1.0-overlap)))
B =  F.pad(B, (partial_overlap_pad_sz[0], partial_overlap_pad_sz[0], partial_overlap_pad_sz[1], partial_overlap_pad_sz[1]))


AB = torch.fft.fftshift(torch.fft.irfft2(torch.fft.rfft2(A) * torch.conj(torch.fft.rfft2(B))))


A = A.detach().cpu().numpy()[0,0,:,:]
B = B.detach().cpu().numpy()[0,0,:,:]
AB = AB.detach().cpu().numpy()[0,0,:,:]

plt.imshow(A)
plt.show()


plt.imshow(B)
plt.show()


plt.imshow(AB)
plt.show()



max_pos = np.unravel_index(np.argmax(AB), AB.shape)

pos = np.array(max_pos) - (np.array(AB.shape) / 2)

print(pos)



