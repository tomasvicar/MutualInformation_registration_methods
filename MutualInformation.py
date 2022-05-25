import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn

from sklearn.metrics import normalized_mutual_info_score

from skimage import data
from skimage.transform import rotate


# implementation based on https://github.com/connorlee77/pytorch-mutual-information/blob/master/MutualInformation.py


class MutualInformation(nn.Module):
    KERNEL_GAUSS = 0
    KERNEL_BSPLINE = 1
    def __init__(self, num_bins=256, sample=None, kernel=KERNEL_GAUSS, sigma=0.4):
        super().__init__() 
        self.num_bins = num_bins
        self.epsilon = 1e-10
        self.bins = nn.Parameter(torch.linspace(0, 255, num_bins).float(), requires_grad=False)
        self.sample = sample
        self.sigma = 2*sigma**2
        self.kernel = kernel
        
        
    def marginalPdf(self, values):
        
 
        
        residuals = values.unsqueeze(0) - self.bins.unsqueeze(1)
        
        if self.kernel == self.KERNEL_BSPLINE:
            residuals = torch.abs(residuals)
            kernel_values = torch.zeros_like(residuals)	
            kernel_values[residuals <= 1] = 1 - 3/2 * residuals[residuals <= 1]**2 + 3/4 * residuals[residuals <= 1]**3
            kernel_values[(residuals > 1) & (residuals <= 2)] = 1/4 * (2 - residuals[(residuals > 1) & (residuals <= 2)])**3
        elif self.kernel == self.KERNEL_GAUSS:
            kernel_values = torch.exp(-0.5*(residuals / self.sigma) ** 2)
        else:
            assert('wrong kernel')
        
        
        pdf = torch.mean(kernel_values, dim=1)
        normalization = torch.sum(pdf) + self.epsilon
        pdf = pdf / normalization
        
        return pdf, kernel_values


    def jointPdf(self, kernel_values1, kernel_values2):
        joint_kernel_values = kernel_values1 @ kernel_values2.T
        normalization = torch.sum(joint_kernel_values) + self.epsilon
        pdf = joint_kernel_values / normalization
        
        return pdf


    def getMutualInformation(self, input1, input2):
		
        input1 = input1*255
        input2 = input2*255
        
        assert((input1.shape == input2.shape))

        x1 = input1.flatten()
        x2 = input2.flatten()
        
        if self.sample != None:
            ind_perm = torch.randperm(x1.size(0))[:self.sample]
            x1 = x1[ind_perm]
            x2 = x2[ind_perm]
		
        pdf_x1, kernel_values1 = self.marginalPdf(x1)
        pdf_x2, kernel_values2 = self.marginalPdf(x2)
        pdf_x1x2 = self.jointPdf(kernel_values1, kernel_values2)

        H_x1 = -torch.sum(pdf_x1*torch.log2(pdf_x1 + self.epsilon))
        H_x2 = -torch.sum(pdf_x2*torch.log2(pdf_x2 + self.epsilon))
        H_x1x2 = -torch.sum(pdf_x1x2*torch.log2(pdf_x1x2 + self.epsilon))
        mutual_information = H_x1 + H_x2 - H_x1x2
		
        mutual_information = 2*mutual_information/(H_x1+H_x2)

        return mutual_information


    def forward(self, input1, input2):  
        return self.getMutualInformation(input1, input2)



if __name__ == '__main__':
	
    device = 'cuda:0'

    ### Create test cases ###
    img1 = data.camera().astype(np.float32)/255
    img1 = img1[::2,::2]
    
    img2 = rotate(img1,5,preserve_range=True)
    img3 = rotate(img1,15,preserve_range=True)

	
    mi_true_1 = normalized_mutual_info_score(img1.ravel(), img1.ravel())
    mi_true_2 = normalized_mutual_info_score(img1.ravel(), img2.ravel())
    mi_true_3 = normalized_mutual_info_score(img1.ravel(), img3.ravel())

    img1 = torch.from_numpy(img1).to(device)
    img2 = torch.from_numpy(img2).to(device)
    img3 = torch.from_numpy(img3).to(device)



    # MI = MutualInformation(num_bins=64, sample=20000, kernel=MutualInformation.KERNEL_GAUSS, sigma=0.4).to(device)
    # MI = MutualInformation(num_bins=256, sample=None, kernel=MutualInformation.KERNEL_BSPLINE).to(device)
    mi_test_1 = MI(img1, img1).detach().cpu().numpy()
    mi_test_2 = MI(img1, img2).detach().cpu().numpy()
    mi_test_3 = MI(img1, img3).detach().cpu().numpy()


    print('Image Pair 1 | sklearn MI: {}, this MI: {}'.format(mi_true_1, mi_test_1))
    print('Image Pair 2 | sklearn MI: {}, this MI: {}'.format(mi_true_2, mi_test_2))
    print('Image Pair 3 | sklearn MI: {}, this MI: {}'.format(mi_true_3, mi_test_3))

    
