import torch
import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(-3,3,100)
residuals = np.abs(x - 0)

residuals = np.abs(residuals)
kernel_values = np.zeros_like(residuals)	
kernel_values[residuals <= 1] = 1 - 3/2 * residuals[residuals <= 1]**2 + 3/4 * residuals[residuals <= 1]**3
kernel_values[(residuals > 1) & (residuals <= 2)] = 1/4 * (2 - residuals[(residuals > 1) & (residuals <= 2)])**3


plt.plot(x,kernel_values)
