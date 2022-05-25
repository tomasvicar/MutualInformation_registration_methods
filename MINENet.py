import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm as wn

# code based on https://github.com/abnan/DRMIME/blob/master/03_MultiModal_Registration_Affine.ipynb

nChannel = 1
n_neurons = 100

sample = 100000


class Multiply(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, tensor):
    return 0.1 * tensor


class MINENet(nn.Module): 
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2*nChannel, n_neurons),
            nn.ReLU(),
            # nn.Dropout(p=0.5),
            # nn.BatchNorm1d(n_neurons),
            nn.Linear(n_neurons, n_neurons),
            nn.ReLU(),
            # nn.Dropout(p=0.5),
            # nn.BatchNorm1d(n_neurons),
            nn.Linear(n_neurons, 1),
            # nn.Sigmoid(),
            # Multiply(),
            # nn.Tanh(),
        )
        

    def forward(self, x, y):
        
        x = x.permute(0,2,3,1)
        y = y.permute(0,2,3,1)
        
        x = x.view(x.size()[1]*x.size()[2],x.size()[3])
        y = y.view(y.size()[1]*y.size()[2],y.size()[3])
        
        perm = torch.randperm(x.size(0))[:sample]
        x = x[perm,:]
        y = y[perm,:]
        
        ind_perm = torch.randperm(x.size(0))
        
        z1 = self.layers(torch.cat((x,y),1))
        z2 = self.layers(torch.cat((x,y[ind_perm,:]),1))
        MI = torch.mean(z1) - torch.log(torch.mean(torch.exp(z2)))
        # MI = torch.mean(z1) -torch.mean(z2)

        return MI