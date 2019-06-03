import numpy as np
from argparse import ArgumentParser
import pdb

import torch
from torch.autograd import Variable
import torch.nn.functional as F

from lcplearn import lcp_utils
from . import dynamics
    
def load_data(path):
    data = dynamics.unmarshal_data(np.load(path)) 
    states = torch.from_numpy(np.vstack((
                data.xdots, data.us, data.lambdas)).transpose())
    ys = lcp_utils.split_sign(torch.from_numpy(data.next_xdots),
                    pos_first=True)

    states, ys = Variable(states.float()), Variable(ys.float())
    return states, ys, data

def learning_setup():
    model = StructuredNet()
    loss = structured_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    return model, loss, optimizer, None

def structured_loss(next_xdots_pred, next_xdots, states, _):
    lambdas = lcp_utils.split_sign(states[:, 2], pos_first=True)
    
    err_term = torch.norm(next_xdots_pred - next_xdots, 2)
    comp_term = torch.norm(next_xdots_pred * lambdas, 1)
    nonneg_term = torch.norm(torch.max(
        torch.zeros(next_xdots_pred.shape), -next_xdots_pred), 1)
    
    w = torch.tensor([1, 1, 0])

    return w[0] * err_term + w[1] * comp_term + w[2] * nonneg_term

class StructuredNet(torch.nn.Module):
    def __init__(self):
        super(StructuredNet, self).__init__()
        self.f = torch.nn.Linear(2, 2, bias=True)
        self.G = torch.nn.Linear(2, 4, bias=True)

    def forward(self, states):
        # An nx2 array with first col neg and second col pos components
        lambdas = lcp_utils.split_sign(states[:, 2], pos_first=True)
        xus = states[:, 0:2]
        fxu = self.f(xus)
        Gxu = self.G(xus).view(-1, 2, 2)
        
        xdots = fxu + torch.bmm(Gxu, lambdas.unsqueeze(2)).squeeze(2) 

        #return xdots
        return F.relu(xdots)

if __name__ == "__main__": main()
