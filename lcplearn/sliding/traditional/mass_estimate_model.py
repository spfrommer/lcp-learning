import numpy as np
from argparse import ArgumentParser
import pdb

import torch
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F

from lcplearn import lcp_utils
from . import dynamics
    
def load_data(path):
    data = dynamics.unmarshal_data(np.load(path)) 
    states = torch.from_numpy(np.vstack((
                data.xdots, data.us, data.poslambdas,
                data.neglambdas, data.gammas)).transpose())
    ys = torch.from_numpy(data.next_xdots)

    states, ys = Variable(states), Variable(ys)
    return states, ys, data

def learning_setup():
    model = MassEstimateNet()
    loss = structured_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                    [90, 150, 250], gamma=0.3)

    return model, loss, optimizer, scheduler

def structured_loss(net_out, next_xdots, states, net):
    lcp_slack = net_out
    lambdas = states[:, 2:5]

    comp_term = torch.norm(lambdas * lcp_slack, 2)
    nonneg_term = torch.norm(torch.clamp(-lcp_slack, min=0), 2)

    loss = 1 * comp_term + 1 * nonneg_term 

    return loss

class MassEstimateNet(torch.nn.Module):
    def __init__(self):
        super(MassEstimateNet, self).__init__()

        self.f = torch.nn.Linear(2, 3, bias=False)
        self.f_bias = torch.nn.Parameter(torch.ones(3))
        self.G = torch.nn.Linear(2, 9, bias=False)
        self.G_bias = torch.nn.Parameter(torch.ones(9))

        # Correct dynamics solution
        self.G.weight.data.fill_(0)
        self.G_bias = Parameter(torch.tensor([1, -1, 1,
                                             -1,  1, 1,
                                             -1, -1, 0]).float())

        self.f.weight = Parameter(torch.tensor([[1,  1],
                                                [-1, -1],
                                                [0,  0]]).float())

        self.f_bias = Parameter(torch.tensor([0, 0, 20]).float())

        self.G.weight.requires_grad = False
        self.G_bias.requires_grad = False
        self.f.weight.requires_grad = False

    def forward(self, states):
        lambdas = states[:, 2:5]
        xus = states[:, 0:2]
        fxu = self.f(xus) + self.f_bias
        
        Gxu = (self.G(xus) + self.G_bias).view(-1, 3, 3)
    
        lcp_slack = fxu + torch.bmm(Gxu,lambdas.unsqueeze(2)).squeeze(2)
        
        return lcp_slack

if __name__ == "__main__": main()
